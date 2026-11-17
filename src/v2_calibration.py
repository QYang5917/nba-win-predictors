"""Past-only out-of-fold calibration and blending of V2 research models."""
import hashlib
import io
import json

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from threadpoolctl import threadpool_limits

from v2_research import ROOT, FULL, CONFIGS, dataset, fit_fold, paired_accuracy, digest
from context_experiment import paired_loss

OUT=ROOT/'reports/v2_calibration'
ARMS=['baseline','linear_full','fixed_blend','platt_linear','learned_blend','stacked']


def stack_features(pred):
    return np.column_stack([logit(np.clip(pred[c],1e-6,1-1e-6))
                            for c in ['P_linear_full','P_boost_classifier']])


def fit_meta(past):
    """Only accepts genuinely out-of-fold predictions from earlier dates."""
    if past.empty or past.WIN.nunique()!=2:
        raise ValueError('Insufficient calibration history')
    if not np.isfinite(past[['P_linear_full','P_boost_classifier']]).all().all():
        raise ValueError('Invalid calibration probabilities')
    if not past[['P_linear_full','P_boost_classifier']].apply(lambda s:s.between(0,1)).all().all():
        raise ValueError('Calibration probabilities outside [0,1]')
    x=stack_features(past)
    platt=LogisticRegression(C=1.,max_iter=2000).fit(x[:,:1],past.WIN)
    stacked=LogisticRegression(C=.1,max_iter=2000).fit(x,past.WIN)
    def objective(w):
        return log_loss(past.WIN,w*past.P_linear_full+(1-w)*past.P_boost_classifier)
    optimized=minimize_scalar(objective,bounds=(0.,1.),method='bounded')
    if not optimized.success:
        raise RuntimeError('Blend fitting failed')
    # Include exact boundaries: the booster need not receive nonzero weight.
    weight=min([0.,1.,float(optimized.x)],key=objective)
    return dict(platt=platt,stacked=stacked,linear_weight=weight,
                trained_through=str(past.GAME_DATE.max()),rows=len(past))


def apply_meta(meta,pred):
    dates=pd.to_datetime(pred.GAME_DATE,errors='coerce')
    if dates.isna().any() or (dates<=pd.Timestamp(meta['trained_through'])).any():
        raise ValueError('Calibration data overlaps prediction dates')
    x=stack_features(pred)
    d=pred.copy()
    d['P_platt_linear']=meta['platt'].predict_proba(x[:,:1])[:,1]
    d['P_stacked']=meta['stacked'].predict_proba(x)[:,1]
    w=meta['linear_weight']
    d['P_learned_blend']=w*d.P_linear_full+(1-w)*d.P_boost_classifier
    return d


def audit_meta(meta):
    return dict(rows=meta['rows'],trained_through=meta['trained_through'],linear_weight=meta['linear_weight'],
        platt_coef=meta['platt'].coef_[0].tolist(),platt_intercept=meta['platt'].intercept_.tolist(),
        stacked_coef=meta['stacked'].coef_[0].tolist(),stacked_intercept=meta['stacked'].intercept_.tolist())


def evaluate(oof):
    chunks,folds=[],[]
    for season in ['2023-24','2024-25','2025-26']:
        past=oof[oof.SEASON<season]
        test=oof[oof.SEASON==season]
        if past.empty or test.empty:
            raise ValueError('Missing chronological calibration fold')
        meta=fit_meta(past)
        chunks.append(apply_meta(meta,test))
        folds.append(dict(season=season,test_start=str(test.GAME_DATE.min()),**audit_meta(meta)))
    return pd.concat(chunks,ignore_index=True),folds


def metrics(pred):
    rows=[]
    for season in ['ALL']+sorted(pred.SEASON.unique()):
        d=pred if season=='ALL' else pred[pred.SEASON==season]
        for arm in ARMS:
            p=d['P_'+arm]
            table=pd.DataFrame({'p':p,'y':d.WIN,'bin':pd.cut(p,np.linspace(0,1,11),include_lowest=True)})
            bins=table.groupby('bin',observed=True).agg(p=('p','mean'),y=('y','mean'),n=('y','size'))
            rows.append(dict(season=season,model=arm,games=len(d),correct=int(((p>=.5)==d.WIN).sum()),
                accuracy=float(((p>=.5)==d.WIN).mean()),log_loss=float(log_loss(d.WIN,p)),
                brier=float(brier_score_loss(d.WIN,p)),ece10=float(((bins.p-bins.y).abs()*bins.n).sum()/len(d))))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        games,provenance=dataset()
        for name in ['src/v2_research.py','src/v2_calibration.py','src/style_experiment.py','src/context_experiment.py']:
            provenance['sha256'][name]=digest(ROOT/name)
        recipe=hashlib.sha256(json.dumps([provenance,CONFIGS,FULL],sort_keys=True).encode()).hexdigest()
        cache=OUT/'cache'/recipe
        cache.mkdir(parents=True,exist_ok=True)
        chunks,base_folds=[],[]
        for season in sorted(games.SEASON.unique()):
            if season<'2020-21': continue
            path=cache/f'oof_{season}.csv'
            audit_path=cache/f'audit_{season}.json'
            if path.exists() and audit_path.exists():
                result=pd.read_csv(io.BytesIO(path.read_bytes()),dtype={'GAME_ID':str})
                audit=json.loads(audit_path.read_text())
            else:
                train,test=games[games.SEASON<season],games[games.SEASON==season]
                result,_,selection=fit_fold(train,test)
                audit=dict(season=season,train_end=str(train.GAME_DATE.max()),test_start=str(test.GAME_DATE.min()),selection=selection)
                result.to_csv(path,index=False)
                audit_path.write_text(json.dumps(audit,indent=2)+'\n')
            chunks.append(result);base_folds.append(audit)
            print(f'{season}: {len(result)} out-of-fold predictions',flush=True)
        oof=pd.concat(chunks,ignore_index=True)
        pred,folds=evaluate(oof)
        table=metrics(pred)
        comparisons=[]
        for arm in ['platt_linear','learned_blend','stacked']:
            for comparator in ['baseline','linear_full','fixed_blend']:
                comparisons.append(dict(log_loss=paired_loss(pred,'P_'+arm,'P_'+comparator),
                                        accuracy=paired_accuracy(pred,'P_'+arm,'P_'+comparator)))
        meta=fit_meta(oof)
        # Calibration bundle references the V2 artifact by hash; base models
        # remain frozen separately. Both are needed for research inference.
        base_path=ROOT/'reports/v2_research/research_bundle.joblib'
        bundle=dict(meta=meta,base_bundle_sha256=digest(base_path),schema_version=1,mode='research_only')
        joblib.dump(bundle,OUT/'calibration_bundle.joblib')
    pred.to_csv(OUT/'predictions.csv',index=False)
    oof.to_csv(OUT/'oof_predictions.csv.gz',index=False,compression='gzip')
    report=dict(**provenance,recipe=recipe,base_folds=base_folds,calibration_folds=folds,
        metrics=table,comparisons=comparisons,forward_meta=audit_meta(meta),
        base_bundle_sha256=bundle['base_bundle_sha256'],
        limitations=['Repeated historical research, not untouched forward evidence.',
            'Calibration only sees past out-of-fold predictions; earlier evaluation seasons enter later training.',
            'Blend maximizes past probability quality, not hit rate.',
            'Date/week bootstrap omits longer dependence, training and selection uncertainty.',
            'No executable odds, profit test, live feature integration or deployment.'])
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(pd.DataFrame(table).to_string(index=False))


if __name__=='__main__':
    main()
