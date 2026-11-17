"""Nested chronological selection of recency weights for V2 linear models."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
from v2_research import ROOT, FULL, dataset, digest, paired_accuracy
from context_experiment import paired_loss

OUT=ROOT/'reports/v2_recency'
CONFIGS=[dict(half_life_days=h,C=c) for h in [None,365.,730.] for c in [.1,1.]]


def weights(dates,half_life):
    dates=pd.to_datetime(dates,errors='raise')
    if dates.isna().any() or not len(dates):
        raise ValueError('Invalid training dates')
    if half_life is None: return np.ones(len(dates))
    if not np.isfinite(half_life) or half_life<=0: raise ValueError('Invalid half life')
    w=np.exp2(-(dates.max()-dates).dt.total_seconds().to_numpy()/86400/half_life)
    return w*len(w)/w.sum()


def fit(train,config):
    w=weights(train.GAME_DATE,config['half_life_days'])
    model=make_pipeline(StandardScaler(),LogisticRegression(C=config['C'],max_iter=2000))
    model.fit(train[FULL],train.WIN,standardscaler__sample_weight=w,logisticregression__sample_weight=w)
    return model


def select(train,configs=CONFIGS):
    last=max(train.SEASON)
    early,val=train[train.SEASON<last],train[train.SEASON==last]
    if early.empty or val.empty or early.GAME_DATE.max()>=val.GAME_DATE.min():
        raise ValueError('Invalid recency validation fold')
    candidates=[]
    for config in configs:
        model=fit(early,config)
        p=model.predict_proba(val[FULL])[:,1]
        candidates.append(dict(config=config,validation_loss=float(log_loss(val.WIN,p))))
    selected=min(candidates,key=lambda x:x['validation_loss'])
    model=fit(train,selected['config'])
    return model,dict(selected=selected,candidates=candidates,validation_season=last,
                      validation_end=str(val.GAME_DATE.max()),trained_through=str(train.GAME_DATE.max()))


def predict_fold(train,test):
    if test.empty or train.empty or train.GAME_DATE.max()>=test.GAME_DATE.min():
        raise ValueError('Recency train/test overlap')
    model,audit=select(train)
    result=test[['GAME_ID','GAME_DATE','SEASON','WIN']].copy()
    result['P_recency']=model.predict_proba(test[FULL])[:,1]
    return result,audit


def predict_bundle(bundle,features):
    cols=bundle['features']
    if not set(cols+['GAME_ID','GAME_DATE'])<=set(features.columns):
        raise ValueError('Missing recency-model features')
    dates=pd.to_datetime(features.GAME_DATE,errors='coerce')
    cutoff=pd.Timestamp(bundle['selection']['trained_through'])
    if dates.isna().any() or (dates<=cutoff).any():
        raise ValueError('Recency inference overlaps training')
    if features.GAME_ID.duplicated().any() or not np.isfinite(features[cols]).all().all():
        raise ValueError('Invalid recency inference input')
    return bundle['model'].predict_proba(features[cols])[:,1]


def main():
    games,provenance=dataset()
    chunks,folds=[],[]
    for season in ['2023-24','2024-25','2025-26']:
        pred,audit=predict_fold(games[games.SEASON<season],games[games.SEASON==season])
        chunks.append(pred);folds.append(dict(season=season,**audit))
    pred=pd.concat(chunks,ignore_index=True)
    previous=ROOT/'reports/v2_research/predictions.csv'
    ref=pd.read_csv(previous,dtype={'GAME_ID':str})
    if set(pred.GAME_ID)!=set(ref.GAME_ID): raise ValueError('Comparison game set changed')
    pred=pred.merge(ref[['GAME_ID','P_baseline','P_linear_full','P_fixed_blend']],on='GAME_ID',validate='one_to_one')
    rows=[]
    for season in ['ALL']+sorted(pred.SEASON.unique()):
        d=pred if season=='ALL' else pred[pred.SEASON==season]
        for name in ['baseline','linear_full','fixed_blend','recency']:
            p=d['P_'+name]
            rows.append(dict(season=season,model=name,games=len(d),correct=int(((p>=.5)==d.WIN).sum()),
                accuracy=float(((p>=.5)==d.WIN).mean()),log_loss=float(log_loss(d.WIN,p)),brier=float(brier_score_loss(d.WIN,p))))
    model,audit=select(games)
    OUT.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(model=model,features=FULL,selection=audit,mode='research_only'),OUT/'recency_bundle.joblib')
    pred.to_csv(OUT/'predictions.csv',index=False)
    for name in ['src/v2_recency.py','src/v2_research.py','src/style_experiment.py','src/context_experiment.py']:
        provenance['sha256'][name]=digest(ROOT/name)
    provenance['sha256'][str(previous.relative_to(ROOT))]=digest(previous)
    report=dict(**provenance,configs=CONFIGS,folds=folds,metrics=rows,forward_selection=audit,
        comparisons=[dict(log_loss=paired_loss(pred,'P_recency','P_'+name),accuracy=paired_accuracy(pred,'P_recency','P_'+name))
                     for name in ['baseline','linear_full','fixed_blend']],
        limitations=['Repeated historical research; no untouched forward holdout.',
            'Only six configurations selected using the most recent past season.',
            'Normalized sample weights change historical relevance, not total loss scale.',
            'Date/week bootstrap omits training, selection and longer dependence.',
            'Research bundle not deployed; live feature integration remains pending.'])
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))
    print('Selected:',[(f['season'],f['selected']) for f in folds])


if __name__=='__main__':
    main()
