"""V2 chronological model research: classifiers, margins, and a fixed blend.

Only consumes past-only rating artifacts and pregame histories. No live writes.
"""
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from context_experiment import CONTEXT, build_context, paired_loss
from style_experiment import style_features, GROUPS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/v2_research'
BASE = ['recent_normalized_DIFF','REST_DAYS_DIFF']
STYLE = [c for v in GROUPS.values() for c in v]
FULL = BASE + CONTEXT + [c+s for c in STYLE for s in ['_DIFF','_H','_A']] + [
    'recent_normalized_H','recent_normalized_A','EXPECTED_MIN_H','EXPECTED_MIN_A']
CONFIGS = [dict(max_leaf_nodes=7,min_samples_leaf=80,l2_regularization=10.,max_iter=150),
           dict(max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=30.,max_iter=150),
           dict(max_leaf_nodes=7,min_samples_leaf=150,l2_regularization=30.,max_iter=250)]
MODELS = ['baseline','linear_full','boost_classifier','boost_margin','fixed_blend']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset():
    """Load immutable upstream recipe; record hashes of every actual cache input."""
    hashes = {}
    def read(path):
        path = Path(path)
        content=path.read_bytes()
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(content).hexdigest()
        return pd.read_csv(io.BytesIO(content),dtype={'GAME_ID':str})
    source=ROOT/'reports/impact_control/summary.json'
    hashes[str(source.relative_to(ROOT))]=digest(source)
    meta=json.loads(source.read_text())
    for name in ['data/raw/team_game_logs.csv','data/raw/player_game_logs.csv.gz','data/raw/player_ages.csv']:
        if digest(ROOT/name)!=meta['sha256'][name]:
            raise ValueError(f'Upstream rating cache no longer matches {name}; rebuild impact-control first')
    cache=ROOT/'reports/impact_control/cache'/meta['historical_recipe']
    paths=sorted(cache.glob('teams_*.csv'))
    expected={f'{y}-{str(y+1)[-2:]}' for y in range(2018,2026)}
    if {p.stem.removeprefix('teams_') for p in paths}!=expected:
        raise ValueError('Missing historical rating seasons')
    teams=pd.concat([read(p) for p in paths],ignore_index=True)
    raw=read(ROOT/'data/raw/team_game_logs.csv')
    raw['GAME_ID']=raw.GAME_ID.str.zfill(10)
    raw['HOME']=raw.MATCHUP.str.contains('vs.',regex=False)
    counts=raw.groupby('GAME_ID').HOME.agg(['sum','count'])
    bad=counts.index[(counts['sum']!=1)|(counts['count']!=2)]
    raw=raw[~raw.GAME_ID.isin(bad)].sort_values(['TEAM_ID','GAME_DATE']).copy()
    if raw.duplicated(['TEAM_ID','GAME_DATE']).any():
        raise ValueError('Multiple team games on one date')
    raw['REST_DAYS']=raw.groupby(['TEAM_ID','SEASON']).GAME_DATE.transform(
        lambda s:pd.to_datetime(s).diff().dt.days.clip(upper=5).fillna(5))
    parts=[]
    for home,suffix in [(True,'_H'),(False,'_A')]:
        r=raw[raw.HOME==home][['GAME_ID','TEAM_ID','GAME_DATE','SEASON','PTS','REST_DAYS']]
        r=r.merge(teams[['GAME_ID','TEAM_ID','recent_normalized','EXPECTED_MIN']],on=['GAME_ID','TEAM_ID'],validate='one_to_one')
        if not home: r=r.drop(columns=['GAME_DATE','SEASON'])
        parts.append(r.rename(columns={c:c+suffix for c in r.columns if c not in ['GAME_ID','GAME_DATE','SEASON']}))
    games=parts[0].merge(parts[1],on='GAME_ID',validate='one_to_one')
    games['WIN']=(games.PTS_H>games.PTS_A).astype(int)
    games['MARGIN']=games.PTS_H-games.PTS_A
    games['recent_normalized_DIFF']=games.recent_normalized_H-games.recent_normalized_A
    games['REST_DAYS_DIFF']=games.REST_DAYS_H-games.REST_DAYS_A
    games=games.merge(build_context(raw),on='GAME_ID',validate='one_to_one')
    games=games.merge(style_features(raw,include_sides=True),on='GAME_ID',validate='one_to_one')
    games=games.sort_values(['GAME_DATE','GAME_ID']).reset_index(drop=True)
    if not np.isfinite(games[FULL+['MARGIN','WIN']]).all().all():
        raise ValueError('Incomplete V2 features')
    return games,dict(sha256=hashes,upstream_recipe=meta['historical_recipe'],excluded_home_away_ids=bad.tolist())


def estimator(kind,config):
    cls=HistGradientBoostingClassifier if kind=='classifier' else HistGradientBoostingRegressor
    return cls(**config,learning_rate=.04,early_stopping=False,random_state=42)


def choose(train,features,kind,configs=CONFIGS):
    """Choose on the last PAST season; test labels are not an input here."""
    validation_season=max(train.SEASON)
    fit=train[train.SEASON<validation_season]
    val=train[train.SEASON==validation_season]
    if fit.empty or val.empty or fit.GAME_DATE.max()>=val.GAME_DATE.min():
        raise ValueError('Invalid inner chronological split')
    candidates=[]
    target='WIN' if kind=='classifier' else 'MARGIN'
    for config in configs:
        model=estimator(kind,config).fit(fit[features],fit[target])
        sigma=None
        if kind=='classifier':
            p=model.predict_proba(val[features])[:,1]
        else:
            margin=model.predict(val[features])
            sigma=float(np.sqrt(np.mean((val.MARGIN-margin)**2)).clip(8,30))
            p=ndtr(margin/sigma)
        candidates.append(dict(config=config,sigma=sigma,validation_log_loss=float(log_loss(val.WIN,p,labels=[0,1]))))
    best=min(candidates,key=lambda d:d['validation_log_loss'])
    model=estimator(kind,best['config']).fit(train[features],train[target])
    return model,dict(selected=best,candidates=candidates,validation_season=validation_season,
                      inner_train_end=str(fit.GAME_DATE.max()),validation_start=str(val.GAME_DATE.min()),
                      validation_end=str(val.GAME_DATE.max()),refit_end=str(train.GAME_DATE.max()))


def fit_fold(train,test,configs=CONFIGS):
    if train.empty or test.empty or train.GAME_DATE.max()>=test.GAME_DATE.min():
        raise ValueError('Training overlaps test dates')
    result=test.copy()
    models={}
    for name,features in [('baseline',BASE),('linear_full',FULL)]:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000))
        model.fit(train[features],train.WIN)
        result['P_'+name]=model.predict_proba(test[features])[:,1]
        models[name]=model
    audit={}
    for kind in ['classifier','margin']:
        model,details=choose(train,FULL,kind,configs)
        p=model.predict_proba(test[FULL])[:,1] if kind=='classifier' else ndtr(
            model.predict(test[FULL])/details['selected']['sigma'])
        name='boost_'+kind
        result['P_'+name]=p
        models[name]=model
        audit[name]=details
    result['P_fixed_blend']=.5*result.P_linear_full+.5*result.P_boost_classifier
    return result,models,audit


def evaluate(games,configs=CONFIGS):
    chunks,folds=[],[]
    for season in ['2023-24','2024-25','2025-26']:
        train=games[games.SEASON<season]
        test=games[games.SEASON==season]
        result,_,audit=fit_fold(train,test,configs)
        chunks.append(result)
        folds.append(dict(season=season,train_games=len(train),test_games=len(test),
                          train_end=str(train.GAME_DATE.max()),test_start=str(test.GAME_DATE.min()),selection=audit))
    return pd.concat(chunks,ignore_index=True),folds


def metrics_table(pred):
    rows=[]
    for season in ['ALL']+sorted(pred.SEASON.unique()):
        part=pred if season=='ALL' else pred[pred.SEASON==season]
        for model in MODELS:
            p=part['P_'+model]
            # Fixed equal-width reliability bins; descriptive, not a calibration fit.
            b=pd.DataFrame({'p':p,'y':part.WIN,'bin':pd.cut(p,np.linspace(0,1,11),include_lowest=True)})
            bins=b.groupby('bin',observed=True).agg(p=('p','mean'),y=('y','mean'),n=('y','size'))
            ece=float(((bins.p-bins.y).abs()*bins.n).sum()/len(part))
            rows.append(dict(season=season,model=model,games=len(part),correct=int(((p>=.5)==part.WIN).sum()),
                accuracy=float(((p>=.5)==part.WIN).mean()),log_loss=float(log_loss(part.WIN,p)),
                brier=float(brier_score_loss(part.WIN,p)),ece10=ece))
    return rows


def paired_accuracy(pred,left,right):
    """Resample calendar weeks to retain some within-week game dependence."""
    delta=((pred[left]>=.5)==pred.WIN).astype(int)-((pred[right]>=.5)==pred.WIN).astype(int)
    blocks=pd.DataFrame({'week':pd.to_datetime(pred.GAME_DATE).dt.to_period('W').astype(str),
                         'delta':delta}).groupby('week').delta.agg(['sum','count'])
    rng=np.random.default_rng(42)
    samples=blocks.to_numpy()[rng.integers(len(blocks),size=(2000,len(blocks)))].sum(axis=1)
    return dict(left=left,right=right,accuracy_delta=float(delta.mean()),
                extra_correct=int(delta.sum()),week_bootstrap_95=np.quantile(samples[:,0]/samples[:,1],[.025,.975]).tolist(),
                limitation='Calendar-week bootstrap omits longer dependence, training uncertainty and model selection.')


def predict_bundle(bundle,features):
    """Research inference for dates strictly after the training history.

    Caller must provide verified pregame feature snapshots. Calendar-date checks
    alone do not establish when inputs actually became available.
    """
    required=['GAME_ID','GAME_DATE']+bundle['features']
    if not set(required)<=set(features.columns):
        raise ValueError('Missing V2 features or game metadata')
    dates=pd.to_datetime(features.GAME_DATE,errors='coerce')
    if dates.isna().any() or (dates<=pd.Timestamp(bundle['trained_through'])).any():
        raise ValueError('Research inference must follow the training period')
    if features.GAME_ID.duplicated().any() or not np.isfinite(features[bundle['features']]).all().all():
        raise ValueError('Duplicate games or nonfinite V2 inputs')
    result=features[['GAME_ID','GAME_DATE']].copy()
    for name,model in bundle['models'].items():
        cols=bundle['baseline_features'] if name=='baseline' else bundle['features']
        if name=='boost_margin':
            sigma=bundle['forward_selection'][name]['selected']['sigma']
            p=ndtr(model.predict(features[cols])/sigma)
        else:
            p=model.predict_proba(features[cols])[:,1]
        result['P_'+name]=p
    weight=bundle['blend_weight']
    result['P_fixed_blend']=weight*result.P_linear_full+(1-weight)*result.P_boost_classifier
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        games,provenance=dataset()
        print(f'V2 dataset: {len(games)} games, {len(FULL)} features',flush=True)
        predictions,folds=evaluate(games)
        metrics=metrics_table(predictions)
        comparisons=[paired_loss(predictions,'P_'+m,'P_baseline') for m in MODELS if m!='baseline']
        comparisons.append(paired_loss(predictions,'P_boost_classifier','P_linear_full'))
        # Research bundle: trained on completed history for future shadow evaluation.
        # The bundle is not wired into production prediction logs.
        models={}
        selection={}
        for name,features in [('baseline',BASE),('linear_full',FULL)]:
            models[name]=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000)).fit(games[features],games.WIN)
        for kind in ['classifier','margin']:
            model,audit=choose(games,FULL,kind)
            models['boost_'+kind]=model;selection['boost_'+kind]=audit
        bundle=dict(models=models,features=FULL,baseline_features=BASE,forward_selection=selection,
            trained_through=str(games.GAME_DATE.max()),mode='research_only',blend_weight=.5,
            units='win probability; margin target in points',schema_version=1)
        joblib.dump(bundle,OUT/'research_bundle.joblib')
    for p in ['src/v2_research.py','src/style_experiment.py','src/context_experiment.py']:
        provenance['sha256'][p]=digest(ROOT/p)
    artifact_hashes={name:digest(OUT/name) for name in ['research_bundle.joblib']}
    report=dict(**provenance,artifact_sha256=artifact_hashes,features=FULL,configs=CONFIGS,folds=folds,
        models=MODELS,metrics=metrics,comparisons=comparisons,forward_selection=selection,
        accuracy_comparisons=[paired_accuracy(predictions,'P_'+m,'P_baseline') for m in MODELS if m!='baseline'],
        limitations=['Historical evaluation seasons have been repeatedly inspected; not untouched forward validation.',
            'Existing ratings use full-game minutes, not possession-level lineup attribution.',
            'Candidates come from previous games; no verified historical injury snapshots.',
            'Inner margin sigma is fitted on its validation season; selection is optimistic within that past season.',
            'Day bootstrap ignores across-date dependence, training uncertainty and multiple comparisons.',
            'No betting returns evaluated; research bundle not deployed.',
            'NBA paint/turnover scoring and individual tracking defense inputs remain unavailable.'])
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    games.to_csv(OUT/'features.csv.gz',index=False,compression='gzip')
    predictions.to_csv(OUT/'predictions.csv',index=False)
    print(pd.DataFrame(metrics).to_string(index=False))
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    main()
