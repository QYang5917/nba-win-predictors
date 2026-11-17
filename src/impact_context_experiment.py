"""Fixed context additions to existing out-of-time rating snapshots."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from context_experiment import build_context, CONTEXT, paired_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/impact_context'
BASE = ['recent_normalized_DIFF','REST_DAYS_DIFF']
ARMS = {'rating': BASE, 'rating_context': BASE+CONTEXT,
        'conservative_context':['impact_120_DIFF','REST_DAYS_DIFF']+CONTEXT}


def predict(games):
    chunks, folds = [], []
    for season in ['2023-24','2024-25','2025-26']:
        tr, te = games[games.SEASON<season], games[games.SEASON==season].copy()
        if tr.empty or te.empty or tr.GAME_DATE.max()>=te.GAME_DATE.min():
            raise ValueError('Invalid chronological split')
        fold = dict(season=season, train_end=str(tr.GAME_DATE.max()), train_games=len(tr))
        for arm,cols in ARMS.items():
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000))
            model.fit(tr[cols],tr.WIN)
            te['P_'+arm]=model.predict_proba(te[cols])[:,1]
            fold[arm]=dict(zip(cols,model[-1].coef_[0].tolist()))
        chunks.append(te); folds.append(fold)
    return pd.concat(chunks,ignore_index=True),folds


def main():
    inputs=['reports/impact_control/predictions.csv','data/raw/team_game_logs.csv',
            'src/impact_context_experiment.py','src/context_experiment.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs}
    ratings=pd.read_csv(ROOT/inputs[0],dtype={'GAME_ID':str})
    raw=pd.read_csv(ROOT/inputs[1],dtype={'GAME_ID':str})
    raw['GAME_ID']=raw.GAME_ID.str.zfill(10)
    counts=raw.assign(HOME=raw.MATCHUP.str.contains('vs.',regex=False)).groupby('GAME_ID').HOME.agg(['sum','count'])
    valid=counts.index[(counts['sum']==1)&(counts['count']==2)]
    context=build_context(raw[raw.GAME_ID.isin(valid)])
    games=ratings.merge(context,on='GAME_ID',validate='one_to_one')
    games['GAME_DATE']=pd.to_datetime(games.GAME_DATE)
    pred,folds=predict(games)
    metrics=[]
    for season in ['ALL']+sorted(pred.SEASON.unique()):
        d=pred if season=='ALL' else pred[pred.SEASON==season]
        for arm in ARMS:
            p=d['P_'+arm]
            metrics.append(dict(season=season,arm=arm,games=len(d),
                accuracy=float(((p>=.5)==d.WIN).mean()), log_loss=float(log_loss(d.WIN,p)),
                brier=float(brier_score_loss(d.WIN,p))))
    comparisons=[paired_loss(pred,'P_'+arm,'P_rating') for arm in ARMS if arm!='rating']
    OUT.mkdir(parents=True,exist_ok=True)
    pred.to_csv(OUT/'predictions.csv',index=False)
    report=dict(sha256=hashes,arms=ARMS,folds=folds,metrics=metrics,comparisons=comparisons,
        limitations=['Historical periods reused; exploratory comparison, not untouched forward evidence.',
            'Training data begins in 2021-22; compare only this experiment baseline.',
            'Team context does not identify individual player impact.',
            'Date bootstrap excludes cross-date dependence and model-selection uncertainty.',
            'No betting-profit test or deployment.'])
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(pd.DataFrame(metrics).to_string(index=False))
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    main()
