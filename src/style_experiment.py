"""Test available user-requested scoring/rebounding/defense features, past games only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from context_experiment import paired_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/style_experiment'
GROUPS = {'three_scoring': ['THREE_PTS100'], 'rebounds': ['OREB_RATE', 'DREB_RATE'],
          'opponent_three': ['OPP_THREE_PCT']}
BASE = ['recent_normalized_DIFF', 'REST_DAYS_DIFF']
ARMS = {'player': BASE, **{k: BASE + [c+'_DIFF' for c in v] for k,v in GROUPS.items()},
        'all_available': BASE + [c+'_DIFF' for v in GROUPS.values() for c in v]}


def style_features(raw, include_sides=False):
    d = raw.copy()
    d['HOME'] = d.MATCHUP.str.contains('vs.', regex=False)
    counts = d.groupby('GAME_ID').HOME.agg(['sum', 'count'])
    good = counts.index[(counts['sum'] == 1) & (counts['count'] == 2)]
    d = d[d.GAME_ID.isin(good)]
    if d.duplicated(['TEAM_ID', 'GAME_DATE']).any():
        raise ValueError('Multiple games for one team-date')
    cols = ['FG3M', 'FG3A', 'FGA', 'FTA', 'OREB', 'DREB', 'TOV']
    opp = d[['GAME_ID', 'TEAM_ID'] + cols].rename(columns={c:'OPP_'+c for c in ['TEAM_ID']+cols})
    d = d.merge(opp, on='GAME_ID')
    d = d[d.TEAM_ID != d.OPP_TEAM_ID].sort_values(['TEAM_ID', 'GAME_DATE']).copy()
    d['POSS'] = d.FGA + .44*d.FTA - d.OREB + d.TOV
    # Sum attempts before computing percentages; do not average individual game percentages.
    sums = ['FG3M', 'POSS', 'OREB', 'DREB', 'OPP_OREB', 'OPP_DREB', 'OPP_FG3M', 'OPP_FG3A']
    history = d.groupby('TEAM_ID')[sums].transform(lambda s:s.shift(1).rolling(20, min_periods=10).sum())
    d['THREE_PTS100'] = 300*history.FG3M/history.POSS
    d['OREB_RATE'] = history.OREB/(history.OREB+history.OPP_DREB)
    d['DREB_RATE'] = history.DREB/(history.DREB+history.OPP_OREB)
    d['OPP_THREE_PCT'] = history.OPP_FG3M/history.OPP_FG3A.replace(0, np.nan)
    features = [c for group in GROUPS.values() for c in group]
    h, a = [d[d.HOME == home].set_index('GAME_ID')[features] for home in [True, False]]
    result = (h-a).rename(columns=lambda c:c+'_DIFF').dropna().reset_index()
    if include_sides:
        for side, suffix in [(h,'_H'),(a,'_A')]:
            result = result.merge(side.add_suffix(suffix).reset_index(),on='GAME_ID',validate='one_to_one')
    return result


def predictions(games):
    out = []
    for season in ['2023-24', '2024-25', '2025-26']:
        tr, te = games[games.SEASON < season], games[games.SEASON == season].copy()
        if tr.empty or te.empty or tr.GAME_DATE.max() >= te.GAME_DATE.min():
            raise ValueError('Invalid chronological split')
        for name, cols in ARMS.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000))
            model.fit(tr[cols], tr.WIN)
            te['P_'+name] = model.predict_proba(te[cols])[:,1]
        out.append(te)
    return pd.concat(out, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = ['data/raw/team_game_logs.csv', 'reports/rating_experiment/predictions.csv',
             'src/style_experiment.py', 'src/context_experiment.py']
    hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    raw = pd.read_csv(ROOT/paths[0], dtype={'GAME_ID':str})
    rated = pd.read_csv(ROOT/paths[1], dtype={'GAME_ID':str})
    games = rated[['GAME_ID','GAME_DATE','SEASON','WIN']+BASE].merge(style_features(raw), on='GAME_ID', validate='one_to_one')
    pred = predictions(games)
    metrics = []
    for season in ['ALL']+sorted(pred.SEASON.unique()):
        part = pred if season=='ALL' else pred[pred.SEASON==season]
        for arm in ARMS:
            p = part['P_'+arm]
            metrics.append(dict(season=season, arm=arm, games=len(part),
                accuracy=float(((p>=.5)==part.WIN).mean()), log_loss=float(log_loss(part.WIN,p)),
                brier=float(brier_score_loss(part.WIN,p))))
    comparisons = [paired_loss(pred,'P_'+arm,'P_player') for arm in ARMS if arm!='player']
    report = dict(sha256=hashes, arms=ARMS, metrics=metrics, comparisons=comparisons,
        training_start=str(games.GAME_DATE.min()), window=20, minimum_history=10,
        unavailable=['paint points','points off turnovers','opponent paint FG percentage'],
        limitations=['Reused historical evaluation; multiple exploratory comparisons.',
            'Team features are not individual player defensive ratings.',
            'Opponent three-point percentage is not adjusted for shooter quality or shot openness.',
            'Date bootstrap ignores dependence across dates and model selection.',
            'Training starts in 2021-22, so compare against this experiment baseline only.'])
    pred.to_csv(OUT/'predictions.csv', index=False)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(pd.DataFrame(metrics).to_string(index=False))
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    main()
