"""Compare rating algorithms with identical pregame candidates and minutes."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ratings import fit_age_curve, REPLACEMENT
from build_player_strength import expected_minutes, season_box_model, REFRESH_DAYS
from context_experiment import paired_loss
from rating_research import ARMS, HALF_LIFE_DAYS, rating_variants

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/rating_experiment'


def candidates_from_past(logs):
    pg = expected_minutes(logs)
    tg = pg[['GAME_ID', 'TEAM_ID', 'GAME_DATE', 'SEASON']].drop_duplicates()
    if tg.duplicated(['TEAM_ID', 'GAME_DATE']).any():
        raise ValueError('Multiple games per team-date')
    tg = tg.sort_values(['TEAM_ID', 'SEASON', 'GAME_DATE'])
    tg['NEXT_ID'] = tg.groupby(['TEAM_ID', 'SEASON']).GAME_ID.shift(-1)
    tg['NEXT_DATE'] = tg.groupby(['TEAM_ID', 'SEASON']).GAME_DATE.shift(-1)
    # Only previous game's identities and expected minutes enter a prediction.
    c = pg[['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'EXP_AFTER']].merge(tg, on=['GAME_ID', 'TEAM_ID'])
    c = c.dropna(subset=['NEXT_ID']).copy()
    if not (c.GAME_DATE < c.NEXT_DATE).all():
        raise ValueError('Noncausal candidates')
    c['GAME_ID'], c['GAME_DATE'] = c.NEXT_ID, c.NEXT_DATE
    c = c.sort_values(['GAME_ID', 'TEAM_ID', 'EXP_AFTER'], ascending=[True, True, False])
    used = c.groupby(['GAME_ID', 'TEAM_ID']).EXP_AFTER.cumsum() - c.EXP_AFTER
    c['EXPECTED_MIN'] = np.clip(240 - used, 0, c.EXP_AFTER)
    return c[['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'SEASON', 'GAME_DATE', 'EXPECTED_MIN']]


def predict_wins(games, arms=ARMS):
    predictions, folds = [], []
    for season in sorted(games.SEASON.unique()):
        if season < '2021-22':
            continue
        tr, te = games[games.SEASON < season], games[games.SEASON == season].copy()
        if tr.empty or tr.GAME_DATE.max() >= te.GAME_DATE.min():
            raise ValueError('Invalid chronological split')
        for arm in arms:
            cols = [arm + '_DIFF', 'REST_DAYS_DIFF']
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000))
            model.fit(tr[cols], tr.WIN)
            te['P_' + arm] = model.predict_proba(te[cols])[:, 1]
        folds.append(dict(season=season, train_games=len(tr), test_games=len(te),
                          train_end=str(tr.GAME_DATE.max()), test_start=str(te.GAME_DATE.min())))
        predictions.append(te)
    return pd.concat(predictions, ignore_index=True), folds


def main(impact_control=False):
    OUT = ROOT / 'reports' / ('impact_control' if impact_control else 'rating_experiment')
    arms = ('recent_normalized', 'impact_120') if impact_control else ARMS
    penalties = (120.,) if impact_control else ()
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    def read(path):
        b = (ROOT / path).read_bytes()
        hashes[path] = hashlib.sha256(b).hexdigest()
        return pd.read_csv(io.BytesIO(b), dtype={'GAME_ID': str},
                           compression='gzip' if path.endswith('.gz') else None)
    logs = read('data/raw/player_game_logs.csv.gz')
    logs = logs[(logs.MIN > 0) & (logs.SEASON <= '2025-26')].copy()
    logs['GAME_ID'] = logs.GAME_ID.str.zfill(10)
    ages = read('data/raw/player_ages.csv')
    raw = read('data/raw/team_game_logs.csv')
    pre = read('data/processed/pregame_features.csv')
    for d in [raw, pre]:
        d['GAME_ID'] = d.GAME_ID.str.zfill(10)
    for name in ['rating_experiment.py', 'rating_research.py', 'ratings.py',
                 'build_player_strength.py', 'context_experiment.py']:
        hashes['src/' + name] = hashlib.sha256((ROOT / 'src' / name).read_bytes()).hexdigest()
    recipe = hashlib.sha256(json.dumps([hashes, arms, penalties], sort_keys=True).encode()).hexdigest()
    cache = OUT / 'cache' / recipe
    cache.mkdir(parents=True, exist_ok=True)
    candidates = candidates_from_past(logs)
    candidates = candidates[candidates.SEASON >= '2018-19']
    teams, snapshots = [], []
    for season in sorted(candidates.SEASON.unique()):
        path, snap = cache / f'teams_{season}.csv', cache / f'snapshots_{season}.csv'
        if path.exists() and snap.exists():
            teams.append(pd.read_csv(path, dtype={'GAME_ID': str}))
            snapshots.append(pd.read_csv(snap))
            print(f'{season}: matching input/code cache', flush=True)
            continue
        box = season_box_model(logs, season)
        curve = fit_age_curve(logs, ages, box, season)
        sp = logs[logs.SEASON == season]
        cutoffs = pd.date_range(sp.GAME_DATE.min(), sp.GAME_DATE.max(), freq=f'{REFRESH_DAYS}D').strftime('%Y-%m-%d')
        c = candidates[candidates.SEASON == season].copy()
        c['CUTOFF'] = cutoffs[np.searchsorted(cutoffs, c.GAME_DATE, side='right') - 1]
        season_snapshots = []
        for cutoff, group in c.groupby('CUTOFF'):
            variants = rating_variants(logs, season, box, curve, ages, cutoff, impact_penalties=penalties)
            for arm in arms:
                ratings = variants[arm]
                c.loc[group.index, arm] = group.PLAYER_ID.map(ratings.RATING).fillna(REPLACEMENT)
                season_snapshots.append(dict(season=season, cutoff=cutoff, arm=arm,
                    home_edge=ratings.attrs['home_edge'], players=len(ratings)))
        for arm in arms:
            c[arm] *= c.EXPECTED_MIN / 48
        team = c.groupby(['GAME_ID', 'TEAM_ID'])[list(arms) + ['EXPECTED_MIN']].sum()
        for arm in arms:
            team[arm] += (240 - team.EXPECTED_MIN) / 48 * REPLACEMENT
        team = team.reset_index()
        team.to_csv(path, index=False)
        pd.DataFrame(season_snapshots).to_csv(snap, index=False)
        teams.append(team)
        snapshots.append(pd.DataFrame(season_snapshots))
        print(f'{season}: {len(team)} team-games / {c.CUTOFF.nunique()} snapshots / {len(arms)} ratings', flush=True)
    team = pd.concat(teams, ignore_index=True)
    counts = raw.assign(HOME=raw.MATCHUP.str.contains('vs.', regex=False)).groupby('GAME_ID').HOME.agg(['sum', 'count'])
    invalid = counts.index[(counts['sum'] != 1) | (counts['count'] != 2)]
    raw = raw[~raw.GAME_ID.isin(invalid)]
    games = pre[['GAME_ID', 'SEASON', 'GAME_DATE', 'REST_DAYS_DIFF']].copy()
    for home, suffix in [(True, '_H'), (False, '_A')]:
        side = raw[raw.MATCHUP.str.contains('vs.', regex=False) == home][['GAME_ID', 'TEAM_ID', 'PTS']]
        side = side.merge(team, on=['GAME_ID', 'TEAM_ID'], validate='one_to_one')
        side = side[['GAME_ID', 'PTS'] + list(arms)].rename(
            columns=lambda col: col if col == 'GAME_ID' else col + suffix)
        games = games.merge(side, on='GAME_ID', validate='one_to_one')
    games['WIN'] = (games.PTS_H > games.PTS_A).astype(int)
    for arm in arms:
        games[arm + '_DIFF'] = games[arm + '_H'] - games[arm + '_A']
    forecasts, folds = predict_wins(games, arms)
    evaluation = forecasts[forecasts.SEASON >= '2023-24'].copy()
    metrics = []
    for season in ['ALL'] + sorted(evaluation.SEASON.unique()):
        d = evaluation if season == 'ALL' else evaluation[evaluation.SEASON == season]
        for arm in arms:
            p = d['P_' + arm]
            metrics.append(dict(season=season, arm=arm, games=len(d),
                accuracy=float(((p >= .5) == d.WIN).mean()), log_loss=float(log_loss(d.WIN, p)),
                brier=float(brier_score_loss(d.WIN, p))))
    comparisons = [paired_loss(evaluation, 'P_' + arm, 'P_'+arms[0]) for arm in arms[1:]]
    if not impact_control:
        comparisons.append(paired_loss(evaluation, 'P_recent_normalized', 'P_recent_365'))
    # Current ratings are an interpretable comparison artifact, not model selection.
    cur = read('data/raw/player_game_logs_current.csv.gz')
    cur = cur[cur.MIN > 0].copy()
    current_season = str(cur.SEASON.max())
    cutoff = (pd.Timestamp(cur.GAME_DATE.max()) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    current_logs = pd.concat([logs[logs.SEASON != current_season], cur], ignore_index=True)
    box = season_box_model(current_logs, current_season)
    curve = fit_age_curve(current_logs, ages, box, current_season)
    variants = rating_variants(current_logs, current_season, box, curve, ages, cutoff, impact_penalties=penalties)
    table = pd.concat([r.reset_index().assign(ARM=arm, CUTOFF=cutoff,
        PLAYED_CURRENT_SEASON=r.index.isin(cur.PLAYER_ID)) for arm, r in variants.items() if arm in arms], ignore_index=True)
    table.to_csv(OUT / 'current_rating_comparison.csv', index=False)
    report = dict(sha256=hashes, historical_recipe=recipe, arms=list(arms), impact_penalties=penalties, half_life_days=HALF_LIFE_DAYS,
        window_seasons=3, ridge_lambda=30, refresh_days=REFRESH_DAYS, folds=folds, metrics=metrics,
        comparisons=comparisons, excluded_ambiguous_games=invalid.tolist(), current_cutoff=cutoff,
        limitations=['Reused historical evaluation seasons, not untouched forward validation.',
            'Game-level minutes cannot fully separate teammate, coach or lineup effects.',
            'Same prior/age models are retained to isolate changes; priors were learned from game-level targets.',
            'Recency changes weighting and effective shrinkage together; 365 days fixed, not tuned.',
            'All arms use previous-game candidates; returns/injuries and season openers remain limitations.',
            'Age is season-level metadata, not independently timestamp-verified.',
            'Date bootstrap ignores cross-date dependence, training and model-selection uncertainty.',
            'No betting evaluation or live model replacement in this experiment.'])
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    forecasts.to_csv(OUT / 'predictions.csv', index=False)
    pd.concat(snapshots).to_csv(OUT / 'rating_snapshots.csv', index=False)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--impact-control',action='store_true')
    main(parser.parse_args().impact_control)
