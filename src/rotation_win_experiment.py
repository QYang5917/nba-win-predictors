"""End-to-end, chronological minute -> strength -> win -> market experiment.

New research artifacts only. Unallocated minutes are an explicit feature rather
than silently being assigned to actual returning players discovered after tip-off.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rotation_research import historical_minutes, MINUTE_FEATURES
from ratings import load_ages, fit_age_curve, season_ratings, REPLACEMENT
from build_player_strength import season_box_model, REFRESH_DAYS
from market_research import match_odds, market_predictions, settle, bet_summary
from context_experiment import build_context, CONTEXT, paired_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/rotation_win_experiment'
ARMS = {
    'legacy': ['LEGACY_DIFF', 'REST_DAYS_DIFF'],
    'learned_fixed': ['LEARNED_FIXED_DIFF', 'REST_DAYS_DIFF'],
    'learned_reserve': ['KNOWN_DIFF', 'RESERVE_DIFF', 'REST_DAYS_DIFF'],
    'legacy_context': ['LEGACY_DIFF', 'REST_DAYS_DIFF'] + CONTEXT,
    'learned_context': ['KNOWN_DIFF', 'RESERVE_DIFF', 'REST_DAYS_DIFF'] + CONTEXT,
}


def minute_predictions(ledger):
    """Cross-fit from 2018 so downstream models can train on honest minute forecasts."""
    chunks, folds = [], []
    for season in sorted(ledger.SEASON.unique()):
        if season < '2018-19':
            continue
        tr = ledger[(ledger.SEASON < season) & ledger.EXP_AFTER.notna()]
        te = ledger[(ledger.SEASON == season) & ledger.EXP_AFTER.notna()].copy()
        if pd.to_datetime(tr.GAME_DATE).max() >= pd.to_datetime(te.GAME_DATE).min():
            raise ValueError('Minute train/test overlap')
        model = HistGradientBoostingRegressor(loss='squared_error', max_iter=120,
            max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=10,
            learning_rate=.05, early_stopping=False, random_state=42)
        model.fit(tr[MINUTE_FEATURES], tr.ACTUAL_REG_MIN)
        te['PRED_MIN'] = np.clip(model.predict(te[MINUTE_FEATURES]), 0, 48)
        # Drop target minutes before any team feature calculation.
        chunks.append(te[['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'SEASON', 'GAME_DATE', 'LEGACY_MIN', 'PRED_MIN']])
        folds.append({'season': season, 'train_end': str(tr.GAME_DATE.max()),
                      'test_start': str(te.GAME_DATE.min()), 'train_rows': len(tr)})
    return pd.concat(chunks, ignore_index=True), folds


def aggregate_strength(rows):
    """Input: pregame candidates ONLY, each with past-only RATING and minute forecast."""
    keys = ['GAME_ID', 'TEAM_ID']
    if rows.duplicated(keys + ['PLAYER_ID']).any():
        raise ValueError('Duplicate candidate')
    if not np.isfinite(rows[['PRED_MIN', 'LEGACY_MIN', 'RATING']]).all().all():
        raise ValueError('Nonfinite strength input')
    if (rows[['PRED_MIN', 'LEGACY_MIN']] < 0).any().any():
        raise ValueError('Negative minutes')
    r = rows.copy()
    total = r.groupby(keys).PRED_MIN.transform('sum')
    # Never scale up a forecast to force an incomplete candidate pool to 240.
    r['USED_MIN'] = r.PRED_MIN * np.minimum(1, 240 / total.clip(lower=1e-12))
    r['KNOWN'] = r.USED_MIN * r.RATING / 48
    r['LEGACY_CONTRIB'] = r.LEGACY_MIN * r.RATING / 48
    sums = r.groupby(keys)[['USED_MIN', 'KNOWN', 'LEGACY_MIN', 'LEGACY_CONTRIB']].sum()
    if (sums.LEGACY_MIN > 240 + 1e-6).any():
        raise ValueError('Legacy rotation exceeds 240')
    sums['RESERVE'] = (240 - sums.USED_MIN).clip(lower=0) / 48
    sums['LEARNED_FIXED'] = sums.KNOWN + sums.RESERVE * REPLACEMENT
    sums['LEGACY'] = sums.LEGACY_CONTRIB + (240 - sums.LEGACY_MIN).clip(lower=0) / 48 * REPLACEMENT
    return sums.reset_index()


def rated_team_features(logs, candidates, cache_dir):
    ages = load_ages()
    chunks = []
    for season in sorted(candidates.SEASON.unique()):
        dest = cache_dir / f'teams_{season}.csv'
        if dest.exists():
            chunks.append(pd.read_csv(dest, dtype={'GAME_ID': str}))
            print(f'{season}: verified-recipe cache', flush=True)
            continue
        box = season_box_model(logs, season)
        curve = fit_age_curve(logs, ages, box, season)
        sp = logs[logs.SEASON == season]
        cutoffs = pd.date_range(sp.GAME_DATE.min(), sp.GAME_DATE.max(), freq=f'{REFRESH_DAYS}D').strftime('%Y-%m-%d')
        c = candidates[candidates.SEASON == season].copy()
        c['CUTOFF'] = cutoffs[np.searchsorted(cutoffs, c.GAME_DATE, side='right') - 1]
        c['RATING'] = np.nan
        for cutoff, group in c.groupby('CUTOFF'):
            # season_ratings excludes every game on/after cutoff.
            r = season_ratings(logs, season, box, curve, ages, cutoff).set_index('PLAYER_ID').RATING
            c.loc[group.index, 'RATING'] = group.PLAYER_ID.map(r).fillna(REPLACEMENT)
        result = aggregate_strength(c)
        result.to_csv(dest, index=False)
        chunks.append(result)
        print(f'{season}: {len(result)} team-games, {c.CUTOFF.nunique()} past-only rating snapshots', flush=True)
    return pd.concat(chunks, ignore_index=True)


def win_predictions(games):
    chunks, folds = [], []
    for season in sorted(games.SEASON.unique()):
        if season < '2021-22':
            continue
        tr, te = games[games.SEASON < season], games[games.SEASON == season].copy()
        if tr.GAME_DATE.max() >= te.GAME_DATE.min():
            raise ValueError('Win-model train/test overlap')
        fold = {'season': season, 'train_end': str(tr.GAME_DATE.max()),
                'test_start': str(te.GAME_DATE.min()), 'train_rows': len(tr)}
        for arm, cols in ARMS.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000))
            model.fit(tr[cols], tr.WIN)
            te[f'P_{arm}'] = model.predict_proba(te[cols])[:, 1]
            # Original-scale coefficient for interpreting reserve assumption.
            fold[arm] = dict(zip(cols, (model[-1].coef_[0] / model[0].scale_).tolist()))
        folds.append(fold)
        chunks.append(te)
    return pd.concat(chunks, ignore_index=True), folds


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT / p for p in ['data/raw/player_game_logs.csv.gz', 'data/raw/player_ages.csv',
        'data/raw/team_game_logs.csv', 'data/processed/pregame_features.csv', 'data/raw/betting_odds_mgm.csv']]
    code = [Path(__file__)] + [ROOT / 'src' / f for f in ['ratings.py', 'build_player_strength.py',
        'rotation_research.py', 'context_experiment.py', 'market_research.py']]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs + code}
    recipe = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    cache = OUT / 'cache' / recipe
    cache.mkdir(parents=True, exist_ok=True)
    logs = pd.read_csv(inputs[0], dtype={'GAME_ID': str})
    logs = logs[(logs.MIN > 0) & (logs.SEASON <= '2025-26')]
    logs['GAME_ID'] = logs.GAME_ID.str.zfill(10)
    candidate_path = cache / 'candidate_minutes.csv.gz'
    if candidate_path.exists():
        candidates = pd.read_csv(candidate_path, dtype={'GAME_ID': str})
        minute_folds = json.loads((cache / 'minute_folds.json').read_text())
    else:
        print('Building out-of-fold minute predictions, 2018–2025…', flush=True)
        candidates, minute_folds = minute_predictions(historical_minutes(logs))
        candidates.to_csv(candidate_path, index=False, compression='gzip')
        (cache / 'minute_folds.json').write_text(json.dumps(minute_folds, indent=2))
    team = rated_team_features(logs, candidates, cache)
    raw = pd.read_csv(inputs[2], dtype={'GAME_ID': str})
    raw['GAME_ID'] = raw.GAME_ID.str.zfill(10)
    counts = raw.assign(HOME=raw.MATCHUP.str.contains('vs.', regex=False)).groupby('GAME_ID').HOME.agg(['sum', 'count'])
    bad = counts.index[(counts['sum'] != 1) | (counts['count'] != 2)]
    raw = raw[~raw.GAME_ID.isin(bad)]
    pre = pd.read_csv(inputs[3], dtype={'GAME_ID': str})
    pre['GAME_ID'] = pre.GAME_ID.str.zfill(10)
    games = pre[['GAME_ID', 'SEASON', 'GAME_DATE', 'TEAM', 'OPP', 'REST_DAYS_DIFF', 'WIN']]
    # Match home/away IDs by team metadata, not by actual player availability.
    for home, suffix in [(True, '_H'), (False, '_A')]:
        side = raw[raw.MATCHUP.str.contains('vs.', regex=False) == home][['GAME_ID', 'TEAM_ID']]
        side = side.merge(team, on=['GAME_ID', 'TEAM_ID'], validate='one_to_one')
        side = side[['GAME_ID', 'LEGACY', 'LEARNED_FIXED', 'KNOWN', 'RESERVE']].rename(
            columns=lambda c: c if c == 'GAME_ID' else c + suffix)
        games = games.merge(side, on='GAME_ID', validate='one_to_one')
    for col in ['LEGACY', 'LEARNED_FIXED', 'KNOWN', 'RESERVE']:
        games[col + '_DIFF'] = games[col + '_H'] - games[col + '_A']
    games = games.merge(build_context(raw), on='GAME_ID', validate='one_to_one')
    games['GAME_DATE'] = pd.to_datetime(games.GAME_DATE)
    base, win_folds = win_predictions(games)
    matched, audit = match_odds(base, pd.read_csv(inputs[4]))
    audit.update(invalid_home_away_ids=bad.tolist(), matched_feature_games=len(games),
                 evaluation_policy='Require past-game candidates for BOTH teams; exclude season openers.')
    predictions, meta_folds = None, {}
    for arm in ARMS:
        scored, meta_folds[arm] = market_predictions(matched.assign(P_PLAYER=matched[f'P_{arm}']))
        if predictions is None:
            predictions = scored.drop(columns=['P_PLAYER', 'P_HYBRID'])
        if not predictions.GAME_ID.equals(scored.GAME_ID):
            raise ValueError('Misaligned market comparison')
        predictions[f'HYBRID_{arm}'] = scored.P_HYBRID.to_numpy()
    metrics, ledgers = [], []
    for col in ['P_MARKET', 'P_CALIBRATED'] + [f'{p}_{a}' for p in ['P', 'HYBRID'] for a in ARMS]:
        ledger = settle(predictions, col)
        ledgers.append(ledger.assign(MODEL=col))
        for season in ['ALL'] + sorted(predictions.SEASON.unique()):
            part = predictions if season == 'ALL' else predictions[predictions.SEASON == season]
            bets = ledger if season == 'ALL' else ledger[ledger.SEASON == season]
            metrics.append({'model': col, 'season': season, 'games': len(part),
                'accuracy': float(((part[col] >= .5) == part.WIN).mean()),
                'log_loss': float(log_loss(part.WIN, part[col])),
                'brier': float(brier_score_loss(part.WIN, part[col])), **bet_summary(bets)})
    comparisons = [paired_loss(predictions, l, r) for l, r in [
        ('P_learned_fixed', 'P_legacy'), ('P_learned_reserve', 'P_legacy'),
        ('P_learned_context', 'P_legacy_context'), ('HYBRID_learned_context', 'P_MARKET')]]
    report = {'arms': ARMS, 'sha256': hashes, 'minute_folds': minute_folds, 'win_folds': win_folds,
        'market_folds': meta_folds, 'matching': audit, 'metrics': metrics, 'comparisons': comparisons,
        'limitations': ['Exploratory reused historical seasons; no untouched holdout.',
            'Previous-game candidates do not capture actual injury/return news.',
            'Reserve coefficient is an aggregate statistical proxy, not identified replacement players.',
            'Minute mean forecasts are capped downward at 240; no forced upward allocation.',
            'Closing odds have no verified observation times; no executable-profit claim.',
            'Date bootstrap ignores serial dependence, model uncertainty and multiple comparisons.']}
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    predictions.to_csv(OUT / 'predictions.csv', index=False)
    pd.concat(ledgers).to_csv(OUT / 'ledger.csv', index=False)
    print(pd.DataFrame(metrics).query("season == 'ALL'").to_string(index=False))
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
