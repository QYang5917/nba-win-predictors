"""Past-five-game candidate pool + learned participation and conditional minutes.

All parameters fixed; research output only. No injury/news labels are fabricated.
"""
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

from build_player_strength import expected_minutes, season_box_model, REFRESH_DAYS
from ratings import load_ages, fit_age_curve, season_ratings, REPLACEMENT
from rotation_win_experiment import aggregate_strength
from context_experiment import build_context, CONTEXT, paired_loss
from market_research import match_odds, market_predictions, settle, bet_summary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/availability_experiment'
FEATURES = ['LAST_TEAM_MIN', 'LAST_PLAYER_MIN', 'EXP_AFTER', 'APPEAR_RATE5', 'AVG_MIN5',
            'MISSED_TEAM_GAMES', 'DAYS_SINCE_PLAYED', 'TEAM_REST', 'POOL_SIZE', 'KNOWN_OTHER_TEAM']
PARAMS = dict(max_iter=120, max_leaf_nodes=15, min_samples_leaf=100,
              l2_regularization=10, learning_rate=.05, early_stopping=False, random_state=42)
ARMS = {
    'legacy': ['LEGACY_DIFF', 'REST_DAYS_DIFF'],
    'pool_direct': ['DIRECT_KNOWN_DIFF', 'DIRECT_RESERVE_DIFF', 'REST_DAYS_DIFF'],
    'pool_hurdle': ['HURDLE_KNOWN_DIFF', 'HURDLE_RESERVE_DIFF', 'REST_DAYS_DIFF'],
    'legacy_context': ['LEGACY_DIFF', 'REST_DAYS_DIFF'] + CONTEXT,
    'pool_context': ['HURDLE_KNOWN_DIFF', 'HURDLE_RESERVE_DIFF', 'REST_DAYS_DIFF'] + CONTEXT,
}


def build_pool(logs):
    """Generate candidates BEFORE observing today's participation or minutes.

    The game schedule/team identities are assumed known. Current results enter
    targets and next-date state only. All teams share a date-level update barrier.
    """
    d = logs[logs.MIN > 0].copy()
    if d.duplicated(['GAME_ID', 'PLAYER_ID']).any():
        raise ValueError('Duplicate player-game')
    d['GAME_DATE'] = pd.to_datetime(d.GAME_DATE)
    pg = expected_minutes(d)
    events = []
    for (day, gid, tid, season), part in pg.groupby(['GAME_DATE', 'GAME_ID', 'TEAM_ID', 'SEASON'], sort=True):
        players = {r.PLAYER_ID: (float(r.MIN), float(r.EXP_AFTER)) for r in part.itertuples()}
        events.append((day, gid, tid, season, players))
    history, latest_team, rows = {}, {}, []
    by_day = {}
    for e in events:
        by_day.setdefault(e[0], []).append(e)
    for day, games in sorted(by_day.items()):
        if len({g[2] for g in games}) != len(games):
            raise ValueError('Multiple games for a team on one date')
        for _, gid, tid, season, actual in games:
            h = history.get((tid, season), deque(maxlen=5))
            if not h or season < '2016-17':
                continue
            ids = sorted(set(p for _, players in h for p in players))
            previous = h[-1][1]
            # Reconstruct the original capped previous-game rotation for control.
            legacy, remaining = {}, 240.
            for p in sorted(previous, key=lambda p: previous[p][1], reverse=True):
                legacy[p] = min(remaining, previous[p][1])
                remaining -= legacy[p]
            for pid in ids:
                seq = [(date, players.get(pid)) for date, players in h]
                appearances = [(i, date, value) for i, (date, value) in enumerate(seq) if value is not None]
                i, last_day, (last_min, exp) = appearances[-1]
                rows.append(dict(GAME_ID=gid, TEAM_ID=tid, PLAYER_ID=pid, SEASON=season, GAME_DATE=day,
                    LAST_TEAM_MIN=previous.get(pid, (0., 0.))[0], LAST_PLAYER_MIN=last_min, EXP_AFTER=exp,
                    APPEAR_RATE5=len(appearances) / len(h),
                    AVG_MIN5=sum(v[0] for _, _, v in appearances) / len(h),
                    MISSED_TEAM_GAMES=len(h) - 1 - i, DAYS_SINCE_PLAYED=min(60, (day-last_day).days),
                    TEAM_REST=min(10, (day-h[-1][0]).days), POOL_SIZE=len(ids),
                    KNOWN_OTHER_TEAM=int(latest_team.get(pid, tid) != tid),
                    LEGACY_MIN=legacy.get(pid, 0.),
                    ACTUAL_MIN=240 * actual.get(pid, (0., 0.))[0] / sum(v[0] for v in actual.values())))
        # Only now may today's participation alter subsequent candidate pools.
        for _, gid, tid, season, actual in games:
            history.setdefault((tid, season), deque(maxlen=5)).append((day, actual))
            for pid in actual:
                latest_team[pid] = tid
    return pd.DataFrame(rows)


def crossfit(pool):
    outputs, folds = [], []
    for season in sorted(pool.SEASON.unique()):
        if season < '2018-19':
            continue
        tr, te = pool[pool.SEASON < season], pool[pool.SEASON == season].copy()
        if tr.GAME_DATE.max() >= te.GAME_DATE.min():
            raise ValueError('Participation/minute training overlaps test')
        played = tr.ACTUAL_MIN > 0
        clf = HistGradientBoostingClassifier(**PARAMS).fit(tr[FEATURES], played)
        conditional = HistGradientBoostingRegressor(**PARAMS).fit(tr.loc[played, FEATURES], tr.loc[played, 'ACTUAL_MIN'])
        direct = HistGradientBoostingRegressor(**PARAMS).fit(tr[FEATURES], tr.ACTUAL_MIN)
        te['P_PLAY'] = clf.predict_proba(te[FEATURES])[:, 1]
        te['CONDITIONAL_MIN'] = np.clip(conditional.predict(te[FEATURES]), 0, 48)
        te['HURDLE_MIN'] = te.P_PLAY * te.CONDITIONAL_MIN
        te['DIRECT_MIN'] = np.clip(direct.predict(te[FEATURES]), 0, 48)
        # A player observed on another team before today cannot be assigned minutes here.
        te.loc[te.KNOWN_OTHER_TEAM == 1, ['P_PLAY', 'HURDLE_MIN', 'DIRECT_MIN']] = 0.
        outputs.append(te)
        folds.append({'season': season, 'train_end': str(tr.GAME_DATE.max()),
                      'test_start': str(te.GAME_DATE.min()), 'train_candidates': len(tr)})
    return pd.concat(outputs, ignore_index=True), folds


def team_strengths(logs, candidates, cache):
    ages, chunks = load_ages(), []
    for season in sorted(candidates.SEASON.unique()):
        path = cache / f'team_{season}.csv'
        if path.exists():
            chunks.append(pd.read_csv(path, dtype={'GAME_ID': str}))
            continue
        box = season_box_model(logs, season)
        curve = fit_age_curve(logs, ages, box, season)
        sp = logs[logs.SEASON == season]
        cutoffs = pd.date_range(sp.GAME_DATE.min(), sp.GAME_DATE.max(), freq=f'{REFRESH_DAYS}D').strftime('%Y-%m-%d')
        c = candidates[candidates.SEASON == season].copy()
        c['CUTOFF'] = cutoffs[np.searchsorted(cutoffs, c.GAME_DATE.dt.strftime('%Y-%m-%d'), side='right') - 1]
        c['RATING'] = np.nan
        for cutoff, group in c.groupby('CUTOFF'):
            r = season_ratings(logs, season, box, curve, ages, cutoff).set_index('PLAYER_ID').RATING
            c.loc[group.index, 'RATING'] = group.PLAYER_ID.map(r).fillna(REPLACEMENT)
        result = None
        for name in ['DIRECT', 'HURDLE']:
            stats = aggregate_strength(c.assign(PRED_MIN=c[name + '_MIN']))
            keep = ['GAME_ID', 'TEAM_ID', 'KNOWN', 'RESERVE'] + (['LEGACY'] if result is None else [])
            stats = stats[keep].rename(columns={'KNOWN': name+'_KNOWN', 'RESERVE': name+'_RESERVE'})
            result = stats if result is None else result.merge(stats, on=['GAME_ID', 'TEAM_ID'], validate='one_to_one')
        result.to_csv(path, index=False)
        chunks.append(result)
        print(f'{season}: rated expanded pool for {len(result)} team-games', flush=True)
    return pd.concat(chunks, ignore_index=True)


def fit_wins(games):
    outputs, folds = [], []
    for season in sorted(games.SEASON.unique()):
        if season < '2021-22':
            continue
        tr, te = games[games.SEASON < season], games[games.SEASON == season].copy()
        if tr.GAME_DATE.max() >= te.GAME_DATE.min():
            raise ValueError('Win training overlaps test')
        for name, cols in ARMS.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000)).fit(tr[cols], tr.WIN)
            te['P_'+name] = model.predict_proba(te[cols])[:, 1]
        outputs.append(te)
        folds.append({'season': season, 'train_end': str(tr.GAME_DATE.max()), 'test_start': str(te.GAME_DATE.min())})
    return pd.concat(outputs, ignore_index=True), folds


def minute_metrics(candidates, logs):
    keys = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']
    eligible = candidates[candidates.SEASON >= '2023-24'][['GAME_ID', 'TEAM_ID']].drop_duplicates()
    c = candidates[candidates.SEASON >= '2023-24'].copy()
    actual = logs[logs.MIN > 0].copy()
    actual['TARGET'] = 240 * actual.MIN / actual.groupby(['GAME_ID', 'TEAM_ID']).MIN.transform('sum')
    actual = actual.merge(eligible, on=['GAME_ID', 'TEAM_ID'], validate='many_to_one')
    rows = c[keys + ['LEGACY_MIN', 'DIRECT_MIN', 'HURDLE_MIN']].merge(actual[keys + ['TARGET']],
        on=keys, how='outer', validate='one_to_one').fillna(0)
    result = {col: {'mae': float((rows[col]-rows.TARGET).abs().mean()),
                    'rmse': float(np.sqrt(np.mean((rows[col]-rows.TARGET)**2)))}
              for col in ['LEGACY_MIN', 'DIRECT_MIN', 'HURDLE_MIN']}
    # Candidate recall is evaluated against actual participants, not used to select them.
    all_played = set(map(tuple, actual[keys].to_numpy()))
    pool = set(map(tuple, c[keys].to_numpy()))
    last = set(map(tuple, c.loc[c.LAST_TEAM_MIN > 0, keys].to_numpy()))
    result['coverage'] = {'actual_participants': len(all_played),
        'previous_game_recall': len(all_played & last)/len(all_played),
        'five_game_recall': len(all_played & pool)/len(all_played), 'scoring_union_rows': len(rows)}
    result['participation_brier'] = float(brier_score_loss(c.ACTUAL_MIN > 0, c.P_PLAY))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [ROOT/p for p in ['data/raw/player_game_logs.csv.gz', 'data/raw/player_ages.csv',
        'data/raw/team_game_logs.csv', 'data/processed/pregame_features.csv', 'data/raw/betting_odds_mgm.csv']]
    code = [Path(__file__)] + [ROOT/'src'/p for p in ['ratings.py', 'build_player_strength.py',
        'rotation_research.py', 'rotation_win_experiment.py', 'context_experiment.py', 'market_research.py']]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+code}
    recipe = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    cache = OUT/'cache'/recipe
    cache.mkdir(parents=True, exist_ok=True)
    logs = pd.read_csv(paths[0], dtype={'GAME_ID': str})
    logs = logs[(logs.MIN>0) & (logs.SEASON<='2025-26')]
    logs['GAME_ID'] = logs.GAME_ID.str.zfill(10)
    candidate_file = cache/'candidates.csv.gz'
    if candidate_file.exists():
        candidates = pd.read_csv(candidate_file, dtype={'GAME_ID': str}, parse_dates=['GAME_DATE'])
        folds = json.loads((cache/'folds.json').read_text())
    else:
        print('Building past-five-game candidates and chronological participation models…', flush=True)
        candidates, folds = crossfit(build_pool(logs))
        candidates.to_csv(candidate_file, index=False, compression='gzip')
        (cache/'folds.json').write_text(json.dumps(folds, indent=2))
    minute_results = minute_metrics(candidates, logs)
    print(json.dumps(minute_results, indent=2), flush=True)
    teams = team_strengths(logs, candidates, cache)
    raw = pd.read_csv(paths[2], dtype={'GAME_ID': str})
    raw['GAME_ID'] = raw.GAME_ID.str.zfill(10)
    counts = raw.assign(HOME=raw.MATCHUP.str.contains('vs.', regex=False)).groupby('GAME_ID').HOME.agg(['sum','count'])
    bad = counts.index[(counts['sum']!=1)|(counts['count']!=2)]
    raw = raw[~raw.GAME_ID.isin(bad)]
    pre = pd.read_csv(paths[3], dtype={'GAME_ID': str})
    pre['GAME_ID'] = pre.GAME_ID.str.zfill(10)
    games = pre[['GAME_ID','SEASON','GAME_DATE','TEAM','OPP','REST_DAYS_DIFF','WIN']]
    cols = ['LEGACY','DIRECT_KNOWN','DIRECT_RESERVE','HURDLE_KNOWN','HURDLE_RESERVE']
    for home, suffix in [(True,'_H'),(False,'_A')]:
        side = raw[raw.MATCHUP.str.contains('vs.', regex=False)==home][['GAME_ID','TEAM_ID']]
        side = side.merge(teams,on=['GAME_ID','TEAM_ID'],validate='one_to_one')[['GAME_ID']+cols]
        games = games.merge(side.rename(columns={c:c+suffix for c in cols}),on='GAME_ID',validate='one_to_one')
    for col in cols:
        games[col+'_DIFF'] = games[col+'_H']-games[col+'_A']
    games = games.merge(build_context(raw),on='GAME_ID',validate='one_to_one')
    games['GAME_DATE'] = pd.to_datetime(games.GAME_DATE)
    base, win_folds = fit_wins(games)
    matched, audit = match_odds(base,pd.read_csv(paths[4]))
    predictions, market_folds = None, {}
    for name in ARMS:
        scored, market_folds[name] = market_predictions(matched.assign(P_PLAYER=matched['P_'+name]))
        if predictions is None:
            predictions = scored.drop(columns=['P_PLAYER','P_HYBRID'])
        if not predictions.GAME_ID.equals(scored.GAME_ID):
            raise ValueError('Misaligned models')
        predictions['HYBRID_'+name] = scored.P_HYBRID.to_numpy()
    metrics, ledgers = [], []
    for col in ['P_MARKET','P_CALIBRATED']+[f'{p}_{a}' for p in ['P','HYBRID'] for a in ARMS]:
        ledger = settle(predictions,col)
        ledgers.append(ledger.assign(MODEL=col))
        for season in ['ALL']+sorted(predictions.SEASON.unique()):
            part = predictions if season=='ALL' else predictions[predictions.SEASON==season]
            bets = ledger if season=='ALL' else ledger[ledger.SEASON==season]
            metrics.append({'model':col,'season':season,'games':len(part),
                'accuracy':float(((part[col]>=.5)==part.WIN).mean()),
                'log_loss':float(log_loss(part.WIN,part[col])), **bet_summary(bets)})
    report = {'sha256':hashes,'features':FEATURES,'parameters':PARAMS,'pool_games':5,'arms':ARMS,
        'minute_metrics':minute_results,'participation_folds':folds,'win_folds':win_folds,
        'market_folds':market_folds,'matching':audit,'metrics':metrics,
        'comparisons':[paired_loss(predictions,l,r) for l,r in [
            ('P_pool_direct','P_legacy'),('P_pool_hurdle','P_legacy'),
            ('P_pool_context','P_legacy_context'),('HYBRID_pool_hurdle','P_MARKET')]],
        'limitations':['Repeated historical research; not a new holdout.',
            'Participation is learned from past games, not timestamped injury or transaction reports.',
            'Five-game pool cannot cover long absences or first appearances; no season-openers.',
            'Not a joint lineup distribution; expected minutes scaled down above 240, never scaled up.',
            'Closing odds have no observation timestamps; no executable-profit claim.',
            'Date bootstrap omits across-date dependence, training uncertainty and model selection.']}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    predictions.to_csv(OUT/'predictions.csv',index=False)
    pd.concat(ledgers).to_csv(OUT/'ledger.csv',index=False)
    print(pd.DataFrame(metrics).query("season == 'ALL'").to_string(index=False))
    print(json.dumps(report['comparisons'],indent=2))


if __name__ == '__main__':
    main()
