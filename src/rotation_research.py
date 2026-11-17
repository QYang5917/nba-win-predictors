"""Research-only rotation allocator and chronological next-game minutes evaluation.

No injury-status probabilities are invented. Questionable/doubtful players are
evaluated under separate all-in/all-out scenarios until probabilities can be learned.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from build_player_strength import expected_minutes

ROOT = Path(__file__).resolve().parents[1]
MINUTE_FEATURES = ['PREV_MIN', 'EXP_AFTER', 'G_BEFORE', 'REST', 'TEAM_PREV_PLAYERS']


def allocate_minutes(weights, caps=None, total=240.):
    """Proportional water filling with caps; return explicit unallocated minutes.

    A zero weight never receives minutes. Insufficient roster capacity is not
    silently assigned to a fictitious player. Caps are modeling constraints.
    """
    w = np.asarray(weights, dtype=float)
    caps = np.full(w.shape, 40.) if caps is None else np.asarray(caps, dtype=float)
    if w.ndim != 1 or w.shape != caps.shape or not np.isfinite(w).all() or not np.isfinite(caps).all():
        raise ValueError('Invalid weights/caps')
    if (w < 0).any() or (caps < 0).any() or (caps > 48).any() or not np.isfinite(total) or total < 0:
        raise ValueError('Weights, caps or total out of bounds')
    result = np.zeros_like(w)
    remaining = float(total)
    active = (w > 0) & (caps > 0)
    for _ in range(len(w) + 1):
        if remaining <= 1e-9 or not active.any():
            break
        proposal = remaining * w[active] / w[active].sum()
        indices = np.flatnonzero(active)
        saturated = proposal >= caps[active] - result[active]
        if not saturated.any():
            result[active] += proposal
            remaining = 0.
            break
        ids = indices[saturated]
        take = caps[ids] - result[ids]
        result[ids] += take
        remaining -= take.sum()
        active[ids] = False
    return result, max(0., remaining)


def roster_priors(logs, roster, season, cutoff_date):
    """Use current roster plus strictly earlier games, including prior-season returnees.

    Roster must be supplied as of the decision time; this function cannot prove
    historical roster provenance. Same-date stats are deliberately excluded.
    """
    if roster.PLAYER_ID.duplicated().any():
        raise ValueError('Duplicate roster player')
    past = logs[pd.to_datetime(logs.GAME_DATE) < pd.Timestamp(cutoff_date)].copy()
    year = int(season[:4])
    previous = f'{year-1}-{str(year)[-2:]}'
    past = past[past.SEASON.isin([previous, season]) & (past.MIN > 0)]
    cur = expected_minutes(past) if len(past) else pd.DataFrame()
    out = roster.copy()
    out['BASE_MIN'] = 12.
    out['PRIOR_SOURCE'] = 'unknown_player_heuristic'
    if len(cur):
        prev = past[past.SEASON == previous].groupby('PLAYER_ID').MIN.mean()
        have_prev = out.PLAYER_ID.isin(prev.index)
        out.loc[have_prev, 'BASE_MIN'] = out.loc[have_prev, 'PLAYER_ID'].map(prev)
        out.loc[have_prev, 'PRIOR_SOURCE'] = 'previous_season'
        latest = cur[cur.SEASON == season].sort_values('GAME_DATE').groupby('PLAYER_ID').last()
        have_cur = out.PLAYER_ID.isin(latest.index)
        out.loc[have_cur, 'BASE_MIN'] = out.loc[have_cur, 'PLAYER_ID'].map(latest.EXP_AFTER)
        out.loc[have_cur, 'PRIOR_SOURCE'] = 'current_season'
    out['BASE_MIN'] = out.BASE_MIN.clip(0, 48)
    return out


def rotation_scenarios(priors, statuses):
    """statuses: one explicit player status per roster player, from validated input.

    Missing/unrecognized statuses block output instead of implying healthy.
    This is a sensitivity range, not a calibrated probability distribution.
    """
    if priors.TEAM_ID.nunique() != 1:
        raise ValueError('Allocate one team at a time')
    allowed = {'Available', 'Probable', 'Questionable', 'Doubtful', 'Out'}
    if statuses.PLAYER_ID.duplicated().any():
        raise ValueError('Duplicate status')
    d = priors.merge(statuses[['PLAYER_ID', 'STATUS']], on='PLAYER_ID', how='left', validate='one_to_one')
    if not d.STATUS.isin(allowed).all():
        raise ValueError('Missing/unverified availability; require a complete snapshot')
    # Probable is also uncertain: all-in/all-out deliberately bounds these cases.
    uncertain = d.STATUS.isin(['Questionable', 'Doubtful', 'Probable'])
    outputs = []
    for scenario in ['uncertain_in', 'uncertain_out']:
        active = d.STATUS != 'Out'
        if scenario == 'uncertain_out':
            active &= ~uncertain
        weights = d.BASE_MIN.where(active, 0)
        caps = d['MIN_CAP'].to_numpy() if 'MIN_CAP' in d else None
        minutes, unallocated = allocate_minutes(weights, caps)
        outputs.append(d.assign(SCENARIO=scenario, EXPECTED_MIN=minutes, UNALLOCATED_TEAM_MIN=unallocated))
    return pd.concat(outputs, ignore_index=True)


def from_snapshot(logs, roster, snapshot, season, as_of):
    """Generate a shadow rotation only from a timely, complete local snapshot."""
    from pregame_snapshots import validate, utc
    payload = snapshot['payload']
    validate(payload, snapshot['received_at'])
    if payload['kind'] != 'availability':
        raise ValueError('Availability snapshot required')
    if not utc(snapshot['received_at']) <= utc(as_of) < utc(payload['tip_utc']):
        raise ValueError('Snapshot unavailable at decision time or game already started')
    if (utc(as_of) - utc(payload['observed_at'])).total_seconds() > 21600:
        raise ValueError('Availability snapshot expired')
    team = roster[roster.TEAM_ID == payload['team_id']]
    statuses = pd.DataFrame(payload['players']).rename(columns={'player_id': 'PLAYER_ID', 'status': 'STATUS', 'minute_cap': 'MIN_CAP'})
    if set(team.PLAYER_ID) != set(statuses.PLAYER_ID):
        raise ValueError('Snapshot must cover the exact supplied team roster')
    if 'MIN_CAP' not in statuses:
        statuses['MIN_CAP'] = 40.
    statuses['MIN_CAP'] = statuses.MIN_CAP.fillna(40.)
    # Logs carry calendar dates rather than exact completion times; use only prior ET dates.
    from zoneinfo import ZoneInfo
    cutoff = utc(as_of).astimezone(ZoneInfo('America/New_York')).date().isoformat()
    priors = roster_priors(logs, team, season, cutoff)
    priors = priors.merge(statuses[['PLAYER_ID', 'MIN_CAP']], on='PLAYER_ID', validate='one_to_one')
    return rotation_scenarios(priors, statuses)


def historical_minutes(logs):
    """Forecast from previous game's roster, then grade ALL missed/returning players.

    Actual participants only enter scoring, never candidate selection. Actual
    minutes are normalized to 240 to remove overtime from this regulation model.
    """
    d = logs[logs.MIN > 0].copy()
    if d.duplicated(['GAME_ID', 'PLAYER_ID']).any():
        raise ValueError('Duplicate player-game')
    pg = expected_minutes(d)
    tg = pg[['GAME_ID', 'TEAM_ID', 'SEASON', 'GAME_DATE']].drop_duplicates()
    if tg.duplicated(['TEAM_ID', 'GAME_DATE']).any():
        raise ValueError('Multiple team games on one date')
    tg = tg.sort_values(['TEAM_ID', 'SEASON', 'GAME_DATE'])
    tg['NEXT_GAME_ID'] = tg.groupby(['TEAM_ID', 'SEASON']).GAME_ID.shift(-1)
    tg['NEXT_DATE'] = tg.groupby(['TEAM_ID', 'SEASON']).GAME_DATE.shift(-1)
    forecast = pg.merge(tg, on=['GAME_ID', 'TEAM_ID', 'SEASON', 'GAME_DATE'], validate='many_to_one').dropna(subset=['NEXT_GAME_ID'])
    forecast = forecast[forecast.SEASON >= '2016-17'].copy()
    if not (pd.to_datetime(forecast.GAME_DATE) < pd.to_datetime(forecast.NEXT_DATE)).all():
        raise ValueError('Non-causal previous game link')
    forecast['SOURCE_GAME_ID'] = forecast.GAME_ID
    forecast['PREV_MIN'] = forecast.MIN
    forecast['REST'] = (pd.to_datetime(forecast.NEXT_DATE) - pd.to_datetime(forecast.GAME_DATE)).dt.days.clip(upper=10)
    forecast['TEAM_PREV_PLAYERS'] = forecast.groupby(['GAME_ID', 'TEAM_ID']).PLAYER_ID.transform('size')
    forecast['GAME_ID'] = forecast.NEXT_GAME_ID
    forecast['GAME_DATE'] = forecast.NEXT_DATE
    forecast = forecast.sort_values(['GAME_ID', 'TEAM_ID', 'EXP_AFTER'], ascending=[True, True, False])
    used = forecast.groupby(['GAME_ID', 'TEAM_ID']).EXP_AFTER.cumsum() - forecast.EXP_AFTER
    forecast['LEGACY_MIN'] = np.clip(240 - used, 0, forecast.EXP_AFTER)
    forecast['ALLOCATED_MIN'] = 0.
    for _, group in forecast.groupby(['GAME_ID', 'TEAM_ID'], sort=False):
        forecast.loc[group.index, 'ALLOCATED_MIN'] = allocate_minutes(group.EXP_AFTER)[0]
    keys = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']
    actual = pg[keys + ['MIN']].copy()
    actual['ACTUAL_REG_MIN'] = 240 * actual.MIN / actual.groupby(['GAME_ID', 'TEAM_ID']).MIN.transform('sum')
    eligible = forecast[['GAME_ID', 'TEAM_ID', 'SEASON', 'GAME_DATE']].drop_duplicates()
    actual = actual.merge(eligible, on=['GAME_ID', 'TEAM_ID'], validate='many_to_one')
    ledger = forecast[keys + ['LEGACY_MIN', 'ALLOCATED_MIN'] + MINUTE_FEATURES].merge(
        actual[keys + ['ACTUAL_REG_MIN']], on=keys, how='outer', validate='one_to_one')
    ledger[['LEGACY_MIN', 'ALLOCATED_MIN', 'ACTUAL_REG_MIN']] = ledger[['LEGACY_MIN', 'ALLOCATED_MIN', 'ACTUAL_REG_MIN']].fillna(0)
    return ledger.merge(eligible, on=['GAME_ID', 'TEAM_ID'], validate='many_to_one')


def learned_minutes(ledger):
    """Predict candidate minutes incl. zero-minute absences; no actual lineup inputs.

    Mean and median models serve different losses. Neither is a joint 240-minute
    rotation or a calibrated player-availability model.
    """
    chunks, folds = [], []
    for season in ['2023-24', '2024-25', '2025-26']:
        train = ledger[(ledger.SEASON < season) & ledger.EXP_AFTER.notna()]
        test = ledger[ledger.SEASON == season].copy()
        if test.empty:
            continue
        if train.empty or pd.to_datetime(train.GAME_DATE).max() >= pd.to_datetime(test.GAME_DATE).min():
            raise ValueError('Invalid chronological minute-model split')
        known = test.EXP_AFTER.notna()
        for name, loss in [('LEARNED_MEAN', 'squared_error'), ('LEARNED_MEDIAN', 'absolute_error')]:
            model = HistGradientBoostingRegressor(loss=loss, max_iter=120, max_leaf_nodes=15,
                min_samples_leaf=100, l2_regularization=10, learning_rate=.05,
                early_stopping=False, random_state=42)
            model.fit(train[MINUTE_FEATURES], train.ACTUAL_REG_MIN)
            test[name] = 0.
            if known.any():
                test.loc[known, name] = np.clip(model.predict(test.loc[known, MINUTE_FEATURES]), 0, 48)
        folds.append({'season': season, 'train_candidates': len(train),
                      'train_end': str(pd.to_datetime(train.GAME_DATE).max().date()),
                      'test_start': str(pd.to_datetime(test.GAME_DATE).min().date()),
                      'test_candidates': int(known.sum()), 'unpredicted_returning_rows': int((~known).sum())})
        chunks.append(test)
    return pd.concat(chunks, ignore_index=True), folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, help='Recorded availability snapshot for shadow rotation')
    parser.add_argument('--season', default='2026-27')
    args = parser.parse_args()
    if args.snapshot:
        import datetime as dt
        from ratings import load_logs
        roster = pd.read_csv(ROOT / 'data/raw/rosters_current.csv').rename(columns={'TeamID': 'TEAM_ID'})
        snapshot = json.loads(args.snapshot.read_text())
        output = from_snapshot(load_logs(), roster, snapshot, args.season, dt.datetime.now(dt.timezone.utc).isoformat())
        print(output[['PLAYER', 'PRIOR_SOURCE', 'SCENARIO', 'EXPECTED_MIN', 'UNALLOCATED_TEAM_MIN']].to_string(index=False))
        return
    input_path = ROOT / 'data/raw/player_game_logs.csv.gz'
    logs = pd.read_csv(input_path, dtype={'GAME_ID': str})
    logs = logs[logs.SEASON <= '2025-26']
    ledger, folds = learned_minutes(historical_minutes(logs))
    rows = []
    for season in ['ALL'] + sorted(ledger.SEASON.unique()):
        part = ledger if season == 'ALL' else ledger[ledger.SEASON == season]
        for col in ['LEGACY_MIN', 'ALLOCATED_MIN', 'LEARNED_MEAN', 'LEARNED_MEDIAN']:
            error = part[col] - part.ACTUAL_REG_MIN
            totals = part.groupby(['GAME_ID', 'TEAM_ID'])[col].sum()
            rows.append({'season': season, 'model': col, 'player_rows': len(part),
                         'team_games': len(totals), 'mae': float(error.abs().mean()),
                         'rmse': float(np.sqrt(np.mean(error ** 2))),
                         'mean_team_unallocated': float((240 - totals).clip(lower=0).mean())})
    out = ROOT / 'reports/rotation_research'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps({'metrics': rows,
        'folds': folds, 'learned_features': MINUTE_FEATURES,
        'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'hyperparameters': {'max_iter': 120, 'max_leaf_nodes': 15, 'min_samples_leaf': 100,
                            'l2_regularization': 10, 'learning_rate': .05, 'early_stopping': False, 'seed': 42},
        'scope': 'Previous-game roster proxy; no historical injury reports, no win or ROI claim.',
        'cap': 40, 'total_minutes': 240, 'actual_minutes_normalized_for_overtime': True,
        'includes_absent_and_returning_players_in_scoring': True}, indent=2) + '\n')
    ledger.to_csv(out / 'minutes.csv.gz', index=False, compression='gzip')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
