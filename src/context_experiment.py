"""Fixed feature-ablation experiment; historical research only, no live changes.

Three arms: previous-game players; team context; players + team context.
All regressions use C=1 and past-season-only standardization/training.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from market_research import FEATURES, match_odds, market_predictions, settle, bet_summary

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ['ELO_DIFF', 'NET10_DIFF', 'NET30_DIFF', 'B2B_DIFF', 'GAMES7_DIFF']
ARMS = {'players': FEATURES, 'context': ['REST_DAYS_DIFF'] + CONTEXT,
        'combined': FEATURES + CONTEXT}


def build_context(raw):
    """Snapshot all teams before each date; update only after that date's games.

    Elo: fixed K=20, home offset=65, season regression=.75.
    Recent net efficiency: EWM spans 10 and 30, initialized at zero.
    These settings are hypotheses, not selected from the evaluation results.
    """
    d = raw.copy()
    d['GAME_DATE'] = pd.to_datetime(d.GAME_DATE)
    if d.duplicated(['GAME_ID', 'TEAM_ID']).any():
        raise ValueError('Duplicate team-game rows')
    if d.duplicated(['GAME_DATE', 'TEAM_ID']).any():
        raise ValueError('Multiple games per team/date require timestamp-level processing')
    home = d[d.MATCHUP.str.contains('vs.', regex=False)]
    away = d[~d.MATCHUP.str.contains('vs.', regex=False)]
    g = home.merge(away, on='GAME_ID', suffixes=('_H', '_A'), validate='one_to_one')
    if len(g) * 2 != len(d) or not (g.GAME_DATE_H == g.GAME_DATE_A).all():
        raise ValueError('Unpaired or inconsistent games')
    if not (g.SEASON_H == g.SEASON_A).all() or not (g.TEAM_ID_H != g.TEAM_ID_A).all():
        raise ValueError('Invalid opponents or season')
    numeric = ['PTS', 'FGA', 'FTA', 'OREB', 'TOV']
    if not np.isfinite(d[numeric]).all().all():
        raise ValueError('Invalid box score')
    state, rows = {}, []
    season = None
    for day, group in g.sort_values(['GAME_DATE_H', 'GAME_ID']).groupby('GAME_DATE_H', sort=True):
        if group.SEASON_H.nunique() != 1:
            raise ValueError('Overlapping seasons')
        current = group.SEASON_H.iloc[0]
        if current != season:
            for s in state.values():
                for metric in ['ELO', 'NET10', 'NET30']:
                    s[metric] *= .75
            season = current
        updates = []
        for r in group.itertuples():
            for tid in [r.TEAM_ID_H, r.TEAM_ID_A]:
                state.setdefault(tid, {'ELO': 0., 'NET10': 0., 'NET30': 0., 'dates': []})
            h, a = state[r.TEAM_ID_H], state[r.TEAM_ID_A]
            def schedule(s):
                dates = s['dates']
                rest = (day - dates[-1]).days if dates else 999
                return int(rest == 1), sum(0 < (day - t).days <= 7 for t in dates)
            hb, hg = schedule(h)
            ab, ag = schedule(a)
            rows.append({'GAME_ID': r.GAME_ID, 'ELO_DIFF': h['ELO'] - a['ELO'],
                         'NET10_DIFF': h['NET10'] - a['NET10'],
                         'NET30_DIFF': h['NET30'] - a['NET30'],
                         'B2B_DIFF': hb - ab, 'GAMES7_DIFF': hg - ag})
            poss_h = r.FGA_H + .44 * r.FTA_H - r.OREB_H + r.TOV_H
            poss_a = r.FGA_A + .44 * r.FTA_A - r.OREB_A + r.TOV_A
            poss = (poss_h + poss_a) / 2
            if poss <= 0 or r.PTS_H == r.PTS_A:
                raise ValueError('Invalid completed game')
            net = 100 * (r.PTS_H - r.PTS_A) / poss
            expected = 1 / (1 + 10 ** (-(h['ELO'] - a['ELO'] + 65) / 400))
            delta = 20 * (int(r.PTS_H > r.PTS_A) - expected)
            updates.extend([(r.TEAM_ID_H, delta, net), (r.TEAM_ID_A, -delta, -net)])
        for tid, delta, net in updates:
            s = state[tid]
            s['ELO'] += delta
            for span in [10, 30]:
                alpha = 2 / (span + 1)
                s[f'NET{span}'] = (1 - alpha) * s[f'NET{span}'] + alpha * net
            s['dates'] = [t for t in s['dates'] if (day - t).days <= 7] + [day]
    return pd.DataFrame(rows)


def predict_arms(games):
    chunks, folds = [], []
    for season in sorted(games.SEASON.unique()):
        if season < '2021-22':
            continue
        tr, te = games[games.SEASON < season], games[games.SEASON == season].copy()
        if tr.GAME_DATE.max() >= te.GAME_DATE.min():
            raise ValueError('Overlapping training/test periods')
        fold = {'season': season, 'train_rows': len(tr), 'test_rows': len(te),
                'train_end': str(tr.GAME_DATE.max().date()), 'test_start': str(te.GAME_DATE.min().date())}
        for arm, cols in ARMS.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000))
            model.fit(tr[cols], tr.WIN)
            te[f'P_{arm}'] = model.predict_proba(te[cols])[:, 1]
            fold[arm] = dict(zip(cols, model[-1].coef_[0].tolist()))
        folds.append(fold)
        chunks.append(te)
    return pd.concat(chunks, ignore_index=True), folds


def paired_loss(frame, left, right):
    """Paired mean log-loss difference, with dates resampled as whole blocks."""
    def losses(p):
        p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
        return -(frame.WIN * np.log(p) + (1 - frame.WIN) * np.log(1 - p))
    delta = losses(frame[left]) - losses(frame[right])
    day = pd.DataFrame({'date': frame.GAME_DATE, 'delta': delta}).groupby('date').delta.agg(['sum', 'count'])
    rng = np.random.default_rng(42)
    sampled = day.to_numpy()[rng.integers(len(day), size=(2000, len(day)))].sum(axis=1)
    ci = np.quantile(sampled[:, 0] / sampled[:, 1], [.025, .975]).tolist()
    return {'left': left, 'right': right, 'mean_delta': float(delta.mean()),
            'day_bootstrap_95': ci, 'negative_favors_left': True}


def main():
    paths = [ROOT / p for p in ['data/raw/team_game_logs.csv', 'data/processed/pregame_features.csv',
             'data/processed/player_strength_features.csv', 'data/raw/betting_odds_mgm.csv']]
    raw, pre, ps, odds = [pd.read_csv(p) for p in paths]
    home_counts = raw.assign(IS_HOME=raw.MATCHUP.str.contains('vs.', regex=False)).groupby('GAME_ID').IS_HOME.agg(['sum', 'count'])
    invalid_ids = home_counts.index[(home_counts['sum'] != 1) | (home_counts['count'] != 2)].tolist()
    raw = raw[~raw.GAME_ID.isin(invalid_ids)]
    # Deliberately do not use legacy full-history-average team features.
    games = pre[['GAME_ID', 'SEASON', 'GAME_DATE', 'TEAM', 'OPP', 'WIN', 'REST_DAYS_DIFF']].merge(
        ps[['GAME_ID', 'LAST_GAME_STRENGTH_DIFF']], on='GAME_ID', validate='one_to_one')
    games = games.merge(build_context(raw), on='GAME_ID', validate='one_to_one')
    games['GAME_DATE'] = pd.to_datetime(games.GAME_DATE)
    base, folds = predict_arms(games)
    matched, audit = match_odds(base, odds)
    audit['invalid_raw_home_away_game_ids'] = invalid_ids
    audit['excluded_pregame_rows'] = len(pre) - len(games)
    predictions = None
    meta_folds = {}
    for arm in ARMS:
        candidate = matched.assign(P_PLAYER=matched[f'P_{arm}'])
        scored, meta_folds[arm] = market_predictions(candidate)
        if predictions is None:
            predictions = scored.drop(columns=['P_PLAYER', 'P_HYBRID'])
        predictions[f'HYBRID_{arm}'] = scored.P_HYBRID.to_numpy()
    columns = ['P_MARKET', 'P_CALIBRATED'] + [f'P_{a}' for a in ARMS] + [f'HYBRID_{a}' for a in ARMS]
    metrics, ledgers = [], []
    for col in columns:
        ledger = settle(predictions, col)
        ledgers.append(ledger.assign(MODEL=col))
        for season in ['ALL'] + sorted(predictions.SEASON.unique()):
            part = predictions if season == 'ALL' else predictions[predictions.SEASON == season]
            bets = ledger if season == 'ALL' else ledger[ledger.SEASON == season]
            metrics.append({'model': col, 'season': season, 'games': len(part),
                            'accuracy': float(((part[col] >= .5) == part.WIN).mean()),
                            'log_loss': float(log_loss(part.WIN, part[col])),
                            'brier': float(brier_score_loss(part.WIN, part[col])), **bet_summary(bets)})
    comparisons = [paired_loss(predictions, 'P_combined', 'P_players'),
                   paired_loss(predictions, 'P_combined', 'P_context'),
                   paired_loss(predictions, 'HYBRID_combined', 'P_MARKET'),
                   paired_loss(predictions, 'HYBRID_combined', 'P_CALIBRATED')]
    out = ROOT / 'reports/context_experiment'
    out.mkdir(parents=True, exist_ok=True)
    report = {'arms': ARMS, 'base_folds': folds, 'meta_folds': meta_folds, 'matching': audit,
              'metrics': metrics, 'comparisons': comparisons,
              'fixed_parameters': {'C': 1., 'elo_K': 20, 'elo_home': 65, 'season_decay': .75,
                                   'ewm_spans': [10, 30], 'bet_threshold': .03, 'unit_stake': 1},
              'limitations': ['Previously examined historical seasons: exploratory, not a fresh holdout.',
                              'Closing odds lack observation timestamps; no executable edge established.',
                              'Day bootstrap ignores across-day dependence and model-selection uncertainty.',
                              'Raw home designation does not distinguish neutral-site games.',
                              'Existing player/rest features reused; only new context features have event-causality tests.'],
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in paths + [Path(__file__), ROOT / 'src/market_research.py']}}
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    predictions.to_csv(out / 'predictions.csv', index=False)
    pd.concat(ledgers).to_csv(out / 'ledger.csv', index=False)
    print(pd.DataFrame(metrics).query("season == 'ALL'").to_string(index=False))
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
