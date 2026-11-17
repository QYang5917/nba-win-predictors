"""Walk-forward moneyline research. Closing-price diagnostics, NOT executable bets.

Never writes frozen models, live logs or raw inputs. No strategy search.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['LAST_GAME_STRENGTH_DIFF', 'REST_DAYS_DIFF']
CITY = dict(zip(
    ['Atlanta', 'Boston', 'Brooklyn', 'Charlotte', 'Chicago', 'Cleveland', 'Dallas',
     'Denver', 'Detroit', 'Golden State', 'Houston', 'Indiana', 'LA Clippers',
     'LA Lakers', 'Memphis', 'Miami', 'Milwaukee', 'Minnesota', 'New Orleans',
     'New York', 'Oklahoma City', 'Orlando', 'Philadelphia', 'Phoenix', 'Portland',
     'Sacramento', 'San Antonio', 'Toronto', 'Utah', 'Washington'],
    ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU',
     'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL',
     'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def match_odds(games, odds):
    """Require exact team/date and unique matches; quarantine conflicting quotes."""
    cols = ['game_id', 'game_date', 'home_team', 'away_team',
            'money_home_decimal_odds', 'money_away_decimal_odds', 'money_home_won']
    o = odds[cols].drop_duplicates().copy()
    audit = {'raw_odds_rows': len(odds), 'identical_quote_duplicates': len(odds) - len(o)}
    conflict = o.game_id.duplicated(keep=False)
    audit['conflicting_quote_rows'] = int(conflict.sum())
    o = o[~conflict].copy()
    o['HOME'], o['AWAY'] = o.home_team.map(CITY), o.away_team.map(CITY)
    o['ODDS_DATE'] = pd.to_datetime(o.game_date.str[:10], errors='coerce')
    valid = o[['HOME', 'AWAY', 'ODDS_DATE']].notna().all(axis=1)
    for col in ['money_home_decimal_odds', 'money_away_decimal_odds']:
        o[col] = pd.to_numeric(o[col], errors='coerce')
        valid &= np.isfinite(o[col]) & (o[col] > 1)
    audit['invalid_quote_rows'] = int((~valid).sum())
    o = o[valid]
    m = games.merge(o, left_on=['TEAM', 'OPP'], right_on=['HOME', 'AWAY'])
    m['DATE_GAP'] = (m.GAME_DATE - m.ODDS_DATE).dt.days.abs()
    m = m[m.DATE_GAP == 0].copy()
    ambiguous = m.GAME_ID.duplicated(keep=False) | m.game_id.duplicated(keep=False)
    audit['ambiguous_match_rows'] = int(ambiguous.sum())
    m = m[~ambiguous].copy()
    if not m.money_home_won.isin([True, False, 0, 1]).all():
        raise ValueError('Invalid source winner labels')
    # Source settlement labels do not select games or bets. NBA outcomes settle
    # all exact-identity matches; retain discrepancies explicitly for review.
    audit['source_outcome_disagreements'] = m.loc[
        m.money_home_won.astype(int) != m.WIN, ['GAME_ID', 'game_id']].to_dict('records')
    ih, ia = 1 / m.money_home_decimal_odds, 1 / m.money_away_decimal_odds
    m['P_MARKET'] = ih / (ih + ia)
    audit.update(matched_games=len(m), one_day_matches=int((m.DATE_GAP == 1).sum()),
                 unmatched_feature_games=len(games) - len(m))
    return m.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True), audit


def base_predictions(games):
    """Every season's player prediction comes from earlier seasons only."""
    chunks, folds = [], []
    for season in sorted(games.SEASON.unique()):
        if season < '2021-22':
            continue
        tr, te = games[games.SEASON < season], games[games.SEASON == season].copy()
        if tr.GAME_DATE.max() >= te.GAME_DATE.min():
            raise ValueError('Overlapping chronological folds')
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(tr[FEATURES], tr.WIN)
        te['P_PLAYER'] = model.predict_proba(te[FEATURES])[:, 1]
        chunks.append(te)
        folds.append({'season': season, 'train_rows': len(tr), 'test_rows': len(te),
                      'train_end': str(tr.GAME_DATE.max().date()),
                      'test_start': str(te.GAME_DATE.min().date())})
    return pd.concat(chunks, ignore_index=True), folds


def market_predictions(matched):
    """Calibrator and hybrid see only prior-season, out-of-fold predictions."""
    m = matched.copy()
    m['MARKET_LOGIT'], m['PLAYER_LOGIT'] = logit(m.P_MARKET), logit(m.P_PLAYER)
    chunks, folds = [], []
    for season in sorted(m.SEASON.unique()):
        if season < '2023-24':
            continue
        tr, te = m[m.SEASON < season], m[m.SEASON == season].copy()
        if len(tr) < 500 or tr.GAME_DATE.max() >= te.GAME_DATE.min():
            raise ValueError('Insufficient or overlapping market training data')
        fold = {'season': season, 'train_rows': len(tr), 'test_rows': len(te),
                'train_end': str(tr.GAME_DATE.max().date()),
                'test_start': str(te.GAME_DATE.min().date())}
        for name, cols in [('P_CALIBRATED', ['MARKET_LOGIT']),
                           ('P_HYBRID', ['MARKET_LOGIT', 'PLAYER_LOGIT'])]:
            model = LogisticRegression(C=1.0, max_iter=2000).fit(tr[cols], tr.WIN)
            te[name] = model.predict_proba(te[cols])[:, 1]
            fold[name] = {'coef': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0])}
        chunks.append(te)
        folds.append(fold)
    return pd.concat(chunks, ignore_index=True), folds


def settle(frame, probability, threshold=0.03, haircut=0.0):
    """One flat unit per game at most; payout haircut applies before selection."""
    if not 0 <= haircut < 1 or threshold < 0:
        raise ValueError('Invalid cost or threshold')
    p = np.asarray(frame[probability], dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities')
    h = 1 + (frame.money_home_decimal_odds.to_numpy() - 1) * (1 - haircut)
    a = 1 + (frame.money_away_decimal_odds.to_numpy() - 1) * (1 - haircut)
    if not np.isfinite(np.r_[h, a]).all() or (np.r_[h, a] <= 1).any():
        raise ValueError('Invalid odds')
    eh, ea = p * h - 1, (1 - p) * a - 1
    home = eh >= ea
    bet = np.maximum(eh, ea) > threshold
    win = np.where(home, frame.WIN.to_numpy() == 1, frame.WIN.to_numpy() == 0)
    out = frame[['GAME_ID', 'GAME_DATE', 'SEASON']].copy()
    out['SIDE'] = np.where(bet, np.where(home, 'HOME', 'AWAY'), 'NONE')
    out['EXPECTED_RETURN'] = np.maximum(eh, ea)
    out['STAKE'] = bet.astype(int)
    out['PNL'] = np.where(bet, np.where(win, np.where(home, h, a) - 1, -1), 0)
    return out


def bet_summary(ledger):
    days = ledger.groupby('GAME_DATE')[['PNL', 'STAKE']].sum().sort_index()
    n = int(days.STAKE.sum())
    curve = np.r_[0, days.PNL.cumsum().to_numpy()]
    ci = [None, None]
    if n:
        rng = np.random.default_rng(42)
        # Resample whole dates, preserving within-date correlations.
        arr = days.to_numpy()
        samples = arr[rng.integers(0, len(arr), size=(2000, len(arr)))].sum(axis=1)
        samples = samples[samples[:, 1] > 0]
        ci = np.quantile(samples[:, 0] / samples[:, 1], [0.025, 0.975]).tolist()
    return {'bets': n, 'profit_units': float(days.PNL.sum()),
            'roi': float(days.PNL.sum() / n) if n else None,
            'roi_95_day_bootstrap': ci,
            'max_daily_drawdown_units': float((np.maximum.accumulate(curve) - curve).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports' / 'market_research')
    args = parser.parse_args()
    paths = [ROOT / 'data/processed/pregame_features.csv',
             ROOT / 'data/processed/player_strength_features.csv', ROOT / 'data/raw/betting_odds_mgm.csv']
    pre, ps, odds = [pd.read_csv(p) for p in paths]
    games = pre.merge(ps[['GAME_ID', 'LAST_GAME_STRENGTH_DIFF']], on='GAME_ID', validate='one_to_one')
    games['GAME_DATE'] = pd.to_datetime(games.GAME_DATE)
    if not games.WIN.isin([0, 1]).all() or not np.isfinite(games[FEATURES]).all().all():
        raise ValueError('Invalid labels or features')
    base, base_folds = base_predictions(games)
    matched, audit = match_odds(base, odds)
    audit['pregame_rows_without_player_features'] = len(pre) - len(games)
    predictions, meta_folds = market_predictions(matched)
    summaries, ledgers, calibration = [], [], []
    for name in ['P_MARKET', 'P_PLAYER', 'P_CALIBRATED', 'P_HYBRID']:
        ledger = settle(predictions, name)
        ledger['MODEL'] = name
        ledgers.append(ledger)
        for season in ['ALL'] + sorted(predictions.SEASON.unique()):
            part = predictions if season == 'ALL' else predictions[predictions.SEASON == season]
            bets = ledger if season == 'ALL' else ledger[ledger.SEASON == season]
            summaries.append({'model': name, 'season': season, 'games': len(part),
                              'accuracy': float(((part[name] >= .5) == part.WIN).mean()),
                              'log_loss': float(log_loss(part.WIN, part[name], labels=[0, 1])),
                              'brier': float(brier_score_loss(part.WIN, part[name])), **bet_summary(bets)})
        bins = pd.cut(predictions[name], np.linspace(0, 1, 11), include_lowest=True)
        for interval, part in predictions.groupby(bins, observed=True):
            calibration.append({'model': name, 'bin': str(interval), 'games': len(part),
                                'mean_probability': float(part[name].mean()), 'home_win_rate': float(part.WIN.mean())})
    report = {'purpose': 'historical closing-price diagnostic; not proof of executable profit',
              'parameters': {'features': FEATURES, 'threshold': .03, 'stake_units': 1,
                             'meta_C': 1.0, 'bootstrap_seed': 42},
              'limitations': ['Historical evaluation seasons have already been examined in this project.',
                              'Odds have no verifiable observation timestamps; closing quotes may not be executable.',
                              'Uses existing historical feature artifacts; no historical injury snapshots.',
                              'Date bootstrap omits across-date dependence, model uncertainty and multiple testing.',
                              'ROI excludes taxes, limits, rejected wagers and account restrictions.'],
              'inputs_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'matching': audit, 'base_folds': base_folds, 'market_folds': meta_folds, 'metrics': summaries,
              'hybrid_1pct_net_payout_haircut': bet_summary(settle(predictions, 'P_HYBRID', haircut=.01))}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    predictions.to_csv(args.output / 'predictions.csv', index=False)
    pd.concat(ledgers).to_csv(args.output / 'ledger.csv', index=False)
    pd.DataFrame(calibration).to_csv(args.output / 'calibration.csv', index=False)
    print(json.dumps(audit, indent=2))
    print(pd.DataFrame(summaries).query("season == 'ALL'").to_string(index=False))


if __name__ == '__main__':
    main()
