import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from market_research import match_odds, market_predictions, base_predictions, settle, bet_summary


class MarketResearchTests(unittest.TestCase):
    def test_payouts_home_away_and_no_bet(self):
        d = pd.DataFrame({'GAME_ID': [1, 2, 3], 'GAME_DATE': ['2025-01-01'] * 3,
                          'SEASON': ['2024-25'] * 3, 'WIN': [1, 0, 1], 'P': [.7, .2, .5],
                          'money_home_decimal_odds': [2., 2., 1.9],
                          'money_away_decimal_odds': [2., 2., 1.9]})
        r = settle(d, 'P')
        self.assertEqual(r.SIDE.tolist(), ['HOME', 'AWAY', 'NONE'])
        self.assertEqual(r.PNL.tolist(), [1., 1., 0.])
        np.testing.assert_allclose(settle(d, 'P', haircut=.1).PNL, [.9, .9, 0.])
        d.loc[0, 'WIN'] = 0
        self.assertEqual(settle(d, 'P').PNL.iloc[0], -1)
        d.loc[0, 'P'] = np.nan
        with self.assertRaises(ValueError):
            settle(d, 'P')

    def test_back_to_back_matches_not_swapped(self):
        g = pd.DataFrame({'GAME_ID': [1, 2], 'GAME_DATE': pd.to_datetime(['2025-01-01', '2025-01-02']),
                          'TEAM': ['BOS'] * 2, 'OPP': ['NYK'] * 2, 'WIN': [1, 0]})
        o = pd.DataFrame({'game_id': ['a', 'b'], 'game_date': ['2025-01-01', '2025-01-02'],
                          'home_team': ['Boston'] * 2, 'away_team': ['New York'] * 2,
                          'money_home_decimal_odds': [1.5, 1.7], 'money_away_decimal_odds': [3., 2.5],
                          'money_home_won': [True, False]})
        m, _ = match_odds(g, pd.concat([o, o.iloc[[0]]]))
        self.assertEqual(m.money_home_decimal_odds.tolist(), [1.5, 1.7])
        # Conflicting prices are not resolved by arbitrary first-row selection.
        changed = o.iloc[[0]].assign(money_home_decimal_odds=1.6)
        m, audit = match_odds(g, pd.concat([o, changed]))
        self.assertEqual(m.GAME_ID.tolist(), [2])
        self.assertEqual(audit['conflicting_quote_rows'], 2)
        # Outcome labels may diagnose source issues but never select matches.
        o['money_home_won'] = ~o.money_home_won
        m, audit = match_odds(g, o)
        self.assertEqual(len(m), 2)
        self.assertEqual(len(audit['source_outcome_disagreements']), 2)

    def test_future_labels_do_not_change_earlier_predictions(self):
        rng = np.random.default_rng(7)
        chunks = []
        for y in range(2020, 2026):
            n = 600
            chunks.append(pd.DataFrame({'GAME_ID': np.arange(n) + y * 1000,
                'SEASON': f'{y}-{str(y+1)[-2:]}',
                'GAME_DATE': pd.date_range(f'{y}-10-01', periods=n, freq='4h'),
                'WIN': rng.integers(0, 2, n),
                'LAST_GAME_STRENGTH_DIFF': rng.normal(size=n), 'REST_DAYS_DIFF': rng.normal(size=n)}))
        g = pd.concat(chunks, ignore_index=True)
        original, _ = base_predictions(g)
        changed = g.copy()
        changed.loc[changed.SEASON >= '2023-24', 'WIN'] = 1 - changed.loc[changed.SEASON >= '2023-24', 'WIN']
        altered, _ = base_predictions(changed)
        np.testing.assert_allclose(original.loc[original.SEASON <= '2023-24', 'P_PLAYER'],
                                   altered.loc[altered.SEASON <= '2023-24', 'P_PLAYER'])
        original['P_MARKET'] = rng.uniform(.15, .85, len(original))
        changed = original.copy()
        changed.loc[changed.SEASON >= '2023-24', 'WIN'] = 1 - changed.loc[changed.SEASON >= '2023-24', 'WIN']
        a, _ = market_predictions(original)
        b, _ = market_predictions(changed)
        for col in ['P_CALIBRATED', 'P_HYBRID']:
            np.testing.assert_allclose(a.loc[a.SEASON == '2023-24', col], b.loc[b.SEASON == '2023-24', col])

    def test_drawdown_includes_initial_loss_and_zero_bets(self):
        d = pd.DataFrame({'GAME_DATE': ['2025-01-01', '2025-01-02'], 'PNL': [-1., -1.], 'STAKE': [1, 1]})
        self.assertEqual(bet_summary(d)['max_daily_drawdown_units'], 2)
        d[['PNL', 'STAKE']] = 0
        self.assertIsNone(bet_summary(d)['roi'])


if __name__ == '__main__':
    unittest.main()
