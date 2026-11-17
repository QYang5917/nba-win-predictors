import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rotation_win_experiment import aggregate_strength, minute_predictions, win_predictions, ARMS
from rotation_research import MINUTE_FEATURES


class EndToEndRotationTests(unittest.TestCase):
    def test_reserve_is_explicit_and_excess_is_scaled_down(self):
        r = pd.DataFrame({'GAME_ID': ['a'] * 7, 'TEAM_ID': [1] * 7, 'PLAYER_ID': range(7),
                          'PRED_MIN': [40.] * 7, 'LEGACY_MIN': [30.] * 7, 'RATING': [1.] * 7})
        s = aggregate_strength(r).iloc[0]
        self.assertAlmostEqual(s.USED_MIN, 240)
        self.assertAlmostEqual(s.KNOWN, 5)
        self.assertAlmostEqual(s.RESERVE, 0)
        r['PRED_MIN'] = 20.
        s = aggregate_strength(r).iloc[0]
        self.assertAlmostEqual(s.USED_MIN, 140)
        self.assertAlmostEqual(s.RESERVE, 100 / 48)
        self.assertAlmostEqual(s.KNOWN, 140 / 48)
        self.assertAlmostEqual(s.LEARNED_FIXED, -60 / 48)
        with self.assertRaises(ValueError):
            aggregate_strength(pd.concat([r, r.iloc[[0]]]))

    def test_minute_crossfit_excludes_future_targets_and_drops_them(self):
        rng = np.random.default_rng(7)
        chunks = []
        for year in [2016, 2017, 2018]:
            d = pd.DataFrame({c: rng.uniform(1, 30, 220) for c in MINUTE_FEATURES})
            d['GAME_ID'] = [f'{year}-{i}' for i in range(len(d))]
            d['TEAM_ID'], d['PLAYER_ID'] = 1, np.arange(len(d))
            d['GAME_DATE'], d['SEASON'] = f'{year}-11-01', f'{year}-{str(year+1)[-2:]}'
            d['LEGACY_MIN'] = 20.
            d['ACTUAL_REG_MIN'] = rng.uniform(0, 40, len(d))
            chunks.append(d)
        ledger = pd.concat(chunks, ignore_index=True)
        a, _ = minute_predictions(ledger)
        ledger.loc[ledger.SEASON == '2018-19', 'ACTUAL_REG_MIN'] = 0
        b, _ = minute_predictions(ledger)
        np.testing.assert_allclose(a.PRED_MIN, b.PRED_MIN)
        self.assertNotIn('ACTUAL_REG_MIN', a.columns)

    def test_win_models_ignore_same_season_labels(self):
        rng = np.random.default_rng(9)
        chunks = []
        columns = set(c for cols in ARMS.values() for c in cols)
        for year in [2018, 2019, 2020, 2021]:
            d = pd.DataFrame({c: rng.normal(size=60) for c in columns})
            d['WIN'] = rng.integers(0, 2, len(d))
            d['GAME_DATE'], d['SEASON'] = f'{year}-11-01', f'{year}-{str(year+1)[-2:]}'
            chunks.append(d)
        data = pd.concat(chunks, ignore_index=True)
        a, _ = win_predictions(data)
        data.loc[data.SEASON == '2021-22', 'WIN'] = 1 - data.loc[data.SEASON == '2021-22', 'WIN']
        b, _ = win_predictions(data)
        for arm in ARMS:
            np.testing.assert_allclose(a[f'P_{arm}'], b[f'P_{arm}'])


if __name__ == '__main__':
    unittest.main()
