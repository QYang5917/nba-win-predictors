import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pregame_snapshots import record, validate
from rotation_research import allocate_minutes, roster_priors, rotation_scenarios, from_snapshot, learned_minutes, MINUTE_FEATURES


class RotationTests(unittest.TestCase):
    def test_caps_and_missing_capacity(self):
        result, missing = allocate_minutes([40, 35, 30, 25, 20, 15, 10, 5])
        self.assertAlmostEqual(result.sum(), 240)
        self.assertLessEqual(result.max(), 40)
        self.assertAlmostEqual(missing, 0)
        result, missing = allocate_minutes([40, 40, 0])
        self.assertEqual(result.tolist(), [40, 40, 0])
        self.assertEqual(missing, 160)
        with self.assertRaises(ValueError):
            allocate_minutes([np.nan])

    def test_returnee_rookie_and_future_exclusion(self):
        roster = pd.DataFrame({'PLAYER_ID': [1, 2, 3], 'TEAM_ID': [10] * 3})
        logs = pd.DataFrame({'PLAYER_ID': [1, 2, 2], 'SEASON': ['2025-26', '2026-27', '2026-27'],
                             'GAME_DATE': ['2026-04-01', '2026-11-01', '2026-11-12'], 'MIN': [32., 25., 48.]})
        result = roster_priors(logs, roster, '2026-27', '2026-11-12')
        self.assertEqual(result.BASE_MIN.tolist(), [32., 25., 12.])
        self.assertEqual(result.PRIOR_SOURCE.iloc[0], 'previous_season')
        changed = logs.copy()
        changed.loc[2, 'MIN'] = 1
        pd.testing.assert_frame_equal(result, roster_priors(changed, roster, '2026-27', '2026-11-12'))

    def test_uncertainty_is_scenarios_not_certain_play(self):
        prior = pd.DataFrame({'PLAYER_ID': range(8), 'TEAM_ID': [1] * 8, 'BASE_MIN': [30.] * 8})
        statuses = pd.DataFrame({'PLAYER_ID': range(8), 'STATUS': ['Out', 'Questionable'] + ['Available'] * 6})
        result = rotation_scenarios(prior, statuses)
        self.assertTrue((result.loc[result.PLAYER_ID == 0, 'EXPECTED_MIN'] == 0).all())
        q = result[result.PLAYER_ID == 1].set_index('SCENARIO').EXPECTED_MIN
        self.assertGreater(q['uncertain_in'], 0)
        self.assertEqual(q['uncertain_out'], 0)
        with self.assertRaises(ValueError):
            rotation_scenarios(prior, statuses.iloc[:-1])

    def test_future_labels_do_not_change_learned_minutes(self):
        rng = np.random.default_rng(42)
        rows = []
        for year in [2022, 2023]:
            d = pd.DataFrame({c: rng.uniform(1, 30, 300) for c in MINUTE_FEATURES})
            d['SEASON'] = f'{year}-{str(year+1)[-2:]}'
            d['GAME_DATE'] = f'{year}-11-01'
            d['ACTUAL_REG_MIN'] = rng.uniform(0, 40, 300)
            rows.append(d)
        ledger = pd.concat(rows, ignore_index=True)
        a, _ = learned_minutes(ledger)
        ledger.loc[ledger.SEASON == '2023-24', 'ACTUAL_REG_MIN'] = 0
        b, _ = learned_minutes(ledger)
        np.testing.assert_allclose(a[['LEARNED_MEAN', 'LEARNED_MEDIAN']], b[['LEARNED_MEAN', 'LEARNED_MEDIAN']])


class SnapshotTests(unittest.TestCase):
    def payload(self):
        return dict(kind='odds', game_id='0022600001', source='test fixture',
                    observed_at='2026-11-12T19:59:00Z', tip_utc='2026-11-13T01:00:00Z',
                    bookmaker='TEST', market='moneyline', home_decimal=1.9, away_decimal=2.0)

    def test_receipt_timestamp_and_no_overwrite(self):
        now = dt.datetime(2026, 11, 12, 20, tzinfo=dt.timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            p = record(self.payload(), tmp, now)
            self.assertEqual(json.loads(p.read_text())['received_at'], now.isoformat())
            with self.assertRaises(FileExistsError):
                record(self.payload(), tmp, now)

    def test_reject_future_stale_posttip_and_bad_odds(self):
        for edits in [dict(observed_at='2026-11-12T20:01:00Z'),
                      dict(observed_at='2026-11-12T19:00:00Z'),
                      dict(tip_utc='2026-11-12T20:00:00Z'), dict(home_decimal=float('nan')),
                      dict(observed_at='2026-11-12T19:59:00')]:
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                validate({**self.payload(), **edits}, '2026-11-12T20:00:00Z')

    def test_snapshot_integration_and_future_receipt(self):
        payload = dict(kind='availability', game_id='0022600001', source='test fixture', team_id=10,
                       observed_at='2026-11-12T19:59:00Z', tip_utc='2026-11-13T01:00:00Z',
                       players=[dict(player_id=p, status='Available') for p in range(8)])
        snapshot = dict(received_at='2026-11-12T20:00:00Z', payload=payload)
        roster = pd.DataFrame({'PLAYER_ID': range(8), 'TEAM_ID': [10] * 8})
        logs = pd.DataFrame(columns=['PLAYER_ID', 'SEASON', 'GAME_DATE', 'MIN'])
        r = from_snapshot(logs, roster, snapshot, '2026-27', '2026-11-12T20:01:00Z')
        self.assertTrue(np.allclose(r.groupby('SCENARIO').EXPECTED_MIN.sum(), 240))
        with self.assertRaises(ValueError):
            from_snapshot(logs, roster, snapshot, '2026-27', '2026-11-12T19:59:30Z')
        with self.assertRaises(ValueError):
            from_snapshot(logs, roster.iloc[:-1], snapshot, '2026-27', '2026-11-12T20:01:00Z')


if __name__ == '__main__':
    unittest.main()
