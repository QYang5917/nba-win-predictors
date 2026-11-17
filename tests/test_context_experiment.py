import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from context_experiment import build_context, predict_arms, CONTEXT


def raw_games():
    rows = []
    for i, day in enumerate(['2023-10-01', '2023-10-02', '2023-10-05', '2024-10-01']):
        for team, home, pts in [(1, True, 110), (2, False, 100)]:
            rows.append(dict(GAME_ID=i, GAME_DATE=day, TEAM_ID=team,
                             SEASON='2023-24' if i < 3 else '2024-25',
                             MATCHUP='A vs. B' if home else 'B @ A',
                             PTS=pts, FGA=90, FTA=20, OREB=10, TOV=10))
    return pd.DataFrame(rows)


class ContextTests(unittest.TestCase):
    def test_current_and_future_scores_cannot_change_pregame_features(self):
        raw = raw_games()
        before = build_context(raw)
        changed = raw.copy()
        changed.loc[(changed.GAME_ID >= 1) & (changed.TEAM_ID == 1), 'PTS'] = 50
        after = build_context(changed)
        pd.testing.assert_frame_equal(before[before.GAME_ID <= 1], after[after.GAME_ID <= 1])
        self.assertFalse(np.allclose(before.loc[2, CONTEXT].astype(float), after.loc[2, CONTEXT].astype(float)))
        # Removing every future game must give identical past features too.
        pd.testing.assert_frame_equal(before.iloc[:2].reset_index(drop=True),
                                      build_context(raw[raw.GAME_ID <= 1]))

    def test_initialization_and_season_decay(self):
        c = build_context(raw_games())
        self.assertTrue((c.loc[0, CONTEXT] == 0).all())
        self.assertGreater(c.loc[1, 'ELO_DIFF'], 0)
        self.assertGreater(c.loc[1, 'NET10_DIFF'], 0)
        self.assertEqual(c.loc[3, 'GAMES7_DIFF'], 0)
        self.assertEqual(c.loc[3, 'B2B_DIFF'], 0)

    def test_reject_duplicate_and_malformed_games(self):
        raw = raw_games()
        with self.assertRaises(ValueError):
            build_context(pd.concat([raw, raw.iloc[[0]]]))
        with self.assertRaises((ValueError, pd.errors.MergeError)):
            build_context(raw.iloc[1:])

    def test_all_arms_ignore_evaluation_labels(self):
        rng = np.random.default_rng(42)
        chunks = []
        for year in [2020, 2021, 2022]:
            n = 40
            d = pd.DataFrame({'SEASON': f'{year}-{str(year+1)[-2:]}',
                              'GAME_DATE': pd.date_range(f'{year}-10-01', periods=n),
                              'WIN': rng.integers(0, 2, n)})
            for col in CONTEXT + ['REST_DAYS_DIFF', 'LAST_GAME_STRENGTH_DIFF']:
                d[col] = rng.normal(size=n)
            chunks.append(d)
        games = pd.concat(chunks, ignore_index=True)
        a, _ = predict_arms(games)
        games.loc[games.SEASON == '2022-23', 'WIN'] = 1 - games.loc[games.SEASON == '2022-23', 'WIN']
        b, _ = predict_arms(games)
        np.testing.assert_allclose(a[['P_players', 'P_context', 'P_combined']],
                                   b[['P_players', 'P_context', 'P_combined']])


if __name__ == '__main__':
    unittest.main()
