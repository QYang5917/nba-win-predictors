import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from availability_experiment import build_pool, crossfit, FEATURES, fit_wins, ARMS


def fixture():
    rows = []
    for i, players in enumerate([[1,2], [2,3], [2,4], [1,2], [2,3], [2,3], [2,3], [1,2]]):
        for p in players:
            rows.append(dict(GAME_ID=str(i), TEAM_ID=10, PLAYER_ID=p, MIN=30.,
                             GAME_DATE=f'2023-11-{i+1:02d}', SEASON='2023-24'))
    return pd.DataFrame(rows)


class AvailabilityTests(unittest.TestCase):
    def test_returnee_in_pool_without_reading_current_lineup(self):
        raw = fixture()
        pool = build_pool(raw)
        at_third = pool[pool.GAME_ID=='2']
        self.assertEqual(set(at_third.PLAYER_ID), {1,2,3})
        self.assertNotIn(4, set(at_third.PLAYER_ID))
        self.assertEqual(at_third.set_index('PLAYER_ID').loc[1, 'MISSED_TEAM_GAMES'], 1)
        # Today's participants change targets, not today's candidate features.
        changed = raw.copy()
        changed.loc[changed.GAME_ID=='2','PLAYER_ID'] = [40,41]
        alternate = build_pool(changed)
        cols = ['GAME_ID','TEAM_ID','PLAYER_ID']+FEATURES+['LEGACY_MIN']
        pd.testing.assert_frame_equal(pool.loc[pool.GAME_ID<='2',cols].reset_index(drop=True),
                                      alternate.loc[alternate.GAME_ID<='2',cols].reset_index(drop=True))

    def test_window_eviction_and_known_other_team(self):
        raw = fixture()
        # Player 4 appears only in game 2 and expires before a ninth game.
        extra = raw.iloc[[-1]].assign(GAME_ID='8',GAME_DATE='2023-11-09')
        pool = build_pool(pd.concat([raw,extra],ignore_index=True))
        self.assertNotIn(4,set(pool.loc[pool.GAME_ID=='8','PLAYER_ID']))
        # Player 1 is observed with another team before original team's next game.
        other = raw.iloc[[0]].assign(GAME_ID='trade',TEAM_ID=20,GAME_DATE='2023-11-02')
        pool = build_pool(pd.concat([raw,other],ignore_index=True))
        p = pool[(pool.GAME_ID=='2') & (pool.PLAYER_ID==1)].iloc[0]
        self.assertEqual(p.KNOWN_OTHER_TEAM,1)

    def test_future_targets_cannot_train_current_participation(self):
        rng = np.random.default_rng(42)
        rows = []
        for year in [2016,2017,2018]:
            d = pd.DataFrame({c:rng.uniform(0,30,250) for c in FEATURES})
            d['KNOWN_OTHER_TEAM'] = 0
            d['SEASON'],d['GAME_DATE'] = f'{year}-{str(year+1)[-2:]}',pd.Timestamp(f'{year}-11-01')
            d['ACTUAL_MIN'] = rng.choice([0.,20.,30.],len(d))
            rows.append(d)
        data = pd.concat(rows,ignore_index=True)
        a,_ = crossfit(data)
        data.loc[data.SEASON=='2018-19','ACTUAL_MIN'] = 0
        b,_ = crossfit(data)
        np.testing.assert_allclose(a[['P_PLAY','HURDLE_MIN','DIRECT_MIN']],b[['P_PLAY','HURDLE_MIN','DIRECT_MIN']])
        self.assertTrue(a.P_PLAY.between(0,1).all())
        np.testing.assert_allclose(a.HURDLE_MIN,a.P_PLAY*a.CONDITIONAL_MIN)

    def test_win_stage_ignores_current_outcomes(self):
        rng = np.random.default_rng(9)
        rows = []
        for year in [2018,2019,2020,2021]:
            d = pd.DataFrame({c:rng.normal(size=60) for cols in ARMS.values() for c in cols})
            d['SEASON'],d['GAME_DATE'] = f'{year}-{str(year+1)[-2:]}',pd.Timestamp(f'{year}-11-01')
            d['WIN'] = rng.integers(0,2,len(d))
            rows.append(d)
        data = pd.concat(rows,ignore_index=True)
        a,_ = fit_wins(data)
        data.loc[data.SEASON=='2021-22','WIN'] = 1-data.loc[data.SEASON=='2021-22','WIN']
        b,_ = fit_wins(data)
        for name in ARMS:
            np.testing.assert_allclose(a['P_'+name],b['P_'+name])


if __name__ == '__main__':
    unittest.main()
