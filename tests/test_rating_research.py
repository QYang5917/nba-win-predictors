import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ratings import STATS
from rating_research import ARMS, joint_fit, recency_weights, rating_variants
from rating_experiment import candidates_from_past, predict_wins


class ZeroBox:
    def predict(self, x):
        return np.zeros(len(x))


class RatingResearchTests(unittest.TestCase):
    def logs(self):
        rows = []
        for year in [2020, 2021, 2022]:
            for day in [1, 3, 5]:
                for pid in range(1, 5):
                    team = 1 if pid <= 2 else 2
                    row = dict(GAME_ID=f'{year}{day:06}', GAME_DATE=f'{year}-11-{day:02}',
                        SEASON=f'{year}-{str(year+1)[-2:]}', PLAYER_ID=pid, PLAYER_NAME=f'P{pid}',
                        TEAM_ID=team, MIN=24., MATCHUP='A vs. B' if team == 1 else 'B @ A')
                    row.update({s: float(pid + day) for s in STATS})
                    rows.append(row)
        return pd.DataFrame(rows)

    def test_joint_fit_matches_independent_dense_normal_equations(self):
        rng = np.random.default_rng(3)
        x, h, y = rng.normal(size=(25, 4)), rng.choice([-1., 1.], 25), rng.normal(size=25)
        prior, w, lam = rng.normal(size=4), rng.uniform(.2, 1., 25), 7.
        z = np.column_stack([x, h])
        penalty = np.diag([lam] * 4 + [0])
        expected = np.linalg.solve(z.T @ (w[:, None] * z) + penalty,
                                   z.T @ (w * y) + np.r_[lam * prior, 0])
        ratings, home = joint_fit(sparse.csr_matrix(x), y, h, prior, w, lam)
        np.testing.assert_allclose(np.r_[ratings, home], expected, atol=1e-8)
        scaled, scaled_home = joint_fit(sparse.csr_matrix(x), y, h, prior, w * 3, lam * 3)
        np.testing.assert_allclose(np.r_[scaled, scaled_home], expected, atol=1e-8)

    def test_decay_and_same_day_exclusion(self):
        cutoff = pd.Timestamp('2025-01-01')
        dates = pd.Series([cutoff - pd.Timedelta(days=d) for d in [365, 730]])
        np.testing.assert_allclose(recency_weights(dates, cutoff), [.5, .25])
        with self.assertRaises(ValueError):
            recency_weights(pd.Series([cutoff]), cutoff)

    def test_stronger_penalty_reduces_aggregate_prior_deviation(self):
        rng=np.random.default_rng(33)
        x=sparse.csr_matrix(rng.normal(size=(80,8)))
        y=rng.normal(size=80); h=rng.choice([-1.,1.],80)
        prior=rng.normal(size=8); w=rng.uniform(.1,1.,80)
        normal,_=joint_fit(x,y,h,prior,w,lam=30)
        strong,_=joint_fit(x,y,h,prior,w,lam=120)
        self.assertLess(np.linalg.norm(strong-prior),np.linalg.norm(normal-prior))

    def test_future_results_and_players_cannot_change_rating(self):
        logs = self.logs()
        ages = pd.DataFrame({'PLAYER_ID': range(1, 5), 'SEASON': ['2022-23']*4, 'AGE': [25]*4})
        kwargs = dict(season='2022-23', box_model=ZeroBox(), age_curve=lambda a: np.zeros(len(a)),
                      ages=ages, cutoff='2022-11-03')
        a = rating_variants(logs, **kwargs)
        logs.loc[logs.GAME_DATE >= '2022-11-03', STATS] = 9999.
        logs.loc[logs.GAME_DATE >= '2022-11-03', 'PLAYER_ID'] += 100
        b = rating_variants(logs.sample(frac=1, random_state=7), **kwargs)
        for arm in ARMS:
            np.testing.assert_allclose(a[arm].sort_index().RATING, b[arm].sort_index().RATING)
            self.assertEqual(set(a[arm].index), {1, 2, 3, 4})
            np.testing.assert_allclose(a[arm].RATING,
                a[arm].BOX_RATING + a[arm].IMPACT_ADJ + a[arm].AGE_ADJ)
        scale = a['recent_normalized'].attrs['weight_scale']
        self.assertGreater(scale, 1)
        np.testing.assert_allclose(a['recent_normalized'].EFFECTIVE_MIN,
                                   a['recent_365'].EFFECTIVE_MIN * scale)
        np.testing.assert_allclose(a['recent_normalized'].AGE_ADJ, a['recent_365'].AGE_ADJ)

    def test_current_participants_cannot_change_same_game_candidates(self):
        logs = self.logs()
        a = candidates_from_past(logs)
        logs.loc[logs.GAME_DATE == '2022-11-05', 'PLAYER_ID'] += 100
        logs.loc[logs.GAME_DATE == '2022-11-05', 'MIN'] = 3.
        b = candidates_from_past(logs)
        cols = ['PLAYER_ID', 'EXPECTED_MIN']
        pd.testing.assert_frame_equal(a.loc[a.GAME_DATE == '2022-11-05', cols].reset_index(drop=True),
                                     b.loc[b.GAME_DATE == '2022-11-05', cols].reset_index(drop=True))

    def test_evaluation_labels_cannot_change_win_forecasts(self):
        rng = np.random.default_rng(12)
        frames = []
        for year in [2018, 2019, 2020, 2021]:
            d = pd.DataFrame({arm+'_DIFF': rng.normal(size=40) for arm in ARMS})
            d['REST_DAYS_DIFF'] = rng.normal(size=40)
            d['WIN'] = rng.integers(0, 2, 40)
            d['SEASON'], d['GAME_DATE'] = f'{year}-{str(year+1)[-2:]}', f'{year}-11-01'
            frames.append(d)
        games = pd.concat(frames, ignore_index=True)
        a, _ = predict_wins(games)
        games.loc[games.SEASON == '2021-22', 'WIN'] = 1 - games.loc[games.SEASON == '2021-22', 'WIN']
        b, _ = predict_wins(games)
        for arm in ARMS:
            np.testing.assert_allclose(a['P_'+arm], b['P_'+arm])


if __name__ == '__main__':
    unittest.main()
