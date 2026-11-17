import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from style_experiment import style_features, predictions, ARMS


class StyleTests(unittest.TestCase):
    def test_same_day_and_future_stats_do_not_change_features(self):
        rows=[]
        for i,day in enumerate(pd.date_range('2023-01-01',periods=30)):
            for team in [1,2]:
                rows.append(dict(GAME_ID=str(i),TEAM_ID=team,GAME_DATE=str(day.date()),
                    MATCHUP='A vs. B' if team==1 else 'B @ A',FG3M=10,FG3A=30,
                    FGA=80,FTA=20,OREB=10,DREB=30,TOV=12))
        d=pd.DataFrame(rows)
        a=style_features(d).set_index('GAME_ID')
        d.loc[d.GAME_DATE>='2023-01-21',['FG3M','OREB','DREB']]=99
        b=style_features(d).set_index('GAME_ID')
        pd.testing.assert_frame_equal(a.loc[[str(i) for i in range(10,21)]],
                                     b.loc[[str(i) for i in range(10,21)]])

    def test_test_labels_cannot_change_predictions(self):
        rng=np.random.default_rng(8)
        cols=set(c for v in ARMS.values() for c in v)
        parts=[]
        for year in range(2021,2026):
            d=pd.DataFrame({c:rng.normal(size=40) for c in cols})
            d['WIN']=rng.integers(0,2,40)
            d['SEASON']=f'{year}-{str(year+1)[-2:]}'
            d['GAME_DATE']=f'{year}-11-01'
            parts.append(d)
        d=pd.concat(parts,ignore_index=True)
        a=predictions(d)
        d.loc[d.SEASON=='2025-26','WIN']=1-d.loc[d.SEASON=='2025-26','WIN']
        b=predictions(d)
        for arm in ARMS:
            np.testing.assert_allclose(a['P_'+arm],b['P_'+arm])
