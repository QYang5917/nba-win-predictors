import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from grade_predictions import score_games, wilson_interval
from audit_data import audit_teams


class DataAuditTests(unittest.TestCase):
    def predictions(self):
        return pd.DataFrame([dict(GAME_ID='1',HOME='A',AWAY='B',P_HOME=.7,
            PREDICTED_AT='2026-11-12T20:00:00Z',TIP_UTC='2026-11-13T00:00:00Z')])

    def test_actual_scores_override_bad_margin_and_home_marker(self):
        teams=pd.DataFrame({'GAME_ID':['1','1'],'TEAM_ABBREVIATION':['A','B'],
                            'PTS':[100,110],'PLUS_MINUS':[50,-50],'MATCHUP':['A @ B','A @ B']})
        r=score_games(self.predictions(),teams)
        self.assertEqual(r.AWAY_PTS.iloc[0],110)
        self.assertFalse(r.CORRECT.iloc[0])
        self.assertFalse(r.HOME_WIN.iloc[0])
        self.assertEqual(r.GRADE_STATUS.iloc[0],'graded')

    def test_incomplete_and_late_predictions_are_not_graded(self):
        teams=pd.DataFrame({'GAME_ID':['1'],'TEAM_ABBREVIATION':['A'],'PTS':[100]})
        self.assertTrue(score_games(self.predictions(),teams).CORRECT.isna().all())
        pred=self.predictions(); pred['PREDICTED_AT']=pred.TIP_UTC
        self.assertEqual(score_games(pred,teams).GRADE_STATUS.iloc[0],'invalid_prediction')

    def test_wilson_does_not_give_zero_uncertainty_for_four_wins(self):
        low,high=wilson_interval(4,4)
        self.assertAlmostEqual(low,.510109,places=5)
        self.assertAlmostEqual(high,1.)
        low,high=wilson_interval(0,4)
        self.assertAlmostEqual(low,0.)
        self.assertGreater(high,.48)

    def test_audit_flags_margin_without_changing_source(self):
        d=pd.DataFrame({'GAME_ID':['1','1'],'TEAM_ID':[1,2],'GAME_DATE':['2026-01-01']*2,
            'SEASON':['2025-26']*2,'MATCHUP':['A vs. B','B @ A'],'PTS':[100,90],
            'FGM':[40,35],'FGA':[80,80],'FG3M':[10,10],'FG3A':[20,20],
            'FTM':[10,10],'FTA':[12,12],'REB':[40,40],'OREB':[10,10],'DREB':[30,30],
            'PLUS_MINUS':[8,-8],'WL':['W','L']})
        stats,bad,_,checked=audit_teams(d,d)
        self.assertEqual(stats['margin_disagreement_games'],1)
        self.assertEqual(stats['score_formula_failures'],0)
        self.assertEqual(d.PLUS_MINUS.tolist(),[8,-8])
        self.assertEqual(checked.DERIVED_MARGIN.tolist(),[10,-10])


if __name__=='__main__':
    unittest.main()
