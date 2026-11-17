import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from advanced_impact import FEATURES, prepare, fit_model, score
from normalize_impact_inputs import normalize
from impact_context_experiment import predict, ARMS


class AdvancedTests(unittest.TestCase):
    def test_real_endpoint_schema_and_units_translation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            tables={
                'advanced':{'PLAYER_ID':[1], **{c:[.2] for c in ['TS_PCT','USG_PCT','AST_PCT','TM_TOV_PCT','OREB_PCT','DREB_PCT']},'OFF_RATING':[115],'DEF_RATING':[110]},
                'misc':{'PLAYER_ID':[1],'PTS_PAINT':[12],'PTS_OFF_TOV':[4]},
                'base_per36':{'PLAYER_ID':[1],'FG3M':[3]},
                'defense_three':{'CLOSE_DEF_PERSON_ID':[1],'D_FGM':[3],'D_FGA':[10],'NORMAL_FG_PCT':[.35]},
                'defense_under_6ft':{'CLOSE_DEF_PERSON_ID':[1],'D_FGM':[5],'D_FGA':[10],'NORMAL_FG_PCT':[.6]}}
            manifest=dict(schema_version=2,season='2026-27',units={'misc':'Per36','base_per36':'Per36'},
                results={k:dict(success=True,received_at='2026-11-12T12:00:00Z') for k in tables})
            for k,v in tables.items(): pd.DataFrame(v).to_csv(root/f'{k}_0.csv',index=False)
            (root/'manifest.json').write_text(json.dumps(manifest))
            d=normalize(root)
            self.assertAlmostEqual(d.THREE_DEF_DIFF.iloc[0],-.05)
            self.assertAlmostEqual(d.RIM_DEF_DIFF.iloc[0],-.1)
            self.assertEqual(d.THREE_PTS_PER36.iloc[0],9)
            manifest['units']['misc']='Totals'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): normalize(root)

    def test_missing_coverage_future_snapshot_and_invalid_percent(self):
        base=pd.DataFrame({'PLAYER_ID':[1,2],'RATING':[3.,2.]})
        advanced=pd.DataFrame({c:[.2] for c in FEATURES if c!='BASE_RATING'})
        advanced['PLAYER_ID']=1;advanced['OBSERVED_AT']='2026-11-12T00:00:00Z'
        self.assertEqual(prepare(base,advanced,'2026-11-13').ADVANCED_COMPLETE.tolist(),[True,False])
        with self.assertRaises(ValueError): prepare(base,advanced,'2026-11-11')
        advanced['USG_PCT']=30
        with self.assertRaises(ValueError): prepare(base,advanced,'2026-11-13')

    def test_forward_target_provenance_and_incomplete_scoring(self):
        rng=np.random.default_rng(5)
        train=pd.DataFrame({c:rng.uniform(.1,.3,120) for c in FEATURES})
        train['TARGET_IMPACT']=rng.normal(size=120)
        train['OBSERVED_AT']='2023-01-01';train['TARGET_START']='2023-02-01';train['TARGET_END']='2023-04-01'
        model=fit_model(train,'2023-05-01')
        advanced=train.drop(columns=['BASE_RATING']).iloc[:1].copy()
        advanced['PLAYER_ID']=1
        base=pd.DataFrame({'PLAYER_ID':[1,2],'RATING':[3.,2.]})
        result=score(base,advanced,model,'2023-06-01')
        self.assertTrue(np.isfinite(result.ADVANCED_IMPACT.iloc[0]))
        self.assertTrue(pd.isna(result.ADVANCED_IMPACT.iloc[1]))
        with self.assertRaises(ValueError): fit_model(train,'2023-03-01')

    def test_test_season_labels_cannot_change_context_forecasts(self):
        rng=np.random.default_rng(1)
        cols=set(c for v in ARMS.values() for c in v)
        frames=[]
        for year in range(2021,2026):
            d=pd.DataFrame({c:rng.normal(size=40) for c in cols})
            d['WIN']=rng.integers(0,2,40);d['SEASON']=f'{year}-{str(year+1)[-2:]}'
            d['GAME_DATE']=f'{year}-11-01';frames.append(d)
        d=pd.concat(frames,ignore_index=True)
        a,_=predict(d)
        d.loc[d.SEASON=='2025-26','WIN']=1-d.loc[d.SEASON=='2025-26','WIN']
        b,_=predict(d)
        for arm in ARMS: np.testing.assert_allclose(a['P_'+arm],b['P_'+arm])
