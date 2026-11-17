import sys
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from v2_calibration import fit_meta, apply_meta, evaluate


class CalibrationTests(unittest.TestCase):
    def history(self):
        rng=np.random.default_rng(73)
        parts=[]
        for year in range(2020,2026):
            y=rng.integers(0,2,120)
            linear=np.clip(.3+.4*y+rng.normal(0,.23,120),.01,.99)
            boost=np.clip(.3+.4*y+rng.normal(0,.25,120),.01,.99)
            parts.append(pd.DataFrame({'SEASON':f'{year}-{str(year+1)[-2:]}',
                'GAME_DATE':f'{year}-11-01','WIN':y,'P_linear_full':linear,
                'P_boost_classifier':boost,'P_fixed_blend':.5*(linear+boost)}))
        return pd.concat(parts,ignore_index=True)

    def test_evaluation_labels_do_not_change_same_season_meta_predictions(self):
        d=self.history()
        a,fa=evaluate(d)
        d.loc[d.SEASON=='2025-26','WIN']=1-d.loc[d.SEASON=='2025-26','WIN']
        b,fb=evaluate(d)
        self.assertEqual(fa,fb)
        for c in ['P_platt_linear','P_learned_blend','P_stacked']:
            np.testing.assert_allclose(a[c],b[c])
        for fold in fa:
            self.assertLess(fold['trained_through'],fold['test_start'])

    def test_blend_can_reject_booster_completely(self):
        d=self.history().iloc[:120].copy()
        d['P_linear_full']=np.where(d.WIN,.9,.1)
        d['P_boost_classifier']=1-d.P_linear_full
        self.assertEqual(fit_meta(d)['linear_weight'],1.)

    def test_reject_overlapping_dates_and_invalid_probability(self):
        d=self.history()
        past=d[d.SEASON<'2023-24']
        meta=fit_meta(past)
        with self.assertRaises(ValueError): apply_meta(meta,past)
        past=past.copy();past.iloc[0,past.columns.get_loc('P_linear_full')]=1.1
        with self.assertRaises(ValueError): fit_meta(past)

    def test_saved_meta_reproduces_probabilities(self):
        d=self.history();past=d[d.SEASON<'2023-24'];test=d[d.SEASON=='2023-24']
        meta=fit_meta(past)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'meta.joblib';joblib.dump(meta,p)
            loaded=joblib.load(p)
            a,b=apply_meta(meta,test),apply_meta(loaded,test)
        for c in ['P_platt_linear','P_learned_blend','P_stacked']:
            np.testing.assert_allclose(a[c],b[c])
            self.assertTrue(b[c].between(0,1).all())
