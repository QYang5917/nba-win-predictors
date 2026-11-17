import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from v2_recency import weights, predict_fold, select, predict_bundle
from v2_research import FULL


class RecencyTests(unittest.TestCase):
    def frames(self):
        rng=np.random.default_rng(99)
        frames=[]
        for year in range(2020,2024):
            d=pd.DataFrame({c:rng.normal(size=50) for c in FULL})
            d['WIN']=rng.integers(0,2,50);d['GAME_ID']=[f'{year}-{i}' for i in range(50)]
            d['SEASON']=f'{year}-{str(year+1)[-2:]}'
            d['GAME_DATE']=f'{year}-11-01';frames.append(d)
        d=pd.concat(frames,ignore_index=True)
        return d[d.SEASON<'2023-24'],d[d.SEASON=='2023-24'].copy()

    def test_weights_decay_but_keep_total_sample_weight(self):
        dates=pd.Series(pd.to_datetime(['2020-01-01','2020-12-31']))
        w=weights(dates,365.)
        self.assertAlmostEqual(w.sum(),2.)
        self.assertAlmostEqual(w[1]/w[0],2.)
        np.testing.assert_allclose(weights(dates,None),[1.,1.])

    def test_target_season_labels_cannot_change_selection(self):
        tr,te=self.frames()
        a,fa=predict_fold(tr,te)
        te['WIN']=1-te.WIN
        b,fb=predict_fold(tr,te)
        self.assertEqual(fa,fb)
        np.testing.assert_allclose(a.P_recency,b.P_recency)
        self.assertLess(fa['validation_end'],str(te.GAME_DATE.min()))

    def test_bundle_agrees_and_rejects_training_dates(self):
        tr,te=self.frames()
        a,_=predict_fold(tr,te)
        model,audit=select(tr)
        bundle=dict(model=model,selection=audit,features=FULL)
        np.testing.assert_allclose(a.P_recency,predict_bundle(bundle,te))
        with self.assertRaises(ValueError): predict_bundle(bundle,tr.iloc[:1])
