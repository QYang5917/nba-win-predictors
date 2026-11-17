import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from v2_research import FULL, BASE, fit_fold, predict_bundle, MODELS
from style_experiment import style_features

CONFIGS=[dict(max_leaf_nodes=3,min_samples_leaf=5,l2_regularization=3.,max_iter=8)]


class V2Tests(unittest.TestCase):
    def frames(self):
        rng=np.random.default_rng(50)
        frames=[]
        for year in range(2018,2024):
            d=pd.DataFrame({c:rng.normal(size=40) for c in FULL})
            d['GAME_ID']=[f'{year}-{i}' for i in range(len(d))]
            d['SEASON']=f'{year}-{str(year+1)[-2:]}'
            d['GAME_DATE']=f'{year}-11-01'
            d['WIN']=rng.integers(0,2,len(d))
            d['MARGIN']=rng.normal(0,12,len(d))
            frames.append(d)
        d=pd.concat(frames,ignore_index=True)
        return d[d.SEASON<'2023-24'],d[d.SEASON=='2023-24'].copy()

    def test_test_scores_do_not_change_selected_models_or_forecasts(self):
        tr,te=self.frames()
        with threadpool_limits(limits=1):
            a,_,audit_a=fit_fold(tr,te,CONFIGS)
            te['WIN']=1-te.WIN;te['MARGIN']=999
            b,_,audit_b=fit_fold(tr,te,CONFIGS)
        self.assertEqual(audit_a,audit_b)
        for m in MODELS: np.testing.assert_allclose(a['P_'+m],b['P_'+m])
        for fold in audit_a.values():
            self.assertLess(fold['validation_end'],str(te.GAME_DATE.min()))
        self.assertFalse(set(FULL)&{'WIN','MARGIN','PTS_H','PTS_A','GAME_ID'})

    def test_bundle_matches_fold_and_rejects_training_dates(self):
        tr,te=self.frames()
        with threadpool_limits(limits=1):
            result,models,audit=fit_fold(tr,te,CONFIGS)
            bundle=dict(models=models,features=FULL,baseline_features=BASE,
                        forward_selection=audit,trained_through=str(tr.GAME_DATE.max()),blend_weight=.5)
            restored=predict_bundle(bundle,te)
        for m in MODELS: np.testing.assert_allclose(result['P_'+m],restored['P_'+m])
        with self.assertRaises(ValueError): predict_bundle(bundle,tr.iloc[:1])
        with self.assertRaises(ValueError): predict_bundle(bundle,pd.concat([te.iloc[:1]]*2))

    def test_overlapping_fold_is_rejected(self):
        tr,te=self.frames()
        te['GAME_DATE']=tr.GAME_DATE.max()
        with self.assertRaises(ValueError): fit_fold(tr,te,CONFIGS)

    def test_side_style_features_exclude_current_and_future_results(self):
        rows=[]
        for i,day in enumerate(pd.date_range('2020-01-01',periods=25)):
            for team in [1,2]:
                rows.append(dict(GAME_ID=str(i),TEAM_ID=team,GAME_DATE=str(day.date()),
                    MATCHUP='A vs. B' if team==1 else 'B @ A',FG3M=10,FG3A=30,
                    FGA=80,FTA=20,OREB=10,DREB=30,TOV=12))
        raw=pd.DataFrame(rows)
        a=style_features(raw,include_sides=True).set_index('GAME_ID')
        raw.loc[raw.GAME_DATE>='2020-01-21',['FG3M','OREB','DREB']]=90
        b=style_features(raw,include_sides=True).set_index('GAME_ID')
        ids=[str(i) for i in range(10,21)]
        pd.testing.assert_frame_equal(a.loc[ids],b.loc[ids])
