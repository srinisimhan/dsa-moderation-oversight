"""Synthetic contract tests; these are NOT results from the DSA data."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_final_research_validation as v


def fixture():
    rng=np.random.default_rng(90210)
    weeks=pd.date_range('2026-03-02',periods=26,freq='W-MON')
    rows=[]
    for pi,platform in enumerate(['A','B','C','D']):
        for ci,cat in enumerate(['CAT1','CAT2','CAT3']):
            for wi,w in enumerate(weeks):
                rate=float(np.clip(.35+.1*pi +.03*ci+.025*np.sin(wi/2+pi)+rng.normal(0,.017),.01,.99))
                rows.append({'week_start':str(w.date()),'platform_name':platform,'category':cat,
                             'complete_study_week':True,'represented_sors':2000+pi*500+10*wi,
                             'fully_automated_rate':rate,'content_removed_rate':.15+pi*.04,
                             'account_or_service_restriction_rate':.25+ci*.05})
    return pd.DataFrame(rows)


class ResearchValidationTests(unittest.TestCase):
    def test_chronological_cohort_and_features(self):
        cand=v.build_candidates(fixture())
        self.assertEqual(cand.week_start.nunique(),25)
        tr,te,meta=v.chronological_features(cand,17)
        self.assertEqual(meta['candidate_test_weeks'],8)
        self.assertEqual(meta['eligible_training_series'],12)
        self.assertEqual(len(tr),12*17)
        self.assertEqual(len(te),12*8)
        self.assertTrue(tr.week_start.max()<te.week_start.min())
        self.assertTrue(np.isfinite(te[v.FEATURES].to_numpy(float)).all())
        self.assertTrue(te.weekly_peer_support_count.eq(3).all())

    def test_future_rows_do_not_change_training_values(self):
        df=fixture()
        cand=v.build_candidates(df)
        tr,te,meta=v.chronological_features(cand,17)
        # Strongly change only dates after cutoff. Train features and robust
        # temporal scores must remain identical.
        edited=df.copy()
        edited.loc[pd.to_datetime(edited.week_start)>=te.week_start.min(),'fully_automated_rate']=.99
        cand2=v.build_candidates(edited)
        tr2,_,_=v.chronological_features(cand2,17)
        a=tr.sort_values(v.KEY).reset_index(drop=True)
        b=tr2.sort_values(v.KEY).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[v.FEATURES],b[v.FEATURES])
        pd.testing.assert_series_equal(a.robust_change_score_train_baseline,b.robust_change_score_train_baseline)

    def test_models_and_rankings(self):
        orig=v.N_ESTIMATORS;seeds=v.SEEDS
        try:
            v.N_ESTIMATORS=20
            v.SEEDS=[11,22,33,44,55,66,77,88,99,111]
            tr,te,meta=v.chronological_features(v.build_candidates(fixture()),17)
            result,compar,sensitivity,meta=v.eval_split(tr,te,meta)
            self.assertEqual(len(compar),3)
            self.assertEqual(len(sensitivity),2)
            self.assertEqual(int(result.in_if_test_top20.sum()),20)
            self.assertEqual(compar.iloc[0].jaccard_with_if_top20,1.)
        finally:
            v.N_ESTIMATORS=orig
            v.SEEDS=seeds

    def test_reject_bad_base_and_frozen(self):
        df=fixture()
        df.loc[1,v.KEY]=df.loc[0,v.KEY].to_list()
        with self.assertRaisesRegex(ValueError,'duplicated weekly keys'):
            v.build_candidates(df)
        self.assertAlmostEqual(v.jaccard([1,2],[2,3]),1/3)

if __name__=='__main__':unittest.main()
