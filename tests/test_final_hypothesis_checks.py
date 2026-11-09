"""Synthetic/permutation smoke tests; no DSA conclusions."""
import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_final_hypothesis_checks as h

class HypothesisTests(unittest.TestCase):
    def test_hhi(self):
        self.assertAlmostEqual(h.hhi([1,1,1,1]),.25)
        self.assertAlmostEqual(h.hhi([1,0]),1.)
        with self.assertRaises(ValueError):h.hhi([0,0])

    def test_loo_and_permutation(self):
        df=pd.DataFrame({'platform_name':['A','B','C','A','B','C'],
          'category':['X']*3+['Y']*3,'fully_automated_rate':[.1,.5,.9,.2,.4,.8]})
        groups=h.cat_groups(df)
        signed=h.peer_signed_rates(df.fully_automated_rate.to_numpy(),groups)
        self.assertAlmostEqual(signed[0],-.6)
        rng=np.random.default_rng(12)
        out=h.permute_within_category(df.fully_automated_rate.to_numpy(),groups,rng)
        self.assertEqual(sorted(out[:3]),[.1,.5,.9])
        self.assertEqual(sorted(out[3:]),[.2,.4,.8])

    def test_run_exploratory(self):
        p=['A','B','C']*3;c=['X']*3+['Y']*3+['Z']*3
        df=pd.DataFrame({'platform_name':p,'category':c,
            'fully_automated_rate':np.linspace(.1,.9,9),'represented_sors':np.arange(9)*100+1000})
        temp=df[['platform_name','category']].copy()
        temp['automation_change_abs']=np.linspace(.01,.25,9)
        out=h.randomization_tests(df,df,temp,n_perm=99)
        self.assertEqual(out.rq.tolist(),['RQ1','RQ2','RQ3'])
        self.assertTrue(out.iloc[:3].p_one_sided.between(0,1).all())

if __name__=='__main__':unittest.main()
