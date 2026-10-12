import unittest
import numpy as np
import pandas as pd
from plssvd_select_components import select_components

class SelectionTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(run=run,partition='test',iteration=i,component=j,
            covariance=(1. if run=='Permuted' else (2. if j != 3 or i<89 else .5)))
            for run in ('Unpermuted','Permuted') for i in range(100) for j in range(1,5)])
    def test_phase_modes(self):
        from plssvd_select_components import _selection_folder
        for mode in ('ieeg','meg','both'):
            _, summary = select_components(self.frame(),4,perm=mode)
            self.assertEqual(summary['permutation'],mode+'__phase')
        self.assertNotEqual(_selection_folder('/tmp',10,'ieeg'),_selection_folder('/tmp',10,'meg'))
        with self.assertRaises(ValueError):select_components(self.frame(),4,perm='invalid')

    def test_prefix_and_boundary(self):
        table,summary=select_components(self.frame(),4)
        self.assertEqual(summary['selected_k'],2)
        self.assertTrue(table.iloc[3].passes)
        self.assertFalse(table.iloc[3].retained)
        _,s=select_components(self.frame(),2)
        self.assertTrue(s['boundary_supported'])
    def test_zero_and_ties(self):
        f=self.frame();f['covariance']=-1
        self.assertEqual(select_components(f,4)[1]['selected_k'],0)
        f['covariance']=1
        self.assertEqual(select_components(f,4)[1]['selected_k'],0)
    def test_missing_nonfinite_and_mismatch(self):
        f=self.frame()
        with self.assertRaises(ValueError):select_components(f.iloc[1:],4)
        f.loc[0,'covariance']=np.nan
        with self.assertRaises(ValueError):select_components(f,4)
        f=self.frame();f.loc[f.run=='Permuted','iteration']+=1
        with self.assertRaises(ValueError):select_components(f,4)
    def test_support_boundary(self):
        f=self.frame();f.loc[(f.run=='Unpermuted') & (f.component==3) & (f.iteration==89),'covariance']=2
        self.assertEqual(select_components(f,4)[1]['selected_k'],4)

if __name__=='__main__':unittest.main()
