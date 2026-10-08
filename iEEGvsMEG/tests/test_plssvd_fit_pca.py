import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from plssvd_eval_utils import fit_plssvd, ValidationOptions
from plssvd_fit_pca import backfill_pca
from test_plssvd_iterations import synthetic_trials


class PCAOnlyTests(unittest.TestCase):
    def test_only_missing_pca_and_resume(self):
        trials=synthetic_trials()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            fit_plssvd(trials,'full_concatenated',ValidationOptions(n_components=2,n_iterations=1,
                perm='both',perm_type='time_point'),root)
            child=root/'iteration_000';expected={}
            for fold in (0,2):
                path=child/f'scores_{fold:03d}.npz'
                with np.load(path) as z:values=dict(z)
                expected[fold]={k:v for k,v in values.items() if k.endswith('_pca')}
                np.savez_compressed(path,**{k:v for k,v in values.items() if not k.endswith('_pca')})
            before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            from plssvd_pca import independent_pca_scores
            calls=[]
            def interrupted(*args,**kwargs):
                calls.append(1)
                if len(calls)==3:raise MemoryError('simulated interruption')
                return independent_pca_scores(*args,**kwargs)
            with patch('plssvd_eval_utils.load_trial_cache',return_value=trials), \
                 patch('plssvd_eval_utils._fit',side_effect=AssertionError('PLSSVD refit')), \
                 patch('plssvd_evaluation.evaluate_statistics',side_effect=AssertionError('metrics recomputed')), \
                 patch('plssvd_pca.temporal_singular_values',side_effect=AssertionError('spectra recomputed')):
                with patch('plssvd_pca.independent_pca_scores',side_effect=interrupted):
                    with self.assertRaises(MemoryError):backfill_pca(root,Path(tmp)/'cache')
                self.assertTrue((child/'pca_scores_000.npz').exists())
                self.assertFalse((child/'pca_scores_002.npz').exists())
                result=backfill_pca(root,Path(tmp)/'cache')
                self.assertEqual(result['computed_folds'],1)
                self.assertEqual(result['reused_folds'],4)
            for path,content in before.items():self.assertEqual(path.read_bytes(),content)
            self.assertFalse(list(child.glob('temporal_spectra_*.npz')))
            self.assertFalse((root/'evaluation').exists())
            for fold,values in expected.items():
                with np.load(child/f'pca_scores_{fold:03d}.npz') as saved:
                    for key,value in values.items():np.testing.assert_allclose(saved[key],value,atol=1e-7)
            with patch('plssvd_eval_utils.load_trial_cache',side_effect=AssertionError('unnecessary cache read')):
                result=backfill_pca(root,Path(tmp)/'missing_cache')
                self.assertEqual(result['computed_folds'],0)
                self.assertEqual(result['reused_folds'],5)

if __name__=='__main__':unittest.main()
