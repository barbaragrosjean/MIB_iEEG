import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coverage_matching_utils import Dataset
from test_plssvd_iterations import synthetic_trials
from plssvd_pca import independent_pca_scores, plssvd_pca_comparison, plot_plssvd_pca_comparison
from plssvd_eval_utils import fit_plssvd, ValidationOptions


class PCATests(unittest.TestCase):
    def test_train_only_pca_matches_direct_svd(self):
        rng=np.random.default_rng(51)
        train_arrays=[rng.normal(size=(2,w,25)) for w in (4,3)]
        test_arrays=[rng.normal(size=(2,w,25))+3 for w in (4,3)]
        def ds(arrays):return Dataset('data',arrays,pd.DataFrame(index=range(7)),np.arange(7),'average')
        train,test=ds(train_arrays),ds(test_arrays)
        x=np.concatenate(list(train.blocks()),axis=1);y=np.concatenate(list(test.blocks()),axis=1)
        for scale in (1.,.3):
            actual=independent_pca_scores(train,test,3,scale)
            u,d,v=np.linalg.svd((x-x.mean(0))*scale,full_matrices=False)
            a=u[:,:3]*d[:3];b=(y-x.mean(0))*scale@v[:3].T
            signs=np.sign(a[np.argmax(np.abs(a),axis=0),np.arange(3)])
            np.testing.assert_allclose(actual['train'],a*signs,atol=1e-10)
            np.testing.assert_allclose(actual['test'],b*signs,atol=1e-10)
            # Changing only held-out input cannot change training PCA.
            changed=independent_pca_scores(train,ds([a*10-20 for a in test_arrays]),3,scale)
            np.testing.assert_array_equal(actual['train'],changed['train'])
            self.assertFalse(np.allclose(actual['test'],changed['test']))
            prefix=independent_pca_scores(train,test,1,scale)
            np.testing.assert_allclose(prefix['test'],actual['test'][:,:1],atol=1e-10)

    def test_comparison_and_backfill(self):
        trials=synthetic_trials()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,perm,kind in [('none',None,None),('meg__time_point','meg','time_point')]:
                fit_plssvd(trials,'full_concatenated',ValidationOptions(n_components=2,n_iterations=2,
                           perm=perm,perm_type=kind),root/name)
            with patch('plssvd_eval_utils._fit',side_effect=AssertionError('PLSSVD refit')):
                data=plssvd_pca_comparison(root,'meg','time_point',n_components=2)
                self.assertEqual(data['iterations'],[0,1])
                self.assertEqual(len(data['summary']),16)
                # Direct held-out Pearson computation, not training scores.
                with np.load(root/'none'/'iteration_000'/'scores_000.npz') as scores:
                    expected=abs(np.corrcoef(scores['test_ieeg'][:,0],scores['test_ieeg_pca'][:,1])[0,1])
                row=data['fold_correlations'].query("run == 'Unpermuted' and iteration == 0 and repeat == 0 and modality == 'ieeg' and pls_component == 1 and pca_component == 2")
                self.assertAlmostEqual(row.value.iloc[0],expected)
                a=data['iteration_correlations'].query("run == 'Unpermuted'").value.to_numpy()
                b=data['iteration_correlations'].query("run == 'Permuted'").value.to_numpy()
                np.testing.assert_allclose(data['differences'].delta,b-a)
                fig=plot_plssvd_pca_comparison(data);fig.savefig('/tmp/plssvd_pca_preview.png');plt.close(fig)
                # Remove PCA scores to simulate pre-feature fits, then backfill.
                for path in (root/'none').glob('iteration_*/scores_*.npz'):
                    with np.load(path) as z:values=dict(z)
                    np.savez_compressed(path,**{k:v for k,v in values.items() if not k.endswith('_pca')})
                with self.assertRaisesRegex(FileNotFoundError,'original trial cache'):
                    plssvd_pca_comparison(root,n_components=2)
                cache=root/'cache';cache.mkdir();(cache/'manifest.json').write_text('{}')
                with patch('plssvd_eval_utils.load_trial_cache',return_value=trials):
                    recovered=plssvd_pca_comparison(root,n_components=2,cache_dir=cache)
                expected=data['fold_correlations'].query("run == 'Unpermuted'").value
                np.testing.assert_allclose(recovered['fold_correlations'].value,expected,atol=1e-7)
                # Cached sidecars remain usable without trial access.
                signed=plssvd_pca_comparison(root,n_components=1,absolute=False)
                self.assertFalse(signed['absolute'])
                self.assertEqual(len(signed['summary']),2)
                (root/'none'/'iteration_000'/'COMPLETE.json').unlink()
                partial=plssvd_pca_comparison(root,'meg','time_point',n_components=1)
                self.assertEqual(partial['iterations'],[1])
                self.assertTrue(partial['summary']['std'].isna().all())

if __name__=='__main__':unittest.main()
