import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from test_plssvd_iterations import synthetic_trials
from plssvd_eval_utils import fit_plssvd,ValidationOptions
from plssvd_full_pca import fit_full_data_pca,correlate_full_pca,plot_full_pca_comparison


class FullPCATests(unittest.TestCase):
    def test_full_reference_and_foldwise_iteration_statistics(self):
        trials=synthetic_trials()
        with patch('plssvd_eval_utils.load_trial_cache',return_value=trials):
            pca=fit_full_data_pca('unused_cache',2)
        for modality in ('ieeg','meg'):
            arrays=[]
            for subject in getattr(trials,modality):
                values=np.stack([a.mean(0) for a in subject.data])
                if modality=='meg':
                    values=(values-values.mean((0,2),keepdims=True))/values.std((0,2),keepdims=True)
                else:values*=1000
                arrays.append(values.astype(np.float32).mean(0).T)
            matrix=np.concatenate(arrays,axis=1).astype(float);matrix-=matrix.mean(0)
            u,d,_=np.linalg.svd(matrix,full_matrices=False)
            expected=u[:,:2]*d[:2]
            signs=np.sign(expected[np.argmax(abs(expected),axis=0),np.arange(2)])
            np.testing.assert_allclose(pca['scores'][modality],expected*signs,rtol=1e-5,atol=1e-5)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run'
            fit_plssvd(trials,'full_concatenated',ValidationOptions(n_components=2,n_iterations=2,
                        perm='meg',perm_type='time_point'),root)
            with patch('plssvd_eval_utils._fit',side_effect=AssertionError('refit')), \
                 patch('cov_models_utils._spectrum',side_effect=AssertionError('PCA recomputed')):
                result=correlate_full_pca(root,pca,2)
            self.assertEqual(result['mean'].shape,(2,2,5,2,2))
            self.assertTrue((result['count']==2).all())
            for f in range(5):
                expected=[]
                for i in range(2):
                    with np.load(root/f'iteration_{i:03d}'/f'scores_{f:03d}.npz') as saved:
                        expected.append(abs(np.corrcoef(saved['test_meg'][:,0],pca['scores']['meg'][:,1])[0,1]))
                self.assertAlmostEqual(result['mean'][1,1,f,0,1],np.mean(expected))
                self.assertAlmostEqual(result['std'][1,1,f,0,1],np.std(expected,ddof=1))
            fig=plot_full_pca_comparison(result);fig.savefig('/tmp/plssvd_full_pca_preview.png');plt.close(fig)
            (root/'iteration_001'/'COMPLETE.json').unlink()
            single=correlate_full_pca(root,pca,1,absolute=False)
            self.assertTrue(np.isnan(single['std']).all())
            self.assertEqual(single['iterations'],[0])
            with self.assertRaisesRegex(ValueError,'between 1'):
                correlate_full_pca(root,pca,3)

if __name__=='__main__':unittest.main()
