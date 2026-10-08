import tempfile
import unittest
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plssvd_eval_utils import fit_plssvd, ValidationOptions
from plssvd_diagnostics import permutation_diagnostics, plot_permutation_diagnostics, _backfill_spectra
from test_plssvd_iterations import synthetic_trials


class DiagnosticTests(unittest.TestCase):
    def test_matched_comparisons_spectra_and_backfill(self):
        trials=synthetic_trials()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,perm,kind,n in [('none',None,None,2),('ieeg__time_point','ieeg','time_point',3)]:
                fit_plssvd(trials,'full_concatenated',ValidationOptions(n_components=2,n_iterations=n,
                           perm=perm,perm_type=kind),root/name)
            data=permutation_diagnostics(root,'ieeg','time_point',n_components=1)
            self.assertEqual(data['matched_iterations'],[0,1])
            self.assertTrue(data['full_spectra'])
            self.assertEqual(data['components'].component.max(),1)
            self.assertEqual(data['spectra']['rank'].max(),len(trials.times))
            for part in ('train','test'):
                frame=data['metrics'].query('partition == @part')
                a=frame.query("run == 'Unpermuted'").set_index('iteration').mean_r
                b=frame.query("run == 'Permuted'").set_index('iteration').mean_r
                np.testing.assert_allclose(data['differences'].query('partition == @part').mean_r,b-a)
            sums=data['spectra'].groupby(['run','iteration','partition','modality']).energy_fraction.sum()
            np.testing.assert_allclose(sums,1)
            for name,fig in plot_permutation_diagnostics(data).items():
                fig.savefig(f'/tmp/plssvd_diagnostic_{name}.png');plt.close(fig)
            # Older compact artifacts: retain metrics but remove spectra.
            expected={}
            for path in (root/'ieeg__time_point'/'iteration_000').glob('scores_*.npz'):
                with np.load(path) as z: values=dict(z)
                expected[path.stem]= {k:v for k,v in values.items() if 'temporal_singular_values' in k}
                np.savez_compressed(path,**{k:v for k,v in values.items() if 'temporal_singular_values' not in k})
            fallback=permutation_diagnostics(root,'ieeg','time_point',n_components=1)
            self.assertFalse(fallback['full_spectra'])
            self.assertEqual(fallback['spectra']['rank'].max(),1)
            _backfill_spectra(root/'ieeg__time_point',0,trials)
            for fold in range(5):
                with np.load(root/'ieeg__time_point'/'iteration_000'/f'temporal_spectra_{fold:03d}.npz') as z:
                    for key,value in expected[f'scores_{fold:03d}'].items():
                        np.testing.assert_allclose(z[key],value,rtol=1e-6,atol=1e-5)
            restored=permutation_diagnostics(root,'ieeg','time_point',n_components=1)
            self.assertTrue(restored['full_spectra'])
            # Reuse only the remaining shared iteration, without SD.
            (root/'none'/'iteration_000'/'COMPLETE.json').unlink()
            single=permutation_diagnostics(root,'ieeg','time_point',n_components=1)
            self.assertEqual(single['matched_iterations'],[1])
            self.assertTrue(single['delta_summary']['std'].isna().all())
            # Identical settings alone do not suffice: verify saved splits.
            import pandas as pd
            audit=root/'ieeg__time_point'/'iteration_001'/'split_audit.csv.gz'
            frame=pd.read_csv(audit);frame.loc[0,'trial_index']=999
            frame.to_csv(audit,index=False)
            with self.assertRaisesRegex(ValueError,'trial assignments'):
                permutation_diagnostics(root,'ieeg','time_point',n_components=1)
            with self.assertRaisesRegex(ValueError,'Select PERM'):
                permutation_diagnostics(root,None,None,n_components=1)

if __name__=='__main__':unittest.main()
