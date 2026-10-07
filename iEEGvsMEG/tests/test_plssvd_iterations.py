import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plssvd_eval_utils import (TrialSubject, TrialData, ValidationOptions, validate_plssvd,
    load_plssvd_results, plot_iteration_summary, plot_permutation_comparison,
    plot_plssvd_validation, _checked_options, _permutation_indices, _apply_permutation)


def synthetic_trials():
    rng = np.random.default_rng(25)
    times = np.arange(24)*.05
    signal = rng.normal(size=(3,24))
    subjects = []
    for modality in ('ieeg','meg'):
        xyz = rng.normal(size=(3,3))
        meta = pd.DataFrame(xyz, columns=['x','y','z'])
        meta['subject'] = modality; meta['channel_index'] = range(3)
        data = [signal[None]+rng.normal(scale=.7,size=(10,3,24)) for _ in range(2)]
        subjects.append(TrialSubject(modality,(1,2),data,xyz,meta,[None,None],[['all']*10]*2))
    return TrialData([subjects[0]],[subjects[1]],times,(1,2))


class RepeatedEvaluationTests(unittest.TestCase):
    def test_permutation_integrity(self):
        x=np.arange(3*24).reshape(3,24)
        times=np.arange(24)*.05
        for kind in ('time_cirular_shift','time_block','time_point','space'):
            options=_checked_options(ValidationOptions(perm='ieeg',perm_type=kind))
            a=_permutation_indices(x.shape,times,options,'ieeg',np.random.default_rng(2))
            b=_permutation_indices(x.shape,times,options,'ieeg',np.random.default_rng(2))
            np.testing.assert_array_equal(a,b)
            y=_apply_permutation(x,a)
            self.assertFalse(np.array_equal(x,y))
            axis=0 if kind=='space' else 1
            np.testing.assert_array_equal(np.sort(x,axis=axis),np.sort(y,axis=axis))
            np.testing.assert_array_equal(_apply_permutation(np.stack([x,x]),a),np.stack([y,y]))
            self.assertIsNone(_permutation_indices(x.shape,times,options,'meg',np.random.default_rng(2)))
        with self.assertRaises(ValueError):_checked_options(ValidationOptions(perm='ieeg'))
        with self.assertRaises(ValueError):_checked_options(ValidationOptions(n_iterations=0))
        with self.assertRaises(ValueError):
            _permutation_indices(x.shape,times,ValidationOptions(perm='both',perm_type='time_block',block_seconds=360),'ieeg',np.random.default_rng(2))

    def test_other_permutation_runs_and_single_iteration(self):
        trials=synthetic_trials()
        with tempfile.TemporaryDirectory() as directory:
            for kind in ('time_cirular_shift','time_block','space'):
                root=Path(directory)/kind
                result=validate_plssvd(trials,'full_concatenated',
                    ValidationOptions(n_iterations=1,n_components=2,perm='ieeg',perm_type=kind),root)
                self.assertTrue(np.isfinite(result['iteration_metrics'].mean_r).all())
                self.assertTrue(result['iteration_stability'].empty)
            plt.close(plot_iteration_summary(root))

    def test_full_repeated_run_and_plots(self):
        trials=synthetic_trials()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            options=ValidationOptions(repeats=5,n_iterations=2,n_components=2)
            baseline=validate_plssvd(trials,'full_concatenated',options,root/'none')
            self.assertEqual(len(baseline['summary']),5)
            self.assertEqual(len(baseline['iteration_summary']),10)
            self.assertEqual(len(baseline['iteration_metrics']),4)
            audits=[pd.read_csv(root/'none'/f'iteration_{i:03d}'/'split_audit.csv') for i in range(2)]
            for audit in audits:
                counts=audit.query("partition == 'test'").groupby(['modality','subject','condition','trial_index']).size()
                self.assertTrue((counts==1).all())
            self.assertFalse(audits[0].equals(audits[1]))
            repeated=validate_plssvd(trials,'full_concatenated',options,root/'reproducible')
            pd.testing.assert_frame_equal(baseline['iteration_fold_metrics'],repeated['iteration_fold_metrics'])
            perm=validate_plssvd(trials,'full_concatenated',
                ValidationOptions(repeats=5,n_iterations=2,n_components=2,perm='both',perm_type='time_point'),
                root/'both__time_point')
            for i in range(2):
                pd.testing.assert_frame_equal(audits[i],pd.read_csv(root/'both__time_point'/f'iteration_{i:03d}'/'split_audit.csv'))
            self.assertFalse(np.allclose(baseline['iteration_metrics'].mean_r,perm['iteration_metrics'].mean_r))
            self.assertFalse((root/'none'/'iteration_000'/'primary_null_tests.csv').exists())
            plot_plssvd_validation(baseline,show=False)
            fig=plot_iteration_summary(root/'none');fig.savefig('/tmp/plssvd_iteration_preview.png');plt.close(fig)
            fig=plot_permutation_comparison(root);self.assertEqual(len(fig.axes),12);plt.close(fig)
            loaded=load_plssvd_results(root/'none')
            self.assertEqual(loaded['iteration_options']['n_iterations'],2)
            with self.assertRaises(FileExistsError):validate_plssvd(trials,'full_concatenated',options,root/'none')

if __name__=='__main__': unittest.main()
