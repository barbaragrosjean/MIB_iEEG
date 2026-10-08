import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from test_plssvd_iterations import synthetic_trials
from plssvd_eval_utils import (ValidationOptions, _permutation_indices, _apply_permutation,
    _temporary_fold, _subject_folds, fit_plssvd)
from plssvd_postprocess import evaluate_run, prepare_comparison
from plssvd_figures import plot_phase_null_comparison, load_comparison
from plssvd_phase_null import resolve_max_components, reusable_iterations


class PhaseTests(unittest.TestCase):
    def test_reuse_without_run_completion_marker(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            saved = dict(meg_kind='full_concatenated', schema_version=6, n_iterations=2)
            with self.assertRaisesRegex(ValueError, 'no completed iterations'):
                reusable_iterations(root, saved, 'full_concatenated')
            child = root/'iteration_000'
            child.mkdir()
            (child/'COMPLETE.json').write_text('{}')
            (root/'iteration_001').mkdir()  # Interrupted iteration is excluded.
            self.assertEqual(reusable_iterations(root, saved, 'full_concatenated'), [0])
            with self.assertRaisesRegex(ValueError, 'saved meg_kind'):
                reusable_iterations(root, saved, 'full_average')
            with self.assertRaisesRegex(ValueError, 'unsupported fit schema'):
                reusable_iterations(root, dict(saved, schema_version=5), 'full_concatenated')

    def test_baseline_component_count_resolution(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(resolve_max_components(root, None, [10]), 100)
            self.assertEqual(resolve_max_components(root, 20, [10]), 20)
            (root/'none').mkdir()
            (root/'none'/'validation_options.json').write_text(json.dumps({'n_components': 20}))
            self.assertEqual(resolve_max_components(root, None, [10]), 20)
            self.assertEqual(resolve_max_components(root, 20, [10]), 20)
            with self.assertRaisesRegex(ValueError, 'Omit --max-components'):
                resolve_max_components(root, 100, [10])
            with self.assertRaisesRegex(ValueError, 'fitted component count'):
                resolve_max_components(root, None, [21])

    def test_preserves_covariance_spectra_means_even_and_odd(self):
        for n in (23,24):
            x=np.random.default_rng(1).normal(size=(2,5,n))
            options=ValidationOptions(perm='meg',perm_type='phase')
            transform=_permutation_indices((5,n),np.arange(n)*.01,options,'meg',np.random.default_rng(2))
            y=_apply_permutation(x,transform)
            self.assertFalse(np.allclose(x,y))
            np.testing.assert_allclose(x.mean(-1),y.mean(-1),atol=1e-12)
            fx=np.fft.rfft(x,axis=-1);fy=np.fft.rfft(y,axis=-1)
            np.testing.assert_allclose(abs(fx),abs(fy),atol=1e-12)
            np.testing.assert_allclose(fx[:,:,None,:]*fx[:,None,:,:].conj(),fy[:,:,None,:]*fy[:,None,:,:].conj(),atol=1e-11)
            np.testing.assert_allclose(x@x.swapaxes(-1,-2),y@y.swapaxes(-1,-2),atol=1e-12)
            self.assertIsNone(_permutation_indices((5,n),np.arange(n),options,'ieeg',np.random.default_rng(2)))

    def test_global_modality_phases_and_repeatability(self):
        trials=synthetic_trials()
        # Two participants per modality, with identical signals but distinct IDs.
        for m in ('ieeg','meg'):
            original=getattr(trials,m)[0]
            other=replace(original,subject=m+'2',metadata=original.metadata.assign(subject=m+'2'))
            getattr(trials,m).append(other)
        rng=np.random.default_rng(3)
        indices={m:{s.subject:_subject_folds(s,rng,'trial',2)[0] for s in getattr(trials,m)} for m in ('ieeg','meg')}
        import plssvd_eval_utils as module
        actual=module._apply_permutation
        def collect(mode):
            recorded=[]
            def apply(a,t):
                recorded.append(t);return actual(a,t)
            with patch.object(module,'_apply_permutation',side_effect=apply):
                with _temporary_fold(trials,'full_concatenated',indices,1,
                        permutation_options=ValidationOptions(perm=mode,perm_type='phase'),permutation_seed=12):
                    pass
            return recorded
        both=collect('both')
        # One stacked training call and one call per test condition/participant.
        midpoint=len(both)//2
        for block in (both[:midpoint],both[midpoint:]):
            for t in block:np.testing.assert_array_equal(t['phase_multiplier'],block[0]['phase_multiplier'])
        self.assertFalse(np.array_equal(both[0]['phase_multiplier'],both[midpoint]['phase_multiplier']))
        meg=collect('meg');ieeg=collect('ieeg')
        np.testing.assert_array_equal(meg[midpoint]['phase_multiplier'],both[midpoint]['phase_multiplier'])
        np.testing.assert_array_equal(ieeg[0]['phase_multiplier'],both[0]['phase_multiplier'])
        self.assertTrue(all(t is None for t in meg[:midpoint]))
        again=collect('both')
        np.testing.assert_array_equal(both[0]['phase_multiplier'],again[0]['phase_multiplier'])

    def test_all_modes_refit_prepare_and_plot(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            for mode in (None,'meg','ieeg','both'):
                name='none' if mode is None else mode+'__phase'
                options=ValidationOptions(n_components=2,n_iterations=1,repeats=2,perm=mode,perm_type='phase' if mode else None)
                fit_plssvd(synthetic_trials(),'full_concatenated',options,root/name)
                # A missing run-level marker must not hide completed iterations.
                (root/name/'COMPLETE.json').unlink()
                saved = json.loads((root/name/'validation_options.json').read_text())
                self.assertEqual(reusable_iterations(root/name, saved, 'full_concatenated'), [0])
                evaluate_run(root/name,[1,2])
                if mode:
                    prepare_comparison(root,name,2)
                    data=load_comparison(root,name,2)
                    self.assertTrue(np.isfinite(data['metrics']['mean_r']).all())
                    self.assertEqual(set(data['metrics'].run),{'Unpermuted','Permuted'})
            fig=plot_phase_null_comparison(root,2)
            self.assertEqual(len(fig.axes),12)
            fig.savefig('/tmp/plssvd-phase-test.png',dpi=100)
            plt.close(fig)


if __name__=='__main__':unittest.main()
