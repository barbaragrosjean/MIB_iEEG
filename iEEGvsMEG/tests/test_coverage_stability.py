import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from coverage_matching_utils import Dataset
from coverage_stability import (matched_correlation, run_subject_count_stability, run_pairing_stability,
    load_stability_results, plot_subject_count_stability, plot_pairing_stability)


def reference(ni=30, nm=32):
    rng = np.random.default_rng(40)
    t = 12
    latent = rng.normal(size=(2, 2, t))
    owners = np.repeat([f'i{i:02d}' for i in range(ni)], 3)
    positions = np.column_stack([np.arange(8)*5., np.zeros((8, 2))])
    coordinates = positions[np.tile([0, 0, 3], ni)]
    metadata = pd.DataFrame(coordinates, columns=['x', 'y', 'z']); metadata['subject'] = owners
    ieeg = np.einsum('fk,ckt->cft', rng.normal(size=(3*ni, 2)), latent)+rng.normal(scale=.3, size=(2, 3*ni, t))
    meg = [np.einsum('fk,ckt->cft', rng.normal(size=(8, 2)), latent)+rng.normal(scale=.3, size=(2, 8, t)) for _ in range(nm)]
    source = dict(ieeg=ieeg, meg=meg, meg_subjects=[f'm{i:02d}' for i in range(nm)],
                  electrode_subjects=owners, electrode_positions=coordinates, electrode_metadata=metadata,
                  meg_positions=[positions.copy() for _ in meg], times=np.arange(t), load_config={'conditions': [1, 2]})
    return Dataset('iEEG', [ieeg], metadata, np.arange(len(owners)), 'stack', source_data=source)


class CorrelationTests(unittest.TestCase):
    def test_sign_and_order_invariance_and_rank(self):
        x = np.random.default_rng(5).normal(size=(40, 3))
        summary, pairs = matched_correlation(x, x[:, [2, 0, 1]]*[-1, 1, -1], 3)
        self.assertAlmostEqual(summary['correlation'], 1)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(matched_correlation(x[:, :2], x, 3)[0]['status'], 'skipped_rank')


class SubjectCountTests(unittest.TestCase):
    def test_requested_counts_in_both_modalities_and_nested_subsets(self):
        ref = reference(); before = ref.arrays[0].copy()
        result = run_subject_count_stability(ref, repeats=2, n_components=2)
        table = result['metrics']
        self.assertEqual(set(table.n_subjects), {5, 10, 20, 30})
        self.assertEqual(set(table.modality), {'meg', 'ieeg'})
        self.assertEqual(set(table.comparison), {'to_full_cohort', 'between_resamples', 'cross_modal'})
        self.assertTrue((result['availability'].status == 'included').all())
        for (repeat, modality), group in result['participants'].groupby(['repeat', 'modality']):
            previous = set()
            for count, rows in group.groupby('n_subjects'):
                subjects = set(rows.subject)
                self.assertEqual(len(subjects), count)
                self.assertTrue(previous <= subjects); previous = subjects
        # The complete 30-person iEEG cohort reproduces itself.
        full = table.query("modality == 'ieeg' and n_subjects == 30 and comparison == 'to_full_cohort'")
        np.testing.assert_allclose(full.correlation, 1, atol=1e-12)
        np.testing.assert_array_equal(before, ref.arrays[0])
        self.assertFalse(table.dataset.isin(['paired_coverage', 'random_control']).any())

    def test_small_meg_cohort_supported_and_unavailable_counts_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_subject_count_stability(reference(ni=6, nm=4), subject_counts=[2, 4, 5, 10],
                repeats=2, n_components=2, output_dir=tmp)
            self.assertEqual(len(result['availability'].query("status == 'skipped_insufficient_subjects'")), 3)
            self.assertEqual(set(result['metrics'].query("modality == 'meg'").n_subjects), {2, 4})
            loaded = load_stability_results(tmp, 'subject_count')
            self.assertEqual(len(loaded['metrics']), len(result['metrics']))
            with self.assertRaises(ValueError): load_stability_results(tmp, 'pairing')
            import matplotlib.pyplot as plt
            for i, fig in enumerate(plot_subject_count_stability(loaded)):
                fig.savefig(Path(tmp)/f'plot{i}.png'); plt.close(fig)
                self.assertTrue((Path(tmp)/f'plot{i}.png').is_file())


class PairingTests(unittest.TestCase):
    def test_fixed_rosters_source_randomization_and_reproducibility(self):
        ref = reference(ni=4, nm=6)
        baseline = {f'i{i:02d}': f'm{i+1:02d}' for i in range(4)}
        with tempfile.TemporaryDirectory() as tmp:
            result = run_pairing_stability(ref, baseline_pairing=baseline, repeats=5, n_components=2, output_dir=tmp)
            again = run_pairing_stability(ref, baseline_pairing=baseline, repeats=5, n_components=2)
            pd.testing.assert_frame_equal(result['metrics'], again['metrics'])
            self.assertEqual(result['config']['baseline_pairing'], baseline)
            for repeat, rows in result['assignments'].groupby('repeat'):
                self.assertEqual(set(rows.ieeg_subject), set(baseline))
                self.assertEqual(set(rows.meg_subject), set(baseline.values()))
            self.assertGreater(len({tuple(g.meg_subject) for _, g in result['assignments'].groupby('repeat')}), 1)
            mapping = result['source_mapping']
            # Same MEG source always maps to the same random-control source.
            self.assertTrue((mapping.groupby(['meg_subject', 'source_index']).random_source_index.nunique() == 1).all())
            for _, rows in mapping.groupby(['repeat', 'meg_subject']):
                a, b = rows.source_index.to_numpy(), rows.random_source_index.to_numpy()
                np.testing.assert_array_equal(a[:, None] == a, b[:, None] == b)
            self.assertEqual(result['metrics'].n_features.nunique(), 1)
            self.assertEqual(result['metrics'].n_subjects.nunique(), 1)
            self.assertEqual(len(result['paired_control_differences']), 6)
            loaded = load_stability_results(tmp, 'pairing')
            self.assertEqual(len(loaded['metrics']), len(result['metrics']))
            import matplotlib.pyplot as plt
            for fig in plot_pairing_stability(loaded):
                fig.savefig(Path(tmp)/'pairing.png'); plt.close(fig)
            self.assertTrue((Path(tmp)/'pairing.png').is_file())
            self.assertTrue((Path(tmp)/'control_source_permutations.npz').is_file())

    def test_fixed_subject_count_and_bad_baseline(self):
        result = run_pairing_stability(reference(ni=4, nm=6), n_subjects=3, repeats=2, n_components=2)
        self.assertEqual(result['config']['n_subjects'], 3)
        with self.assertRaises(ValueError):
            run_pairing_stability(reference(ni=4, nm=6), baseline_pairing={'i00': 'm00', 'i01': 'm00'})


if __name__ == '__main__': unittest.main()
