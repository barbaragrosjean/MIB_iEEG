"""Synthetic numerical and data-flow checks; no project recordings required."""
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import numpy as np
import pandas as pd

from coverage_matching_utils import Dataset
from coverage_sampling import run_coverage_sampling, load_coverage_sampling, plot_coverage_sampling
from compare_models import (compare_representations, compare_cov_models_within,
                            compare_fitted_models, plot_within_models, space_metrics)
from cov_models_utils import fit_cov_models
from compare_subspace import fit_models, run_comparison, load_results, plot_results


def synthetic_reference():
    rng = np.random.default_rng(52)
    nt = 18
    latent = rng.normal(size=(2, 3, nt))
    positions = np.column_stack([np.arange(12)*4., np.zeros((12, 2))])
    # Include repeated nearest-source matches to exercise the random control.
    electrode_positions = positions[[0, 0, 2, 3, 4, 4, 6, 7]]
    owners = np.repeat(['i0', 'i1'], 4)
    metadata = pd.DataFrame(electrode_positions, columns=['x', 'y', 'z'])
    metadata['subject'] = owners
    ieeg = np.einsum('pk,ckt->cpt', rng.normal(size=(8, 3)), latent)+rng.normal(scale=.15, size=(2, 8, nt))
    meg = [np.einsum('pk,ckt->cpt', rng.normal(size=(12, 3)), latent)+rng.normal(scale=.15, size=(2, 12, nt))
           for _ in range(3)]
    source = dict(ieeg=ieeg, electrode_metadata=metadata, electrode_positions=electrode_positions,
                  electrode_subjects=owners, meg=meg, meg_subjects=['m0', 'm1', 'm2'],
                  meg_positions=[positions.copy() for _ in meg], times=np.arange(nt)/250,
                  load_config={'conditions': [1, 2]})
    return Dataset('iEEG', [ieeg], metadata, np.arange(8), 'stack', source_data=source)


def synthetic_cache(root):
    ref = synthetic_reference()
    rng = np.random.default_rng(9)
    records = []
    for modality, subjects, arrays in [
            ('ieeg', ['i0', 'i1'], [ref.arrays[0][:, :4], ref.arrays[0][:, 4:]]),
            ('meg', ref.source_data['meg_subjects'], ref.source_data['meg'])]:
        for index, (subject, mean) in enumerate(zip(subjects, arrays)):
            files = []
            for condition in range(2):
                filename = f'{subject}_{condition}.npy'
                np.save(root/filename, mean[condition]+rng.normal(scale=.1, size=(12, *mean[condition].shape)))
                files.append(filename)
            positions = (ref.source_data['electrode_positions'][index*4:(index+1)*4]
                         if modality == 'ieeg' else ref.source_data['meg_positions'][index])
            meta = pd.DataFrame(positions, columns=['x', 'y', 'z'])
            meta['subject'] = subject
            records.append(dict(modality=modality, subject=subject, files=files,
                positions=positions.tolist(), metadata=meta.to_dict('list'),
                split_groups=[None, None], permutation_blocks=[['all']*12, ['all']*12]))
    (root/'manifest.json').write_text(json.dumps(dict(
        records=records, times=ref.source_data['times'].tolist(),
        config=dict(conditions=[1, 2], ieeg_subjects=['i0', 'i1'], meg_subjects=['m0', 'm1', 'm2']))))


class WithinModelTests(unittest.TestCase):
    def test_frozen_matching_signs_and_no_test_rematching(self):
        rng = np.random.default_rng(3)
        train = rng.normal(size=(80, 3))
        test = rng.normal(size=(80, 3))
        permutation = [2, 0, 1]
        signs = np.array([-1, 1, -1])
        a = {'train': train, 'test': test}
        b = {'train': train[:, permutation]*signs, 'test': -test[:, permutation]*signs}
        reps = {'pca': {'temporal_scores': a, 'spatial_patterns': a},
                'pls': {'temporal_scores': b, 'spatial_patterns': b}}
        result = compare_representations(reps, modality='ieeg', k=3)
        rows = result['within_model_metrics']
        np.testing.assert_allclose(rows.query("partition == 'train'").matched_signed_r, 1)
        np.testing.assert_allclose(rows.query("partition == 'test'").matched_signed_r, -1)
        np.testing.assert_allclose(rows.overlap, 1)
        self.assertEqual(len(result['within_model_pairs']), 3)
        self.assertEqual(len(result['within_model_correlations']), 36)
        # Changing test data cannot change training assignments.
        reps['pls']['temporal_scores']['test'] = rng.normal(size=(80, 3))
        again = compare_representations(reps, modality='ieeg', k=3)
        pd.testing.assert_frame_equal(result['within_model_pairs'], again['within_model_pairs'])

    def test_rotation_rank_and_undefined_correlations(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(40, 3))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        self.assertAlmostEqual(space_metrics(x, x@q)['overlap'], 1)
        x[:, 2] = 0
        self.assertAlmostEqual(space_metrics(x, x)['overlap'], 2/3)
        result = compare_representations({n: {'temporal_scores': {'train': x}} for n in ['a', 'b']}, modality='meg', k=3)
        self.assertEqual(result['within_model_metrics'].iloc[0].n_valid_pairs, 2)

    def test_descriptive_and_heldout_agree_on_identical_partitions(self):
        ref = synthetic_reference()
        meg = Dataset('meg', [ref.source_data['meg'][0]], pd.DataFrame(index=range(12)),
                      np.arange(8), 'stack', source_data=ref.source_data)
        models = fit_cov_models(ref, meg, n_components=2, block_scaling='equal_variance')
        descriptive = compare_cov_models_within(models, ref, meg, dimensions=[1, 2])
        train = {'ieeg': ref, 'meg': meg}
        fitted = fit_models(train, ['separate_pca', 'joint_pca', 'plssvd'], 2)
        held = compare_fitted_models({'train': train, 'test': train}, fitted, [1, 2])
        cols = ['modality', 'model_a', 'model_b', 'k', 'space']
        a = descriptive['within_model_metrics'].sort_values(cols)
        b = held['within_model_metrics'].query("partition == 'train'").sort_values(cols)
        np.testing.assert_allclose(a.overlap, b.overlap, atol=1e-10)
        # Signed matching survives arbitrary fit-level sign orientations for temporal scores.
        np.testing.assert_allclose(a.matched_abs_r, b.matched_abs_r, atol=1e-10)
        import matplotlib.pyplot as plt
        for fig in plot_within_models(descriptive, k=2, partition='in_sample'): plt.close(fig)


class CoverageTests(unittest.TestCase):
    def test_reproducible_nested_controls_and_reference(self):
        kwargs = dict(participant_counts=[2, 3], feature_counts=[4, 6, None], repeats=2,
                      dimensions=[1, 2], seed=17)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_coverage_sampling(synthetic_reference(), output_dir=tmp, **kwargs)
            again = run_coverage_sampling(synthetic_reference(), **kwargs)
            for key in result: pd.testing.assert_frame_equal(result[key], again[key])
            self.assertTrue((Path(tmp)/'COMPLETE.json').exists())
            config, loaded = load_coverage_sampling(tmp)
            self.assertEqual(config['repeats'], 2)
            self.assertEqual(len(result['metrics']), len(loaded['metrics']))
            self.assertEqual(set(loaded['metrics'].feature_budget), {'4', '6', 'native'})
            with self.assertRaises(FileExistsError):
                run_coverage_sampling(synthetic_reference(), output_dir=tmp, **kwargs)
        metrics = result['metrics']
        self.assertEqual(len(metrics), 2*2*3*5*2)
        self.assertTrue((metrics.status == 'ok').all())
        for k, rows in metrics.groupby('k'):
            self.assertEqual(rows.ieeg_pca_variance_fraction.nunique(), 1)
        selections = result['selections']
        for (repeat, count), rows in selections.groupby(['repeat', 'participant_pool_count']):
            for kind in ['coverage_average', 'paired_coverage', 'random_control']:
                small = set(rows.query('dataset == @kind and feature_budget == "4"').electrode_index)
                large = set(rows.query('dataset == @kind and feature_budget == "6"').electrode_index)
                self.assertTrue(small <= large)
            for budget in ['4', '6', 'native']:
                paired = rows.query('dataset == "paired_coverage" and feature_budget == @budget')
                random = rows.query('dataset == "random_control" and feature_budget == @budget')
                np.testing.assert_array_equal(paired.electrode_index, random.electrode_index)
                np.testing.assert_array_equal(paired.meg_subject, random.meg_subject)
                for subject in paired.meg_subject.unique():
                    p = paired.query('meg_subject == @subject').source_index.to_numpy()
                    r = random.query('meg_subject == @subject').source_index.to_numpy()
                    np.testing.assert_array_equal(p[:, None] == p, r[:, None] == r)
        self.assertEqual(len(result['paired_control_deltas']), 2*2*3*2)
        import matplotlib.pyplot as plt
        for fig in plot_coverage_sampling(result, k=2): plt.close(fig)

    def test_rank_skip_and_grid_guard(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = run_coverage_sampling(synthetic_reference(), feature_counts=[1],
                kinds=['paired_coverage'], repeats=1, dimensions=[1, 2])
        self.assertEqual(result['metrics'].iloc[1].status, 'skipped_rank')
        self.assertTrue(np.isnan(result['metrics'].iloc[1].meg_pca_variance_fraction))
        ref = synthetic_reference()
        ref.source_data['meg_positions'][1][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'registered source grids'):
            run_coverage_sampling(ref, repeats=1)


class BatchIntegrationTests(unittest.TestCase):
    def test_real_trial_cache_through_batch_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cache = root/'cache'; cache.mkdir()
            synthetic_cache(cache)
            result = run_comparison(cache, root/'out', dimensions=[1, 2], repeats=1,
                models=['separate_pca', 'plssvd'], cluster_counts=[2], ridge_grid=[.01])
            _, loaded = load_results(root/'out')
            for key in ['within_model_metrics', 'within_model_pairs', 'within_model_correlations']:
                self.assertFalse(loaded[key].empty)
                self.assertEqual(len(result[key]), len(loaded[key]))
            metric = loaded['within_model_metrics']
            self.assertEqual(set(metric.partition), {'train', 'tune', 'test_a', 'test_b', 'test'})
            self.assertEqual(set(metric.space), {'temporal_scores', 'spatial_patterns', 'condition_contrast'})
            self.assertEqual(set(metric.modality), {'ieeg', 'meg'})
            figures = plot_results(root/'out', k=2, show=False)
            self.assertTrue(all(Path(path).is_file() for path in figures))
            self.assertTrue(any('within_models' in path for path in figures))


if __name__ == '__main__':
    unittest.main()
