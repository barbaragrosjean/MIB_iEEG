"""Regression tests for coverage construction, input validation and exact PCA.
Run: python -m unittest test_coverage_matching.py
"""
import json
import pickle
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils_updated import (
    construct_five_datasets, synthetic_inputs, fit_block_pca,
    observations, make_ieeg_dataset, variance_summary, compare_to_ieeg,
    load_project_data, load_dataset, compute_variance, compute_pca,
    correlate_timecourses, correlate_weights,
)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.inputs = synthetic_inputs(7)

    def build(self, **kwargs):
        d = self.inputs
        return construct_five_datasets(d['meg'], d['meg_positions'], d['meg_subjects'],
                                       d['electrode_positions'], d['electrode_subjects'], **kwargs)

    def test_pairing_reads_the_paired_participant_signal(self):
        # Nonidentity pairing is essential: the original helper silently read i instead of j.
        pairing = {'I1': 'M4', 'I2': 'M1', 'I3': 'M2'}
        ds, audit, got = self.build(pairing=pairing, seed=10)
        self.assertEqual(pairing, got)
        for row in audit.itertuples():
            j = self.inputs['meg_subjects'].index(row.meg_subject)
            np.testing.assert_allclose(ds['paired_coverage'].arrays[0][:, row.electrode_index],
                                       self.inputs['meg'][j][:, row.source_index], rtol=1e-6)
            np.testing.assert_allclose(ds['random_control'].arrays[0][:, row.electrode_index],
                                       self.inputs['meg'][j][:, row.random_source_index], rtol=1e-6)
        # The first iEEG participant contains a duplicated nearest source.
        for _, rows in audit.groupby('ieeg_subject'):
            self.assertEqual(sorted(rows.source_index.value_counts()),
                             sorted(rows.random_source_index.value_counts()))
        full_x = np.concatenate(list(ds['full_concatenated'].blocks()), axis=1)
        np.testing.assert_allclose(full_x[:, ds['full_concatenated'].electrode_to_feature],
                                   observations(ds['paired_coverage'].arrays[0]), rtol=1e-6)

    def test_averaging_and_dimensions(self):
        ds, _, _ = self.build()
        expected = np.mean(self.inputs['meg'], axis=0)
        np.testing.assert_allclose(ds['full_average'].arrays[0], expected, rtol=1e-5, atol=1e-6)
        ix = ds['full_average'].electrode_to_feature
        np.testing.assert_allclose(ds['coverage_average'].arrays[0], expected[:, ix], rtol=1e-5, atol=1e-6)
        self.assertEqual(ds['full_concatenated'].n_features, 160)
        self.assertTrue(all(ds[n].n_features == 12 for n in ['coverage_average', 'paired_coverage', 'random_control']))
        self.assertIs(ds['full_concatenated'].arrays, self.inputs['meg'])

    def test_reproducibility_and_independent_randomisation(self):
        ds1, audit1, p1 = self.build(seed=13)
        ds2, audit2, p2 = self.build(seed=13)
        pd.testing.assert_frame_equal(audit1, audit2)
        self.assertEqual(p1, p2)
        ds3, audit3, _ = self.build(seed=14, pairing=p1)
        np.testing.assert_array_equal(ds1['paired_coverage'].arrays[0], ds3['paired_coverage'].arrays[0])
        self.assertFalse(np.array_equal(audit1.random_source_index, audit3.random_source_index))

    def test_exact_pca_matches_sklearn_for_all_setups_and_condition_modes(self):
        for mode in ['average', 'stack']:
            ds, _, _ = self.build(condition_mode=mode)
            for name, d in ds.items():
                with self.subTest(mode=mode, dataset=name):
                    x = np.concatenate(list(d.blocks()), axis=1).astype(float)
                    expected = PCA(n_components=10, svd_solver='full').fit(x)
                    actual = fit_block_pca(d, n_components=10, feature_chunk=7)
                    np.testing.assert_allclose(actual.explained_variance, expected.explained_variance_, rtol=1e-8, atol=1e-10)
                    np.testing.assert_allclose(actual.explained_variance_ratio, expected.explained_variance_ratio_, rtol=1e-8, atol=1e-10)
                    # Compare reconstructions; eigenvector signs are arbitrary.
                    np.testing.assert_allclose(actual.scores @ actual.weights.T,
                                               expected.transform(x) @ expected.components_, rtol=1e-6, atol=1e-6)
                    self.assertAlmostEqual(actual.total_variance, x.var(axis=0, ddof=1).sum(), places=7)

    def test_comparison_uses_electrode_mapping(self):
        ds, _, _ = self.build()
        ref_ds = make_ieeg_dataset(self.inputs['ieeg'], self.inputs['electrode_metadata'])
        ref = fit_block_pca(ref_ds)
        d = ds['full_concatenated']
        r = fit_block_pca(d)
        c = compare_to_ieeg(d, r, ref)
        expected = np.corrcoef(r.weights[d.electrode_to_feature].T, ref.weights.T)[:10, 10:]
        np.testing.assert_allclose(c['weights'], expected, atol=1e-12)
        self.assertEqual(c['time'].shape, (10, 10))
        self.assertEqual(c['assignment'].ieeg_pc.nunique(), 10)
        self.assertEqual(c['assignment'].meg_pc.nunique(), 10)

    def test_invalid_grids_pairings_and_memory_budget(self):
        self.inputs['meg_positions'][1][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'ordered source grid'):
            self.build()
        self.inputs = synthetic_inputs(7)
        with self.assertRaisesRegex(ValueError, 'one-to-one'):
            self.build(pairing={'I1': 'M1', 'I2': 'M1', 'I3': 'M2'})
        ds, _, _ = self.build()
        with self.assertRaises(MemoryError):
            fit_block_pca(ds['full_concatenated'], max_gram_gib=0)

    def test_zero_variance_and_rank_deficiency(self):
        meta = pd.DataFrame({'x': [0, 1], 'y': [0, 1], 'z': [0, 1]})
        d = make_ieeg_dataset(np.zeros((2, 2, 20)), meta)
        with self.assertRaisesRegex(ValueError, 'no temporal variance'):
            fit_block_pca(d)
        a = np.tile(np.arange(20, dtype=float), (2, 2, 1))
        d = make_ieeg_dataset(a, meta)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = fit_block_pca(d)
        self.assertEqual(result.scores.shape[1], 1)
        self.assertAlmostEqual(result.explained_variance_ratio.sum(), 1.)

    def test_simple_api_selects_one_setup_and_preserves_comparisons(self):
        reference = make_ieeg_dataset(self.inputs['ieeg'], self.inputs['electrode_metadata'])
        reference.source_data = self.inputs
        expected, _, _ = self.build(seed=2026)
        reference_pca = compute_pca(reference)
        for kind in expected:
            selected = load_dataset(kind, reference=reference)
            self.assertIs(selected.source_data, self.inputs)
            for a, b in zip(selected.arrays, expected[kind].arrays):
                np.testing.assert_array_equal(a, b)
            selected_pca = compute_pca(selected)
            self.assertAlmostEqual(compute_variance(selected)['total_variance'],
                                   selected_pca.total_variance, places=6)
            c = compare_to_ieeg(selected, selected_pca, reference_pca)
            np.testing.assert_allclose(correlate_timecourses(selected_pca, reference_pca, plot=False), c['time'])
            np.testing.assert_allclose(correlate_weights(selected_pca, reference_pca, plot=False), c['weights'])
        paired = load_dataset('paired_coverage', reference=reference)
        random = load_dataset('random_control', reference=reference)
        self.assertEqual(paired.pairing, random.pairing)
        with self.assertRaises(ValueError):
            load_dataset('invalid', reference=reference)

    def test_original_file_layout_loading_and_temporal_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            meg, ieeg = root / 'meg', root / 'ieeg'
            meg.mkdir(); ieeg.mkdir()
            t = np.arange(20) / 250 - .1
            epochs = np.arange(60 * 3 * 20, dtype=float).reshape(60, 3, 20)
            with (ieeg / 'I_epochs.p').open('wb') as f:
                pickle.dump(epochs, f)
            (ieeg / 'I_info.json').write_text(json.dumps({'event_id': [1]*30 + [2]*30, 'time_epoch': t.tolist()}))
            # Reverse metadata row order to verify key-based reordering.
            meta = pd.DataFrame({'subject': ['I']*3, 'channel_index': [2, 1, 0],
                                 'x': [30, 20, 10], 'y': [0]*3, 'z': [0]*3})
            meta.to_csv(root / 'meta.csv', index=False)
            a = np.arange(2 * 4 * 21, dtype=float).reshape(2, 4, 21)
            with (meg / 'M_source.p').open('wb') as f:
                pickle.dump(a, f)
            pd.DataFrame([[.01, 0, 0], [.02, 0, 0], [.03, 0, 0], [.04, 0, 0]]).to_csv(meg / 'M_pos.csv')
            kwargs = dict(metadata_csv=root / 'meta.csv', meg_tmin=-.1,
                          meg_scaling='none', ieeg_scaling='none', ieeg_multiplier=1.)
            data = load_project_data(meg, ieeg, **kwargs)
            np.testing.assert_array_equal(data['meg'][0], a[..., :-1])
            np.testing.assert_allclose(data['ieeg'][0], epochs[:30].mean(0))
            np.testing.assert_array_equal(data['electrode_positions'][:, 0], [10, 20, 30])
            self.assertEqual(data['trial_counts'].n_trials.tolist(), [30, 30])
            loaded = load_dataset('ieeg', meg_dir=meg, ieeg_dir=ieeg, **kwargs)
            self.assertEqual(loaded.name, 'iEEG')
            np.testing.assert_array_equal(loaded.arrays[0], data['ieeg'])
            self.assertEqual(load_dataset('paired_coverage', reference=loaded).n_features, 3)
            prepared = load_project_data(meg, ieeg,
                **{**kwargs, 'metadata_csv': None, 'electrode_metadata': meta,
                   'meg_subjects': ['M'], 'ieeg_subjects': ['I']})
            np.testing.assert_array_equal(prepared['electrode_positions'], data['electrode_positions'])
            self.assertEqual(prepared['load_config']['metadata_source'], 'prepared_dataframe')
            with self.assertRaisesRegex(ValueError, 'not both'):
                load_project_data(meg, ieeg, **kwargs, electrode_metadata=meta)
            with self.assertRaisesRegex(ValueError, 'duplicates'):
                load_project_data(meg, ieeg, **kwargs, meg_subjects=['M', 'M'])
            with self.assertRaisesRegex(ValueError, 'not aligned'):
                load_project_data(meg, ieeg, **{**kwargs, 'meg_tmin': 0})
            with self.assertRaisesRegex(ValueError, 'verified MEG_TMIN'):
                load_project_data(meg, ieeg, **{**kwargs, 'meg_tmin': None})


if __name__ == '__main__':
    unittest.main()
