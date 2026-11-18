"""Cache correctness and one-pass dataset construction on synthetic recordings."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import test_matched_selection as fixtures
import vs0_load_and_config as raw
from vs0_prepare_data import prepare_data, load_prepared_data
from vs1_coverage_matching_utils import load_dataset, load_meg_datasets, SETUP_NAMES
from vs1_run_coverage import load_datasets


class PreparedDataTests(unittest.TestCase):
    setUp = fixtures.MatchedSelectionTests.setUp

    def test_round_trip_equals_raw_preprocessing_and_memory_maps_meg(self):
        options = dict(self.options, prepared_cache_dir=self.root/'prepared')
        expected = raw.load_project_data(**self.options)
        prepare_data(**options)
        with patch('vs0_prepare_data.raw.load_project_data', side_effect=AssertionError('raw load')):
            actual = load_prepared_data(**options)
        for name in ['ieeg', 'times', 'electrode_subjects', 'electrode_positions']:
            np.testing.assert_array_equal(actual[name], expected[name])
        for first, second in zip(actual['meg'], expected['meg']):
            self.assertIsInstance(first, np.memmap)
            self.assertFalse(first.flags.writeable)
            np.testing.assert_array_equal(first, second)
        pd.testing.assert_frame_equal(actual['electrode_metadata'], expected['electrode_metadata'])
        pd.testing.assert_frame_equal(actual['trial_counts'], expected['trial_counts'])
        self.assertEqual(actual['pairing'], expected['pairing'])

    def test_loading_missing_cache_never_prepares(self):
        with patch('vs0_prepare_data.raw.load_project_data') as prepare:
            with self.assertRaisesRegex(FileNotFoundError, 'Run prepare_data'):
                load_prepared_data(**dict(self.options, prepared_cache_dir=self.root/'prepared'))
        prepare.assert_not_called()

    def test_assignment_and_csv_order_change_reuse_prepared_recordings(self):
        options = dict(self.options, prepared_cache_dir=self.root/'prepared')
        directory = prepare_data(**options)
        table = pd.read_csv(self.csv)
        table['meg_subject'] = ['M3', 'M1']
        table['path_to_meg'] = [self.files['M3'], self.files['M1']]
        table.iloc[::-1].to_csv(self.csv, index=False)
        with patch('vs0_prepare_data.raw.load_project_data', side_effect=AssertionError('reprocessed')):
            self.assertEqual(prepare_data(**options), directory)
            loaded = load_prepared_data(**options)
        self.assertEqual(loaded['pairing'], {'I1': 'M1', 'I2': 'M3'})
        self.assertEqual(loaded['ieeg_subjects'], ['I1', 'I2'])
        self.assertEqual(loaded['meg_subjects'], ['M1', 'M3'])
        # Reordering participants must also reorder electrode metadata and signal blocks.
        expected = raw.load_project_data(**self.options)
        np.testing.assert_array_equal(loaded['ieeg'], expected['ieeg'])
        pd.testing.assert_frame_equal(loaded['electrode_metadata'], expected['electrode_metadata'])

    def test_preprocessing_and_source_changes_invalidate_cache(self):
        options = dict(self.options, prepared_cache_dir=self.root/'prepared')
        prepare_data(**options)
        with self.assertRaisesRegex(FileNotFoundError, 'No compatible prepared data'):
            load_prepared_data(**dict(options, meg_scaling='none'))
        path = Path(self.files['M1'])
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        with self.assertRaisesRegex(FileNotFoundError, 'No compatible prepared data'):
            load_prepared_data(**options)

    def test_all_setups_equal_separate_construction_with_one_spatial_pass(self):
        options = dict(self.options, prepared_cache_dir=self.root/'prepared')
        prepare_data(**options)
        reference = load_dataset('ieeg', block_scaling=True, **options)
        for mode in ('average', 'stack'):
            reference.condition_mode = mode
            expected = {name: load_dataset(name, reference=reference, seed=12) for name in SETUP_NAMES}
            with patch('vs1_coverage_matching_utils.cKDTree', wraps=cKDTree) as tree:
                actual = load_meg_datasets(reference, seed=12)
            self.assertEqual(tree.call_count, len(reference.source_data['meg']))
            for name in SETUP_NAMES:
                self.assertEqual(actual[name].pairing, expected[name].pairing)
                self.assertEqual(actual[name].scale, expected[name].scale)
                pd.testing.assert_frame_equal(actual[name].metadata, expected[name].metadata)
                pd.testing.assert_frame_equal(actual[name].matching, expected[name].matching)
                for a, b in zip(actual[name].arrays, expected[name].arrays):
                    np.testing.assert_array_equal(a, b)

    def test_random_coverage_loader_uses_one_pass_and_preserves_pairings(self):
        options = dict(self.options, prepared_cache_dir=self.root/'prepared', use_matched_pairs=False)
        # data_options forwards the top-level seed to cohort selection.
        options['selection_seed'] = 12
        prepare_data(**options)
        config = dict(data=options, condition_mode='average', block_scaling=True, seed=12)
        with patch('vs1_coverage_matching_utils.cKDTree', wraps=cKDTree) as tree:
            datasets = load_datasets(config)
        self.assertEqual(tree.call_count, 3)
        expected = {name: load_dataset(name, reference=datasets['iEEG'], seed=12) for name in SETUP_NAMES}
        for name in SETUP_NAMES:
            self.assertEqual(datasets[name].pairing, expected[name].pairing)
            for a, b in zip(datasets[name].arrays, expected[name].arrays):
                np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
