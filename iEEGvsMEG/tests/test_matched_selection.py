"""Synthetic integration checks; no remote recordings are required."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.io import savemat
import vs0_load_and_config as shared
from vs1_coverage_matching_utils import load_dataset as _load_dataset
from vs0_prepare_data import prepare_data
from vs2_1_plssvd_eval_utils import prepare_trial_cache, load_trial_cache


def load_dataset(kind, **options):
    if options.get('reference') is None:
        file_options = {k: v for k, v in options.items() if k not in ('condition_mode', 'block_scaling', 'seed', 'pairing')}
        file_options.setdefault('prepared_cache_dir', Path(file_options['ieeg_dir']).parent / 'prepared')
        prepare_data(**file_options)
        options['prepared_cache_dir'] = file_options['prepared_cache_dir']
    return _load_dataset(kind, **options)


class MatchedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.meg = self.root / 'meg'
        self.ieeg = self.root / 'ieeg'
        self.meg.mkdir()
        self.ieeg.mkdir()
        rng = np.random.default_rng(11)
        self.times = np.arange(12) / 250
        self.metadata = pd.DataFrame([
            dict(subject=s, channel_index=i, x=i * 8, y=0, z=0)
            for s in ['I1', 'I2', 'I3'] for i in range(2)])
        for s in ['I1', 'I2', 'I3']:
            with (self.ieeg / f'{s}_epochs.p').open('wb') as f:
                pickle.dump(rng.normal(size=(16, 2, 12)), f)
            (self.ieeg / f'{s}_info.json').write_text(json.dumps(
                dict(event_id=[1]*8+[2]*8, time_epoch=self.times.tolist())))
        self.files = {}
        for s in ['M1', 'M2', 'M3', 'M4']:
            cells = np.empty((2, 1), object)
            for c in range(2):
                cells[c, 0] = rng.normal(size=(3, 12, 8))
            path = self.meg / f'{s}_norm0_abs_0.mat'
            savemat(path, dict(OUT=dict(sources_ERFs=cells, time=self.times,
                                      pos_brainsources_MNI8=np.array([[0, 0, 0], [.008, 0, 0], [.016, 0, 0]]))))
            self.files[s] = str(path)
        self.csv = self.root / 'matched_pairs.csv'
        pd.DataFrame(dict(ieeg_subject=['I2', 'I1'], meg_subject=['M1', 'M3'],
                          path_to_meg=[self.files['M1'], self.files['M3']])).to_csv(self.csv, index=False)
        self.options = dict(meg_dir=self.meg, ieeg_dir=self.ieeg,
                            electrode_metadata=self.metadata, ieeg_coordinate_unit='mm',
                            use_matched_pairs=True, matched_pairs_csv=self.csv)

    def test_matched_averages_and_assignment_ignore_pairing_seed(self):
        ref = load_dataset('ieeg', **self.options)
        self.assertEqual(ref.source_data['ieeg_subjects'], ['I2', 'I1'])
        self.assertEqual(ref.source_data['meg_subjects'], ['M1', 'M3'])
        a = load_dataset('paired_coverage', reference=ref, seed=1)
        b = load_dataset('paired_coverage', reference=ref, seed=99)
        self.assertEqual(a.pairing, {'I2': 'M1', 'I1': 'M3'})
        np.testing.assert_array_equal(a.arrays[0], b.arrays[0])
        self.assertEqual(set(ref.source_data['meg_files'].values()), {self.files['M1'], self.files['M3']})

    def test_random_selection_is_reproducible_and_csv_independent(self):
        options = dict(self.options, use_matched_pairs=False, matched_pairs_csv=self.root/'absent.csv')
        first = load_dataset('ieeg', **options)
        second = load_dataset('ieeg', **options)
        self.assertEqual(first.source_data['meg_subjects'], second.source_data['meg_subjects'])
        self.assertEqual(len(first.source_data['meg_subjects']), 3)
        pairs = [load_dataset('paired_coverage', reference=first, seed=i).pairing for i in range(5)]
        self.assertGreater(len({tuple(sorted(v.items())) for v in pairs}), 1)

    def test_duplicate_pairing_rejected(self):
        table = pd.read_csv(self.csv)
        table.loc[1, 'meg_subject'] = 'M1'
        table.to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, 'one-to-one'):
            load_dataset('ieeg', **self.options)

    def test_missing_csv_never_falls_back_to_random(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset('ieeg', **dict(self.options, matched_pairs_csv=self.root/'absent.csv'))

    def test_trial_cache_uses_exact_csv_files_and_pairing(self):
        cache = self.root / 'cache'
        trials = prepare_trial_cache(self.meg, self.ieeg, cache, ['M1', 'M2', 'M3', 'M4'],
            ['I1', 'I2', 'I3'], self.metadata, ieeg_coordinate_unit='mm',
            use_matched_pairs=True, matched_pairs_csv=self.csv)
        self.assertEqual(trials.pairing, {'I2': 'M1', 'I1': 'M3'})
        self.assertEqual([s.subject for s in trials.ieeg], ['I2', 'I1'])
        self.assertEqual(trials.meg[0].data[0].shape, (8, 3, 12))
        # Editing the CSV must not silently load unrelated cached recordings.
        table = pd.read_csv(self.csv)
        table.loc[0, 'path_to_meg'] = self.files['M4']
        table.to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, 'CSV MEG files'):
            load_trial_cache(cache, use_matched_pairs=True, matched_pairs_csv=self.csv)

    def test_actual_fit_preserves_csv_pairing_in_every_fold(self):
        from vs2_1_plssvd_eval_utils import ValidationOptions, fit_plssvd
        cache = self.root / 'cache'
        trials = prepare_trial_cache(self.meg, self.ieeg, cache, list(self.files),
            ['I1', 'I2', 'I3'], self.metadata, ieeg_coordinate_unit='mm',
            use_matched_pairs=True, matched_pairs_csv=self.csv)
        output = self.root / 'fit'
        fit_plssvd(trials, 'paired_coverage', ValidationOptions(
            n_iterations=3, repeats=2, n_components=1, perm='none', perm_type='none'), output)
        self.assertTrue((output/'iteration_000/COMPLETE.json').exists())
        self.assertFalse((output/'iteration_001').exists())
        for path in output.rglob('pairing.json'):
            self.assertEqual(json.loads(path.read_text()), {'I2': 'M1', 'I1': 'M3'})

    def test_single_assignment_summary_has_no_estimated_sd(self):
        from vs3_0_weight_stability import roi_statistics, roi_pair_correlations
        meta = pd.DataFrame(dict(subject=['I1', 'I2'], region=['ROI', 'ROI']))
        weights = np.array([[[1.], [2.]]])
        rows, _ = roi_statistics(weights, meta)
        self.assertEqual(rows[0]['mean_magnitude'], 1.5)
        self.assertTrue(np.isnan(rows[0]['sd_across_iterations']))
        self.assertTrue(np.isnan(roi_pair_correlations(weights, weights, meta)[0]['mean_r']))

    def test_single_matched_iteration_is_enforced(self):
        from vs2_1_plssvd_eval_utils import TrialData, ValidationOptions, fit_plssvd
        trials = TrialData([], [], self.times, (1, 2), {'I1': 'M1'})
        with patch('vs2_1_plssvd_eval_utils._validate_iteration') as fit:
            fit_plssvd(trials, 'paired_coverage', ValidationOptions(n_iterations=9), self.root/'fit')
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(json.loads((self.root/'fit/validation_options.json').read_text())['n_iterations'], 1)


if __name__ == '__main__':
    unittest.main()
