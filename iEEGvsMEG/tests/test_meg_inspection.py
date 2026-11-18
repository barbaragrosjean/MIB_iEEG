"""Validate notebook inspection against small MATLAB fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy.io import savemat

NOTEBOOK = Path(__file__).resolve().parents[1] / 'vs0_inspect_matched_meg.ipynb'
namespace = dict(np=np, Path=Path, json=json)
helpers = next(c for c in json.loads(NOTEBOOK.read_text())['cells']
               if c.get('id') == 'meg-inspection-05')
exec(compile(''.join(helpers['source']), str(NOTEBOOK), 'exec'), namespace)


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.time = np.arange(12) / 250.
        self.pos = np.zeros((4, 3))

    def test_conventional_mat_preserves_conditions_and_single_trial_axis(self):
        cells = np.empty((2, 1), object)
        cells[0, 0] = np.zeros((4, 12, 7))
        cells[1, 0] = np.zeros((4, 12, 1))
        path = self.root/'regular.mat'
        savemat(path, dict(OUT=dict(time=self.time, pos_brainsources_MNI8=self.pos, sources_ERFs=cells)))
        info, rows, _ = namespace['inspect_file'](path)
        self.assertAlmostEqual(info['sampling_frequency_hz'], 250.)
        self.assertEqual([r['n_trials'] for r in rows], [7, 1])
        self.assertEqual(info['n_stored_conditions'], 2)

    def test_hdf5_reads_matlab_dimensions_and_nested_cells(self):
        path = self.root/'v73.mat'
        with h5py.File(path, 'w') as handle:
            out = handle.create_group('OUT')
            time = out.create_dataset('time', data=self.time[:, None])
            time.attrs['MATLAB_class'] = np.bytes_('double')
            pos = out.create_dataset('pos_brainsources_MNI8', data=self.pos.T)
            pos.attrs['MATLAB_class'] = np.bytes_('double')
            cells = out.create_dataset('sources_ERFs', shape=(1, 2), dtype=h5py.ref_dtype)
            cells.attrs['MATLAB_class'] = np.bytes_('cell')
            for i, count in enumerate([7, 5]):
                # Shape metadata for a large signal; no signal values are needed.
                data = handle.create_dataset(f'signal_{i}', shape=(count, 12, 4), dtype='f8')
                data.attrs['MATLAB_class'] = np.bytes_('double')
                wrapper = handle.create_dataset(f'wrapper_{i}', shape=(1, 1), dtype=h5py.ref_dtype)
                wrapper.attrs['MATLAB_class'] = np.bytes_('cell')
                wrapper[0, 0] = data.ref
                cells[0, i] = wrapper.ref
        info, rows, _ = namespace['inspect_file'](path)
        self.assertEqual(info['n_sources'], 4)
        self.assertAlmostEqual(info['sampling_frequency_hz'], 250.)
        self.assertEqual([r['n_trials'] for r in rows], [7, 5])
        self.assertEqual(rows[0]['array_shape'], '(4, 12, 7)')

    def test_averaged_and_ambiguous_arrays_do_not_invent_trial_counts(self):
        averaged = namespace['interpret_shape']((4, 12), 4, 12)
        ambiguous = namespace['interpret_shape']((4, 12, 4), 4, 12)
        self.assertTrue(np.isnan(averaged['n_trials']))
        self.assertTrue(np.isnan(ambiguous['n_trials']))

    def test_time_units_and_nonuniform_axis(self):
        info = namespace['time_summary'](np.arange(12)*4., 'ms')
        self.assertAlmostEqual(info['sampling_frequency_hz'], 250.)
        irregular = namespace['time_summary']([0, .004, .009])
        self.assertFalse(irregular['uniform_time_axis'])
        self.assertTrue(np.isnan(irregular['sampling_frequency_hz']))


if __name__ == '__main__':
    unittest.main()
