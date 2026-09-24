import unittest
import numpy as np
import pandas as pd
from coverage_matching_utils import Dataset, inspect_meg_grids, reorder_meg_grids, construct_five_datasets


class GridTests(unittest.TestCase):
    def reference(self, positions, values):
        source = dict(meg=[self.a, values], meg_positions=[self.pos, positions],
                      meg_subjects=['a', 'b'])
        return Dataset('iEEG', [self.a[:, :1]], pd.DataFrame(), np.array([0]), source_data=source)

    def setUp(self):
        self.pos = np.array([[0., 0, 0], [8., 0, 0], [16., 0, 0]])
        self.a = np.arange(24.).reshape(2, 3, 4)

    def test_permutation_reorders_signals_and_coordinates(self):
        order = [2, 0, 1]
        ref = self.reference(self.pos[order], self.a[:, order])
        self.assertEqual(inspect_meg_grids(ref).status.tolist(), ['aligned', 'reordered_grid'])
        fixed = reorder_meg_grids(ref)
        np.testing.assert_array_equal(fixed.source_data['meg'][1], self.a)
        np.testing.assert_array_equal(fixed.source_data['meg_positions'][1], self.pos)
        np.testing.assert_array_equal(ref.source_data['meg'][1], self.a[:, order])
        s = fixed.source_data
        datasets, _, _ = construct_five_datasets(s['meg'], s['meg_positions'], s['meg_subjects'],
            self.pos[:1], ['i0'], kinds=['full_average'])
        np.testing.assert_array_equal(datasets['full_average'].arrays[0], self.a)

    def test_real_grid_difference_is_not_silently_repaired(self):
        ref = self.reference(self.pos+np.array([1., 0, 0]), self.a)
        self.assertEqual(inspect_meg_grids(ref).status.iloc[1], 'different_grid_or_shape')
        with self.assertRaises(ValueError): reorder_meg_grids(ref)

    def test_source_count_difference_and_roundoff(self):
        ref = self.reference(self.pos[:2], self.a[:, :2])
        self.assertFalse(inspect_meg_grids(ref).same_shape.iloc[1])
        ref = self.reference(self.pos+1e-8, self.a)
        self.assertEqual(inspect_meg_grids(ref).status.iloc[1], 'aligned')


if __name__ == '__main__': unittest.main()
