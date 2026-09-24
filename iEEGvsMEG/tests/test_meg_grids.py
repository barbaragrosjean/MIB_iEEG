import unittest
import numpy as np
from coverage_matching_utils import construct_five_datasets


class FullAverageTests(unittest.TestCase):
    def test_existing_row_order_is_preserved_with_different_coordinates(self):
        a = np.arange(24.).reshape(2, 3, 4)
        b = a[:, [2, 0, 1]] + 10
        positions = np.array([[0., 0, 0], [8., 0, 0], [16., 0, 0]])
        second = positions[[2, 0, 1]] + 1
        electrode_positions = positions[:1].copy()
        datasets, _, _ = construct_five_datasets([a, b], [positions, second], ['a', 'b'],
            electrode_positions, ['i0'], kinds=['full_average'])
        np.testing.assert_array_equal(datasets['full_average'].arrays[0], (a+b)/2)
        np.testing.assert_array_equal(electrode_positions, positions[:1])
        np.testing.assert_array_equal(b, a[:, [2, 0, 1]]+10)

    def test_incompatible_source_counts_raise(self):
        a = np.zeros((2, 3, 4))
        with self.assertRaisesRegex(ValueError, 'same MEG source count'):
            construct_five_datasets([a, a[:, :2]], [np.zeros((3, 3)), np.zeros((2, 3))],
                ['a', 'b'], np.zeros((1, 3)), ['i0'], kinds=['full_average'])


if __name__ == '__main__': unittest.main()
