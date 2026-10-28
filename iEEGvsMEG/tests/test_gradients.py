import unittest
import numpy as np
import pandas as pd
from vs3_1_gradients import analyze_gradients, fit_decay, gaussian_decay


class GradientTests(unittest.TestCase):
    def metadata(self, n):
        return pd.DataFrame(dict(x=np.arange(n, dtype=float), y=np.zeros(n),
                                 z=np.zeros(n), subject=['a'] * n))

    def test_gaussian_recovery(self):
        d = np.arange(0., 61.)
        a = gaussian_decay(d, .2, 2., 12.)
        a[0] = 100  # Selection peak must not control the radial fit.
        fit = fit_decay(d, a)
        self.assertEqual(fit['fit_status'], 'ok')
        np.testing.assert_allclose([fit['baseline'], fit['amplitude'], fit['sigma_mm']],
                                   [.2, 2., 12.], rtol=1e-4)
        self.assertAlmostEqual(fit['r_squared'], 1.)

    def test_local_peaks_and_other_peak_distances(self):
        w = np.array([1., 4., 2., 1., 3., 1.])[:, None]
        r = analyze_gradients(w, self.metadata(6), n_components=1,
                              neighborhood_mm=1.1)['pooled']
        self.assertEqual(r['maxima'].feature_row.tolist(), [1, 4])
        self.assertEqual(r['other_maxima']['P1'].distance_mm.tolist(), [3.])
        self.assertEqual(r['profiles']['P1'].distance_mm.tolist(), [1., 0., 1., 2., 3., 4.])

    def test_percentage_and_component_count(self):
        w = np.column_stack([np.arange(20.), np.arange(20.)[::-1] * 100])
        a = analyze_gradients(w, self.metadata(20), n_components=1,
                              neighborhood_mm=None)['pooled']
        b = analyze_gradients(w, self.metadata(20), n_components=2,
                              neighborhood_mm=None, top_percent=10)['pooled']
        self.assertEqual(a['maxima'].feature_row.tolist(), [19])
        self.assertEqual(b['maxima'].feature_row.tolist(), [0, 1])

    def test_subjects_ties_and_plot_cap(self):
        m = self.metadata(20)
        m['subject'] = ['a'] * 10 + ['b'] * 10
        w = np.ones((20, 1))
        r = analyze_gradients(w, m, n_components=1, neighborhood_mm=None,
                              top_percent=20, spatial_scope='within_subject', max_peaks=1)
        self.assertEqual(r['a']['maxima'].feature_row.tolist(), [0, 1])
        self.assertEqual(r['b']['maxima'].feature_row.tolist(), [10, 11])
        self.assertEqual(len(r['a']['selected_maxima']), 1)
        self.assertEqual(len(r['a']['other_maxima']['P1']), 1)

    def test_degenerate_and_invalid_inputs(self):
        self.assertEqual(fit_decay(np.arange(10), np.ones(10))['fit_status'], 'constant_amplitude')
        self.assertEqual(fit_decay(np.arange(3), np.ones(3))['fit_status'], 'insufficient_data')
        r = analyze_gradients(np.zeros((10, 1)), self.metadata(10), n_components=1,
                              neighborhood_mm=None)['pooled']
        self.assertTrue(r['maxima'].empty)
        with self.assertRaises(ValueError):
            analyze_gradients(np.ones((10, 1)), self.metadata(10), n_components=2)
        m = self.metadata(10); m['meg_subject'] = 'participant_average'
        with self.assertRaisesRegex(ValueError, 'Averaged'):
            analyze_gradients(np.ones((10, 1)), m, n_components=1, spatial_scope='within_subject')


if __name__ == '__main__':
    unittest.main()
