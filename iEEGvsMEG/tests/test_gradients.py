import unittest
import numpy as np
import pandas as pd
from vs3_1_gradients import (analyze_gradients, fit_decay, gaussian_decay,
                           summarize_subject_gradients)


class GradientTests(unittest.TestCase):
    def test_folding_retains_features_and_changes_distance_geometry(self):
        m = pd.DataFrame(dict(x=[-30., 30., -30., 30.], y=[0., 0., 10., 20.],
                              z=[0.] * 4, subject=['a', 'b', 'a', 'b']))
        before = m.copy(deep=True)
        w = np.array([[4.], [3.], [2.], [1.]])
        options = dict(n_components=1, neighborhood_mm=None, top_percent=50)
        native = analyze_gradients(w, m, **options)['pooled']
        left = analyze_gradients(w, m, fold_hemispheres=True, **options)['pooled']
        right = analyze_gradients(w, m, fold_hemispheres=True,
                                  target_hemisphere='right', **options)['pooled']
        pd.testing.assert_frame_equal(m, before)
        self.assertEqual(len(left['features']), 4)
        np.testing.assert_array_equal(left['features'].original_x, m.x)
        np.testing.assert_array_equal(left['features'].weight_norm, w[:, 0])
        np.testing.assert_array_equal(left['features'].x, -30.)
        np.testing.assert_array_equal(right['features'].x, 30.)
        self.assertEqual(native['profiles']['P1'].distance_mm.iloc[1], 60.)
        np.testing.assert_allclose(left['profiles']['P1'].distance_mm, [0., 0., 10., 20.])
        np.testing.assert_allclose(left['profiles']['P1'].distance_mm,
                                   right['profiles']['P1'].distance_mm)
        self.assertFalse(left['profiles']['P1'].in_fit_radius.iloc[1])
        self.assertEqual(left['other_maxima']['P1'].distance_mm.iloc[0], 0.)
        # Reflection also changes which features compete for a local maximum.
        local = analyze_gradients(w, m, n_components=1, fold_hemispheres=True,
                                  neighborhood_mm=15.)['pooled']
        self.assertEqual(local['maxima'].feature_row.tolist(), [0])
        within = analyze_gradients(w, m, spatial_scope='within_subject',
                                   fold_hemispheres=True, **options)
        self.assertEqual(set(within), {'a', 'b'})
        self.assertEqual(len(within['a']['features']), 2)
        with self.assertRaisesRegex(ValueError, 'target_hemisphere'):
            analyze_gradients(w, m, target_hemisphere='invalid', **options)

    def test_subject_summary_equal_weight_and_shared_range(self):
        def result(baseline, distances, status='ok'):
            return dict(profiles={'P1': pd.DataFrame(dict(
                distance_mm=distances, in_fit_radius=np.ones(len(distances), dtype=bool)))},
                fits=pd.DataFrame([dict(peak_id='P1', fit_status=status,
                                        baseline=baseline, amplitude=2., sigma_mm=10.)]))
        # Different numbers of observations must not alter subject weights.
        scopes = {'a': result(1., [1., 10., 20.]),
                  'b': result(3., np.arange(5., 31.)),
                  'bad': result(100., [1., 20.], 'weakly_identified_or_boundary'),
                  'missing_rank': dict(profiles={})}
        s = summarize_subject_gradients(scopes, 'P1')
        self.assertEqual(s['subjects'], ['a', 'b'])
        self.assertEqual((s['distance_mm'][0], s['distance_mm'][-1]), (5., 20.))
        np.testing.assert_allclose(s['mean'], gaussian_decay(s['distance_mm'], 2., 2., 10.))
        np.testing.assert_allclose(s['std'], np.sqrt(2.))
        single = summarize_subject_gradients({'a': scopes['a']}, 'P1')
        self.assertTrue(np.isnan(single['std']).all())
        self.assertEqual(len(summarize_subject_gradients(scopes, 'P2')['mean']), 0)
        scopes['b'] = result(3., [21., 30.])
        self.assertEqual(len(summarize_subject_gradients(scopes, 'P1')['mean']), 0)

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
