import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vs3_1_gradients import (fit_gradient_models, exponential_decay, gaussian_decay,
    compare_gradient_models, plot_gradient_model_comparison)


class GradientModelTests(unittest.TestCase):
    def test_parameter_recovery_and_adjustment(self):
        x = np.linspace(1, 50, 80)
        cases = [
            ('linear', 4-.03*x, 'slope', -.03),
            ('quadratic', 4-.03*x+.0004*x*x, 'curvature', .0008),
            ('gaussian', gaussian_decay(x, .2, 2, 12), 'sigma_mm', 12),
            ('exponential', exponential_decay(x, .2, 2, 12), 'tau_mm', 12),
        ]
        for model, y, parameter, expected in cases:
            with self.subTest(model=model):
                row = fit_gradient_models(x, y, [model]).iloc[0]
                self.assertEqual(row.fit_status, 'ok')
                self.assertAlmostEqual(row[parameter], expected, places=4)
                self.assertAlmostEqual(row.r_squared, 1.)
        y = 4-.03*x + np.random.default_rng(1).normal(0, .1, len(x))
        row = fit_gradient_models(x, y, ['linear']).iloc[0]
        self.assertAlmostEqual(row.adjusted_r_squared,
                               1-(1-row.r_squared)*79/78)
        # Zero-distance peaks and invalid values are excluded for every model.
        table = fit_gradient_models(np.r_[0, x, np.nan], np.r_[1e6, y, 1])
        self.assertTrue(table.n_fit.eq(len(x)).all())
        self.assertTrue(fit_gradient_models(x, np.ones(len(x))).fit_status.eq('constant_amplitude').all())
        self.assertTrue(fit_gradient_models(x[:3], y[:3]).fit_status.eq('insufficient_data').all())

    def test_geometry_is_reused_and_figures_include_both_modalities(self):
        x = np.linspace(0, 60, 121)
        y = gaussian_decay(x, .2, 2, 12)
        profile = pd.DataFrame(dict(distance_mm=x, weight_norm=y, in_fit_radius=(x>0)&(x<=40)))
        result = dict(features=pd.DataFrame({'scope': ['pooled']}),
            selected_maxima=pd.DataFrame({'peak_id': ['P1']}), profiles={'P1': profile},
            fold_hemispheres=True, target_hemisphere='left', fit_radius_mm=40., n_components=2)
        table = compare_gradient_models({m: {'pooled': result} for m in ('iEEG', 'MEG')})
        self.assertEqual(len(table), 8)
        self.assertTrue(table.n_fit.eq(80).all())
        self.assertTrue(table.fold_hemispheres.all())
        self.assertTrue(table.target_hemisphere.eq('left').all())
        self.assertTrue(table.matched_peak.all())
        figs = plot_gradient_model_comparison(table)
        self.assertEqual(len(figs), 5)
        self.assertEqual([t.get_text() for t in figs['linear_parameter'].axes[0].get_xticklabels()], ['iEEG', 'MEG'])
        self.assertEqual(len(figs['goodness_of_fit'].axes[0].collections), 8)
        for fig in figs.values():
            fig.canvas.draw()
            plt.close(fig)
        # Failed fits exclude the whole peak from every model comparison.
        result['profiles']['P1'] = profile.assign(weight_norm=1.)
        bad = compare_gradient_models({'iEEG': {'pooled': result}})
        self.assertFalse(bad.matched_peak.any())
        for fig in plot_gradient_model_comparison(bad).values():
            fig.canvas.draw()
            plt.close(fig)

    def test_subject_medians_not_feature_weighted(self):
        rows = []
        for model in ('linear', 'quadratic', 'gaussian', 'exponential'):
            for scope, values in [('a', [1., 3., 8.]), ('b', [10.])]:
                for value in values:
                    rows.append(dict(model=model, modality='iEEG', scope=scope,
                        matched_peak=True, within_subject=True, adjusted_r_squared=.5,
                        slope=value, curvature=value, sigma_mm=value, tau_mm=value))
        figs = plot_gradient_model_comparison(pd.DataFrame(rows))
        points = figs['linear_parameter'].axes[0].collections[0].get_offsets()
        np.testing.assert_allclose(points[:, 1], [3., 10.])
        for fig in figs.values():
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
