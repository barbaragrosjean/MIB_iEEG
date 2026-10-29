import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vs3_1_gradient_radius_search import radius_search, select_radius_candidates, plot_radius_search


class RadiusSearchTests(unittest.TestCase):
    def test_radius_search_all_fit_models(self):
        x = np.arange(61.)
        metadata = pd.DataFrame(dict(x=x, y=0., z=0., subject='a'))
        models = {
            'linear': 5-.02*x,
            'quadratic': 5-.0005*x*x,
            'gaussian': .2+2*np.exp(-.5*(x/12)**2),
            'exponential': .2+2*np.exp(-x/12),
        }
        for model, y in models.items():
            with self.subTest(model=model):
                result = radius_search(y[:, None], metadata, neighborhood_radii=[10],
                    fit_radii=[30, 50], metric=model, n_components=1, min_peaks=1)
                self.assertTrue(result['grid'].metric.eq(model).all())
                np.testing.assert_allclose(result['grid'].value, 1., atol=1e-8)
                candidates = select_radius_candidates(result['grid'])
                self.assertFalse(candidates.empty)
                self.assertTrue(candidates.selection_rule.eq('maximum mean adjusted R²').all())
                fig = plot_radius_search({'iEEG': result})
                self.assertIn(model, fig.axes[-1].get_ylabel())
                plt.close(fig)

    def test_semivariance_matches_explicit_pairs(self):
        y = np.array([8., 1., 2., 4., 3., 7., 5., 9.])
        d = np.arange(8.)
        profile = pd.DataFrame(dict(distance_mm=d, weight_norm=y))
        result = dict(features=profile, selected_maxima=pd.DataFrame(dict(peak_id=['P1'], feature_row=[0])),
                      profiles={'P1': profile})
        with patch('vs3_1_gradient_radius_search.analyze_gradients', return_value={'pooled': result}):
            out = radius_search(y[:, None], pd.DataFrame(), neighborhood_radii=[10],
                fit_radii=[6, 7], min_peaks=1, metric='semivariance')
        for row in out['grid'].itertuples():
            local = y[(d > 0) & (d <= row.fit_radius_mm)]
            pairs = [.5*(local[i]-local[j])**2 for i in range(len(local)) for j in range(i+1, len(local))]
            self.assertAlmostEqual(row.value, np.mean(pairs)/np.var(y))
        self.assertTrue(out['grid'].eligible.all())

    def test_sparse_estimates_shown_but_not_recommended(self):
        d = np.arange(8.)
        y = 8-.05*d*d
        profile = pd.DataFrame(dict(distance_mm=d, weight_norm=y))
        result = dict(features=profile, selected_maxima=pd.DataFrame(dict(peak_id=['P1'], feature_row=[0])),
                      profiles={'P1': profile})
        for metric in ('semivariance', 'quadratic'):
            with patch('vs3_1_gradient_radius_search.analyze_gradients', return_value={'pooled': result}):
                out = radius_search(y[:, None], pd.DataFrame(), neighborhood_radii=[10],
                    fit_radii=[1, 2, 3], metric=metric)
            grid = out['grid']
            self.assertTrue(np.isnan(grid.iloc[0].value))
            self.assertTrue(np.isfinite(grid.iloc[-1].value))
            self.assertFalse(grid.eligible.any())
            self.assertTrue(select_radius_candidates(grid).empty)
            fig = plot_radius_search({'iEEG': out})
            array = fig.axes[0].collections[0].get_array()
            self.assertLess(np.ma.getmaskarray(array).sum(), array.size)
            self.assertIn('Limited support', fig.axes[0].get_legend_handles_labels()[1])
            plt.close(fig)

    def test_plateau_and_unresolved_cases(self):
        frame = pd.DataFrame(dict(neighborhood_mm=[10]*7, fit_radius_mm=np.arange(10, 80, 10),
            value=[.2, .5, .91, 1., .98, 1.01, 1.], eligible=True, metric='semivariance'))
        out = select_radius_candidates(frame)
        self.assertEqual(out.iloc[0].fit_radius_mm, 30)
        self.assertTrue(out.iloc[0].selected)
        rising = frame.assign(value=np.arange(1., 8.))
        self.assertTrue(select_radius_candidates(rising).empty)
        flat = frame.assign(value=1.)
        self.assertTrue(select_radius_candidates(flat).empty)
        frame.loc[6, 'eligible'] = False
        self.assertTrue(select_radius_candidates(frame).empty)

    def test_quadratic_ranking_masks_and_ties(self):
        grid = pd.DataFrame(dict(neighborhood_mm=[10, 10, 20, 20], fit_radius_mm=[10, 20, 10, 20],
            value=[.8, .8, .9, -.2], eligible=[True, True, False, True], metric='quadratic'))
        out = select_radius_candidates(grid)
        self.assertEqual((out.iloc[0].neighborhood_mm, out.iloc[0].fit_radius_mm), (10, 10))
        results = {m: {'grid': grid.copy()} for m in ('iEEG', 'MEG')}
        fig = plot_radius_search(results, candidates={m: out for m in results})
        self.assertEqual(len(fig.axes), 3)  # Two modality axes and one colorbar.
        self.assertIs(fig.axes[0].collections[0].norm, fig.axes[1].collections[0].norm)
        fig.canvas.draw()
        plt.close(fig)
        for r in results.values():
            r['grid']['eligible'] = False
        fig = plot_radius_search(results)
        fig.canvas.draw()
        plt.close(fig)

    def test_actual_grid_and_subject_aggregation(self):
        x = np.tile(np.arange(40.), 2)
        metadata = pd.DataFrame(dict(x=x, y=0., z=0., subject=['a']*40+['b']*40))
        weights = np.r_[5-.001*np.arange(40.)**2, 9-.002*np.arange(40.)**2][:, None]
        for metric in ('quadratic', 'semivariance'):
            out = radius_search(weights, metadata, neighborhood_radii=[5, 10], fit_radii=[10, 20],
                n_components=1, spatial_scope='within_subject', fold_hemispheres=True,
                metric=metric, min_peaks=2)
            self.assertEqual(len(out['grid']), 4)
            self.assertTrue(out['grid'].n_units.eq(2).all())
            self.assertTrue(out['grid'].eligible.all())
            self.assertTrue(out['details'].n_points.isin([10, 20]).all())
            if metric == 'quadratic':
                np.testing.assert_allclose(out['grid'].value, 1.)
        for radii in ([0, 10], [10, 10], [20, 10]):
            with self.assertRaises(ValueError):
                radius_search(weights, metadata, neighborhood_radii=radii, fit_radii=[10])


if __name__ == '__main__':
    unittest.main()
