"""Synthetic checks for direct cross-modal shells and uncertainty units."""
import unittest
import numpy as np
import pandas as pd
from vs3_1_cross_modal_profiles import analyze_cross_modal_profiles


class CrossModalTests(unittest.TestCase):
    def setUp(self):
        self.anchor = pd.DataFrame(dict(subject=['s']*4, x=[0., 2., 5., 10.], y=[0.]*4, z=[0.]*4))
        mm = self.anchor.assign(ieeg_subject='s', electrode_index=np.arange(4)).copy()
        mm.x = [0., 2., 5., 5.]
        self.data = dict(anchor=self.anchor, meg_metadata=[mm, mm.copy()], weights={
            'iEEG': np.array([[[1., 0.]]*4]*2),
            'MEG': np.array([[[1., 0.], [0., 2.], [3., 0.], [3., 0.]],
                             [[0., 1.], [0., 2.], [3., 0.], [3., 0.]]])})

    def test_shells_and_sem(self):
        r = analyze_cross_modal_profiles(self.data, normalize_weights=False)
        c = r['electrode_curves']
        a = c.query("electrode_row == 0 and metric == 'cosine_distance' and bin == 0").sort_values('iteration')
        np.testing.assert_allclose(a['mean'], [.5, 1.])
        np.testing.assert_allclose(a.within_sem, [.5, 0.])
        b = c.query("electrode_row == 0 and metric == 'cosine_distance' and bin == 1")
        self.assertTrue((b.n_sources == 1).all())
        self.assertTrue(b.within_sem.isna().all())
        s = r['summary'].query("electrode_row == 0 and metric == 'cosine_distance' and bin == 0").iloc[0]
        self.assertAlmostEqual(s['mean'], .75)
        self.assertAlmostEqual(s.across_sem, .25)
        # Magnitudes are 1 (iEEG) and [1,2,3] (unique MEG); V=1/3.
        gamma = c.query("iteration == 0 and electrode_row == 0 and metric == 'magnitude_semivariance' and bin == 0").iloc[0]
        self.assertAlmostEqual(gamma['mean'], .75)
        self.assertEqual(r['normalization_audit'].n_unique_meg_sources.iloc[0], 3)

    def test_normalization_and_undefined_profiles(self):
        raw = analyze_cross_modal_profiles(self.data, normalize_weights=False)
        norm = analyze_cross_modal_profiles(self.data, normalize_weights=True)
        mask = raw['electrode_curves'].metric == 'cosine_distance'
        np.testing.assert_allclose(raw['electrode_curves'].loc[mask, 'mean'], norm['electrode_curves'].loc[mask, 'mean'])
        self.data['weights']['iEEG'][:, 0] = 0
        r = analyze_cross_modal_profiles(self.data)
        z = r['electrode_curves'].query("electrode_row == 0 and metric == 'cosine_distance'")
        self.assertTrue(z['mean'].isna().all())
        self.assertTrue((z.n_valid == 0).all())

    def test_owner_validation(self):
        self.data['meg_metadata'][0].loc[0, 'ieeg_subject'] = 'other'
        with self.assertRaises(ValueError):
            analyze_cross_modal_profiles(self.data)


if __name__ == '__main__':
    unittest.main()
