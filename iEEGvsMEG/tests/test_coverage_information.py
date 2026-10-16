import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from coverage_information import (MEG_KINDS, temporal_spectrum, reference_fraction,
    compare_meg_information, split_half_reliability, _trial_halves,
    save_information, load_information, plot_meg_information, plot_split_half_reliability)


class MatrixDataset:
    condition_mode = 'average'
    source_data = None

    def __init__(self, x):
        self.x = x
        self.n_observations, self.n_features = x.shape

    def blocks(self):
        yield self.x


class InformationTests(unittest.TestCase):
    def test_spectrum_and_weighted_projection(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(30, 8))
        x -= x.mean(0)
        fit = temporal_spectrum(MatrixDataset(x))
        singular = np.linalg.svd(x, compute_uv=False)
        eigenvalues = singular**2/29
        np.testing.assert_allclose(fit['eigenvalues'][:8], eigenvalues)
        self.assertAlmostEqual(fit['effective_dimensionality'], eigenvalues.sum()**2/(eigenvalues**2).sum())
        modified = temporal_spectrum(MatrixDataset(rng.normal(size=(30, 5))))
        scores = fit['basis'][:, :3] * np.sqrt(fit['eigenvalues'][:3])
        q = modified['basis'][:, :3]
        expected = 1-np.sum((scores-q@q.T@scores)**2)/np.sum(scores**2)
        self.assertAlmostEqual(reference_fraction(fit, modified, 3), expected)
        self.assertAlmostEqual(reference_fraction(fit, fit, 3), 1)
        self.assertTrue(np.isnan(reference_fraction(fit, modified, 9)))

    def test_rank_one_and_rotation_invariance(self):
        x = np.outer(np.arange(20.), [1, 2, 3])
        fit = temporal_spectrum(MatrixDataset(x))
        self.assertEqual(fit['rank'], 1)
        self.assertAlmostEqual(fit['effective_dimensionality'], 1)
        with self.assertRaises(ValueError):
            temporal_spectrum(MatrixDataset(np.ones((10, 2))))
        rng = np.random.default_rng(5)
        x = rng.normal(size=(20, 4))
        q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        a, b = [temporal_spectrum(MatrixDataset(v)) for v in (x, x@q)]
        self.assertAlmostEqual(reference_fraction(a, b, 3), 1)

    def test_disjoint_halves(self):
        data = np.arange(7*2*9.).reshape(7, 2, 9)
        parts, audit = _trial_halves([data], np.random.default_rng(7))
        ids = [[r['trial_index'] for r in audit if r['half'] == h] for h in (0, 1)]
        self.assertFalse(set(ids[0]) & set(ids[1]))
        self.assertEqual(sorted(ids[0]+ids[1]), list(range(7)))
        for h in (0, 1):
            np.testing.assert_allclose(parts[h][0], data[ids[h]].mean(0))

    def test_cache_reliability_and_figures(self):
        from coverage_matching_utils import construct_five_datasets
        rng = np.random.default_rng(7)
        positions = [np.array([[0., 0, 0], [8., 0, 0], [16., 0, 0]])]*2
        means = [rng.normal(size=(2, 3, 18)) for _ in range(2)]
        subjects = ['m1', 'm2']
        coords = np.array([[0., 0, 0], [16., 0, 0]])
        owners = np.array(['i1', 'i2'])
        pairing = {'i1': 'm1', 'i2': 'm2'}
        context = dict(meg_subjects=subjects, meg_positions=positions,
                       electrode_positions=coords, electrode_subjects=owners,
                       times=np.arange(18.), load_config=dict(conditions=[1, 2], meg_scaling='channel_zscore'))
        reference = SimpleNamespace(source_data=context, condition_mode='average')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for s, mu, pos in zip(subjects, means, positions):
                files = []
                for c in range(2):
                    filename = f'{s}_{c}.npy'
                    np.save(root/filename, np.repeat(mu[c][None], 5, axis=0))
                    files.append(filename)
                records.append(dict(modality='meg', subject=s, positions=pos.tolist(), files=files))
            (root/'manifest.json').write_text(json.dumps(dict(config=dict(conditions=[1, 2]),
                times=list(range(18)), records=records)))
            result = split_half_reliability(reference, root, pairing=pairing, repeats=2, dimensions=[1, 2])
            np.testing.assert_allclose(result['reliability'].overlap, 1., atol=1e-12)
            datasets, _, _ = construct_five_datasets(means, positions, subjects, coords, owners,
                                                    pairing=pairing, kinds=MEG_KINDS)
            information = compare_meg_information(datasets, dimensions=[1, 2, 10])
            self.assertEqual(len(information['summary']), 4)
            save_information(information, root/'saved')
            loaded = load_information(root/'saved')
            np.testing.assert_allclose(loaded['summary'].effective_dimensionality,
                                       information['summary'].effective_dimensionality)
            with self.assertRaises(FileExistsError):
                save_information(information, root/'saved')
            import matplotlib.pyplot as plt
            for name, fig in [('information', plot_meg_information(loaded)),
                              ('reliability', plot_split_half_reliability(result))]:
                fig.savefig(root/f'{name}.png')
                plt.close(fig)


if __name__ == '__main__':
    unittest.main()
