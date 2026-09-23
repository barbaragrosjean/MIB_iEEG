"""Numerical and synthetic end-to-end tests; no experimental data required."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSSVD
from compare_subspace import (subspace_metrics, fit_alignment, predict_alignment,
    evaluate_alignment, evaluate_clusters, fit_models, run_comparison, plot_results, load_results,
    MODELS, MEG_KINDS)
from plssvd_eval_utils import load_trial_cache, _split_subject, _build_fold, _project
from utils_updated import Dataset


def make_cache(root):
    rng = np.random.default_rng(34)
    times = np.linspace(-.1, .5, 18)
    latent = np.stack([np.sin(times*11), np.cos(times*8), np.sin(times*21)])
    records = []
    subjects = {'ieeg': ['i0', 'i1'], 'meg': ['m0', 'm1', 'm2']}
    for modality in subjects:
        for subject in subjects[modality]:
            channels = 6 if modality == 'ieeg' else 9
            positions = np.c_[np.arange(channels)*4., np.zeros(channels), np.zeros(channels)]
            weights = rng.normal(size=(channels, 3))
            metadata = pd.DataFrame(positions, columns=['x', 'y', 'z'])
            metadata['subject'] = subject
            metadata['channel_index'] = np.arange(channels)
            files = []
            for condition in (1, 2):
                signal = weights @ (latent * (1 + .2*condition))
                array = signal[None] + rng.normal(scale=.25, size=(20, channels, len(times)))
                if modality == 'ieeg':
                    array /= 1000
                file = f'{modality}_{subject}_{condition}.npy'
                np.save(root/file, array.astype('float32')); files.append(file)
            records.append(dict(modality=modality, subject=subject, files=files,
                                positions=positions.tolist(), metadata=metadata.to_dict(orient='list'),
                                split_groups=[None, None], permutation_blocks=[['all']*20, ['all']*20]))
    (root/'manifest.json').write_text(json.dumps(dict(
        config=dict(ieeg_subjects=subjects['ieeg'], meg_subjects=subjects['meg'], conditions=[1, 2]),
        times=times.tolist(), records=records, source_files=[])))


class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(14)

    def test_overlap_is_rotation_invariant_and_penalizes_rank_loss(self):
        x = self.rng.normal(size=(70, 3))
        rotation = np.linalg.qr(self.rng.normal(size=(3, 3)))[0]
        self.assertAlmostEqual(subspace_metrics(x, x@rotation+7)['overlap'], 1.)
        deficient = np.c_[x[:, 0], x[:, 0], x[:, 0]]
        result = subspace_metrics(deficient, deficient)
        self.assertEqual(result['rank_a'], 1)
        self.assertAlmostEqual(result['overlap'], 1/3)
        zero = subspace_metrics(np.zeros((5, 2)), np.zeros((5, 2)))
        self.assertEqual(zero['overlap'], 0.)

    def test_alignment_recovers_rotation_and_affine_intercept(self):
        train = self.rng.normal(size=(60, 3))
        test = self.rng.normal(size=(40, 3))
        rotation = np.linalg.qr(self.rng.normal(size=(3, 3)))[0]
        fitted = fit_alignment(train, train@rotation, 'orthogonal')
        np.testing.assert_allclose(predict_alignment(fitted, test), test@rotation, atol=1e-12)
        transform = self.rng.normal(size=(3, 3)); offset = np.array([4, -2, 3])
        fitted = fit_alignment(train+5, (train+5)@transform+offset, 'affine', 1e-12)
        np.testing.assert_allclose(predict_alignment(fitted, test), test@transform+offset, atol=1e-8)

    def test_test_values_do_not_change_fitted_alignment(self):
        x = {p: self.rng.normal(size=(40, 2)) for p in ('train', 'tune', 'test_a', 'test_b', 'test')}
        y = {p: a@np.array([[1, 2], [-1, 3]])+.1*self.rng.normal(size=a.shape) for p, a in x.items()}
        _, first = evaluate_alignment({'ieeg': x, 'meg': y}, [1e-4, 1., 100.])
        altered = {p: (a*100 if p.startswith('test') else a) for p, a in y.items()}
        _, second = evaluate_alignment({'ieeg': x, 'meg': altered}, [1e-4, 1., 100.])
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_unavailable_clustering_is_explicit(self):
        parts = {p: self.rng.normal(size=(4, 2)) for p in ('train', 'tune', 'test_a', 'test_b', 'test')}
        rows, artifacts, tuning = evaluate_clusters({'ieeg': parts, 'meg': parts}, [8], 1)
        self.assertEqual(rows[0]['metric'], 'unavailable')
        self.assertEqual(rows[0]['partition'], 'test')
        self.assertEqual(artifacts, {})
        self.assertEqual(tuning, [])

    def test_models_agree_with_dense_reference(self):
        x = self.rng.normal(size=(50, 7))+3
        y = x@self.rng.normal(size=(7, 9))+.2*self.rng.normal(size=(50, 9))-2
        def dataset(a):
            return Dataset('synthetic', [a.T[None]], pd.DataFrame(index=range(a.shape[1])),
                           np.arange(a.shape[1]), 'stack')
        data = {'ieeg': dataset(x), 'meg': dataset(y)}
        fitted = fit_models(data, MODELS, 3, block_scaling='none')
        independent = fitted['separate_pca']
        for m, array in [('ieeg', x), ('meg', y)]:
            ref = PCA(3).fit(array)
            w = independent[m+'_weights']
            np.testing.assert_allclose(w@w.T, ref.components_.T@ref.components_, atol=1e-10)
            np.testing.assert_allclose(_project(data[m], independent, m),
                                       (array-array.mean(0))@w, atol=1e-10)
        joint = fitted['joint_pca']
        w = np.r_[joint['ieeg_weights'], joint['meg_weights']]
        ref = PCA(3).fit(np.c_[x, y])
        np.testing.assert_allclose(w@w.T, ref.components_.T@ref.components_, atol=1e-10)
        ref = PLSSVD(n_components=3, scale=False).fit(x, y)
        for m, refw in [('ieeg', ref.x_weights_), ('meg', ref.y_weights_)]:
            w = fitted['plssvd'][m+'_weights']
            np.testing.assert_allclose(w@w.T, refw@refw.T, atol=1e-10)


class PipelineTests(unittest.TestCase):
    def test_all_meg_setups_have_valid_anatomical_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); make_cache(root)
            trials = load_trial_cache(root); rng = np.random.default_rng(1)
            indices = {m: {s.subject: _split_subject(s, rng, 'trial') for s in getattr(trials, m)}
                       for m in ('ieeg', 'meg')}
            for kind in MEG_KINDS:
                scratch = root/kind; scratch.mkdir()
                fold, _, _, _ = _build_fold(trials, kind, indices, scratch, 42)
                for part, datasets in fold.items():
                    meg = datasets['meg']; ix = meg.electrode_to_feature
                    self.assertEqual(len(ix), datasets['ieeg'].n_features)
                    self.assertTrue(np.all((ix >= 0) & (ix < meg.n_features)))
                    np.testing.assert_array_equal(ix, fold['train']['meg'].electrode_to_feature)

    def test_end_to_end_outputs_and_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cache = root/'cache'; cache.mkdir(); make_cache(cache)
            out = root/'result'
            result = run_comparison(cache, out, dimensions=[1, 2], repeats=2, cluster_counts=[2, 3])
            self.assertEqual(set(result['overlap'].model), set(MODELS))
            self.assertEqual(set(result['overlap'].space), {'temporal_scores', 'spatial_patterns'})
            self.assertTrue(result['overlap'].overlap.between(0, 1+1e-12).all())
            self.assertTrue(np.isfinite(result['alignment'].nrmse).all())
            for _, group in result['splits'].groupby(['repeat', 'modality', 'subject', 'condition']):
                self.assertEqual(len(group), group.trial_index.nunique())
            for model in MODELS:
                saved = np.load(out/f'{model}_000_model.npz')
                self.assertEqual(saved['ieeg_weights'].shape[1], 2)
                self.assertEqual(saved['test_ieeg_scores'].shape, (36, 2))
            figures = plot_results(out, k=2, show=False)
            self.assertEqual(len(figures), 4)
            for path in figures:
                self.assertGreater(Path(path).stat().st_size, 1000)
            config, loaded = load_results(out)
            self.assertEqual(config['repeats'], 2)
            self.assertEqual(len(loaded['alignment']), len(result['alignment']))
            with self.assertRaises(FileExistsError):
                run_comparison(cache, out, dimensions=[1], repeats=1)


if __name__ == '__main__':
    unittest.main()
