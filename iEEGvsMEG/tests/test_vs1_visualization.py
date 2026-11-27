"""Numerical and executable-notebook checks using synthetic saved outputs."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vs1_utils_component_matching import match_components, cumulative_matched_correlation
from vs1_utils_coverage_matching import fit_block_pca
from vs1_run_coverage import load_pca_results


class ArrayDataset:
    name = 'synthetic'
    def __init__(self, values, scale=1.):
        self.values = values
        self.scale = scale
        self.n_observations, self.n_features = values.shape
    def blocks(self):
        yield self.values * self.scale


class NumericalTests(unittest.TestCase):
    def test_full_spectrum_dimensionality_and_unscaled_variance(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(30, 7)) * np.arange(1, 8)
        centered = x - x.mean(0)
        singular = np.linalg.svd(centered, compute_uv=False)
        eigenvalues = singular**2 / (len(x)-1)
        expected = eigenvalues.sum()**2 / np.sum(eigenvalues**2)
        result = fit_block_pca(ArrayDataset(x, scale=.003), n_components=2)
        self.assertAlmostEqual(result.effective_dimensionality, expected)
        self.assertGreater(result.effective_dimensionality, 2.)
        self.assertAlmostEqual(result.pre_block_total_variance, x.var(axis=0, ddof=1).sum())
        self.assertAlmostEqual(result.total_variance, result.pre_block_total_variance * .003**2)

    def test_matching_preserves_weight_rows_and_reports_original_rank(self):
        rng = np.random.default_rng(5)
        reference = np.linalg.qr(rng.normal(size=(40, 3)))[0]
        weights = rng.normal(size=(8, 3))
        permutation, signs = [2, 0, 1], [-1, 1, -1]
        result = match_components(reference, reference[:, permutation]*signs,
                                  weights=weights[:, permutation]*signs)
        np.testing.assert_allclose(result['scores'], reference, atol=1e-12)
        np.testing.assert_allclose(result['weights'], weights, atol=1e-12)
        self.assertEqual(result['audit'].original_rank.tolist(), [2, 3, 1])
        self.assertEqual(result['audit'].sign_correction.tolist(), [1, -1, -1])

    def test_cumulative_metric_is_restricted_to_original_first_k(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(40, 3));x -= x.mean(0)
        reference = np.linalg.qr(x)[0]
        result = cumulative_matched_correlation(reference, reference[:, [2, 0, 1]])
        self.assertAlmostEqual(result.iloc[0].mean_absolute_r, 0., places=12)
        self.assertAlmostEqual(result.iloc[-1].mean_absolute_r, 1., places=12)

    def test_new_export_metadata_does_not_break_existing_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            np.savez(path/'iEEG.npz', scores=np.ones((4, 2)), weights=np.ones((3, 2)),
                     explained_variance=np.ones(2), explained_variance_ratio=np.ones(2)/3,
                     total_variance=3., effective_dimensionality=3., pre_block_total_variance=12.,
                     dataset_scale=.5, n_features=3, n_observations=4,
                     times=np.arange(4), condition_mode='average', pca_schema_version=2)
            results = load_pca_results(path, {'iEEG': object()})
            self.assertEqual(float(results['iEEG'].effective_dimensionality), 3.)


class NotebookTests(unittest.TestCase):
    def test_notebook_executes_all_modes_from_saved_outputs(self):
        notebook = nbformat.read(ROOT/'vs1_coverage_visualization.ipynb', as_version=4)
        nbformat.validate(notebook)
        baseline = ROOT/'OLD/vs1_coverage_matching.ipynb'
        original_hash = hashlib.sha256(baseline.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/'out'
            names = ['iEEG', 'full_concatenated', 'full_average', 'coverage_average', 'paired_coverage', 'random_control']
            for analysis in ['vs1_construct', 'vs1_pca', 'vs1_manipulation', 'vs1_channels']:
                folder = out/analysis;folder.mkdir(parents=True)
                (folder/'config.json').write_text((ROOT/'out'/analysis/'config.json').read_text())
            rng = np.random.default_rng(10)
            base = rng.normal(size=(60, 8));base -= base.mean(0)
            basis = np.linalg.qr(base)[0]
            eigenvalues = np.array([5., 4., 3., 2., 1., .7, .3, .1])
            pairings = ['match', 'order', *[f'random_{i:03d}' for i in range(1, 21)]]
            for pairing in pairings:
                for analysis in ['vs1_pca', 'vs1_channels']:
                    folder=out/analysis/pairing;folder.mkdir()
                    config_hash=hashlib.sha256((out/analysis/'config.json').read_bytes()).hexdigest()
                    (folder/'manifest.json').write_text(json.dumps(dict(status='complete',pairing=pairing,
                        provenance=dict(configs={analysis:config_hash},pairs={}))))
                for j, name in enumerate(names):
                    candidate = basis[:, :5].copy()
                    if pairing != 'match' and name != 'iEEG':
                        candidate = candidate[:, rng.permutation(5)] * rng.choice([-1, 1], 5)
                    scores = candidate * np.sqrt(eigenvalues[:5] * 59)
                    np.savez_compressed(out/'vs1_pca'/pairing/f'{name}.npz', scores=scores,
                        # Corrupting this unused field would be harmless; the reader never asks for weights.
                        weights=np.zeros((8, 5)), explained_variance=eigenvalues[:5],
                        explained_variance_ratio=eigenvalues[:5]/eigenvalues.sum(),
                        total_variance=eigenvalues.sum(), pre_block_total_variance=eigenvalues.sum()*(j+1),
                        effective_dimensionality=eigenvalues.sum()**2/np.sum(eigenvalues**2),
                        n_features=8, n_observations=60, times=np.arange(60)/250,
                        condition_mode='average', conditions=[1,2])
                pd.DataFrame([dict(repeat=r,n_channels=n,variance_fraction_retained=v,
                    subspace_overlap=v-.01) for r in range(3) for n,v in [(2,.5),(4,.8),(8,1.)]]).to_csv(
                        out/'vs1_channels'/pairing/'metrics.csv',index=False)
            preview=Path('/tmp/mib_vs1_visual_preview');preview.mkdir(exist_ok=True)
            for mode in ('match','order','random'):
                figures=[]
                def show():
                    figure=plt.gcf();figures.append(figure)
                    if mode=='random' and len(figures) in (1,2,3,4,5,10,16):
                        figure.savefig(preview/f'{len(figures):02d}.png',dpi=90,bbox_inches='tight')
                    plt.close(figure)
                namespace={'__name__':'__main__'}
                with patch.dict(os.environ,{'MIB_PROJECT_ROOT':str(ROOT),'MIB_OUTPUT_ROOT':str(out)}),patch.object(plt,'show',show):
                    for i,cell in enumerate(notebook.cells):
                        if cell.cell_type!='code':continue
                        source=cell.source.replace("MODE = 'random'",f"MODE = '{mode}'",1)
                        exec(compile(source,f'notebook cell {i}','exec'),namespace)
                self.assertEqual(len(figures),16)
                self.assertEqual(set(namespace['HEATMAPS']),set(names)-{'iEEG'})
                if mode=='random':
                    self.assertEqual(int(namespace['VARIANCE_SUMMARY'].n.min()),20)
                    self.assertTrue(np.all(namespace['HEATMAPS']['paired_coverage']['n']==20))
                    np.testing.assert_allclose(namespace['TIMECOURSE_SUMMARIES']['paired_coverage']['aligned_sd'],0.,atol=1e-12)
            self.assertEqual(original_hash,hashlib.sha256(baseline.read_bytes()).hexdigest())


if __name__=='__main__':unittest.main()
