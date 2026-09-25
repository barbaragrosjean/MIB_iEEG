import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from compare_subspace import plot_full_data_clusters


class ClusterVisualizationTests(unittest.TestCase):
    def test_selections_and_label_permutation(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(12, 3))
        rows = []
        for count, ari in [(2, 1.), (3, .1)]:
            rows.append(dict(model='separate_pca', partition='in_sample', k=3,
                             metric='cross_modal_ari', modality='both', n_clusters=count, value=ari))
            for m in ('ieeg','meg'):
                score = (.9 if count == 3 else .2) if m == 'ieeg' else (.8 if count == 2 else .3)
                rows.append(dict(model='separate_pca',partition='in_sample',k=3,
                                 metric='silhouette',modality=m,n_clusters=count,value=score))
        config = dict(schema_version=2,dimensions=[3],models=['separate_pca'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = {}
            for m in ('ieeg','meg'):
                artifacts[m+'_normalization_center'] = np.zeros(3)
                artifacts[m+'_normalization_rms'] = 1.
                for count in (2,3):
                    labels = np.arange(12) % count
                    if m == 'meg': labels = count-1-labels
                    artifacts[f'c{count}_{m}_train'] = labels
                    artifacts[f'c{count}_{m}_centers'] = np.array([x[labels==c].mean(0) for c in range(count)])
            np.savez(root/'separate_pca_-01_k3_clusters.npz',**artifacts)
            np.savez(root/'separate_pca_-01_k3_spatial_patterns.npz',ieeg_train=x,meg_train=x)
            with patch('compare_subspace.load_results',return_value=(config,{'clusters':pd.DataFrame(rows)})):
                paths, selected = plot_full_data_clusters(root,show=False)
            self.assertEqual(len(paths),2)
            self.assertTrue(all(Path(p).exists() and Path(p).with_suffix('.pdf').exists() for p in paths))
            self.assertEqual(selected.query("criterion == 'silhouette'").n_clusters.tolist(),[3,2])
            self.assertEqual(selected.query("criterion == 'ari'").n_clusters.tolist(),[2,2])
            self.assertTrue((selected.query("criterion == 'ari'").cross_modal_ari==1).all())

    def test_requires_actual_three_component_fit(self):
        with tempfile.TemporaryDirectory() as tmp, patch('compare_subspace.load_results',return_value=({'schema_version':2,'dimensions':[5]},{})):
            with self.assertRaisesRegex(ValueError,'k=3'): plot_full_data_clusters(tmp,show=False)
