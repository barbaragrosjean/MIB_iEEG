import tempfile
import unittest
import numpy as np
import pandas as pd
from vs3_1_weight_clustering import cluster_weights, export_inputs, run_clustering, load_results, METHODS

class WeightClusteringTests(unittest.TestCase):
    def test_methods_and_saved_results(self):
        rng = np.random.default_rng(4)
        w = np.r_[rng.normal([1,.1,.1], .03, (20,3)), rng.normal([.1,1,.1], .03, (20,3)), np.zeros((1,3))]
        meta = pd.DataFrame(dict(x=np.tile([-20,20],21)[:41], y=0., z=10., subject='s', region_shared=np.tile(['A','B'],21)[:41]))
        for method in METHODS:
            a = cluster_weights(w, meta, method=method, n_clusters=2)
            b = cluster_weights(-w, meta, method=method, n_clusters=2)
            np.testing.assert_array_equal(a['memberships'].cluster_id, b['memberships'].cluster_id)
            self.assertEqual((a['memberships'].cluster_id < 0).sum(), 1)
            np.testing.assert_allclose(a['region_contributions'].groupby('cluster_id').percent.sum(),100)
            for row in a['hemisphere_centroids'].itertuples():
                members = a['memberships']
                group = members[(members.cluster_id == row.cluster_id) & (members.hemisphere == row.hemisphere)]
                np.testing.assert_allclose([row.centroid_x,row.centroid_y,row.centroid_z],group[['x','y','z']].mean())
        a = cluster_weights(w, meta, n_clusters=2, use_thresholded=True)
        x = a['memberships'][['input_weight_1','input_weight_2','input_weight_3']].to_numpy()
        self.assertTrue(np.all((x == 0) | (x >= a['thresholds'].threshold.to_numpy())))
        with tempfile.TemporaryDirectory() as tmp:
            export_inputs({'MEG':w}, {'MEG':meta},tmp)
            run_clustering(tmp, ['kmeans'], n_clusters=2)
            load_results(tmp,'kmeans',2)
            export_inputs({'MEG':w * 2}, {'MEG':meta},tmp)
            with self.assertRaisesRegex(ValueError,'inputs changed'):
                load_results(tmp,'kmeans',2)

if __name__ == '__main__':
    unittest.main()
