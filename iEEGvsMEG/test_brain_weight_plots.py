import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coverage_matching_utils import plot_voxel_weights, plot_glasser_weights, plot_pca_weights


class BrainPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_voxels_mean_counts_and_boundary_points(self):
        xyz=np.array([[-1,0,0],[0,0,0],[19,0,0],[20,0,0],[40,0,0]],float)
        values=np.array([2,-2,4,6,8],float)
        with patch('coverage_matching_utils.plotting.plot_markers') as plot:
            _, table=plot_voxel_weights(values,xyz,voxel_size=20,show=False)
        np.testing.assert_array_equal(table['count'],[1,2,1,1])
        np.testing.assert_allclose(table.weight,[2,1,6,8])
        np.testing.assert_allclose(table.node_size,[7,12,7,7])
        np.testing.assert_allclose(table.x,[-1,9.5,20,40])
        np.testing.assert_allclose(plot.call_args.kwargs['node_size'],table.node_size)
        with patch('coverage_matching_utils.plotting.plot_markers'):
            _, absolute=plot_voxel_weights(values,xyz,absolute=True,show=False)
        self.assertEqual(absolute.weight.iloc[1],3)
        np.testing.assert_array_equal(values,[2,-2,4,6,8])

    def test_surface_interpolation_and_shared_colour_range(self):
        template=nib.Nifti1Image(np.zeros((11,11,11)),np.eye(4))
        meshes=SimpleNamespace(**{f'{kind}_{hemi}':f'{kind}_{hemi}'
                                  for kind in ('pial','inflated','sulc') for hemi in ('left','right')})
        with patch('nilearn.surface.vol_to_surf',return_value=np.array([-1.,0.,2.])), \
             patch('coverage_matching_utils.plotting.plot_surf_stat_map') as plot:
            _, data=plot_glasser_weights([-2,4],[[4,5,5],[6,5,5]],sigma=1,
                                         template=template,meshes=meshes,show=False)
        self.assertAlmostEqual(data['image'].get_fdata()[5,5,5],1.)
        self.assertEqual(data['support_mask'].get_fdata()[0,0,0],0)
        self.assertEqual(plot.call_count,2)
        for call in plot.call_args_list:
            self.assertEqual(call.kwargs['vmin'],-4)
            self.assertEqual(call.kwargs['vmax'],4)

    def test_dispatch_preserves_raw_counts_and_weights(self):
        weights=np.array([[1.,-1.],[3.,-3.],[2.,2.]])
        result=SimpleNamespace(weights=weights.copy(),dataset=SimpleNamespace(name='iEEG',
                    metadata=pd.DataFrame([[0,0,0],[0,0,0],[20,0,0]],columns=['x','y','z'])))
        with patch('coverage_matching_utils.plotting.plot_markers') as plot:
            fig=plot_pca_weights(result,plot_type='voxel',show=False)
        self.assertEqual(plot.call_count,2)
        np.testing.assert_array_equal(plot.call_args_list[0].kwargs['node_size'],[12,7])
        np.testing.assert_array_equal(result.weights,weights)
        self.assertIsInstance(fig,plt.Figure)
        with patch('coverage_matching_utils.plotting.plot_markers') as plot:
            plot_pca_weights(result,show=False)
        self.assertEqual(plot.call_count,2)
        with patch('coverage_matching_utils.plot_glasser_weights') as plot:
            plot_pca_weights(result,plot_type='glasser',sigma=5,show=False)
        self.assertEqual(plot.call_count,2)
        self.assertEqual(plot.call_args.kwargs['sigma'],5)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            plot_voxel_weights([1],[[0,0,0]],voxel_size=0,show=False)
        with self.assertRaises(ValueError):
            plot_voxel_weights([np.nan],[[0,0,0]],show=False)
        with self.assertRaises(ValueError):
            plot_voxel_weights([1,2],[[0,0,0]],show=False)


if __name__=='__main__':
    unittest.main()
