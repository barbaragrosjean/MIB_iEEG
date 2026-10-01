import unittest,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from compare_subspace import temporal_representations,subspace_metrics,_representations
from coverage_matching_utils import Dataset
from mni_coordinates import read_mni_coordinates

class TemporalCoordinateTests(unittest.TestCase):
    def test_temporal_ignores_spatial_map(self):
        rng=np.random.default_rng(4)
        scores={'train':{m:rng.normal(size=(20,3)) for m in ('ieeg','meg')}}
        i=Dataset('iEEG',[rng.normal(size=(2,4,20))],pd.DataFrame(index=range(4)),np.arange(4),'average')
        m=Dataset('full_concatenated',[rng.normal(size=(2,11,20))],pd.DataFrame(index=range(11)),np.arange(4),'average')
        fold={'train':{'ieeg':i,'meg':m}}
        a=_representations(fold,scores,3)
        m.electrode_to_feature=np.array([10,8,6,4])
        b=_representations(fold,scores,3)
        for modality in ('ieeg','meg'):
            np.testing.assert_array_equal(a['temporal_scores'][modality]['train'],b['temporal_scores'][modality]['train'])
        self.assertFalse(np.allclose(a['spatial_patterns']['meg']['train'],b['spatial_patterns']['meg']['train']))
        self.assertEqual(subspace_metrics(a['temporal_scores']['ieeg']['train'],a['temporal_scores']['meg']['train']),
                         subspace_metrics(b['temporal_scores']['ieeg']['train'],b['temporal_scores']['meg']['train']))

    def test_coordinate_formats_units_and_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'coords.csv';x=np.array([[1.,2.,3.],[-4.,5.,6.]])
            pd.DataFrame(x,columns=['x','y','z']).to_csv(p,index=False)
            np.testing.assert_array_equal(read_mni_coordinates(p,2),x)
            np.savetxt(p,x/1000,delimiter=',')
            np.testing.assert_allclose(read_mni_coordinates(p,2,'m'),x)
            with self.assertRaisesRegex(ValueError,'expected 3'):read_mni_coordinates(p,3,'m')
