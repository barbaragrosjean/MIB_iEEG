import unittest
import numpy as np
import pandas as pd
from vs3_1_spatial_profiles import analyze_spatial_profiles, _patch_labels

class SpatialProfilesTests(unittest.TestCase):
    def test_matching_null_and_patches(self):
        rng=np.random.default_rng(5)
        xyz=np.array([[x,y,0.] for x in [-20.,20.] for y in np.arange(0,60,5)])
        im=pd.DataFrame(np.tile(xyz,(3,1)),columns=['x','y','z'])
        im['subject']=np.repeat(['a','b','c'],len(xyz))
        mm=im.copy();mm['ieeg_subject']=im.subject;mm['subject']='meg'
        w=rng.uniform(.01,.05,(len(im),3));w[:,0]+=(im.y<30)*2;w[:,1]+=(im.y>=30)*2
        r=analyze_spatial_profiles({'iEEG':w,'MEG':w},im,mm,n_permutations=19,n_bootstrap=20,patch_resamples=3)
        self.assertTrue(len(r['patches'])>0)
        self.assertTrue(all(pd.api.types.is_numeric_dtype(r['curves'][c]) for c in r['curves'] if c!='modality'))
        self.assertTrue((r['patches'].coherence>=.9).all())
        self.assertTrue(np.nanmax(np.abs(r['difference'].meg_minus_ieeg))<.25)
        self.assertTrue((r['audit'].n_matched_features==len(xyz)).all())
        for row in r['patches'].itertuples():
            group=r['memberships'].query('modality == @row.modality and patch_id == @row.patch_id')
            self.assertTrue(np.any(np.all(group[['x','y','z']].to_numpy()==[row.medoid_x,row.medoid_y,row.medoid_z],axis=1)))
        # Identical profile fields have exactly zero excess, including sparse empty bins.
        same=np.ones_like(w)
        r=analyze_spatial_profiles({'iEEG':same,'MEG':same},im,mm,n_permutations=3,n_bootstrap=3,patch_resamples=0)
        np.testing.assert_allclose(r['curves'].excess_similarity.dropna(),0,atol=1e-12)
        self.assertTrue(r['curves'].magnitude_semivariance.isna().all())


    def test_no_chaining(self):
        theta=np.array([0,.35,.7]);unit=np.c_[np.cos(theta),np.sin(theta)]
        labels,_,_= _patch_labels(np.c_[np.arange(3),np.zeros((3,2))],unit,1.1,.9)
        self.assertGreater(len(np.unique(labels)),1)

if __name__=='__main__':unittest.main()
