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


    def test_pooled_scope_pairs_and_uncertainty(self):
        # Two nearby left-hemisphere pairs belonging to different participants;
        # right-hemisphere points must never form pairs with the left cloud.
        im=pd.DataFrame(dict(subject=['a','a','b','b','a','b'],
            x=[-20.,-20.,-20.,-20.,20.,20.], y=[0.,2.,4.,6.,0.,3.], z=np.zeros(6)))
        mm=im.copy();mm['ieeg_subject']=im.subject
        w=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.],[1.,1.],[1.,1.]])
        options=dict(bin_edges=[0,10,100],n_permutations=3,n_bootstrap=4,patch_resamples=0)
        within=analyze_spatial_profiles({'iEEG':w,'MEG':w},im,mm,**options)
        pooled=analyze_spatial_profiles({'iEEG':w,'MEG':w},im,mm,spatial_scope='pooled',**options)
        self.assertEqual(within['audit'].n_matched_pairs.sum(),2)
        self.assertEqual(pooled['audit'].n_matched_pairs.sum(),7)  # C(4,2) + C(2,2)
        first=pooled['curves'].query("modality == 'iEEG'").iloc[0]
        self.assertAlmostEqual(first.similarity,3/7)
        self.assertEqual(first.n_subjects,2)
        self.assertEqual(first.n_analysis_units,1)
        self.assertTrue(pooled['curves'][['ci_low','ci_high']].isna().all().all())
        self.assertTrue(pooled['difference'][['ci_low','ci_high']].isna().all().all())
        self.assertEqual(pooled['audit'].iloc[0].n_original_features,6)
        displaced=mm.copy();displaced.loc[5,'y']=12.
        matched=analyze_spatial_profiles({'iEEG':w,'MEG':w},im,displaced,spatial_scope='pooled',**options)
        self.assertEqual(matched['audit'].n_matched_pairs.sum(),6)
        self.assertAlmostEqual(matched['curves'].iloc[0].similarity,2/6)
        explicit=analyze_spatial_profiles({'iEEG':w,'MEG':w},im,mm,spatial_scope='within_participant',**options)
        for key in within:
            pd.testing.assert_frame_equal(within[key],explicit[key])

    def test_pooled_patches_preserve_participant_ownership(self):
        # Identical coordinates in different people remain distinct measurements.
        im=pd.DataFrame(dict(subject=['a','a','b','b'],x=[-20.]*4,
                             y=[0.,2.,0.,4.],z=[0.]*4))
        mm=im.copy();mm['ieeg_subject']=im.subject
        result=analyze_spatial_profiles({'iEEG':np.ones((4,2)),'MEG':np.ones((4,2))},
            im,mm,spatial_scope='pooled',n_permutations=3,n_bootstrap=3,patch_resamples=0)
        self.assertEqual(result['audit'].n_matched_features.iloc[0],4)
        self.assertEqual(result['audit'].n_matched_pairs.iloc[0],5)  # exclude co-located pair
        self.assertEqual(set(result['memberships'].participant),{'a','b'})
        self.assertTrue(result['patches'].subject.eq('pooled').all())
        self.assertTrue(result['patches'].n_sources.eq(4).all())
        with self.assertRaisesRegex(ValueError,'spatial_scope'):
            analyze_spatial_profiles({},im,mm,spatial_scope='invalid')

    def test_hemisphere_folding_coordinates_pairs_and_scopes(self):
        im=pd.DataFrame(dict(subject=['a','a','b','b'],x=[-20.,20.,-20.,20.],
                             y=[0.,2.,4.,6.],z=[1.]*4))
        mm=im.copy();mm['ieeg_subject']=im.subject
        before_im=im.copy(deep=True);before_mm=mm.copy(deep=True)
        weights={'iEEG':np.ones((4,2)), 'MEG':np.ones((4,2))}
        options=dict(bin_edges=[0,10,100],n_permutations=3,n_bootstrap=3,patch_resamples=0)
        native=analyze_spatial_profiles(weights,im,mm,**options)
        explicit=analyze_spatial_profiles(weights,im,mm,fold_hemispheres=False,**options)
        for key in native:
            pd.testing.assert_frame_equal(native[key],explicit[key])
        self.assertEqual(native['audit'].n_matched_pairs.sum(),0)
        within=analyze_spatial_profiles(weights,im,mm,fold_hemispheres=True,**options)
        self.assertEqual(within['audit'].n_matched_pairs.sum(),2)
        results=[]
        for target,sign in [('left',-1),('right',1)]:
            r=analyze_spatial_profiles(weights,im,mm,spatial_scope='pooled',
                fold_hemispheres=True,target_hemisphere=target,**options)
            results.append(r)
            self.assertEqual(r['audit'].n_matched_pairs.sum(),6)
            self.assertTrue(r['patches'].hemisphere.eq(target).all())
            self.assertTrue(r['patches'].n_sources.eq(4).all())
            members=r['memberships']
            np.testing.assert_allclose(members.x,sign*np.abs(members.original_x))
            np.testing.assert_allclose(members.y,members.original_y)
            np.testing.assert_allclose(members.z,members.original_z)
            self.assertEqual(set(members.original_hemisphere),{'left','right'})
            np.testing.assert_allclose(r['patches'].medoid_x,sign*20.)
        pd.testing.assert_frame_equal(results[0]['curves'],results[1]['curves'])
        pd.testing.assert_frame_equal(im,before_im)
        pd.testing.assert_frame_equal(mm,before_mm)
        # Reflection-created co-locations must not be mistaken for duplicate sources.
        im.loc[1,'y']=0.;mm.loc[1,'y']=0.
        r=analyze_spatial_profiles(weights,im,mm,spatial_scope='pooled',fold_hemispheres=True,**options)
        self.assertEqual(r['audit'].n_matched_features.sum(),4)
        self.assertEqual(r['audit'].n_matched_pairs.sum(),5)
        with self.assertRaisesRegex(ValueError,'target_hemisphere'):
            analyze_spatial_profiles(weights,im,mm,target_hemisphere='invalid',**options)

    def test_no_chaining(self):
        theta=np.array([0,.35,.7]);unit=np.c_[np.cos(theta),np.sin(theta)]
        labels,_,_= _patch_labels(np.c_[np.arange(3),np.zeros((3,2))],unit,1.1,.9)
        self.assertGreater(len(np.unique(labels)),1)

    def test_magnitude_patches_compare_norms_not_component_proportions(self):
        im=pd.DataFrame(dict(subject=['a']*6,x=[-20.]*6,y=np.arange(6)*2.,z=[0.]*6))
        mm=im.copy();mm['ieeg_subject']=im.subject
        w=np.array([[1.,0.],[0.,-1.],[1.,0.],[10.,0.],[0.,10.],[10.,0.]])
        result=analyze_spatial_profiles({'iEEG':w,'MEG':w},im,mm,
            patch_metric='magnitude_semivariance',patch_max_semivariance=.1,
            n_permutations=3,n_bootstrap=3,patch_resamples=3)
        self.assertEqual(len(result['patches']),4)
        self.assertTrue(result['patches'].n_sources.eq(3).all())
        np.testing.assert_allclose(result['patches'].within_magnitude_semivariance,0)
        self.assertTrue((result['patches'].magnitude_contrast>0).all())
        self.assertTrue((result['patches'].coherence<.9).all())
        saved=result['weights'].query("modality == 'iEEG'").sort_values('feature_row')
        np.testing.assert_allclose(saved[['weight_1','weight_2']],w)
        for row in result['patches'].itertuples():
            members=result['memberships'].query('modality == @row.modality and patch_id == @row.patch_id')
            self.assertEqual(len(np.unique(np.linalg.norm(w[members.feature_row],axis=1))),1)

    def test_magnitude_no_chaining_and_zero_variance(self):
        xyz=np.c_[np.arange(3),np.zeros((3,2))]
        unit=np.tile([1.,0.],(3,1))
        labels,_,_=_patch_labels(xyz,unit,1.1,.9,metric='magnitude_semivariance',
            magnitude=np.array([1.,2.,3.]),magnitude_variance=1.,max_semivariance=.6)
        self.assertEqual(len(np.unique(labels)),2)  # Adjacent pairs pass; endpoints fail.
        labels,_,_=_patch_labels(xyz,unit,1.1,.9,metric='magnitude_semivariance',
            magnitude=np.ones(3),magnitude_variance=0.,max_semivariance=.6)
        self.assertEqual(len(np.unique(labels)),3)

if __name__=='__main__':unittest.main()
