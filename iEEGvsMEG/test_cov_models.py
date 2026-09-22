import unittest
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSSVD
from cov_models_utils import fit_cov_models, evaluate_cov_models, compare_cov_models

class Data:
    def __init__(self,x,name,split=False):
        self.name=name
        self.arrays=[x[:,:x.shape[1]//2].T[None],x[:,x.shape[1]//2:].T[None]] if split else [x.T[None]]
        self.condition_mode='average'
        self.source_data={'times':np.arange(len(x))/250,'load_config':{'conditions':[1]}}
        self.n_observations,self.n_features=x.shape
    def blocks(self):
        for a in self.arrays:yield a[0].T

class Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(40)
        z=rng.normal(size=(50,5))
        self.x=z@rng.normal(size=(5,9))+rng.normal(size=(50,9))*.2+4
        self.y=z@rng.normal(size=(5,13))+rng.normal(size=(50,13))*.2-3
        self.a,self.b=Data(self.x,'iEEG'),Data(self.y,'MEG',split=True)
        self.models=fit_cov_models(self.a,self.b,n_components=5)
    def test_separate_pca(self):
        m=self.models['separate_pca']
        for data,scores,w in [(self.x,m.ieeg_scores,m.ieeg_weights),(self.y,m.meg_scores,m.meg_weights)]:
            ref=PCA(5,svd_solver='full').fit(data)
            np.testing.assert_allclose(scores@w.T,ref.transform(data)@ref.components_,atol=1e-10)
    def test_joint_pca_split_scores_and_reconstruction(self):
        m=self.models['joint_pca'];x=self.x-self.x.mean(0);y=self.y-self.y.mean(0)
        ref=PCA(5,svd_solver='full').fit(np.c_[x,y])
        rec=m.joint_scores@np.r_[m.ieeg_weights,m.meg_weights].T
        np.testing.assert_allclose(rec,ref.transform(np.c_[x,y])@ref.components_,atol=1e-10)
        np.testing.assert_allclose(m.ieeg_scores,x@m.ieeg_weights,atol=1e-10)
        np.testing.assert_allclose(m.meg_scores,y@m.meg_weights,atol=1e-10)
        np.testing.assert_allclose(m.ieeg_scores+m.meg_scores,m.joint_scores,atol=1e-10)
    def test_plssvd_matches_sklearn_and_is_not_truncated_pca(self):
        m=self.models['plssvd'];ref=PLSSVD(n_components=5,scale=False).fit(self.x,self.y)
        np.testing.assert_allclose(np.abs(m.ieeg_weights.T@ref.x_weights_),np.eye(5),atol=1e-9)
        np.testing.assert_allclose(np.abs(m.meg_weights.T@ref.y_weights_),np.eye(5),atol=1e-9)
        np.testing.assert_allclose(m.ieeg_scores.T@m.meg_scores/49,np.diag(m.singular_values),atol=1e-10)
    def test_explained_variance_is_reconstruction_r2(self):
        table=evaluate_cov_models(self.models,plot=False)
        for row in table.itertuples():
            m=self.models[row.model];k=row.k
            for data,w,t,value in [(self.x,m.ieeg_weights,m.ieeg_scores,row.ieeg_variance_explained),
                                   (self.y,m.meg_weights,m.meg_scores,row.meg_variance_explained)]:
                data=data-data.mean(0)
                scores=m.joint_scores if m.joint_scores is not None else t
                rec=scores[:,:k]@w[:,:k].T
                self.assertAlmostEqual(value,1-np.sum((data-rec)**2)/np.sum(data**2),places=10)
    def test_shared_covariance_fraction_and_pls_optimality(self):
        table=evaluate_cov_models(self.models,plot=False)
        c=(self.x-self.x.mean(0)).T@(self.y-self.y.mean(0))/49
        for row in table.itertuples():
            m=self.models[row.model];k=row.k
            qx=np.linalg.svd(m.ieeg_weights[:,:k],full_matrices=False)[0]
            qy=np.linalg.svd(m.meg_weights[:,:k],full_matrices=False)[0]
            direct=np.sum((qx.T@c@qy)**2)/np.sum(c*c)
            self.assertAlmostEqual(row.shared_crosscov_fraction,direct,places=10)
        pivot=table.pivot(index='k',columns='model',values='shared_crosscov_fraction')
        self.assertTrue((pivot.plssvd+1e-10>=pivot.separate_pca).all())
        self.assertTrue((pivot.plssvd+1e-10>=pivot.joint_pca).all())
    def test_balanced_block_scaling(self):
        models=fit_cov_models(self.a,self.b,5,block_scaling='equal_variance')
        table=evaluate_cov_models(models,plot=False)
        np.testing.assert_allclose(table.ieeg_total_variance,1)
        np.testing.assert_allclose(table.meg_total_variance,1)
        m=models['plssvd'];sx,sy=m.scales
        ref=PLSSVD(n_components=5,scale=False).fit(self.x*sx,self.y*sy)
        np.testing.assert_allclose(np.abs(m.ieeg_weights.T@ref.x_weights_),np.eye(5),atol=1e-9)
    def test_comparison_uses_fixed_reference_and_modality_scores(self):
        table,pairs,anchor=compare_cov_models(self.models,n_components=(1,3,5),plot=False)
        self.assertEqual(len(table),9)
        self.assertEqual(len(pairs),27)
        self.assertTrue(np.isfinite(table.select_dtypes('number')).all().all())
        row=table.query("model=='joint_pca' and k==1").iloc[0]
        m=self.models['joint_pca']
        expected=abs(np.corrcoef(m.ieeg_scores[:,0],m.meg_scores[:,0])[0,1])
        self.assertAlmostEqual(row.native_matched_abs_r,expected)
    def test_invalid_time_rank_and_budget(self):
        with self.assertRaises(MemoryError):fit_cov_models(self.a,self.b,5,max_gram_gib=0)
        with self.assertRaises(ValueError):fit_cov_models(self.a,self.b,20)
        self.b.source_data['times']=self.b.source_data['times']+.1
        with self.assertRaises(ValueError):fit_cov_models(self.a,self.b,5)

if __name__=='__main__':unittest.main()
