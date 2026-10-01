import unittest
import numpy as np
from spatial_clouds import distances,evaluate_clouds,transport

class CloudTests(unittest.TestCase):
    def test_identical_and_unequal_clouds(self):
        x=np.array([[0.,0.],[1.,0.],[0.,2.]])
        a=np.ones(3)/3
        r=distances(x,x,a,a,1.)
        for key in ('wasserstein2','gw_squared_loss','mmd2'):self.assertAlmostEqual(r[key],0,places=8)
        y=np.repeat(x,2,axis=0);b=np.ones(6)/6
        r=distances(x,y,a,b,1.)
        self.assertAlmostEqual(r['wasserstein2'],0);self.assertAlmostEqual(r['mmd2'],0)
        rot=x@np.array([[0.,1.],[-1.,0.]])
        r=distances(x,rot,a,a,1.)
        self.assertAlmostEqual(r['gw_squared_loss'],0,places=7)

    def test_train_only_preparation(self):
        rng=np.random.default_rng(4)
        patterns={m:{'train':rng.normal(size=(n,3)),'test':rng.normal(size=(n,3))} for m,n in [('ieeg',7),('meg',11)]}
        scores={p:{m:rng.normal(size=(20,3)) for m in ('ieeg','meg')} for p in ('train','test')}
        rows,art=evaluate_clouds(patterns,scores,prototypes=3)
        changed={m:{p:x.copy() for p,x in v.items()} for m,v in patterns.items()}
        changed['meg']['test']*=3
        other,new=evaluate_clouds(changed,scores,prototypes=3)
        for key in ('temporal_rotation','ieeg_membership','meg_membership','mmd_bandwidth','meg_center','meg_scale'):
            np.testing.assert_array_equal(art[key],new[key])
        self.assertNotEqual(rows[1]['wasserstein2'],other[1]['wasserstein2'])
        self.assertAlmostEqual(art['meg_mass'].sum(),1.)
