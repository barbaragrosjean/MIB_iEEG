import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from weight_stability import controlled_seeds, roi_statistics, summarize_weights, plot_weight_stability
from spatial_reliability import fit_iterations

class WeightStabilityTests(unittest.TestCase):
    def test_seeds(self):
        for i in range(5):
            t, p = controlled_seeds(4, i, 'both')
            self.assertEqual(controlled_seeds(4, i, 'trials'), (t, controlled_seeds(4, 0, 'both')[1]))
            self.assertEqual(controlled_seeds(4, i, 'pairing'), (controlled_seeds(4, 0, 'both')[0], p))

    def test_roi_sd_is_across_iterations(self):
        meta = pd.DataFrame(dict(subject=['a','a','b'], region=['R']*3))
        values = np.array([[[1.],[3.],[10.]], [[3.],[5.],[14.]]])
        rows, _ = roi_statistics(values, meta)
        self.assertEqual(rows[0]['mean_magnitude'], 7.5)  # ROI trajectory: 6, 9
        self.assertAlmostEqual(rows[0]['sd_across_iterations'], np.std([6.,9.],ddof=1))

    def test_controlled_fits_and_plots(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cache=root/'cache'; cache.mkdir()
            subjects={'ieeg':['i1','i2'], 'meg':['m1','m2','m3']}; records=[]
            xyz=np.column_stack([np.arange(1,7),np.ones(6),np.ones(6)])
            for m in subjects:
                for s in subjects[m]:
                    signal=rng.normal(size=(6,2)) @ rng.normal(size=(2,20))
                    files=[]
                    for c in (1,2):
                        f=f'{s}_{c}.npy'; files.append(f)
                        np.save(cache/f,(signal[None]+rng.normal(size=(8,6,20))*.2).astype('float32'))
                    meta=pd.DataFrame(xyz,columns=['x','y','z']);meta['subject']=s
                    records.append(dict(modality=m,subject=s,files=files,positions=xyz.tolist(),
                        metadata=meta.to_dict('list'),split_groups=[None]*2,permutation_blocks=[['all']*8]*2))
            (cache/'manifest.json').write_text(json.dumps(dict(records=records,times=list(range(20)),
                config=dict(conditions=[1,2],ieeg_subjects=subjects['ieeg'],meg_subjects=subjects['meg']))))
            def labeler(coords,*args):
                return pd.DataFrame({level:['ROI'+str(int(x)//2) for x in coords[:,0]] for level in ('parcel','region','group')})
            runs={}
            for condition in ('trials','pairing','both'):
                args=SimpleNamespace(output_dir=root/condition,cache_dir=cache,weight_only=True,variation=condition,
                    iterations=3,components=2,seed=42,split_unit='trial',block_scaling='equal_variance',
                    pool_hemispheres=False,ieeg_coords_dir=None,meg_coords_dir=None,ieeg_coords_unit='mm',meg_coords_unit='mm',
                    atlas_image=None,atlas_labels=None,scratch_dir=root/'scratch',max_gram_gib=1.)
                with patch('spatial_reliability.annotate_coordinates',side_effect=labeler): fit_iterations(args)
                frames=summarize_weights(args.output_dir);runs[condition]=args.output_dir
                self.assertEqual(len(frames['iteration_correlations']),3*3*2*2)
                if condition=='pairing':
                    subset=frames['feature_stability'].query("model == 'separate_pca' and modality == 'ieeg'")
                    np.testing.assert_allclose(subset.sd_across_iterations,0,atol=1e-14)
                import matplotlib.pyplot as plt
                for f in plot_weight_stability(args.output_dir): f.canvas.draw();plt.close(f)
            for i in range(3):
                folder=f'iteration_{i:03d}'
                read=lambda mode,file: json.loads((runs[mode]/folder/file).read_text())
                self.assertEqual(read('trials','splits.json'),read('both','splits.json'))
                self.assertEqual(read('pairing','pairing.json'),read('both','pairing.json'))
                self.assertEqual(read('trials','pairing.json'),json.loads((runs['trials']/'iteration_000/pairing.json').read_text()))
                self.assertEqual(read('pairing','splits.json'),json.loads((runs['pairing']/'iteration_000/splits.json').read_text()))

if __name__=='__main__':unittest.main()
