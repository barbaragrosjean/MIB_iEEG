import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from coverage_matching_utils import Dataset,fit_block_pca
from elect_number_effect import run_elect_number_effect,plot_elect_number_effect,_SelectedChannels,_channel_counts

class ChannelCountTests(unittest.TestCase):
    def test_counts_variance_matching_and_reproducibility(self):
        rng=np.random.default_rng(8)
        arrays=[rng.normal(size=(2,n,20)) for n in (100,105)]
        meg=Dataset('full_concatenated',arrays,pd.DataFrame({'source':range(205)}),np.arange(8),'average')
        ieeg=Dataset('iEEG',[rng.normal(size=(2,8,20))],pd.DataFrame(index=range(8)),np.arange(8),'average')
        with tempfile.TemporaryDirectory() as tmp:
            result=run_elect_number_effect(meg,ieeg,repeats=2,output_dir=tmp)
            self.assertEqual(result['config']['channel_counts'],[100,200])
            self.assertEqual(len(result['metrics']),4)
            self.assertEqual(len(result['component_pairs']),20)
            order=np.load(Path(tmp)/'channel_order_000.npy')
            selected=_SelectedChannels(meg,order[:100])
            dense=np.concatenate(list(meg.blocks()),axis=1)[:,np.sort(order[:100])]
            np.testing.assert_allclose(np.concatenate(list(selected.blocks()),axis=1),dense)
            dense-=dense.mean(0)
            singular=np.linalg.svd(dense,compute_uv=False)
            row=result['metrics'].query('repeat == 0 and n_channels == 100').iloc[0]
            self.assertAlmostEqual(row.variance_fraction_first5,np.sum(singular[:5]**2)/np.sum(singular**2))
            fig=plot_elect_number_effect(result);fig.savefig('/tmp/channel_effect_preview.png')
            self.assertTrue((Path(tmp)/'COMPLETE.json').exists())
            with self.assertRaises(FileExistsError):run_elect_number_effect(meg,ieeg,output_dir=tmp)

    def test_cap_and_complete_steps(self):
        self.assertEqual(_channel_counts(35000,100),list(range(100,20001,100)))
        self.assertEqual(_channel_counts(35000,300)[-1],19800)
        self.assertEqual(_channel_counts(205,100),[100,200])
