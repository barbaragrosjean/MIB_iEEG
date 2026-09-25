import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from test_plssvd_fixed import trials
from compare_subspace import run_comparison, load_results, plot_results


class SubspaceKfoldTests(unittest.TestCase):
    def test_full_data_then_disjoint_folds_and_exports(self):
        data=trials()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache';cache.mkdir()
            (cache/'manifest.json').write_text(json.dumps({'config':{m+'_subjects':[s.subject for s in getattr(data,m)] for m in ('ieeg','meg')}}))
            with patch('plssvd_eval_utils.load_trial_cache',return_value=data):
                result=run_comparison(cache,root/'out',dimensions=(1,3),repeats=5,cluster_counts=(2,),ridge_grid=(.01,))
            self.assertEqual(set(result['overlap'].partition),{'in_sample','train','test'})
            self.assertEqual(set(result['overlap'].query("analysis == 'in_sample'").repeat),{-1})
            audit=result['splits'].query("partition == 'test'")
            counts=audit.groupby(['modality','subject','condition','trial_index']).size()
            self.assertTrue((counts==1).all());self.assertEqual(len(counts),5*2*24)
            matching=pd.read_csv(root/'out'/'matching_-01.csv')
            for i in range(5):pd.testing.assert_frame_equal(matching,pd.read_csv(root/'out'/f'matching_{i:03d}.csv'))
            for name in ['separate_pca','plssvd','joint_pca']:
                with np.load(root/'out'/f'{name}_000_model.npz') as model:
                    self.assertEqual(model['train_ieeg_scores'].shape,(12,3))
                    self.assertNotIn('test_a_ieeg_scores',model.files)
            self.assertNotIn('reliability',result)
            self.assertEqual(set(result['clusters'].n_clusters),{2})
            config,loaded=load_results(root/'out');self.assertEqual(config['condition_mode'],'average')
            self.assertEqual(len(plot_results(root/'out',k=3,show=False)),3)
            self.assertEqual(set(loaded['alignment'].query("complexity == 'affine'").alpha),{.01})
