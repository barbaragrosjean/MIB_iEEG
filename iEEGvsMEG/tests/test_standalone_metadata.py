"""Metadata preparation and legacy result helpers must work without a src package."""
import ast
from contextlib import contextmanager
import importlib.abc
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import vs0_load_and_config as shared


@contextmanager
def forbid_legacy_imports():
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname=='src' or fullname.startswith('src.'):
                raise AssertionError(f'Unexpected legacy dependency: {fullname}')
            return None
    finder=Block();sys.meta_path.insert(0,finder)
    try:yield
    finally:sys.meta_path.remove(finder)


class StandaloneMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.data=self.root/'ieeg';self.project=self.root/'metadata'
        self.data.mkdir();(self.project/'misc'/'electrodes').mkdir(parents=True)
        self.coords=self.data/'I1_coords.csv'
        pd.DataFrame(dict(channels=['B1_B2','A1_A2'],x=[10.,20.],y=[0.,0.],z=[0.,0.])).to_csv(self.coords,index=False)
        self.labels=self.project/'misc'/'electrodes'/'I1_electrodes.csv'
        pd.DataFrame(dict(electrode=['A1','B1'],label=['HPC','IFS'])).to_csv(self.labels,index=False)

    def test_getinfo_keeps_row_order_regions_and_raw_coordinates(self):
        with forbid_legacy_imports():
            coord,areas,electrodes,subjects,regions=shared.GetInfo(['I1'],project_path=self.project,data_path=self.data)
        np.testing.assert_array_equal(coord,[[10,0,0],[20,0,0]])
        np.testing.assert_array_equal(areas,[['IFS','IFS'],['HPC','HPC']])
        self.assertEqual(electrodes,['B1_B2','A1_A2'])
        self.assertEqual(subjects,['I1','I1'])
        self.assertEqual(regions,['DLPFC','HPC'])
        self.assertEqual(shared.FindRegion('unknown'),'N')

    def test_default_project_setting_does_not_depend_on_working_directory(self):
        previous=Path.cwd();other=self.root/'other';other.mkdir()
        try:
            os.chdir(other)
            with patch.object(shared,'PROJECT_PATH',str(self.project)):
                self.assertEqual(shared.GetInfo(['I1'],data_path=self.data)[-1],['DLPFC','HPC'])
        finally:os.chdir(previous)
        self.assertTrue(Path(shared.PROJECT_PATH).is_absolute())

    def test_unnamed_channel_column_and_save_interface(self):
        frame=pd.read_csv(self.coords).rename(columns={'channels':'Unnamed: 0'});frame.to_csv(self.coords,index=False)
        self.assertIsNone(shared.GetInfo(['I1'],project_path=self.project,data_path=self.data,save=True))
        saved=pd.read_csv(self.data/'grp_info.csv')
        self.assertEqual(saved.elect.tolist(),['B1_B2','A1_A2'])
        self.assertEqual(saved.region.tolist(),['DLPFC','HPC'])

    def test_missing_or_duplicate_electrode_labels_have_explicit_errors(self):
        pd.DataFrame(dict(electrode=['A1'],label=['HPC'])).to_csv(self.labels,index=False)
        with self.assertRaisesRegex(ValueError,'missing electrode labels'):
            shared.GetInfo(['I1'],project_path=self.project,data_path=self.data)
        pd.DataFrame(dict(electrode=['A1','A1','B1'],label=['HPC','HPC','IFS'])).to_csv(self.labels,index=False)
        with self.assertRaisesRegex(ValueError,'duplicate electrode'):
            shared.GetInfo(['I1'],project_path=self.project,data_path=self.data)

    def make_project(self):
        behavior=self.root/'behavior';meg=self.root/'meg';behavior.mkdir();meg.mkdir()
        rng=np.random.default_rng(7)
        for subject in (1,2):
            pd.DataFrame(dict(subj=[subject]*2,trial=['old.wav','new.wav'],response=[1,2])).to_csv(behavior/f'Subj_{subject:04d}_Block_3.csv',index=False)
            with (meg/f'SUBJ_{subject:04d}_source.p').open('wb') as f:pickle.dump(rng.normal(size=(2,2,9)),f)
            pd.DataFrame(dict(x=[10.,20.],y=[0.,0.],z=[0.,0.])).to_csv(meg/f'SUBJ_{subject:04d}_pos.csv',index=False)
        with (self.data/'I1_epochs.p').open('wb') as f:pickle.dump(rng.normal(size=(4,2,9)),f)
        (self.data/'I1_info.json').write_text(json.dumps(dict(event_id=[1,2,1,2],time_epoch=(np.arange(9)/250).tolist())))
        matching=self.root/'matched.csv'
        pd.DataFrame(dict(ieeg_subject=['I1'],meg_subject=['1'],path_to_meg=[str(meg/'SUBJ_0001_source.p')])).to_csv(matching,index=False)
        options=dict(meg_dir=str(meg),ieeg_dir=str(self.data),project_path=str(self.project),matched_pairs_csv=str(matching),meg_coordinate_unit='mm',meg_tmin=0,prepared_cache_dir=str(self.root/'prepared'))
        return behavior,options

    def test_prepare_without_metadata_csv_or_src_and_invalidate_on_label_changes(self):
        from vs0_prepare_data import prepare_data,load_prepared_data
        behavior,options=self.make_project()
        with patch.dict(os.environ,dict(MEG_BEHAVIOR_DIR=str(behavior),SUBJECT_PAIRING_ITERATION='-1')),forbid_legacy_imports():
            first=prepare_data(**options);data=load_prepared_data(**options)
            self.assertEqual(data['electrode_metadata'].region.tolist(),['DLPFC','HPC'])
            np.testing.assert_allclose(data['electrode_positions'],[[10,0,0],[20,0,0]])
            dependencies={f['path'] for f in json.loads((first/'manifest.json').read_text())['request']['source_files']}
            self.assertIn(str(self.coords.resolve()),dependencies)
            self.assertIn(str(self.labels.resolve()),dependencies)
            pd.DataFrame(dict(electrode=['A1','B1'],label=['AMY','IFS'])).to_csv(self.labels,index=False)
            second=prepare_data(**options)
            self.assertNotEqual(first,second)
            self.assertEqual(load_prepared_data(**options)['electrode_metadata'].region.tolist(),['DLPFC','AMY'])

    def test_actual_preparation_command_runs_from_another_directory(self):
        behavior,options=self.make_project()
        file=self.root/'config.json';file.write_text(json.dumps(dict(data=options)))
        script=Path(shared.__file__).with_name('vs0_prepare_data.py')
        env=dict(os.environ,MEG_BEHAVIOR_DIR=str(behavior));env.pop('PYTHONPATH',None)
        result=subprocess.run([sys.executable,str(script),'--config',str(file)],cwd=self.root,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        manifests=list((self.root/'prepared').glob('*/manifest.json'))
        self.assertEqual(len(manifests),1)
        self.assertEqual(json.loads(manifests[0].read_text())['load_config']['metadata_source'],'GetInfo')

    def test_decoding_cleaner_statistics_and_numpy_representations(self):
        frame=pd.DataFrame([dict(iter=2,iter_perm=2,
            y_pred='[np.int64(0), np.int64(1), 1, 1]',y_test='[0,1,0,1]',
            y_pred_sh='[0,1,0,0,1,1,1,0]',entropy='[np.float64(0.1), np.float64(0.2)]',
            entropy_sh='[0.1,0.2,0.3,0.4]',weight='[[1,2,3],[3,4,5]]',
            weight_sh='[[1,2,3],[2,3,4],[3,4,5],[4,5,6]]',pca_weight='[1.e-06, 2.e-06]',pca_weight_sh='[np.float64(3e-6)]')])
        with forbid_legacy_imports():result=shared.cleandfdecodingTS(frame,3)
        self.assertEqual(result['acc'],.75);self.assertEqual(result['acc_sh'],.25)
        np.testing.assert_allclose(result['weight_mean'],[2,3,4])
        np.testing.assert_allclose(result['weight_std'],[1,1,1])
        np.testing.assert_allclose(result['weight_sh_mean'],[2.5,3.5,4.5])
        np.testing.assert_allclose(result['entropy_sh_mean'],[.15,.35])
        np.testing.assert_allclose(result['pca_weight'],[1e-6,2e-6])
        np.testing.assert_allclose(result['pca_weight_sh'],[3e-6])

    def test_postprojection_augmentation_preserves_shapes_and_condition_means(self):
        data=np.arange(12.).reshape(4,3)
        random.seed(17)
        with forbid_legacy_imports():samples,truth=shared.DataAugmentation(data,[[0,1],[2,3]],nb_trials=3)
        self.assertEqual(samples.shape,(6,3));self.assertEqual(truth.shape,(3,2))
        np.testing.assert_allclose(samples[2],data[:2].mean(0))
        np.testing.assert_allclose(samples[5],data[2:].mean(0))
        np.testing.assert_array_equal(truth,[[True,True],[True,True],[False,False]])

    def test_active_code_and_notebooks_have_no_legacy_imports_or_path_injections(self):
        root=Path(shared.__file__).parent
        sources=[(p.name,p.read_text()) for p in root.glob('*.py')]
        for path in root.glob('*.ipynb'):
            for i,cell in enumerate(json.loads(path.read_text())['cells']):
                if cell['cell_type']=='code':sources.append((f'{path.name}:{i}',''.join(cell['source'])))
        for name,source in sources:
            if any(line.lstrip().startswith(('%','!')) for line in source.splitlines()):continue
            for node in ast.walk(ast.parse(source,filename=name)):
                if isinstance(node,ast.ImportFrom):self.assertFalse((node.module or '').startswith('src'),name)
                if isinstance(node,ast.Import):self.assertFalse(any(n.name=='src' or n.name.startswith('src.') for n in node.names),name)
            self.assertNotIn("parent / 'LB'",source,name)
            self.assertNotIn("parent/'LB'",source,name)


if __name__=='__main__':unittest.main()
