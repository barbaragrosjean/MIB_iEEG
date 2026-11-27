"""Pair selection/output tests; scientific decoder is called without modification."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import vs0_run_analysis as workflow
from vs0_utils_pairing import PAIRING_IDS,pairing_context,pairing_paths,digest_file,active_pairing_id,active_pairing
from vs4_1_run_decoding import mainTS


class DecodingRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.base=Path(cls.tmp.name).resolve();cls.out=cls.base/'out'
        raw=cls.base/'raw';raw.mkdir();project=cls.base/'project';(project/'misc/electrodes').mkdir(parents=True)
        for subject in ['I0','I1']:
            for suffix in ['_epochs.p','_info.json','_coords.csv']:(raw/f'{subject}{suffix}').write_text('provenance fixture')
            (project/'misc/electrodes'/f'{subject}_electrodes.csv').write_text('provenance fixture')
        for i in range(4):(raw/f'M{i}_norm0_abs_0.mat').write_text('provenance fixture')
        dem=cls.base/'dem.csv';dem.write_text('subject,age,sex\n')
        questionnaire=cls.base/'questionnaire.xlsx';questionnaire.write_bytes(b'provenance fixture')
        for analysis in ('vs0_pairs','vs1_construct','vs4_decoding'):
            folder=cls.out/analysis;folder.mkdir(parents=True)
            cfg=json.loads((ROOT/'out'/analysis/'config.json').read_text())
            cfg=json.loads(json.dumps(cfg).replace('{ROOT}',str(ROOT)).replace('{OUT}',str(cls.out)))
            if analysis=='vs0_pairs':cfg['parameters'].update(ieeg_dir=str(raw),meg_dir=str(raw),project_path=str(project),ieeg_dem_csv=str(dem),meg_dem_xlsx=str(questionnaire))
            if analysis=='vs1_construct':cfg['parameters']['data'].update(ieeg_dir=str(raw),meg_dir=str(raw),project_path=str(project))
            if analysis=='vs4_decoding':cfg['parameters'].update(iteration=2,iter_perm=2,n_tr=5,method='PLSSVD')
            (folder/'config.json').write_text(json.dumps(cfg,indent=2))
        cache=cls.out/'vs1_construct/shared/trials';cache.mkdir(parents=True)
        rng=np.random.default_rng(42);records=[];times=np.arange(12)/250-.02
        for modality,subjects in [('ieeg',['I0','I1']),('meg',['M0','M1','M2','M3'])]:
            positions=np.array([[10.,0,0],[20.,0,0]]) if modality=='ieeg' else np.array([[10.,0,0],[20.,0,0],[30.,0,0]])
            for subject in subjects:
                files=[]
                for condition in (1,2):
                    filename=f'{modality}_{subject}_condition{condition}.npy'
                    np.save(cache/filename,rng.normal(size=(8,len(positions),12))+.1*condition)
                    files.append(filename)
                meta=pd.DataFrame(positions,columns=['x','y','z']);meta['subject']=subject;meta['channel_index']=np.arange(len(meta));meta['region']='HPC'
                records.append(dict(modality=modality,subject=subject,files=files,positions=positions.tolist(),metadata=meta.to_dict('list'),split_groups=[None,None],permutation_blocks=[['all']*8,['all']*8]))
        (cache/'manifest.json').write_text(json.dumps(dict(config=dict(conditions=[1,2],ieeg_subjects=['I0','I1'],meg_subjects=['M0','M1','M2','M3']),times=times.tolist(),records=records,source_files=[])))
        for i,name in enumerate(PAIRING_IDS):
            ids=[f'M{i%4}',f'M{(i+1)%4}']
            dem=pd.DataFrame(dict(ieeg_subject=['I0','I1'],meg_subject=ids))
            rows=[]
            for subject,meg in zip(dem.ieeg_subject,dem.meg_subject):
                for channel,x in enumerate([10.,20.]):
                    rows.append(dict(ieeg_subject=subject,meg_subject=meg,ieeg_electrode=f'A{channel}',channel_index=channel,source_index=channel,
                        ieeg_x=x,ieeg_y=0.,ieeg_z=0.,meg_x=x,meg_y=0.,meg_z=0.,distance_mm=0.,random_source_index=(channel+1)%3,
                        coverage_source_indices=json.dumps({s:channel for s in ids})))
            for path,table in zip(pairing_paths(cls.out/'vs0_pairs',name),(dem,pd.DataFrame(rows))):table.to_csv(path,index=False)
        p=cls.out/'vs0_pairs'
        cfg=workflow.read_config(p/'config.json')
        (p/'manifest.json').write_text(json.dumps(dict(status='complete',config=cfg,inputs={},outputs={path.name:digest_file(path) for path in p.glob('*_pairing_*.csv')})))

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def config(self,selection='match',method='PLSSVD'):
        path=self.out/'vs4_decoding/config.json';cfg=json.loads(path.read_text());cfg['pairings']=selection;cfg['parameters']['method']=method;path.write_text(json.dumps(cfg,indent=2));return path

    def test_01_pairing_selection_in_config(self):
        for selection in ('match','order','random_001','random_020'):
            self.assertEqual([j['pairing'] for j in workflow.plan(self.config(selection))],[selection])
        jobs=workflow.plan(self.config('all'))
        self.assertEqual([j['pairing'] for j in jobs],list(PAIRING_IDS))
        self.assertTrue(all(j['dependencies']==[] for j in jobs))
        self.assertEqual(len({j['output'] for j in jobs}),22)
        with self.assertRaises(ValueError):workflow.plan(self.config('random_999'))

    def test_02_all_pairings_reach_decoder_with_distinct_paths(self):
        config=self.config('all');observed=[]
        def fake_decoder(**kwargs):
            observed.append((active_pairing_id(),dict(zip(active_pairing().ieeg_subject,active_pairing().meg_subject)),Path(kwargs['out_path'])))
            for modality in ('ieeg','meg'):
                folder=Path(kwargs['out_path'])/modality/'broadband/fullmodel';folder.mkdir(parents=True)
                (folder/'summary.csv').write_text('completed\n')
                np.savez(folder/'inputs_iteration0_time_courses.npz',data=np.zeros(1))
            return {}
        with patch('vs4_1_run_decoding.mainTS',side_effect=fake_decoder),patch('vs0_run_analysis.ensure_trial_cache',side_effect=AssertionError('Do not export/remap union trials.')):
            for pairing in PAIRING_IDS:workflow.run_analysis(config,pairing)
        self.assertEqual([x[0] for x in observed],list(PAIRING_IDS))
        # Batch selection is orchestration only: a completed pairing is reusable.
        with patch('vs4_1_run_decoding.mainTS',side_effect=AssertionError('Do not refit when only batch selection changes.')):
            workflow.run_analysis(self.config('match'),'match')
        for name,mapping,path in observed:
            self.assertEqual(path,self.out/'vs4_decoding'/name)
            pairs=pd.read_csv(pairing_paths(self.out/'vs0_pairs',name)[0])
            self.assertEqual(mapping,dict(zip(pairs.ieeg_subject,pairs.meg_subject)))
            provenance=json.loads((path/'manifest.json').read_text())['provenance']['trial_cache']
            self.assertEqual(len(provenance['files']),8)  # two conditions × two participants × two modalities
            self.assertFalse(any('/vs4_decoding/shared/' in record['path'] for record in provenance['files']))

    def test_03_real_decoder_matches_direct_call(self):
        from vs2_1_utils_plssvd_eval import load_trial_cache
        for method,pairing in [('PCA','order'),('PLSSVD','random_001')]:
            with self.subTest(method=method):
                config=self.config(pairing,method);parameters=workflow.parameters(self.out,'vs4_decoding')
                direct=self.base/f'direct_{method}'
                with pairing_context(self.out/'vs0_pairs',pairing):
                    selected=load_trial_cache(parameters['cache_dir'])
                    expected=pd.read_csv(pairing_paths(self.out/'vs0_pairs',pairing)[0])
                    self.assertEqual([s.subject for s in selected.meg],expected.meg_subject.tolist())
                    mainTS(out_path=direct,save=True,**parameters)
                with patch('vs0_utils_load_and_config.match_sources',side_effect=AssertionError('No rematching during decoding.')):
                    result=workflow.run_analysis(config,pairing,recompute=True)
                for modality in ('ieeg','meg'):
                    relative=f'{modality}/broadband/fullmodel/{method}_mean_mean_0_LRfull.csv'
                    self.assertEqual((direct/relative).read_text(),(result/relative).read_text())
                    relative=relative[:-4]+'_iteration0_time_courses.npz'
                    with np.load(direct/relative) as first,np.load(result/relative) as second:
                        for key in first.files:np.testing.assert_array_equal(first[key],second[key])

    def test_04_selected_trial_changes_invalidate_result(self):
        config=self.config('random_010')
        # Selection already ran with the fake decoder, but config selection is a
        # run parameter; publish one new equivalent orchestration result first.
        def fake(**kwargs):
            for modality in ('ieeg','meg'):
                folder=Path(kwargs['out_path'])/modality/'broadband/fullmodel';folder.mkdir(parents=True)
                (folder/'summary.csv').write_text('ok\n');np.savez(folder/'x_iteration0_time_courses.npz',x=[1])
        with patch('vs4_1_run_decoding.mainTS',side_effect=fake):workflow.run_analysis(config,'random_010',recompute=True)
        dem=pd.read_csv(pairing_paths(self.out/'vs0_pairs','random_010')[0])
        file=self.out/'vs1_construct/shared/trials'/f'meg_{dem.meg_subject.iloc[0]}_condition1.npy'
        previous=file.read_bytes();file.write_bytes(previous+b'changed')
        try:
            with self.assertRaisesRegex(ValueError,'Stale results'):workflow.validate_result(self.out,'vs4_decoding','random_010')
        finally:file.write_bytes(previous)


if __name__=='__main__':unittest.main()
