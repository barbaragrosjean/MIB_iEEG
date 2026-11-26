"""Synthetic integration checks; no cluster data or atlas downloads required."""
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.io import savemat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vs0_run_analysis import ANALYSES,plan,run_analysis,read_config,execute,output_path
from vs0_utils_pairing import PAIRING_IDS,pairing_paths,pairing_context,digest_file
from vs0_utils_load_and_config import match_sources,load_project_data
from vs0_run_create_pairs import create_pairs
from vs0_utils_meg_behavior import meg_behavior_performance


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory();cls.base=Path(cls.temporary.name).resolve()
        b=cls.base;cls.out=b/'out';ieeg=b/'ieeg';meg=b/'meg';behavior=b/'behavior';project=b/'project'
        for folder in (ieeg,meg,behavior,project/'misc/electrodes'):folder.mkdir(parents=True)
        rng=np.random.default_rng(4);times=np.arange(12)/250-.02
        for i in range(2):
            subject=f'I{i:02d}'
            with (ieeg/f'{subject}_epochs.p').open('wb') as f:pickle.dump(rng.normal(size=(32,2,12)),f)
            (ieeg/f'{subject}_info.json').write_text(json.dumps(dict(event_id=[1]*16+[2]*16,time_epoch=times.tolist())))
            pd.DataFrame(dict(channels=['A_1','A_2'],x=[10.,20.],y=[0.,0.],z=[0.,0.])).to_csv(ieeg/f'{subject}_coords.csv',index=False)
            pd.DataFrame(dict(electrode=['A'],label=['HPC'])).to_csv(project/'misc/electrodes'/f'{subject}_electrodes.csv',index=False)
        for i in range(4):
            cells=np.empty((2,1),dtype=object)
            for c in range(2):cells[c,0]=rng.normal(size=(3,12,16))
            savemat(meg/f'SUBJ_{i+1:04d}_norm0_abs_0.mat',dict(OUT=dict(time=times,pos_brainsources_MNI8=np.array([[.01,0,0],[.02,0,0],[.03,0,0]]),sources_ERFs=cells)))
            pd.DataFrame(dict(trial=['old']*16+['new']*16,response=[1]*16+[2]*16)).to_csv(behavior/f'Subj_{i+1:04d}_Block_3.csv',index=False)
        pd.DataFrame(dict(subject=['I00','I01'],age=[20,30],sex=['F','M'])).to_csv(b/'ieeg_dem.csv',index=False)
        pd.DataFrame(dict(Subject=[1,2,3,4],Age=[21,31,25,35],Sex=['F','M','F','M'])).to_excel(b/'meg_dem.xlsx',sheet_name='Background questionnaire',index=False)
        for name in ('vs0_pairs',*ANALYSES):
            folder=cls.out/name;folder.mkdir(parents=True)
            cfg=json.loads((ROOT/'out'/name/'config.json').read_text())
            # Localize all templates before passing through read_config.
            cfg=json.loads(json.dumps(cfg).replace('{OUT}',str(cls.out)).replace('{ROOT}',str(ROOT)))
            p=cfg['parameters']
            if name=='vs0_pairs':p.update(ieeg_dem_csv=str(b/'ieeg_dem.csv'),meg_dem_xlsx=str(b/'meg_dem.xlsx'),ieeg_dir=str(ieeg),meg_dir=str(meg),behavior_dir=str(behavior),project_path=str(project))
            if name=='vs1_construct':p['data'].update(ieeg_dir=str(ieeg),meg_dir=str(meg),project_path=str(project));p['export_trials']=True
            if 'n_components' in p:p['n_components']=2
            if name=='vs1_channels':p.update(channel_step=2,channel_repeats=2)
            if name=='vs1_stability':p['comparison_pairings']=['random_001','random_002']
            if name=='vs2_plssvd_fit':p['options'].update(n_components=2,repeats=2,n_iterations=2)
            if name in ('vs2_plssvd_evaluation','vs2_plssvd_permutation'):p['components']=[2]
            if name=='vs2_plssvd_pca':p['components']=2
            if name=='vs2_plssvd_selection':p['n_components']=2
            if name=='vs3_subspace':p.update(dimensions=[1,2],repeats=2,cluster_counts=[2])
            if name=='vs3_weight_stability':p.update(components=2,iterations=2)
            if name in ('vs3_weight_clustering','vs3_pairing_clustering'):p.update(component_counts=[2],methods=['kmeans'],k_min=2,k_max=2)
            if name in ('vs3_radius_search','vs3_percent_search'):p.update(fit_radii=[10,20],min_peaks=1)
            if name=='vs3_radius_search':p['neighborhood_radii']=[10,20]
            if name=='vs3_percent_search':p['top_percents']=[50.,100.]
            (folder/'config.json').write_text(json.dumps(cfg,indent=2))
        with patch('vs0_utils_region_labels.get_region_label',side_effect=lambda f:['HPC']*len(f)):
            create_pairs(cls.out/'vs0_pairs/config.json')
        cls.no_rematching=patch('vs0_utils_load_and_config.match_sources',side_effect=AssertionError('Analyses must use saved CSV source indices.'))
        cls.no_rematching.start()

    @classmethod
    def tearDownClass(cls):
        cls.no_rematching.stop();cls.temporary.cleanup()

    def test_01_pairs(self):
        files=list((self.out/'vs0_pairs').glob('*_pairing_*.csv'));self.assertEqual(len(files),44)
        for name in PAIRING_IDS:
            dem,elect=pairing_paths(self.out/'vs0_pairs',name)
            d=pd.read_csv(dem);e=pd.read_csv(elect)
            self.assertEqual(d.ieeg_subject.tolist(),['I00','I01'])
            self.assertTrue(d.meg_subject.is_unique);self.assertEqual(len(e),4)
            np.testing.assert_allclose(e.distance_mm,0)
        self.assertEqual(pd.read_csv(self.out/'vs0_pairs/order_pairing_dem.csv').meg_subject.tolist(),['SUBJ_0001','SUBJ_0002'])

    def test_02_condition_denominators(self):
        folder=self.base/'unequal';folder.mkdir(exist_ok=True)
        # 6/12 old, 4/8 new: accepted at exactly .5 despite unequal denominators;
        # add 4 old omissions -> old 6/16, rejected, even if pooled accuracy is high elsewhere.
        pd.DataFrame(dict(trial=['old']*16+['new']*8,response=[1]*8+[None]*8+[2]*4+[None]*4)).to_csv(folder/'Subj_0001_Block_3.csv',index=False)
        result=meg_behavior_performance(folder).iloc[0]
        self.assertTrue(result.eligible);self.assertEqual(result.accuracy_old,.5);self.assertEqual(result.accuracy_new,.5)
        pd.DataFrame(dict(trial=['old']*8+['new']*24,response=[1]*3+[None]*5+[2]*24)).to_csv(folder/'Subj_0002_Block_3.csv',index=False)
        result=meg_behavior_performance(folder).set_index('subject').loc['2']
        self.assertFalse(result.eligible);self.assertGreater(result.accuracy,.5)

    def test_03_ties(self):
        a,d=match_sources([[0,0,0]]*40,[[-1,0,0],[1,0,0]],seed=42)
        b,_=match_sources([[0,0,0]]*40,[[-1,0,0],[1,0,0]],seed=42)
        np.testing.assert_array_equal(a,b);self.assertEqual(set(a),{0,1});np.testing.assert_allclose(d,1)

    def test_04_explicit_loader_and_source_map(self):
        from vs1_utils_coverage_matching import construct_five_datasets
        cfg=read_config(self.out/'vs1_construct/config.json')['parameters']['data']
        for name in ('match','order','random_001'):
            with pairing_context(self.out/'vs0_pairs',name) as dem:
                source=load_project_data(**cfg)
                self.assertEqual(source['meg_subjects'],dem.meg_subject.tolist())
                ds,audit,pairs=construct_five_datasets(source['meg'],source['meg_positions'],source['meg_subjects'],source['electrode_positions'],source['electrode_subjects'])
                self.assertEqual(pairs,dict(zip(dem.ieeg_subject,dem.meg_subject)))
                elect=pd.read_csv(pairing_paths(self.out/'vs0_pairs',name)[1])
                np.testing.assert_array_equal(audit.source_index,elect.source_index)
                np.testing.assert_array_equal(audit.random_source_index,elect.random_source_index)
                _,other,_=construct_five_datasets(source['meg'],source['meg_positions'],source['meg_subjects'],source['electrode_positions'],source['electrode_subjects'],seed=999)
                np.testing.assert_array_equal(other.source_index,audit.source_index)
                np.testing.assert_array_equal(other.random_source_index,audit.random_source_index)

    def test_05_all_plans(self):
        outputs=set()
        for name in ANALYSES:
            jobs=plan(self.out/name/'config.json');self.assertEqual(len(jobs),22)
            for job in jobs:
                self.assertEqual(Path(job['output']).parent,self.out/name)
                self.assertNotIn(job['output'],outputs);outputs.add(job['output'])
        self.assertEqual(len(outputs),550)

    def test_06_small_analysis_runs(self):
        for analysis in ('vs1_pca','vs1_manipulation','vs1_channels','vs1_stability','vs2_cov_models','vs1_channel_space','vs1_roi_shell_maps'):
            folder=run_analysis(self.out/analysis/'config.json','match')
            self.assertEqual(folder,self.out/analysis/'match')
            self.assertEqual(json.loads((folder/'manifest.json').read_text())['status'],'complete')

    def test_07_plssvd_routing(self):
        from vs2_1_run_plssvd_postprocess import evaluation_directory,comparison_directory
        from vs2_1_run_plssvd_fit_pca import comparison_path
        from vs2_1_run_plssvd_select_components import _selection_folder
        run=self.out/'vs2_plssvd_fit/random_001/none'
        self.assertTrue(evaluation_directory(run).is_relative_to(self.out/'vs2_plssvd_evaluation/random_001'))
        self.assertTrue(comparison_directory(run,2).is_relative_to(self.out/'vs2_plssvd_permutation/random_001'))
        self.assertTrue(comparison_path(run,2).is_relative_to(self.out/'vs2_plssvd_pca/random_001'))
        self.assertTrue(_selection_folder(run.parent,2,'both').is_relative_to(self.out/'vs2_plssvd_selection/random_001'))

    def test_08_plssvd_runs(self):
        for analysis in ('vs2_plssvd_fit','vs2_plssvd_evaluation','vs2_plssvd_pca','vs2_plssvd_permutation','vs2_plssvd_selection'):
            folder=run_analysis(self.out/analysis/'config.json','match')
            self.assertEqual(folder,self.out/analysis/'match')


    def test_09_specificity_and_subspace_runs(self):
        def labeler(frame,return_atlas_labels=False):
            return (['Hippocampus_L']*len(frame),['HPC']*len(frame)) if return_atlas_labels else ['HPC']*len(frame)
        analyses=['vs3_subspace','vs3_weight_stability','vs3_weights','vs3_pairing_weights',
                  'vs3_spatial_profiles','vs3_weight_clustering','vs3_pairing_clustering',
                  'vs3_cross_modal_profiles','vs3_gradients','vs3_radius_search',
                  'vs3_percent_search','vs3_gradient_comparison']
        with patch('vs0_utils_region_labels.get_region_label',side_effect=labeler):
            for analysis in analyses:
                with self.subTest(analysis=analysis):
                    folder=run_analysis(self.out/analysis/'config.json','match')
                    self.assertEqual(folder,self.out/analysis/'match')

    def test_10_all_22_pca_runs_and_cache_reuse(self):
        from vs0_run_analysis import validate_result
        cache=self.out/'vs1_construct/shared/trials/manifest.json'
        before=cache.stat().st_mtime_ns
        for pairing in PAIRING_IDS:
            run_analysis(self.out/'vs1_pca/config.json',pairing)
        self.assertEqual(cache.stat().st_mtime_ns,before)
        for pairing in PAIRING_IDS:
            self.assertEqual(validate_result(self.out,'vs1_pca',pairing),self.out/'vs1_pca'/pairing)


    def test_11_reproducibility_and_stale_detection(self):
        from vs0_run_analysis import validate_result
        expected={p.name:digest_file(p) for p in (self.out/'vs0_pairs').glob('*_pairing_*.csv')}
        with patch('vs0_utils_region_labels.get_region_label',side_effect=lambda f:['HPC']*len(f)):
            create_pairs(self.out/'vs0_pairs/config.json',overwrite=True)
        self.assertEqual(expected,{p.name:digest_file(p) for p in (self.out/'vs0_pairs').glob('*_pairing_*.csv')})
        path=self.out/'vs1_pca/config.json';before=path.read_text();cfg=json.loads(before)
        cfg['parameters']['n_components']=1;path.write_text(json.dumps(cfg))
        try:
            with self.assertRaisesRegex(ValueError,'Stale results'):
                validate_result(self.out,'vs1_pca','match')
        finally:path.write_text(before)
        path=self.out/'vs3_gradients/config.json';before=path.read_text();cfg=json.loads(before)
        cfg['parameters']['fit_radius_mm']=31;path.write_text(json.dumps(cfg))
        try:validate_result(self.out,'vs1_pca','match')
        finally:path.write_text(before)

    def test_12_missing_demographics_excluded(self):
        from vs0_utils_load_and_config import select_subject
        cfg=read_config(self.out/'vs0_pairs/config.json')['parameters'].copy()
        missing=self.base/'missing.csv'
        pd.DataFrame(dict(subject=['I00','I01'],age=[20,None],sex=['F','M'])).to_csv(missing,index=False)
        cfg['ieeg_dem_csv']=str(missing)
        keys=('ieeg_dem_csv','meg_dem_xlsx','ieeg_dir','meg_dir','behavior_dir','project_path','meg_sheet')
        ieeg,meg,audit=select_subject(**{k:cfg[k] for k in keys})
        self.assertEqual(ieeg.subject.tolist(),['I00'])
        row=audit.loc[(audit.modality=='ieeg') & (audit.subject=='I01')].iloc[0]
        self.assertFalse(row.eligible);self.assertIn('age',row.reason)


    def test_13_fit_resume_keeps_completed_iterations(self):
        from vs2_1_utils_plssvd_eval import load_trial_cache,ValidationOptions,fit_plssvd
        with pairing_context(self.out/'vs0_pairs','match'):
            trials=load_trial_cache(self.out/'vs1_construct/shared/trials')
            options=ValidationOptions(repeats=2,n_components=2,n_iterations=2,perm='none',perm_type='none')
            folder=self.base/'resume_fit'
            fit_plssvd(trials,'paired_coverage',options,folder)
            first=folder/'iteration_000/model_000.npz'
            before=first.stat().st_mtime_ns
            expected=digest_file(folder/'iteration_001/model_000.npz')
            (folder/'COMPLETE.json').unlink()
            (folder/'iteration_001/COMPLETE.json').unlink()
            fit_plssvd(trials,'paired_coverage',options,folder,resume=True)
            self.assertEqual(first.stat().st_mtime_ns,before)
            self.assertEqual(digest_file(folder/'iteration_001/model_000.npz'),expected)
            self.assertTrue((folder/'COMPLETE.json').exists())


    def test_14_duplicate_demographics_keep_first(self):
        from vs0_utils_load_and_config import select_subject
        cfg=read_config(self.out/'vs0_pairs/config.json')['parameters'].copy()
        duplicate=self.base/'duplicate.csv'
        pd.DataFrame(dict(subject=['I00',' I00 ','I01'],age=[20,99,30],sex=['F','M','M'])).to_csv(duplicate,index=False)
        cfg['ieeg_dem_csv']=str(duplicate)
        keys=('ieeg_dem_csv','meg_dem_xlsx','ieeg_dir','meg_dir','behavior_dir','project_path','meg_sheet')
        ieeg,meg,audit=select_subject(**{k:cfg[k] for k in keys})
        self.assertEqual(ieeg.subject.tolist(),['I00','I01'])
        self.assertEqual(ieeg.iloc[0].age,20)
        self.assertEqual(ieeg.iloc[0].sex,'female')
        ignored=audit.loc[audit.reason.eq('duplicate_demographic_row_keep_first')]
        self.assertEqual(len(ignored),1)
        self.assertFalse(ignored.iloc[0].eligible)


    def test_15_saved_correspondence_required(self):
        from vs1_utils_coverage_matching import construct_five_datasets
        with self.assertRaisesRegex(ValueError,'explicit CSV pairing'):
            construct_five_datasets([np.ones((2,2,4))],[np.zeros((2,3))],['M'],np.zeros((1,3)),['I'])
        assignments=pd.read_csv(self.out/'vs1_stability/match/assignments.csv')
        self.assertEqual(set(assignments.pairing_id),{'match','random_001','random_002'})
        for name,rows in assignments.groupby('pairing_id'):
            dem=pd.read_csv(pairing_paths(self.out/'vs0_pairs',name)[0])
            self.assertEqual(dict(zip(rows.ieeg_subject,rows.meg_subject)),dict(zip(dem.ieeg_subject,dem.meg_subject)))


    def test_16_csv_indices_are_authoritative_and_missing_fields_fail(self):
        import shutil
        from vs1_utils_coverage_matching import construct_five_datasets
        folder=self.base/'authoritative';folder.mkdir()
        for source in pairing_paths(self.out/'vs0_pairs','match'):shutil.copy2(source,folder/source.name)
        elect_path=pairing_paths(folder,'match')[1]
        elect=pd.read_csv(elect_path)
        subject=elect.loc[0,'meg_subject']
        original=int(elect.loc[0,'source_index'])
        replacement=(original+1)%3
        elect.loc[0,'source_index']=replacement
        elect.loc[0,'meg_x']=10.*(replacement+1)
        elect.loc[0,'distance_mm']=abs(elect.loc[0,'meg_x']-elect.loc[0,'ieeg_x'])
        coverage=json.loads(elect.loc[0,'coverage_source_indices']);coverage[subject]=replacement
        elect.loc[0,'coverage_source_indices']=json.dumps(coverage)
        elect.to_csv(elect_path,index=False)
        cfg=read_config(self.out/'vs1_construct/config.json')['parameters']['data']
        with pairing_context(folder,'match'):
            source=load_project_data(**cfg)
            args=(source['meg'],source['meg_positions'],source['meg_subjects'],source['electrode_positions'],source['electrode_subjects'])
            datasets,audit,_=construct_five_datasets(*args)
            self.assertEqual(audit.iloc[0].source_index,replacement)
            j=source['meg_subjects'].index(subject)
            np.testing.assert_array_equal(datasets['paired_coverage'].arrays[0][:,0,:],source['meg'][j][:,replacement,:])
        elect.drop(columns=['coverage_source_indices']).to_csv(elect_path,index=False)
        with pairing_context(folder,'match'):
            with self.assertRaisesRegex(ValueError,'lacks coverage indices'):
                construct_five_datasets(*args)


if __name__=='__main__':unittest.main()
