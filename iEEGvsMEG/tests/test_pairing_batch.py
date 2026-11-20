import json
import os
import pickle
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

import vs0_load_and_config as shared
from vs0_run_pairings import run,pairing_context,assignment_config,activate
from vs0_prepare_data import prepare_data,data_options
from vs0_pairing_results import numeric_summary,aggregate_results,aligned_arrays,config_view


@pytest.fixture
def project(tmp_path,monkeypatch):
    for key in ('SUBJECT_PAIRING_ITERATION','SUBJECT_PAIRING_SEED','SUBJECT_PAIRING_VIEW','SUBJECT_PAIRING_COUNT','SUBJECT_PAIRING_BATCH_CHILD'):
        monkeypatch.delenv(key,raising=False)
    behavior=tmp_path/'behavior';meg=tmp_path/'meg';ieeg=tmp_path/'ieeg'
    for folder in (behavior,meg,ieeg):folder.mkdir()
    monkeypatch.setenv('MEG_BEHAVIOR_DIR',str(behavior))
    rng=np.random.default_rng(19)
    for i in range(1,4):
        pd.DataFrame(dict(subj=[i,i],trial=['old.wav','new.wav'],response=[1,2])).to_csv(behavior/f'Subj_{i:04d}_Block_3.csv',index=False)
        with (meg/f'SUBJ_{i:04d}_source.p').open('wb') as f:pickle.dump(rng.normal(size=(2,2,9)),f)
        pd.DataFrame(dict(x=[0,10],y=[0,0],z=[0,0])).to_csv(meg/f'SUBJ_{i:04d}_pos.csv',index=False)
    for subject in ('I1','I2'):
        with (ieeg/f'{subject}_epochs.p').open('wb') as f:pickle.dump(rng.normal(size=(8,1,9)),f)
        (ieeg/f'{subject}_info.json').write_text(json.dumps(dict(event_id=[1,2]*4,time_epoch=(np.arange(9)/250).tolist())))
    csv=tmp_path/'matched.csv'
    pd.DataFrame(dict(ieeg_subject=['I1','I2'],meg_subject=['1','2'],path_to_meg=[str(meg/f'SUBJ_{i:04d}_source.p') for i in (1,2)])).to_csv(csv,index=False)
    monkeypatch.setattr(shared,'MATCHED_PAIRS_CSV',csv)
    metadata=tmp_path/'metadata.csv'
    pd.DataFrame(dict(subject=['I1','I2'],channel_index=[0,0],x=[0,10],y=[0,0],z=[0,0],region=['A','B'])).to_csv(metadata,index=False)
    with pairing_context(-1,77):root=shared.analysis_output_root(tmp_path)
    config=dict(data=dict(meg_dir=str(meg),ieeg_dir=str(ieeg),metadata_csv=str(metadata),matched_pairs_csv=str(csv),meg_coordinate_unit='mm',meg_tmin=0,prepared_cache_dir=str(tmp_path/'prepared')),
                output_dir=str(root/'coverage'),n_components=2,condition_mode='average',block_scaling=False,
                seed=11,pairing_seed=77,n_pairings=2,max_gram_gib=2.,channel_step=2,channel_repeats=2,stability_repeats=2,meg_kind='paired_coverage')
    return config


def test_default_batch_is_101_with_matched_first_for_both_views(project,monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(subprocess,'run',lambda args,**kwargs:calls.append((args,kwargs['env'])))
    configfile=tmp_path/'config.json';configfile.write_text(json.dumps(project))
    run(['python','analysis.py','--config',str(configfile)],'matched',100,77)
    assert [int(e['SUBJECT_PAIRING_ITERATION']) for _,e in calls]==[-1,*range(100)]
    assert all(e['SUBJECT_PAIRING_BATCH_CHILD']=='1' for _,e in calls)
    for args,env in calls:
        saved=json.loads(Path(args[-1]).read_text())
        assert f"iteration_{env['SUBJECT_PAIRING_ITERATION']}/" in saved['output_dir']
    # Switching only the view reuses completed computations.
    run(['python','analysis.py','--config',str(configfile)],'full',100,77)
    assert len(calls)==101


def test_once_scope_and_display_only_context(project,monkeypatch,tmp_path):
    activate('full',100,77,display_only=True)
    assert shared.subject_pairing_iteration()==-1
    calls=[]
    monkeypatch.setattr(subprocess,'run',lambda args,**kwargs:calls.append(kwargs['env']))
    file=tmp_path/'config.json';file.write_text(json.dumps(project))
    run(['python','analysis.py','--config',str(file)],'full',100,77,scope='once')
    assert [e['SUBJECT_PAIRING_ITERATION'] for e in calls]==['-1']


def test_prepare_uses_all_eligible_meg_and_reuses_one_cache_for_every_assignment(project,monkeypatch):
    calls=[];raw=shared.load_project_data
    def track(**kwargs):calls.append(kwargs);return raw(**kwargs)
    monkeypatch.setattr(shared,'load_project_data',track)
    paths=[]
    for i in (-1,0,1):
        with pairing_context(i,77):
            config=assignment_config(project,i)
            paths.append(prepare_data(**data_options(config)))
    assert len(set(paths))==1 and len(calls)==1
    manifest=json.loads((paths[0]/'manifest.json').read_text())
    assert manifest['meg_subjects']==['SUBJ_0001','SUBJ_0002','SUBJ_0003']
    assert manifest['ieeg_subjects']==['I1','I2']


def test_real_coverage_and_covariance_views_include_all_exported_metrics(project):
    from vs1_run_coverage import load_datasets,run_analysis
    from vs2_run_cov_models import run as covariance_run
    from vs0_pairing_results import matched_analysis_directory
    for i in (-1,0,1):
        config=assignment_config(project,i)
        with pairing_context(i,77):
            prepare_data(**data_options(config));datasets=load_datasets(config)
            for analysis in ('manipulations','pca'):
                run_analysis(config,analysis,datasets)
            cov=dict(config,output_dir=str(Path(config['output_dir']).parent/'covariance'))
            covariance_run(cov)
    full=config_view(project,'pca','full',2);matched=config_view(project,'pca','matched',2)
    assert full.assignment_ids==[0,1] and matched.assignment_ids==[-1]
    assert {'pca_spectra','pca_comparison','pca_component_pairs'}<=set(full.tables)
    assert any('aligned_scores' in k for k in full.arrays)
    assert all((a['count']==2).all() for a in full.arrays.values())
    cov=dict(project,output_dir=str(Path(project['output_dir']).parent/'covariance'))
    view=config_view(cov,'models','full',2)
    assert {'variance','model_comparison','component_pairs','reference_pairs','within_model_metrics','within_model_pairs','within_model_correlations'}<=set(view.tables)
    assert any('signed_correlation' in key for key in view.arrays)
    assert all((v['count']==2).all() for v in view.arrays.values())


def test_coverage_stability_runs_exact_same_outer_random_assignments(project):
    from vs1_run_coverage import load_datasets,run_analysis
    with pairing_context(-1,77):
        prepare_data(**data_options(project));datasets=load_datasets(project)
        folder=run_analysis(project,'stability',datasets)
    assignments=pd.read_csv(folder/'assignments.csv')
    assert set(assignments.repeat)=={-1,0,1}
    for i in (0,1):
        with pairing_context(i,77):expected=shared.random_subject_pairing(['I1','I2'],['SUBJ_0001','SUBJ_0002','SUBJ_0003'])
        frame=assignments[assignments.repeat==i]
        assert dict(zip(frame.ieeg_subject,frame.meg_subject))==expected
    assert (folder/'full_component_pairs.csv').exists()


def test_equal_assignment_weight_and_sample_sd_not_pooled_fold_sd():
    a=pd.DataFrame(dict(k=[1,1],metric=[0.,2.]))
    b=pd.DataFrame(dict(k=[1]*4,metric=[5.]*4))
    result=numeric_summary([a,b])
    assert result['mean'].metric.iloc[0]==3.
    assert np.isclose(result['std'].metric.iloc[0],np.sqrt(8))
    assert result['count'].metric.iloc[0]==2


def test_missing_assignment_fails_and_matched_is_excluded():
    results={i:dict(tables={'metric':pd.DataFrame(dict(k=[1],value=[v]))},arrays={'matrix':np.array([[v]])}) for i,v in [(-1,999),(0,1),(1,5)]}
    result=aggregate_results(results,'full',2)
    assert result.arrays['matrix']['mean'][0,0]==3
    assert result.tables['metric']['mean'].value.iloc[0]==3
    with pytest.raises(FileNotFoundError):aggregate_results(results,'full',3)


def test_sign_and_component_alignment_before_array_means():
    reference=np.array([[1.,3.],[2.,-2.],[-1.,2.],[-2.,-3.]])
    reversed_scores=reference[:,::-1]*[-1,1]
    aligned=aligned_arrays(reference,reversed_scores)
    np.testing.assert_allclose(aligned['scores'],reference)


def test_vector_metrics_are_not_grouped_as_text_labels():
    frames=[pd.DataFrame(dict(k=[2],angles_deg=[value])) for value in ('[1,3]','[3,7]')]
    result=numeric_summary(frames)
    assert result['mean'].angles_deg_1.iloc[0]==2
    assert result['mean'].angles_deg_2.iloc[0]==5


@pytest.fixture
def trial_cache(project):
    root=Path(project['data']['meg_dir']).parent/'trials';root.mkdir()
    rng=np.random.default_rng(33);records=[]
    for modality,subjects in [('ieeg',['I1','I2']),('meg',['SUBJ_0001','SUBJ_0002','SUBJ_0003'])]:
        for si,subject in enumerate(subjects):
            n=1 if modality=='ieeg' else 2
            xyz=np.array([[10.*si,0,0]]) if modality=='ieeg' else np.array([[0.,0,0],[10.,0,0]])
            files=[]
            for condition in range(2):
                name=f'{modality}_{subject}_{condition}.npy';np.save(root/name,rng.normal(size=(6,n,9)));files.append(name)
            metadata=pd.DataFrame(dict(subject=[subject]*n,channel_index=range(n),source_index=range(n),x=xyz[:,0],y=xyz[:,1],z=xyz[:,2]))
            records.append(dict(modality=modality,subject=subject,files=files,positions=xyz.tolist(),metadata=metadata.to_dict(orient='list'),split_groups=[None,None],permutation_blocks=[['all']*6,['all']*6]))
    megfiles={p.name.removesuffix('_source.p'):str(p) for p in Path(project['data']['meg_dir']).glob('*_source.p')}
    manifest=dict(config=dict(conditions=[1,2],full_meg_pool=True,meg_subjects=list(megfiles),ieeg_subjects=['I1','I2'],meg_files=megfiles),records=records,times=(np.arange(9)/250).tolist(),source_files=[dict(path=p,size=Path(p).stat().st_size,mtime_ns=Path(p).stat().st_mtime_ns) for p in megfiles.values()])
    (root/'manifest.json').write_text(json.dumps(manifest))
    return root


def test_real_plssvd_evaluation_null_pca_and_selection_views(project,trial_cache):
    from vs2_1_plssvd_eval_utils import load_trial_cache,fit_plssvd,ValidationOptions
    from vs2_1_plssvd_postprocess import evaluate_run,prepare_comparison
    from vs2_1_plssvd_results import prepare_summary
    from vs2_1_plssvd_fit_pca import run as fit_pca
    from vs2_1_plssvd_select_components import prepare_selection
    from vs0_pairing_results import evaluation_view,pca_comparison_view,permutation_view,selection_view
    base=Path(project['output_dir']).parent/'plssvd_eval'/'paired_coverage'
    for i in (-1,0,1):
        with pairing_context(i,77):
            trials=load_trial_cache(trial_cache)
            runs=Path(assignment_config(str(base),i))
            for name,perm,kind in [('none',None,None),('both__time_point','both','time_point')]:
                options=ValidationOptions(n_iterations=2,repeats=2,n_components=2,seed=11,perm=perm,perm_type=kind)
                fit_plssvd(trials,'paired_coverage',options,runs/name)
                evaluate_run(runs/name,[2],trial_cache)
            prepare_summary(runs/'none',2)
            fit_pca(runs/'none',trial_cache,2,meg_kind='paired_coverage',block_scaling='none')
            prepare_comparison(runs,'both__time_point',2)
            prepare_selection(runs,2,perm='both',perm_type='time_point')
    result=evaluation_view(base/'none',2,'full',2,meg_kind='paired_coverage',block_scaling='none')
    assert {'fold_metrics','components','iteration_metrics','metric_summary'}<=set(result.tables)
    assert result.assignment_ids==[0,1]
    assert 'test_covariance' in result.arrays
    pca=pca_comparison_view(base/'none',2,'full',2,meg_kind='paired_coverage',block_scaling='none')
    assert pca.arrays['full_data_pca_correlations']['mean'].shape[-2:]==(2,2)
    null=permutation_view(base,'both__time_point',2,'full',2)
    assert {'metrics','components','differences','delta_summary'}<=set(null.tables)
    selected=selection_view(base,2,'both','time_point','full',2)
    assert 'selection_summary' in selected.tables
    assert 'selected_k' in selected.tables['selection_summary']['metrics']


def test_real_modality_comparison_exports_reference_metrics(project,trial_cache):
    from vs3_0_compare_subspace import run_comparison
    from vs0_pairing_results import path_view
    base=Path(project['output_dir']).parent/'compare_subspace'/'paired_coverage'
    for i in (-1,0,1):
        with pairing_context(i,77):
            run_comparison(trial_cache,Path(assignment_config(str(base),i)),meg_kind='paired_coverage',models=['separate_pca','plssvd'],dimensions=[1,2],repeats=2,seed=11,cluster_counts=[2])
    result=path_view(base,'full',2)
    assert {'overlap','alignment','clusters','ieeg_pca_reference_matches','within_model_metrics'}<=set(result.tables)
    assert result.tables['ieeg_pca_reference_matches']['mean'].matched_abs_r.notna().all()


def test_weight_stability_is_summarized_across_outer_assignments(project,trial_cache,monkeypatch):
    from types import SimpleNamespace
    import vs3_0_spatial_reliability as reliability
    from vs0_pairing_results import finalize_weight_batch
    monkeypatch.setattr(reliability,'annotate_coordinates',lambda xyz,*args:pd.DataFrame(dict(parcel=['P']*len(xyz),region=['R']*len(xyz),group=['G']*len(xyz))))
    base=Path(project['output_dir']).parent/'weight_stability'/'paired_coverage'
    for i in (-1,0,1):
        with pairing_context(i,77):
            args=SimpleNamespace(cache_dir=trial_cache,output_dir=Path(assignment_config(str(base),i))/'pairing',weight_only=True,
                variation='pairing',meg_kind='paired_coverage',models=['separate_pca','plssvd'],iterations=1,components=2,seed=11,
                split_unit='trial',block_scaling='none',max_gram_gib=2.,pool_hemispheres=False,scratch_dir=None,
                ieeg_coords_dir=None,meg_coords_dir=None,ieeg_coords_unit='mm',meg_coords_unit='mm',atlas_image=None,atlas_labels=None)
            reliability.fit_iterations(args)
    root=finalize_weight_batch(base,2)/'pairing'
    cfg=json.loads((root/'config.json').read_text())
    assert cfg['assignment_ids']==[0,1] and cfg['reference_assignment']==-1
    roi=pd.read_csv(root/'roi_stability.csv')
    assert (roi.n_iterations==2).all()
    assert (root/'WEIGHT_SUMMARY_COMPLETE.json').exists()


def test_real_terminal_preparation_and_source_count_exception(project,tmp_path):
    import sys
    configfile=tmp_path/'config.json';configfile.write_text(json.dumps(project))
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'vs0_prepare_data.py'),'--config',str(configfile)],check=True,capture_output=True,text=True)
    prepared=list(Path(project['data']['prepared_cache_dir']).glob('*/manifest.json'))
    assert len(prepared)==1
    assert len(json.loads(prepared[0].read_text())['meg_subjects'])==3
    subprocess.run([sys.executable,str(root/'vs1_run_coverage.py'),'--config',str(configfile),'--analysis','channels'],check=True,capture_output=True,text=True)
    baseline=Path(project['output_dir']).parent
    assert len(list(baseline.glob('coverage/channels/*/RUN_COMPLETE.json')))==1
    assert not Path(assignment_config(str(baseline),0)).exists()


def test_real_terminal_computation_defaults_to_matched_plus_random(project,tmp_path):
    import sys
    configfile=tmp_path/'config.json';configfile.write_text(json.dumps(project))
    root=Path(__file__).resolve().parents[1]
    completed=subprocess.run([sys.executable,str(root/'vs1_run_coverage.py'),'--config',str(configfile),'--analysis','manipulations'],check=True,capture_output=True,text=True)
    assert 'Subject pairing -1:' in completed.stdout
    for i in (-1,0,1):
        folder=Path(assignment_config(project['output_dir'],i))
        assert len(list(folder.glob('manipulations/*/RUN_COMPLETE.json')))==1


def test_scalar_metrics_and_array_plots_render_for_full_view(monkeypatch):
    import types,sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from vs0_pairing_results import show
    displayed=[]
    display_module=types.ModuleType('IPython.display');display_module.display=lambda value:displayed.append(value)
    monkeypatch.setitem(sys.modules,'IPython.display',display_module)
    monkeypatch.setattr(plt,'show',lambda:plt.close('all'))
    results={i:dict(tables={'variance':pd.DataFrame(dict(model=['plssvd'],k=[1],variance=[float(i)]))},
                       arrays={'rank':np.array(float(i)),'covariance':np.eye(2)*(i+1),'aligned_scores':np.ones((4,2))*i},times=np.arange(4)) for i in (-1,0,1)}
    show(aggregate_results(results,'full',2),max_components=1)
    assert any('variance_std' in frame for frame in displayed)
    assert any(set(frame)=={'mean','std','count'} for frame in displayed)


def test_notebook_display_context_never_bypasses_terminal_batch(monkeypatch):
    import sys
    import vs0_run_pairings as batch
    calls=[]
    monkeypatch.delenv('SUBJECT_PAIRING_BATCH_CHILD',raising=False)
    monkeypatch.delenv('SUBJECT_PAIRING_SINGLE_RUN',raising=False)
    monkeypatch.setattr(sys,'argv',['analysis.py'])
    monkeypatch.setattr(batch,'run',lambda *args,**kwargs:calls.append(args))
    for view in ('matched','full'):
        activate(view,100,77,display_only=True)
        assert batch.terminal_batch('analysis.py')
    assert len(calls)==2
    assert all(args[2]==100 and args[-1]=='all' for args in calls)
