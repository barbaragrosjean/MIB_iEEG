import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import vs0_load_and_config as shared
from vs0_run_pairings import activate, run, summarize
from vs2_1_plssvd_eval_utils import load_trial_cache, TrialData, TrialSubject, ValidationOptions


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(shared, 'eligible_meg_subjects', lambda subjects: list(map(str, subjects)))
    monkeypatch.setattr(shared, 'behavior_provenance', lambda: dict(sha256='test-behavior'))
    for key in ('SUBJECT_PAIRING_ITERATION', 'SUBJECT_PAIRING_SEED', 'SUBJECT_PAIRING_VIEW', 'SUBJECT_PAIRING_COUNT'):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    meg = tmp_path/'meg'
    meg.mkdir()
    for s in range(12):
        (meg/f'M{s:02d}_source.p').touch()
    csv = tmp_path/'matched_pairs.csv'
    pd.DataFrame({'ieeg_subject':['I2','I1'], 'meg_subject':['M00','M01'],
                  'path_to_meg':[str(meg/'M00_source.p'), str(meg/'M01_source.p')]}).to_csv(csv,index=False)
    monkeypatch.setattr(shared,'MATCHED_PAIRS_CSV',csv)
    return meg,csv


def test_baseline_exact_csv_and_random_uses_full_pool(cohort):
    meg,csv=cohort
    activate('matched', 10)
    ids, ieeg, files, pairing = shared.select_subjects(meg,['I1','I2','I3'])
    assert ids == ['M00','M01'] and ieeg == ['I2','I1']
    assert pairing == {'I2':'M00','I1':'M01'}
    seen=set()
    assignments=[]
    activate('full', 100)
    for i in range(100):
        os.environ['SUBJECT_PAIRING_ITERATION']=str(i)
        ids, ieeg, files, pairing=shared.select_subjects(meg,['I1','I2','I3'])
        assert len(ids)==12 and len(files)==12
        assert ieeg==['I2','I1']
        assert len(set(pairing.values()))==2
        assert pairing==shared.random_subject_pairing(ieeg, reversed(ids))
        seen.update(pairing.values());assignments.append(pairing)
    assert seen==set(ids)
    assert len({tuple(p.values()) for p in assignments})>1


def test_reject_restricted_random_roster(cohort):
    activate('full',3)
    with pytest.raises(ValueError,match='entire MEG'):
        shared.select_subjects(cohort[0], ['I1','I2'], meg_subjects=['M00','M01'])


def make_cache(tmp_path, full=True):
    records=[]
    for modality, ids in [('ieeg',['I1','I2','I3']),('meg',[f'M{i:02d}' for i in range(12)])]:
        for subject in ids:
            name=f'{modality}_{subject}.npy'
            np.save(tmp_path/name,np.zeros((4,1,5)))
            records.append(dict(modality=modality,subject=subject,files=[name,name],
                positions=[[0,0,0]],metadata={'subject':[subject]},split_groups=[None,None],permutation_blocks=[None,None]))
    manifest=dict(config=dict(conditions=[1,2],full_meg_pool=full),records=records,times=list(range(5)),source_files=[])
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))


def test_trial_and_average_assignment_agree_and_all_meg_retained(tmp_path,cohort):
    make_cache(tmp_path)
    activate('full',5,seed=71)
    for i in range(5):
        os.environ['SUBJECT_PAIRING_ITERATION']=str(i)
        trial=load_trial_cache(tmp_path)
        _,ieeg,_,pairing=shared.select_subjects(cohort[0],['I1','I2','I3'])
        assert len(trial.meg)==12
        assert [s.subject for s in trial.ieeg]==ieeg
        assert trial.pairing==pairing


def test_reject_old_subset_cache(tmp_path,cohort):
    make_cache(tmp_path,full=False)
    activate('full',5)
    with pytest.raises(ValueError,match='full MEG cache'):
        load_trial_cache(tmp_path)


def test_assignment_does_not_truncate_validation_or_change_surrogate(tmp_path,monkeypatch):
    import vs2_1_plssvd_eval_utils as utils
    calls=[]
    monkeypatch.setattr(utils,'_validate_iteration',lambda trials,kind,options,*args,**kwargs:calls.append((trials.pairing,options,kwargs)))
    trials=TrialData([],[],np.arange(5),(1,2),{'I1':'M01'})
    options=ValidationOptions(n_iterations=3,repeats=2,n_components=1,perm='meg',perm_type='time_point')
    utils.fit_plssvd(trials,'paired_coverage',options,tmp_path/'fit')
    assert len(calls)==3
    assert all(c[0]=={'I1':'M01'} and c[1].perm_type=='time_point' for c in calls)
    assert len({c[2]['permutation_seed'] for c in calls})==3
    assert len({c[2]['matching_seed'] for c in calls})==1


def test_runner_paths_and_seeds(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(subprocess,'run',lambda args,**kwargs:calls.append((args,kwargs)))
    root=tmp_path/'subject_pairings'/'abc'/'iteration_0'
    root.mkdir(parents=True)
    cfg=root/'config.json';cfg.write_text(json.dumps(dict(output_dir=str(root/'coverage'))))
    run(['python','analysis.py','--config',str(cfg)],'full',3,55)
    assert len(calls)==3
    for i,(args,kwargs) in enumerate(calls):
        assert kwargs['env']['SUBJECT_PAIRING_ITERATION']==str(i)
        assert kwargs['env']['SUBJECT_PAIRING_SEED']=='55'
        assert json.loads(Path(args[-1]).read_text())['output_dir'].endswith(f'iteration_{i}/coverage')
    assert 'iteration_0' in json.loads(cfg.read_text())['output_dir']


def test_summary_equal_assignment_weight_and_excludes_baseline(tmp_path):
    for i,values in [(-1,[999]),(0,[0,2]),(1,[5,5,5,5])]:
        folder=tmp_path/f'iteration_{i}';folder.mkdir()
        pd.DataFrame(dict(k=[1]*len(values),metric=values)).to_csv(folder/'metrics.csv',index=False)
    values,summary=summarize(tmp_path,2,'metrics.csv',['k'],['metric'])
    assert set(values.pairing_iteration)=={0,1}
    assert summary.loc[1,('metric','mean')]==3
    assert np.isclose(summary.loc[1,('metric','std')],np.sqrt(8))
    with pytest.raises(FileNotFoundError):summarize(tmp_path,3,'metrics.csv',['k'],['metric'])


def test_real_fit_keeps_subject_assignment_separate_from_time_permutation(tmp_path,cohort):
    import vs2_1_plssvd_eval_utils as utils
    rng=np.random.default_rng(11)
    def subject(name):
        positions=np.array([[0.,0.,0.],[10.,0.,0.]])
        metadata=pd.DataFrame(dict(subject=[name]*2,channel_index=[0,1],x=positions[:,0],y=positions[:,1],z=positions[:,2]))
        return TrialSubject(name,(1,2),[rng.normal(size=(6,2,10)) for _ in range(2)],positions,metadata,[None,None],[['all']*6,['all']*6])
    trials=TrialData([subject('I1'),subject('I2')],[subject(f'M{i:02d}') for i in range(4)],np.arange(10)/250,(1,2),{'I1':'M03','I2':'M02'})
    options=ValidationOptions(n_iterations=2,repeats=2,n_components=1,perm='meg',perm_type='time_point')
    utils.fit_plssvd(trials,'paired_coverage',options,tmp_path/'fit')
    for iteration in range(2):
        folder=tmp_path/'fit'/f'iteration_{iteration:03d}'
        assert (folder/'COMPLETE.json').exists()
        assert json.loads((folder/'pairing.json').read_text())==trials.pairing
        assert json.loads((folder/'validation_options.json').read_text())['perm_type']=='time_point'
        assert (folder/'model_001.npz').exists()


def test_assignment_namespaces_have_stable_child_hashes():
    a=dict(output_dir='/root/out/subject_pairings/abc/iteration_0/results',matching=dict(subject_pairing_iteration=0))
    b=dict(output_dir='/root/out/subject_pairings/abc/iteration_8/results',matching=dict(subject_pairing_iteration=8))
    assert shared.pairing_cache_identity(a)==shared.pairing_cache_identity(b)


def test_resuming_runner_skips_only_successful_commands(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(subprocess,'run',lambda args,**kwargs:calls.append(args))
    path=tmp_path/'subject_pairings'/'abc'/'iteration_0'/'results'
    args=['python','analysis.py','--output-dir',str(path)]
    run(args,'full',2,4)
    run(args,'full',2,4)
    assert len(calls)==2


def test_ensemble_panel_reads_rank_spectra_and_current_snapshot(tmp_path,monkeypatch):
    import types
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from vs0_run_pairings import display_ensemble
    display=types.ModuleType('IPython.display');display.display=lambda value:None
    monkeypatch.setitem(sys.modules,'IPython.display',display)
    monkeypatch.setattr(plt,'show',lambda:plt.close('all'))
    for i in range(2):
        folder=tmp_path/f'iteration_{i}'
        folder.mkdir()
        (folder/'RUN_COMPLETE.json').write_text('{}')
        np.savez(folder/'spectrum.npz',explained_variance=np.array([2.,1.])+i,explained_variance_ratio=np.array([.6,.4]))
        evaluation=folder/'eval';snapshot=evaluation/'snapshot_abc';snapshot.mkdir(parents=True)
        (evaluation/'CURRENT.json').write_text(json.dumps(dict(snapshot='snapshot_abc')))
        pd.DataFrame(dict(k=[1],metric=[i*4])).to_csv(snapshot/'metrics.csv',index=False)
    results=display_ensemble(tmp_path,2,['spectrum.npz','eval/CURRENT/metrics.csv'])
    assert results['spectrum.npz']['summary'].loc[1,('explained_variance','mean')]==2.5
    assert results['eval/CURRENT/metrics.csv']['summary'].loc[1,('metric','mean')]==2


def test_trial_exporter_builds_full_pool_once_for_both_views(tmp_path,monkeypatch):
    import pickle
    import vs2_1_plssvd_eval_utils as utils
    meg=tmp_path/'meg';ieeg=tmp_path/'ieeg';cache=tmp_path/'cache'
    meg.mkdir();ieeg.mkdir()
    for i in range(4):(meg/f'M{i:02d}.mat').touch()
    times=np.arange(5)/250
    for subject in ['I2','I1']:
        (ieeg/f'{subject}_info.json').write_text(json.dumps(dict(event_id=[1,2,1,2,1,2],time_epoch=times.tolist())))
        with (ieeg/f'{subject}_epochs.p').open('wb') as f:pickle.dump(np.ones((6,1,5)),f)
    csv=tmp_path/'match.csv'
    pd.DataFrame(dict(ieeg_subject=['I2','I1'],meg_subject=['M00','M01'],path_to_meg=[str(meg/'M00.mat'),str(meg/'M01.mat')])).to_csv(csv,index=False)
    monkeypatch.setattr(shared,'MATCHED_PAIRS_CSV',csv)
    monkeypatch.setattr(shared,'load_meg_mat',lambda path:dict(time=times,pos_brainsources_MNI8=np.zeros((1,3))))
    monkeypatch.setattr(shared,'meg_condition_arrays',lambda out,n:[np.ones((1,5,3)) for _ in range(n)])
    metadata=pd.DataFrame(dict(subject=['I2','I1'],channel_index=[0,0],x=[0,0],y=[0,0],z=[0,0]))
    activate('matched',3)
    baseline=utils.prepare_trial_cache(meg,ieeg,cache,['M00','M01'],['I2','I1'],metadata)
    assert baseline.pairing=={'I2':'M00','I1':'M01'}
    assert len(baseline.meg)==4
    manifest=(cache/'manifest.json').read_text()
    activate('full',3)
    random=utils.prepare_trial_cache(meg,ieeg,cache,['M00'],['I2','I1'],metadata)
    assert len(random.meg)==4
    assert (cache/'manifest.json').read_text()==manifest
    assert random.pairing==shared.random_subject_pairing(['I2','I1'],['M00','M01','M02','M03'])


def test_coverage_stability_draws_outside_initial_meg_cohort():
    from vs1_coverage_stability import run_pairing_stability
    from vs1_coverage_matching_utils import Dataset
    rng=np.random.default_rng(22)
    metadata=pd.DataFrame(dict(subject=['I1','I2'],x=[0,10],y=[0,0],z=[0,0]))
    ieeg=rng.normal(size=(2,2,10))
    source=dict(ieeg=ieeg,electrode_subjects=np.array(['I1','I2']),electrode_metadata=metadata,
                electrode_positions=metadata[['x','y','z']].to_numpy(),
                meg_subjects=[f'M{i:02d}' for i in range(12)],
                meg=[rng.normal(size=(2,2,10)) for _ in range(12)],
                meg_positions=[np.array([[0,0,0],[10,0,0]]) for _ in range(12)])
    reference=Dataset('iEEG',[ieeg],metadata,np.arange(2),source_data=source)
    result=run_pairing_stability(reference,repeats=4,n_components=1)
    assignments=result['assignments']
    baseline=set(assignments.loc[assignments.repeat==-1,'meg_subject'])
    random=set(assignments.loc[assignments.repeat>=0,'meg_subject'])
    assert random-baseline
    assert len(result['config']['meg_subjects'])==12
