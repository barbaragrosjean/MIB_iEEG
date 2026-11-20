import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

import vs0_load_and_config as shared
from vs0_meg_behavior import eligible_meg_subjects, filter_meg_demographics, meg_behavior_performance
from vs0_prepare_data import prepare_data, load_prepared_data
from vs2_1_plssvd_eval_utils import load_trial_cache


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    behavior=tmp_path/'behavior';behavior.mkdir()
    monkeypatch.setenv('MEG_BEHAVIOR_DIR',str(behavior))
    for key in ('SUBJECT_PAIRING_ITERATION','SUBJECT_PAIRING_SEED','SUBJECT_PAIRING_VIEW','SUBJECT_PAIRING_COUNT'):
        monkeypatch.delenv(key,raising=False)
    for subject,responses in [(1,[1,2]),(2,[2,1]),(4,[1,1])]:
        pd.DataFrame(dict(subj=[subject]*2,trial=['old.wav','new.wav'],response=responses)).to_csv(behavior/f'Subj_{subject:04d}_Block_3.csv',index=False)
    # Participant 3 deliberately has no behavioral file.
    meg=tmp_path/'meg';ieeg=tmp_path/'ieeg';meg.mkdir();ieeg.mkdir()
    rng=np.random.default_rng(6)
    for subject in range(1,5):
        with (meg/f'SUBJ_{subject:04d}_source.p').open('wb') as f:pickle.dump(rng.normal(size=(2,2,5)),f)
        pd.DataFrame(dict(x=[0,10],y=[0,0],z=[0,0])).to_csv(meg/f'SUBJ_{subject:04d}_pos.csv',index=False)
    for subject in ['I1','I2']:
        with (ieeg/f'{subject}_epochs.p').open('wb') as f:pickle.dump(rng.normal(size=(4,1,5)),f)
        (ieeg/f'{subject}_info.json').write_text(json.dumps(dict(event_id=[1,2,1,2],time_epoch=(np.arange(5)/250).tolist())))
    csv=tmp_path/'matched.csv'
    pd.DataFrame(dict(ieeg_subject=['I1','I2'],meg_subject=['SUBJ_0001','SUBJ_0003'],path_to_meg=[str(meg/f'SUBJ_{i:04d}_source.p') for i in (1,3)])).to_csv(csv,index=False)
    monkeypatch.setattr(shared,'MATCHED_PAIRS_CSV',csv)
    return tmp_path,meg,ieeg,behavior


def test_missing_and_low_performance_excluded_boundary_retained(cohort):
    with pytest.warns(UserWarning,match='missing behavioral performance: 3'):
        assert eligible_meg_subjects(['SUBJ_0001','SUBJ_0002','SUBJ_0003','SUBJ_0004'])==['SUBJ_0001','SUBJ_0004']
    with pytest.warns(UserWarning):
        result=filter_meg_demographics(pd.DataFrame(dict(subject=[1,2,3,4])))
    assert result.subject.tolist()==[1,4]


def test_full_prepare_and_reload_exclude_missing_behavior(cohort,monkeypatch):
    tmp,meg,ieeg,_=cohort
    monkeypatch.setenv('SUBJECT_PAIRING_ITERATION','0')
    options=dict(meg_dir=meg,ieeg_dir=ieeg,meg_coordinate_unit='mm',meg_tmin=0,
                 prepared_cache_dir=tmp/'prepared',
                 electrode_metadata=pd.DataFrame(dict(subject=['I1','I2'],channel_index=[0,0],x=[0,10],y=[0,0],z=[0,0])))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        folder=prepare_data(**options)
        data=load_prepared_data(**options)
    assert any('missing behavioral performance' in str(w.message) for w in caught)
    assert data['meg_subjects']==['SUBJ_0001','SUBJ_0004']
    assert data['ieeg_subjects']==['I1']
    assert set(data['pairing'].values())<=set(data['meg_subjects'])
    assert json.loads((folder/'manifest.json').read_text())['request']['meg_behavior']['missing_behavior']=='exclude'


def test_saved_match_excludes_missing_behavior(cohort,monkeypatch):
    _,meg,_,_=cohort
    monkeypatch.setenv('SUBJECT_PAIRING_ITERATION','-1')
    with pytest.warns(UserWarning):
        ids,ieeg,_,pairing=shared.select_subjects(meg,['I1','I2'])
    assert ids==['SUBJ_0001'] and ieeg==['I1'] and pairing=={'I1':'SUBJ_0001'}


def test_cached_missing_subject_skipped_before_array_loading(cohort):
    tmp,_,_,_=cohort
    cache=tmp/'cache';cache.mkdir();records=[]
    for modality,subjects in [('ieeg',['I1']),('meg',['SUBJ_0001','SUBJ_0002','SUBJ_0003','SUBJ_0004'])]:
        for subject in subjects:
            filename=f'{subject}.npy'
            if subject not in ('SUBJ_0002','SUBJ_0003'):np.save(cache/filename,np.ones((3,1,5)))
            records.append(dict(modality=modality,subject=subject,files=[filename]*2,positions=[[0,0,0]],metadata={'subject':[subject]},split_groups=[None,None],permutation_blocks=[None,None]))
    (cache/'manifest.json').write_text(json.dumps(dict(config=dict(conditions=[1,2]),records=records,times=list(range(5)),source_files=[])))
    with pytest.warns(UserWarning,match='missing behavioral performance'):
        trials=load_trial_cache(cache,use_matched_pairs=False)
    assert [s.subject for s in trials.meg]==['SUBJ_0001','SUBJ_0004']


def test_no_scorable_trials_are_ineligible_instead_of_blocking_others(cohort):
    _,_,_,behavior=cohort
    pd.DataFrame(dict(subj=[3],trial=['distractor.wav'],response=[1])).to_csv(behavior/'Subj_0003_Block_3.csv',index=False)
    performance=meg_behavior_performance().set_index('subject')
    assert performance.loc['3','n_trials']==0
    assert pd.isna(performance.loc['3','accuracy'])
    assert not performance.loc['3','eligible']
    assert eligible_meg_subjects(['1','3','4'])==['1','4']
