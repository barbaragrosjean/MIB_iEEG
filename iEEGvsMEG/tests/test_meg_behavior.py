import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import vs0_load_and_config as shared
import vs0_meg_behavior as behavior
from vs2_1_plssvd_eval_utils import load_trial_cache


@pytest.fixture
def behavioral_data(tmp_path, monkeypatch):
    directory=tmp_path/'behavior';directory.mkdir()
    monkeypatch.setenv('MEG_BEHAVIOR_DIR',str(directory))
    for key in ('SUBJECT_PAIRING_ITERATION','SUBJECT_PAIRING_SEED','SUBJECT_PAIRING_VIEW','SUBJECT_PAIRING_COUNT'):
        monkeypatch.delenv(key, raising=False)
    for subject in range(1,6):
        for block in range(3,7):
            if subject==1:stimuli,responses=['OLD.wav','new.wav'],[1,2]
            elif subject==2:stimuli,responses=['old.wav','old.wav','new.wav','new.wav'],[2,2,1,2]
            elif subject==3:stimuli,responses=['old.wav','new.wav'],[1,1]
            elif subject==4:stimuli,responses=['old.wav','new.wav','old.wav','new.wav'],[None,0,'invalid',2]
            else:
                stimuli=['old.wav']*(8 if block==3 else 2)
                responses=[1 if block==3 else 2]*len(stimuli)
            frame=pd.DataFrame(dict(subj=[f'{subject:04d}']*len(stimuli),trial=stimuli,response=responses))
            # Non-old/new stimuli and export footers must not enter the denominator.
            frame.loc[len(frame)]=[f'{subject:04d}','distractor.wav',1]
            frame.loc[len(frame)]=['encoding ',None,None]
            frame.to_csv(directory/f'Subj_{subject:04d}_Block_{block}.csv',sep=';',index=False)
    return directory


def test_scoring_boundary_invalid_responses_and_trial_weighting(behavioral_data):
    result=behavior.meg_behavior_performance().set_index('subject')
    assert result.loc['1','accuracy']==1
    assert result.loc['2','accuracy']==.25
    assert result.loc['3','accuracy']==.5 and result.loc['3','eligible']
    assert result.loc['4','accuracy']==.25
    assert result.loc['5','accuracy']==8/14 and result.loc['5','eligible']
    assert result.loc['1','n_trials']==8
    assert result.loc['4','n_trials']==16


def test_id_aliases_and_demographic_filter(behavioral_data):
    ids=['SUBJ_0001','2','0003','4.0','Subj_0005']
    assert behavior.eligible_meg_subjects(ids)==['SUBJ_0001','0003','Subj_0005']
    frame=pd.DataFrame(dict(subject=[1.,2.,3.],age=[20,21,22]))
    assert behavior.filter_meg_demographics(frame).subject.tolist()==[1.,3.]
    with pytest.raises(ValueError,match='Missing MEG behavioral data'):
        behavior.eligible_meg_subjects(['9'])


def test_matched_and_random_filters_preserve_same_ieeg_cohort(behavioral_data,tmp_path,monkeypatch):
    meg=tmp_path/'meg';meg.mkdir()
    for i in range(1,6):(meg/f'SUBJ_{i:04d}_source.p').touch()
    csv=tmp_path/'matched.csv'
    pd.DataFrame(dict(ieeg_subject=['I1','I2','I3'],meg_subject=['SUBJ_0001','SUBJ_0002','SUBJ_0003'],path_to_meg=[str(meg/f'SUBJ_{i:04d}_source.p') for i in [1,2,3]])).to_csv(csv,index=False)
    monkeypatch.setattr(shared,'MATCHED_PAIRS_CSV',csv)
    monkeypatch.setenv('SUBJECT_PAIRING_ITERATION','-1')
    with pytest.warns(UserWarning,match='below 50%'):
        ids,ieeg,_,pairing=shared.select_subjects(meg,['I1','I2','I3'])
    assert ids==['SUBJ_0001','SUBJ_0003'] and ieeg==['I1','I3']
    monkeypatch.setenv('SUBJECT_PAIRING_ITERATION','0')
    with pytest.warns(UserWarning):
        ids,random_ieeg,_,pairing=shared.select_subjects(meg,['I1','I2','I3'])
    assert set(ids)=={'SUBJ_0001','SUBJ_0003','SUBJ_0005'}
    assert random_ieeg==ieeg and set(pairing.values())<=set(ids)
    assert set(shared.random_subject_pairing(['I1','I3'],['1','2','3','4','5']).values())<={'1','3','5'}


def test_existing_trial_cache_exclusions_before_loading_arrays(behavioral_data,tmp_path,monkeypatch):
    records=[]
    for modality,ids in [('ieeg',['I1','I2','I3']),('meg',['SUBJ_0001','SUBJ_0002','SUBJ_0003','SUBJ_0004','SUBJ_0005'])]:
        for subject in ids:
            filename=f'{subject}.npy'
            if subject not in ('SUBJ_0002','SUBJ_0004'):np.save(tmp_path/filename,np.zeros((3,1,5)))
            records.append(dict(modality=modality,subject=subject,files=[filename,filename],positions=[[0,0,0]],metadata={'subject':[subject]},split_groups=[None,None],permutation_blocks=[None,None]))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(config=dict(conditions=[1,2],full_meg_pool=True),records=records,times=list(range(5)),source_files=[])))
    trial=load_trial_cache(tmp_path,use_matched_pairs=False)
    assert [s.subject for s in trial.meg]==['SUBJ_0001','SUBJ_0003','SUBJ_0005']
    csv=tmp_path/'matched.csv'
    paths=[str(tmp_path/f'SUBJ_{i:04d}.mat') for i in (1,2,3)]
    pd.DataFrame(dict(ieeg_subject=['I1','I2','I3'],meg_subject=['1','2','3'],path_to_meg=paths)).to_csv(csv,index=False)
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    manifest['source_files']=[dict(path=p) for p in paths]
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.warns(UserWarning,match='below 50%'):
        trial=load_trial_cache(tmp_path,use_matched_pairs=True,matched_pairs_csv=csv)
    assert trial.pairing=={'I1':'SUBJ_0001','I3':'SUBJ_0003'}
    assert [s.subject for s in trial.ieeg]==['I1','I3']



def test_behavior_changes_invalidate_provenance_and_output_namespaces(behavioral_data,tmp_path,monkeypatch):
    csv=tmp_path/'match.csv';csv.write_text('content')
    monkeypatch.setenv('SUBJECT_PAIRING_ITERATION','0')
    old_provenance=behavior.behavior_provenance()
    old_root=shared.analysis_output_root(tmp_path,matched_pairs_csv=csv)
    old_cache=shared.trial_cache_directory(tmp_path)
    path=behavioral_data/'Subj_0001_Block_3.csv'
    table=pd.read_csv(path,sep=';');table.loc[0,'response']=2;table.to_csv(path,sep=';',index=False)
    assert behavior.behavior_provenance()!=old_provenance
    assert shared.analysis_output_root(tmp_path,matched_pairs_csv=csv)!=old_root
    assert shared.trial_cache_directory(tmp_path)!=old_cache
    assert behavior.meg_behavior_performance().set_index('subject').loc['1','accuracy']==7/8


def test_ambiguous_stimuli_fail_and_available_blocks_are_scored(behavioral_data):
    path=behavioral_data/'Subj_0001_Block_3.csv'
    table=pd.read_csv(path,sep=';');table.loc[0,'trial']='old_new.wav';table.to_csv(path,sep=';',index=False)
    with pytest.raises(ValueError,match='both old and new'):behavior.meg_behavior_performance()
    table.loc[0,'trial']='old.wav';table.to_csv(path,sep=';',index=False)
    path.unlink()
    assert behavior.meg_behavior_performance().set_index('subject').loc['1','n_blocks']==3


def test_prepare_filters_raw_arrays_and_saves_audit(behavioral_data,tmp_path):
    from vs0_prepare_data import prepare_data,load_prepared_data
    meg=tmp_path/'meg';ieeg=tmp_path/'ieeg';meg.mkdir();ieeg.mkdir()
    rng=np.random.default_rng(2)
    for i in range(1,6):
        with (meg/f'SUBJ_{i:04d}_source.p').open('wb') as f:pickle.dump(rng.normal(size=(2,2,5)),f)
        pd.DataFrame(dict(x=[0,10],y=[0,0],z=[0,0])).to_csv(meg/f'SUBJ_{i:04d}_pos.csv',index=False)
    for subject in ['I1','I2']:
        with (ieeg/f'{subject}_epochs.p').open('wb') as f:pickle.dump(rng.normal(size=(4,1,5)),f)
        (ieeg/f'{subject}_info.json').write_text(json.dumps(dict(event_id=[1,2,1,2],time_epoch=(np.arange(5)/250).tolist())))
    options=dict(meg_dir=meg,ieeg_dir=ieeg,meg_coordinate_unit='mm',use_matched_pairs=False,meg_tmin=0,
                 electrode_metadata=pd.DataFrame(dict(subject=['I1','I2'],channel_index=[0,0],x=[0,10],y=[0,0],z=[0,0])),prepared_cache_dir=tmp_path/'prepared')
    folder=prepare_data(**options)
    data=load_prepared_data(**options)
    assert data['meg_subjects']==['SUBJ_0001','SUBJ_0003','SUBJ_0005']
    audit=pd.read_csv(folder/'meg_behavior_eligibility.csv')
    assert len(audit)==5 and audit.eligible.sum()==3
    assert 'meg_behavior' in json.loads((folder/'manifest.json').read_text())['request']


def _write_behavior_xlsx(path, *, strict=True):
    """Small fixture covering shared strings, inline strings and sparse cells."""
    import zipfile
    from xml.sax.saxutils import escape
    namespace=('http://purl.oclc.org/ooxml/spreadsheetml/main' if strict else
               'http://schemas.openxmlformats.org/spreadsheetml/2006/main')
    relationship='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    strings=['subj','response','0002bis2','old.wav','new.wav']
    shared='<sst xmlns="'+namespace+'">'+''.join('<si><t>'+escape(v)+'</t></si>' for v in strings)+'</sst>'
    worksheet=f'''<worksheet xmlns="{namespace}"><sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>trial</t></is></c><c r="C1" t="s"><v>1</v></c></row>
    <row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2" t="s"><v>3</v></c><c r="C2"><v>1</v></c></row>
    <row r="3"><c r="A3"><v>2</v></c><c r="B3" t="s"><v>4</v></c></row>
    <row r="4"><c r="A4"><v>2</v></c><c r="B4" t="s"><v>4</v></c><c r="C4"><v>2</v></c></row>
    </sheetData></worksheet>'''
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('xl/sharedStrings.xml',shared)
        z.writestr('xl/worksheets/sheet1.xml',worksheet)
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{namespace}" xmlns:r="{relationship}"><sheets><sheet name="Trials" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')


@pytest.mark.parametrize('strict',[True,False])
def test_xlsx_formats_retake_aliases_and_sparse_responses(tmp_path,monkeypatch,strict):
    monkeypatch.setenv('MEG_BEHAVIOR_DIR',str(tmp_path))
    _write_behavior_xlsx(tmp_path/'Subj_0002bis_Block_3.xlsx',strict=strict)
    (tmp_path/'._Subj_0002bis_Block_3.xlsx').write_text('not an Excel workbook')
    (tmp_path/'Subj_0002bis_Project_PD.xlsx').write_text('different task: do not score')
    (tmp_path/'~$Subj_0002bis_Block_4.xlsx').write_text('temporary lock file')
    result=behavior.meg_behavior_performance().set_index('subject')
    assert result.loc['2','n_trials']==3
    assert result.loc['2','n_correct']==2
    assert result.loc['2','n_blocks']==1
    assert result.loc['2','accuracy']==2/3
    assert behavior.eligible_meg_subjects(['SUBJ_0002','2','0002bis2'])==['SUBJ_0002','2','0002bis2']


def test_duplicate_retake_files_rejected(tmp_path,monkeypatch):
    monkeypatch.setenv('MEG_BEHAVIOR_DIR',str(tmp_path))
    _write_behavior_xlsx(tmp_path/'Subj_0002bis_Block_3.xlsx')
    _write_behavior_xlsx(tmp_path/'Subj_0002_Block_3.xlsx')
    with pytest.raises(ValueError,match='Duplicate behavioral files'):
        behavior.meg_behavior_performance()


def test_default_behavior_directory_is_final_xlsx(monkeypatch):
    monkeypatch.delenv('MEG_BEHAVIOR_DIR',raising=False)
    assert behavior.behavior_directory().name=='Final_xlsx'


def test_coverage_against_provided_roster_reports_missing_subjects_and_blocks(behavioral_data):
    (behavioral_data/'Subj_0001_Block_3.csv').unlink()
    report=behavior.meg_behavior_coverage(['1','SUBJ_0002','9']).set_index('subject')
    assert report.loc['1','has_behavior']
    assert report.loc['1','missing_blocks']=='3'
    assert not report.loc['9','has_behavior']
    assert report.loc['9','missing_blocks']=='3,4,5,6'
    assert set(report.reference)=={'provided_roster'}
