"""Participant discovery must ignore group-level MATLAB outputs before screening."""
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import vs0_load_and_config as shared


class RecordingDiscoveryTests(unittest.TestCase):
    def test_auxiliary_files_never_enter_behavioral_screening(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('SUBJ_0001_norm0_abs_0.mat','subj_0002bis_norm0_abs_0.mat',
                         'DCMfinal_ROIsXtimeXtrXsubsXconds.mat','grand_average_source.p',
                         'SUBJ_0002_notes.mat','123_summary.mat','._SUBJ_0001_norm0_abs_0.mat'):
                (root/name).touch()
            # A directory with a recording-like name is not a recording either.
            (root/'SUBJ_0003_norm0_abs_0.mat').mkdir()
            def screened(subjects):
                ids=list(subjects)
                self.assertEqual(ids,['SUBJ_0001','subj_0002bis'])
                return ids
            with patch.object(shared,'eligible_meg_subjects',side_effect=screened) as check:
                files=shared.meg_recordings(root)
            self.assertEqual(set(files),{'SUBJ_0001','subj_0002bis'})
            self.assertEqual(check.call_count,1)

    def test_legacy_pickle_and_numeric_recordings_remain_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('SUBJ_0001_source.p','0002.mat','3bis2_source.p'):(root/name).touch()
            with patch.object(shared,'eligible_meg_subjects',side_effect=lambda ids:list(ids)):
                self.assertEqual(set(shared.meg_recordings(root)),{'SUBJ_0001','0002','3bis2'})

    def test_duplicate_participant_recordings_still_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('SUBJ_0001_source.p','SUBJ_0001_norm0_abs_0.mat'):(root/name).touch()
            with self.assertRaisesRegex(ValueError,'Ambiguous MEG files'):
                shared.meg_recordings(root)

    def test_full_preparation_and_reload_with_auxiliary_matlab_file(self):
        from vs0_prepare_data import prepare_data,load_prepared_data
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);meg=root/'meg';ieeg=root/'ieeg';behavior=root/'behavior'
            for folder in (meg,ieeg,behavior):folder.mkdir()
            rng=np.random.default_rng(8)
            for subject in (1,2):
                with (meg/f'SUBJ_{subject:04d}_source.p').open('wb') as stream:
                    pickle.dump(rng.normal(size=(2,2,9)),stream)
                pd.DataFrame(dict(x=[0,10],y=[0,0],z=[0,0])).to_csv(meg/f'SUBJ_{subject:04d}_pos.csv',index=False)
                pd.DataFrame(dict(subj=[subject]*2,trial=['old.wav','new.wav'],response=[1,2])).to_csv(behavior/f'Subj_{subject:04d}_Block_3.csv',index=False)
            (meg/'DCMfinal_ROIsXtimeXtrXsubsXconds.mat').write_text('aggregate output: not a participant recording')
            with (ieeg/'I1_epochs.p').open('wb') as stream:pickle.dump(rng.normal(size=(4,1,9)),stream)
            (ieeg/'I1_info.json').write_text(json.dumps(dict(event_id=[1,2,1,2],time_epoch=(np.arange(9)/250).tolist())))
            csv=root/'matched.csv'
            pd.DataFrame(dict(ieeg_subject=['I1'],meg_subject=['1'],path_to_meg=[str(meg/'SUBJ_0001_source.p')])).to_csv(csv,index=False)
            options=dict(meg_dir=meg,ieeg_dir=ieeg,matched_pairs_csv=csv,meg_coordinate_unit='mm',
                         meg_tmin=0,prepared_cache_dir=root/'prepared',
                         electrode_metadata=pd.DataFrame(dict(subject=['I1'],channel_index=[0],x=[0],y=[0],z=[0])))
            with patch.dict(os.environ,dict(MEG_BEHAVIOR_DIR=str(behavior),SUBJECT_PAIRING_ITERATION='0',SUBJECT_PAIRING_SEED='77')):
                prepared=prepare_data(**options)
                data=load_prepared_data(**options)
            self.assertEqual(data['meg_subjects'],['SUBJ_0001','SUBJ_0002'])
            self.assertNotIn('DCMfinal',json.dumps(json.loads((prepared/'manifest.json').read_text())))
            self.assertEqual(len(data['pairing']),1)


if __name__=='__main__':
    unittest.main()
