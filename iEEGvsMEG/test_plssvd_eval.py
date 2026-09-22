import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import json,pickle
import numpy as np
import pandas as pd
from plssvd_eval_utils import (TrialSubject,TrialData,ValidationOptions,_split_subject,_build_fold,_fit,
    _project,_predictor,_q2,_r,_permuted_contrast,prepare_trial_cache,validate_plssvd)


def synthetic_trials(seed=7):
    rng=np.random.default_rng(seed);t=np.arange(40)/100
    base=np.stack([np.sin(2*np.pi*4*t),np.cos(2*np.pi*7*t),np.sin(2*np.pi*2*t)])
    signal=np.stack([base.copy(),base.copy()]);signal[0,2]*=-1
    mix=rng.normal(size=(12,3));pos=rng.uniform(-60,60,(12,3))
    meg=[];ieeg=[]
    def subject(name,positions,signal,scale):
        arrays=[np.asarray([v+rng.normal(scale=.25,size=v.shape) for _ in range(20)])*scale for v in signal]
        meta=pd.DataFrame(positions,columns=['x','y','z']);meta['subject']=name;meta['region']='all';meta['channel_index']=range(len(positions))
        groups=[np.repeat(np.arange(10),2).astype(str).tolist() for _ in range(2)]
        blocks=[['all']*20 for _ in range(2)]
        return TrialSubject(name,(1,2),arrays,positions.copy(),meta,groups,blocks)
    for i in range(3):
        s=np.einsum('pf,cft->cpt',mix+rng.normal(scale=.05,size=mix.shape),signal)
        meg.append(subject(f'M{i}',pos,s,1.))
    for i,ix in enumerate([np.array([1,3,5,7]),np.array([2,4,6,8])]):
        s=np.einsum('pf,cft->cpt',mix[ix],signal)
        ieeg.append(subject(f'I{i}',pos[ix]+.01,s,.001))
    return TrialData(ieeg,meg,t,(1,2))


class Tests(unittest.TestCase):
    def test_trial_and_group_disjointness(self):
        data=synthetic_trials()
        for unit in ['trial','group']:
            s=data.meg[0];parts=_split_subject(s,np.random.default_rng(3),unit)
            for c,a in enumerate(s.data):
                all_ix=np.concatenate([parts[n][c] for n in ['train','tune','test_a','test_b']])
                self.assertEqual(sorted(all_ix),list(range(len(a))))
            if unit=='group':
                sets=[set(np.concatenate([np.asarray(g)[ix] for g,ix in zip(s.split_groups,parts[n])])) for n in ['train','tune','test_a','test_b']]
                for i in range(4):
                    for j in range(i):self.assertFalse(sets[i]&sets[j])
    def test_test_data_cannot_change_scaling_weights_or_tuning(self):
        data=synthetic_trials();rng=np.random.default_rng(4)
        indices={m:{s.subject:_split_subject(s,rng,'trial') for s in getattr(data,m)} for m in ['ieeg','meg']}
        options=ValidationOptions(candidates=(1,2,3))
        with TemporaryDirectory() as a,TemporaryDirectory() as b:
            f,sc,_,_=_build_fold(data,'paired_coverage',indices,a,9);model=_fit(f['train'],3,options)
            train=_project(f['train']['meg'],model,'meg')
            tune=_project(f['tune']['meg'],model,'meg')
            coef=_predictor(train,f['train']['ieeg'],model['ieeg_mean'],options.ridge)
            before=_q2(tune,f['tune']['ieeg'],model['ieeg_mean'],coef)
            for modality in ['ieeg','meg']:
                for subject in getattr(data,modality):
                    for c,array in enumerate(subject.data):array[indices[modality][subject.subject]['test'][c]]+=rng.normal(scale=100,size=array[indices[modality][subject.subject]['test'][c]].shape)
            g,sc2,_,_=_build_fold(data,'paired_coverage',indices,b,9);model2=_fit(g['train'],3,options)
            for key in ['ieeg_weights','meg_weights','ieeg_mean','meg_mean']:
                np.testing.assert_array_equal(model[key],model2[key])
            for modality in sc:
                for subject in sc[modality]:
                    for x,y in zip(sc[modality][subject],sc2[modality][subject]):np.testing.assert_array_equal(x,y)
            after=_q2(_project(g['tune']['meg'],model2,'meg'),g['tune']['ieeg'],model2['ieeg_mean'],coef)
            self.assertEqual(before,after)
    def test_all_five_kinds_and_null_trial_projection(self):
        # _null_tests asserts exact agreement between trial projections and the
        # observed pooled test contrast for each aggregation/matching scheme.
        for kind in ['full_average','full_concatenated','coverage_average','paired_coverage','random_control']:
            with self.subTest(kind=kind),TemporaryDirectory() as out:
                result=validate_plssvd(synthetic_trials(),kind,ValidationOptions(repeats=1,n_null=5,candidates=(1,2,3)),out)
                self.assertEqual(len(result['null_tests']),3)
                self.assertTrue(np.isfinite(result['summary'].test_mean_r).all())
                self.assertTrue((result['null_tests'].n_null==5).all())
                self.assertTrue(((result['null_tests'].tail_fraction>=1/6)&(result['null_tests'].tail_fraction<=1)).all())
                saved=np.load(Path(out)/'model_000.npz')
                self.assertEqual(saved['ieeg_weights'].shape[1],result['summary'].selected_k.iloc[0])
    def test_participant_resampling_and_reproducibility(self):
        with TemporaryDirectory() as a,TemporaryDirectory() as b:
            options=ValidationOptions(repeats=2,n_null=0,candidates=(1,2),subject_fraction=.5)
            ra=validate_plssvd(synthetic_trials(),'full_concatenated',options,a)
            rb=validate_plssvd(synthetic_trials(),'full_concatenated',options,b)
            pd.testing.assert_frame_equal(ra['summary'],rb['summary'])
            pd.testing.assert_frame_equal(ra['split_audit'],rb['split_audit'])
            self.assertEqual(ra['summary'].n_ieeg.tolist(),[2,1])
            self.assertEqual(ra['summary'].n_meg.tolist(),[3,2])
    def test_group_mode_end_to_end(self):
        with TemporaryDirectory() as out:
            result=validate_plssvd(synthetic_trials(),'coverage_average',ValidationOptions(repeats=1,n_null=3,candidates=(1,2),split_unit='group'),out)
            self.assertEqual(len(result['summary']),1)
    def test_nonexchangeable_condition_blocks_are_detected(self):
        values=np.arange(24).reshape(4,3,2);labels=np.array([0,0,1,1]);blocks=np.array(['a','a','b','b'])
        a,movable=_permuted_contrast([(values,labels,blocks)],np.random.default_rng(0))
        self.assertFalse(movable)
        np.testing.assert_array_equal(a,values[2:].mean(0)-values[:2].mean(0))
    def test_cache_original_layout_and_staleness(self):
        data=synthetic_trials()
        with TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir();idir=root/'ieeg';idir.mkdir()
            for s in data.ieeg:
                with (idir/f'{s.subject}_epochs.p').open('wb') as f:pickle.dump(np.concatenate(s.data),f)
                (idir/f'{s.subject}_info.json').write_text(json.dumps({'event_id':[1]*20+[2]*20,'time_epoch':data.times.tolist()}))
            for s in data.meg:(raw/f'{s.subject}_norm0_abs_0.mat').write_text('mock MAT file')
            def loadmat(path):
                s=next(s for s in data.meg if Path(path).name.startswith(s.subject+'_'))
                return {'OUT':{'sources_ERFs':[[a.transpose(1,2,0)] for a in s.data],
                               'pos_brainsources_MNI8':s.positions/1000,'time':data.times}}
            kwargs=dict(meg_raw_dir=raw,ieeg_dir=idir,cache_dir=root/'cache',
                meg_subjects=[s.subject for s in data.meg],ieeg_subjects=[s.subject for s in data.ieeg],
                electrode_metadata=pd.concat([s.metadata for s in data.ieeg]),ieeg_coordinate_unit='mm')
            with patch.dict('sys.modules',{'mat73':SimpleNamespace(loadmat=loadmat)}):
                cached=prepare_trial_cache(**kwargs)
            np.testing.assert_allclose(cached.meg[0].data[0],data.meg[0].data[0],rtol=1e-6)
            self.assertIsInstance(cached.meg[0].data[0],np.memmap)
            self.assertEqual(len(prepare_trial_cache(**kwargs).ieeg),2)
            (raw/'M0_norm0_abs_0.mat').write_text('source changed')
            with self.assertRaisesRegex(ValueError,'source file changed'):prepare_trial_cache(**kwargs)

if __name__=='__main__':unittest.main()
