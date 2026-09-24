import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from coverage_matching_utils import Dataset
from plssvd_eval_utils import (TrialData, TrialSubject, ValidationOptions, _split_train_test,
    _split_subject, _temporary_fold, _fit, _project, _predictor, _evaluate_fixed_model,
    validate_plssvd, load_plssvd_results, plot_plssvd_validation)


def trials():
    rng = np.random.default_rng(99)
    latent = rng.normal(size=(2, 3, 12))
    modalities = {}
    for modality, number, channels in [('ieeg', 2, 3), ('meg', 3, 6)]:
        subjects = []
        for i in range(number):
            signal = np.einsum('fk,ckt->cft', rng.normal(size=(channels, 3)), latent)
            arrays = [signal[c]+rng.normal(scale=.3, size=(24, channels, 12)) for c in range(2)]
            positions = np.column_stack([np.arange(channels)*5., np.zeros((channels, 2))])
            name = f'{modality}{i}'
            meta = pd.DataFrame(positions, columns=['x','y','z']); meta['subject'] = name
            subjects.append(TrialSubject(name, (1,2), arrays, positions, meta, [None,None], [['all']*24]*2))
        modalities[modality] = subjects
    return TrialData(modalities['ieeg'], modalities['meg'], np.arange(12)/250, (1,2))


class SplitTests(unittest.TestCase):
    def test_train_test_exhaustive_no_tuning_and_group_integrity(self):
        subject = trials().ieeg[0]
        a = _split_train_test(subject, np.random.default_rng(7), 'trial', .7)
        b = _split_train_test(subject, np.random.default_rng(8), 'trial', .7)
        self.assertNotIn('tune', a)
        for c in range(2):
            self.assertEqual(len(a['train'][c]), 16)
            self.assertFalse(set(a['train'][c]) & set(a['test'][c]))
            self.assertEqual(set(a['train'][c]) | set(a['test'][c]), set(range(24)))
            np.testing.assert_array_equal(np.sort(a['test'][c]), np.sort(np.r_[a['test_a'][c],a['test_b'][c]]))
        self.assertNotEqual(set(a['train'][0]), set(b['train'][0]))
        tiny = copy.deepcopy(subject); tiny.data = [x[:6] for x in tiny.data]
        small = _split_train_test(tiny, np.random.default_rng(1), 'trial', .7)
        self.assertEqual([len(small[p][0]) for p in ['train','test_a','test_b']], [2,2,2])
        subject.split_groups = [np.repeat(np.arange(6),4).tolist()]*2
        parts = _split_train_test(subject, np.random.default_rng(9), 'group', .7)
        group_sets = [set(np.array(subject.split_groups[0])[parts[p][0]]) for p in ['train','test_a','test_b']]
        for i in range(3):
            for j in range(i): self.assertFalse(group_sets[i] & group_sets[j])

    def test_existing_compare_subspace_tuning_path_unchanged(self):
        data = trials()
        indices = {m: {s.subject: _split_subject(s,np.random.default_rng(i),'trial') for i,s in enumerate(getattr(data,m))}
                   for m in ('ieeg','meg')}
        with _temporary_fold(data,'paired_coverage',indices,4) as prepared:
            self.assertEqual(set(prepared[0]), {'train','tune','test_a','test_b','test'})


class MetricTests(unittest.TestCase):
    def test_dense_covariance_and_reconstruction_reference(self):
        data = trials()
        indices = {m: {s.subject: _split_train_test(s,np.random.default_rng(i),'trial',.7) for i,s in enumerate(getattr(data,m))}
                   for m in ('ieeg','meg')}
        with _temporary_fold(data,'full_concatenated',indices,4) as prepared:
            fold = prepared[0]; opts = ValidationOptions(n_components=5,block_scaling='equal_variance')
            model = _fit(fold['train'],5,opts)
            scores = {p: {m: _project(d,model,m) for m,d in ds.items()} for p,ds in fold.items()}
            predictors = {m: _predictor(scores['train']['meg' if m=='ieeg' else 'ieeg'],fold['train'][m],model[m+'_mean'],opts.ridge)
                          for m in ('ieeg','meg')}
            metrics, cov = _evaluate_fixed_model(fold['test'],scores['test'],model,predictors)
            dense = {m: np.concatenate(list(fold['test'][m].blocks()),axis=1).astype(float) for m in ('ieeg','meg')}
            centered = {m: (x-x.mean(0))*model[m+'_scale'] for m,x in dense.items()}
            c = centered['ieeg'].T@centered['meg']/(len(centered['ieeg'])-1)
            expected = model['ieeg_weights'].T@c@model['meg_weights']
            np.testing.assert_allclose(cov,expected,rtol=1e-10,atol=1e-10)
            self.assertAlmostEqual(metrics['crosscov_energy_fraction'],np.sum(expected**2)/np.sum(c**2))
            for m,x in dense.items():
                x = x-model[m+'_mean']; w = model[m+'_weights']
                actual = 1-np.sum((x-x@w@w.T)**2)/np.sum(x*x)
                self.assertAlmostEqual(metrics[m+'_reconstruction_fraction'],actual)
            self.assertTrue(0 <= metrics['crosscov_energy_fraction'] <= 1)


class EvaluationTests(unittest.TestCase):
    def test_fixed_k_same_cohort_pairing_all_folds_exports_plots(self):
        data = trials()
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_plssvd(data,'paired_coverage',ValidationOptions(repeats=3,n_components=5,n_null=3),tmp)
            self.assertEqual(set(result['summary'].n_components), {5})
            self.assertEqual(set(result['participants'].groupby(['repeat','modality']).size()), {2,3})
            self.assertEqual(set(result['split_audit'].partition), {'train','test_a','test_b'})
            self.assertNotIn('selection',result)
            pairs = [json.loads((Path(tmp)/f'pairing_{i:03d}.json').read_text()) for i in range(3)]
            self.assertTrue(all(p==pairs[0] for p in pairs))
            audits = [pd.read_csv(Path(tmp)/f'matching_{i:03d}.csv') for i in range(3)]
            for a in audits[1:]: pd.testing.assert_frame_equal(audits[0],a)
            split = result['split_audit']
            for _, rows in split.groupby(['repeat','modality','subject','condition']):
                self.assertEqual(len(rows),24); self.assertEqual(rows.trial_index.nunique(),24)
            first = split.query("repeat == 0 and modality == 'ieeg' and subject == 'ieeg0' and condition == 1 and partition == 'train'")
            second = split.query("repeat == 1 and modality == 'ieeg' and subject == 'ieeg0' and condition == 1 and partition == 'train'")
            self.assertNotEqual(set(first.trial_index),set(second.trial_index))
            self.assertEqual(len(result['fold_stability']),3*2*2)
            loaded = load_plssvd_results(tmp)
            self.assertEqual(loaded['validation_options']['n_components'],5)
            self.assertNotIn('tune',loaded['primary_scores'])
            plot_plssvd_validation(loaded,output_dir=tmp,show=False)
            for name in ['heldout_model_goodness','heldout_crosscovariance','fold_performance_consistency','fold_temporal_stability','primary_crosscovariance']:
                self.assertTrue((Path(tmp)/(name+'.png')).is_file())
            with self.assertRaises(FileExistsError): validate_plssvd(data,'paired_coverage',output_dir=tmp)

    def test_test_only_perturbation_cannot_change_trained_parameters(self):
        data = trials(); options = ValidationOptions(repeats=1,n_components=3,n_null=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = validate_plssvd(data,'full_concatenated',options,root/'a')
            changed = copy.deepcopy(data)
            audit = result['split_audit']
            for m in ('ieeg','meg'):
                for subject in getattr(changed,m):
                    for ci,c in enumerate(data.conditions):
                        rows = audit[(audit.modality==m)&(audit.subject==subject.subject)&(audit.condition==c)&(audit.partition!='train')]
                        subject.data[ci][rows.trial_index.to_numpy(int)] *= -4
            other = validate_plssvd(changed,'full_concatenated',options,root/'b')
            with np.load(root/'a'/'model_000.npz') as a, np.load(root/'b'/'model_000.npz') as b:
                for key in ['ieeg_weights','meg_weights','ieeg_mean','meg_mean','predict_ieeg','predict_meg']:
                    np.testing.assert_array_equal(a[key],b[key])
            self.assertNotEqual(result['summary'].predict_ieeg_q2.iloc[0],other['summary'].predict_ieeg_q2.iloc[0])
            self.assertTrue(other['fold_stability'].empty)

    def test_rank_shortfall_fails_instead_of_selecting_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'fixed n_components=50'):
                validate_plssvd(trials(),'paired_coverage',ValidationOptions(repeats=1,n_components=50,n_null=0),tmp)
            self.assertFalse((Path(tmp)/'COMPLETE.json').exists())


if __name__ == '__main__': unittest.main()
