"""Adapted from LB/LB5_decodingCluster.py mainTS and LB/src/decoding.py.

submit_to_cluster -n 2 -q short.q "python decoding.py --pc_use 0 --model LR --meg-kind paired_coverage --iteration 20"  --project MINDLAB2021_MEG-TempSeqAges

"""
import os
import json
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / 'out' / 'trial_cache'
OUT_PATH = ROOT / 'out' / 'decoding'
MEG_KIND = 'paired_coverage'
BLOCK_SCALING = 'equal_variance'

def select_model(model_name) : 
    classifiers = {
        'LR': LogisticRegression(max_iter=1000),
        'SVC_linear': SVC(kernel='linear'),
        'SVC_rbf': SVC(kernel='rbf',  probability=True),
        'RandomForest': RandomForestClassifier(oob_score=True),
    }

    param_grids = {
        'LR': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear']},
        'SVC_linear': {'C': [0.01, 0.1, 1, 10]},
        'SVC_rbf': {'C': [0.01, 0.1, 1, 10], 'gamma': ['scale', 'auto']},
        'RandomForest': {'n_estimators': [50, 100], 'max_depth': [None, 5, 10], 'max_features': ['sqrt', 'log2']},
    }

    return classifiers[model_name], param_grids[model_name]

def DataAugmentation(TFRtr, event_ids, data_aug_method='mean', nb_trials=23) :
    # Adapted from src.decomposition.DataAugmentation; applied after linear projection.
    # This avoids materializing trials across all concatenated MEG sources.
    if data_aug_method not in ['mean', 'duplicat'] :
        raise ValueError('Select mean or duplicat augmentation.')
    samples = []
    truth = []
    for ids in event_ids :
        ids = list(ids)
        if not ids :
            raise ValueError('Each condition needs at least one training trial.')
        selected = random.sample(ids, min(nb_trials, len(ids)))
        data = TFRtr[selected].copy()
        true_trials = np.arange(nb_trials) < len(selected)
        missing = nb_trials - len(selected)
        if missing > 0 :
            if data_aug_method == 'mean' :
                extra = np.repeat(data.mean(axis=0, keepdims=True), missing, axis=0)
            else :
                extra = TFRtr[random.choices(ids, k=missing)]
            data = np.concatenate([data, extra], axis=0)
        samples.append(data)
        truth.append(true_trials)
    return np.concatenate(samples, axis=0), np.stack(truth, axis=1)

def decodingTS(method_pls, data_aug_method, iteration=100, PC_use=0, save=True,
               out_path=OUT_PATH, iter_perm=50, cache_dir=CACHE_DIR, model_name='LR',
               crop_arg=None, n_tr=24, meg_kind=MEG_KIND, block_scaling=BLOCK_SCALING,
               seed=2026, classifier_params=None) :
    from plssvd_eval_utils import (load_trial_cache, ValidationOptions,
        _temporary_fold, _fit, _raw_weights, _project)

    if method_pls not in ['mean', 'concat'] :
        raise ValueError('method_pls must be mean or concat.')
    if min(iteration, n_tr-1) < 1 or iter_perm < 0 or PC_use < 0 :
        raise ValueError('Invalid iteration, trial, permutation or component count.')
    random.seed(seed)
    rng = np.random.default_rng(seed)
    trials = load_trial_cache(cache_dir)
    if len(trials.conditions) != 2 :
        raise ValueError('Exactly two conditions are required.')
    if any(len(a)<2 for m in ['ieeg','meg'] for s in getattr(trials,m) for a in s.data) :
        raise ValueError('Need at least two trials per participant and condition.')
    crop_arg = crop_arg or {'crop':False, 't_id_min':None, 't_id_max':None}
    time_ids = np.arange(len(trials.times))
    if crop_arg['crop'] :
        time_ids = time_ids[crop_arg['t_id_min']:crop_arg['t_id_max']]
    if len(time_ids)==0 :
        raise ValueError('Empty decoding time window.')

    # SELECT MODEL: fixed settings; no CV on overlapping augmented pseudo-trials.
    clf, param_grid = select_model(model_name)
    clf.set_params(random_state=seed)
    if classifier_params :
        clf.set_params(**classifier_params)
    options = ValidationOptions(block_scaling=block_scaling)
    out_path = Path(out_path)
    if save :
        out_path.mkdir(parents=True, exist_ok=True)
        if any(out_path.iterdir()) :
            raise FileExistsError('Choose a new decoding output directory.')
    metrics = []
    predictions = []
    split_audit = []
    matching_seed = int(rng.integers(2**31-1))

    for i in range(iteration) :
        # PREP DATA: one test trial per condition and participant; no paired trials assumed.
        indices = {}
        for modality in ['ieeg', 'meg'] :
            indices[modality] = {}
            for subject in getattr(trials, modality) :
                train_ids = []
                test_ids = []
                for condition, array in zip(trials.conditions, subject.data) :
                    ids = rng.permutation(len(array))
                    test_ids.append(ids[:1])
                    train_ids.append(ids[1:])
                    for part, values in [('train',ids[1:]),('test',ids[:1])] :
                        split_audit.extend(dict(iteration=i, modality=modality,
                            subject=subject.subject, condition=condition,
                            partition=part, trial_index=int(j)) for j in values)
                indices[modality][subject.subject] = {'train':train_ids, 'test':test_ids}

        with _temporary_fold(trials, meg_kind, indices, matching_seed,
                             condition_mode='average' if method_pls=='mean' else 'stack') as prepared :
            fold, scalers, audit, pairing = prepared
            model = _fit(fold['train'], PC_use+1, options)
            if model['k_max'] <= PC_use :
                raise ValueError('Requested PLS component exceeds training rank.')
            if save :
                np.savez_compressed(out_path / f'PLS_weights_{i:03d}.npz', **model)
                np.savez_compressed(out_path / f'preprocessing_{i:03d}.npz',
                    **{f'{m}_{subject}_{label}':value for m,subjects in scalers.items()
                       for subject,params in subjects.items()
                       for label,value in zip(['mean','std','multiplier'],params)})
                audit.to_csv(out_path / f'matching_{i:03d}.csv', index=False)
                for modality in ['ieeg','meg'] :
                    fold['train'][modality].metadata.to_csv(out_path / f'{modality}_features_{i:03d}.csv', index=False)

            for modality in ['ieeg', 'meg'] :
                weights = _raw_weights(trials, fold, model, scalers, audit,
                                       meg_kind, modality, PC_use+1)
                Train_all = np.zeros((2*(n_tr-1), len(trials.times)))
                Test_all = np.zeros((2, len(trials.times)))
                raw_test = np.zeros_like(Test_all)
                for subject in getattr(trials, modality) :
                    parts = indices[modality][subject.subject]
                    # PROJECT before augmentation: linear averaging commutes with projection.
                    train_samples = []
                    test_samples = []
                    for array, train_id, test_id in zip(subject.data, parts['train'], parts['test']) :
                        # Read one trial at a time to bound memory use.
                        train_samples.append(np.stack([weights[subject.subject][:,PC_use] @ array[j] for j in train_id]))
                        test_samples.append(weights[subject.subject][:,PC_use] @ array[test_id[0]])
                    raw_test += np.stack(test_samples)
                    # Training-only per-participant temporal mean; cancels affine projection offsets.
                    center = np.concatenate(train_samples).mean(axis=0)
                    joined = np.concatenate(train_samples)
                    events = [np.arange(len(train_samples[0])),
                              np.arange(len(train_samples[0]), len(joined))]
                    augmented, true_trials = DataAugmentation(joined, events, data_aug_method, n_tr-1)
                    for condition in range(2) :
                        values = augmented[condition*(n_tr-1):(condition+1)*(n_tr-1)]
                        Train_all[condition*(n_tr-1):(condition+1)*(n_tr-1)] += values[rng.permutation(n_tr-1)] - center
                    Test_all += np.stack(test_samples) - center
                # Verify raw-space mapping against the fixed fitted feature-space projection.
                expected = _project(fold['test'][modality], model, modality)[:,PC_use]
                actual = raw_test.mean(axis=0) if method_pls=='mean' else raw_test.reshape(-1)
                # Affine preprocessing contributes only a time-invariant offset.
                if not np.allclose(actual-actual.mean(), expected-expected.mean(), rtol=1e-4, atol=1e-4*max(1.,np.max(abs(expected)))) :
                    raise ValueError('Trial projection disagrees with fitted PLS mapping.')
                Train_transformed = Train_all[:,time_ids]
                Test_transformed = Test_all[:,time_ids]
                y_train = np.repeat([0,1], n_tr-1)
                y_test = np.array([0,1])

                # NORMAL DECODING AND CONDITIONAL SHUFFLED-CLASSIFIER CONTROL
                for j in range(-1, iter_perm) :
                    labels = y_train if j==-1 else rng.permutation(y_train)
                    model_clf = clone(clf)
                    model_clf.fit(Train_transformed, labels)
                    y_pred = model_clf.predict(Test_transformed)
                    metrics.append(dict(iteration=i, modality=modality, permutation=j,
                        accuracy=accuracy_score(y_test,y_pred)))
                    predictions.extend(dict(iteration=i,modality=modality,permutation=j,
                        y_test=int(a),y_pred=int(b)) for a,b in zip(y_test,y_pred))
        if save :
            pd.DataFrame(metrics).to_csv(out_path/'metrics.csv',index=False)
            pd.DataFrame(predictions).to_csv(out_path/'predictions.csv',index=False)
            pd.DataFrame(split_audit).to_csv(out_path/'split_audit.csv',index=False)
        print(f'Decoding iteration {i+1}/{iteration}',flush=True)
    if save :
        (out_path/'settings.json').write_text(json.dumps(dict(method_pls=method_pls,
            data_aug_method=data_aug_method,iteration=iteration,PC_use=PC_use,
            iter_perm=iter_perm,n_tr=n_tr,meg_kind=meg_kind,block_scaling=block_scaling,
            model_name=model_name,seed=seed,crop_arg=crop_arg,
            classifier_params=clf.get_params(),cache_dir=str(cache_dir)),indent=2,default=str))
    return pd.DataFrame(metrics), pd.DataFrame(predictions)

def mainTS(pc_use=0, model='LR', method_pls='mean', data_aug_method='mean',
           bs_decoding=False, **kwargs) :
    # Same mainTS entry-point style; data come from the shared MEG/iEEG trial cache.
    crop_arg = {'crop':False, 't_id_min':None, 't_id_max':None}
    if bs_decoding :
        from plssvd_eval_utils import load_trial_cache
        time = load_trial_cache(kwargs.get('cache_dir',CACHE_DIR)).times
        crop_arg = {'crop':True, 't_id_min':0, 't_id_max':int(np.searchsorted(time,-0.5,side='right'))}
    return decodingTS(method_pls, data_aug_method, PC_use=pc_use,
                      model_name=model, crop_arg=crop_arg, **kwargs)


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Decode conditions using training-only MEG/iEEG PLSSVD.')
    parser.add_argument('--pc_use',type=int,default=0)
    parser.add_argument('--model',choices=['LR','SVC_linear','SVC_rbf','RandomForest'],default='LR')
    parser.add_argument('--method_pls',choices=['mean','concat'],default='mean')
    parser.add_argument('--method_data_aug',choices=['mean','duplicat'],default='mean')
    parser.add_argument('--iteration',type=int,default=100)
    parser.add_argument('--iter_perm',type=int,default=50)
    parser.add_argument('--n_tr',type=int,default=24)
    parser.add_argument('--cache-dir',type=Path,default=CACHE_DIR)
    parser.add_argument('--out-path',type=Path,default=OUT_PATH)
    parser.add_argument('--meg-kind',default=MEG_KIND)
    parser.add_argument('--block-scaling',choices=['none','equal_variance'],default=BLOCK_SCALING)
    parser.add_argument('--seed',type=int,default=2026)
    args = parser.parse_args()
    mainTS(args.pc_use,args.model,args.method_pls,args.method_data_aug,
        iteration=args.iteration,iter_perm=args.iter_perm,n_tr=args.n_tr,
        cache_dir=args.cache_dir,out_path=args.out_path,meg_kind=args.meg_kind,
        block_scaling=args.block_scaling,seed=args.seed)
