"""Adapted from LB/LB5_decodingCluster.py mainTS and LB/src/decoding.py.

submit_to_cluster -n 2 -q short.q "python vs4_1_decoding.py --pc_use 0 --model LR --meg-kind paired_coverage --iteration 20" --project MINDLAB2021_MEG-TempSeqAges

Two condition pseudo-trial decoding, separately within MEG and iEEG.
PC_use is zero-based. PCA or PLSSVD is fitted on training condition data only.
Repeated holdouts overlap: accuracy is descriptive, not independent-trial inference.
Shuffled classifier labels are conditional controls, NOT full-pipeline null tests.
One-row summary CSVs and first-iteration network time courses are saved by modality.
Time-course NPZ files contain the actual decoder inputs before cropping,
with the same training-feature centering and block scaling as evaluation,
with training pseudo-trials ordered by condition and one held-out trace per condition.
Legacy pca_weight fields contain the selected projection weights; best_param contains
fixed classifier settings. entropy retains the original model-specific meaning
(LR log loss, SVM hinge loss, RandomForest OOB score).
"""
import os
import json
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss, hinge_loss
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


def _fit_pca(train, n_components, options):
    """Fit independent, centered, unwhitened PCA within each training modality."""
    from vs2_0_cov_models_utils import _gram, _spectrum, _weights
    from vs2_1_plssvd_eval_utils import _means

    model = {'k_max': n_components}
    for modality in ['ieeg', 'meg']:
        dataset = train[modality]
        n = dataset.n_observations
        if 12*n*n*8/2**30 > options.max_gram_gib:
            raise MemoryError('Increase max_gram_gib or shorten epochs.')
        gram = _gram(dataset)
        if options.block_scaling == 'none':
            scale = 1.
        elif options.block_scaling == 'equal_variance':
            scale = np.sqrt((n-1)/np.trace(gram))
        else:
            raise ValueError('Unknown block scaling.')
        temporal, singular = _spectrum(gram, dataset.n_features)
        if n_components > len(singular):
            raise ValueError(f'Requested PCA component exceeds {modality} training rank.')
        # Scaling the whole modality does not change its PCA axes.
        model[modality+'_weights'] = _weights(
            dataset, temporal[:, :n_components]/singular[:n_components], 1.)
        model[modality+'_mean'] = _means(dataset)
        model[modality+'_scale'] = scale
    return model


def _projection_offset(model, scalers, weights, modality, component):
    """Return the constant raw-space offset in evaluation's affine projection."""
    # Raw weights already include source mapping, preprocessing scale, and block scale.
    preprocessing = sum(
        scalers[modality][subject][0] @ value[:, component]
        for subject, value in weights.items())
    feature_center = (model[modality+'_mean'] @
                      model[modality+'_weights'][:, component]) * model[modality+'_scale']
    return float(preprocessing + feature_center)


def _save_network_time_courses(path, train, test, times, conditions, time_ids,
                               method, component, seed):
    """Save first-iteration, unshuffled decoder inputs in condition x trial x time order.

    Traces sum participant projections and subtract the fitted, time-independent
    preprocessing and feature-centering offset used by evaluation. Training traces include
    the same augmentation and pseudo-trial pairing used by the decoder.
    Full epochs are retained; decoding_time_ids identifies the selected window.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_conditions = len(conditions)
    train = np.asarray(train).reshape(n_conditions, -1, len(times))
    np.savez_compressed(
        path, times=np.asarray(times), conditions=np.asarray(conditions, dtype=str),
        train_time_courses=train, test_time_courses=np.asarray(test),
        train_condition_mean=train.mean(axis=1),
        decoding_time_ids=np.asarray(time_ids), method=np.asarray(method),
        pc_use=np.asarray(component), iteration=np.asarray(0), seed=np.asarray(seed),
        description=np.asarray(
            'Unshuffled decoder inputs; participant-summed projections with evaluation '
            'training-feature centering and block scaling; no time-course centering. Training includes '
            'augmentation and pseudo-trial pairing; test has one held-out trace per condition.'))


def decodingTS(method_pls, data_aug_method, iteration=100, PC_use=0, save=True,
               out_path=OUT_PATH, iter_perm=50, cache_dir=CACHE_DIR, model_name='LR',
               crop_arg=None, n_tr=24, meg_kind=MEG_KIND, block_scaling=BLOCK_SCALING,
               seed=2026, classifier_params=None, band='broadband', method='PLSSVD') :
    from vs2_1_plssvd_eval_utils import (load_trial_cache, ValidationOptions,
        _temporary_fold, _fit, _raw_weights, _project)

    method = method.upper()
    if method not in ['PCA', 'PLSSVD']:
        raise ValueError('method must be PCA or PLSSVD.')
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
        l_crop = ('_crp'+str(crop_arg['t_id_min'])+'_'+str(crop_arg['t_id_max'])+'_'
                  if crop_arg['crop'] else '')
        for modality in ['ieeg','meg'] :
            target = out_path / modality / band / 'fullmodel' / f'{method}_{method_pls}_{data_aug_method}_{PC_use}_{model_name}full{l_crop}.csv'
    # Original decodingTS fields retained; pca_weight contains the selected projection weights.
    # band is an output label: it must match the supplied cache; no filtering is done here.
    stored = {m: {key: [] for key in ['y_pred','y_pred_sh','y_test','entropy',
        'entropy_sh','weight','weight_sh','pca_weight','pca_weight_sh']}
        for m in ['ieeg','meg']}
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
                indices[modality][subject.subject] = {'train':train_ids, 'test':test_ids}

        with _temporary_fold(trials, meg_kind, indices, matching_seed,
                             condition_mode='average' if method_pls=='mean' else 'stack') as prepared :
            fold, scalers, audit, pairing = prepared
            fit = _fit_pca if method == 'PCA' else _fit
            model = fit(fold['train'], PC_use+1, options)
            if model['k_max'] <= PC_use :
                raise ValueError(f'Requested {method} component exceeds training rank.')
            for modality in ['ieeg', 'meg'] :
                weights = _raw_weights(trials, fold, model, scalers, audit,
                                       meg_kind, modality, PC_use+1)
                projection_offset = _projection_offset(
                    model, scalers, weights, modality, PC_use)
                Train_all = np.zeros((2*(n_tr-1), len(trials.times)))
                Test_all = np.zeros((2, len(trials.times)))
                for subject in getattr(trials, modality) :
                    parts = indices[modality][subject.subject]
                    # PROJECT before augmentation: linear averaging commutes with projection.
                    train_samples = []
                    test_samples = []
                    for array, train_id, test_id in zip(subject.data, parts['train'], parts['test']) :
                        # Read one trial at a time to bound memory use.
                        train_samples.append(np.stack([weights[subject.subject][:,PC_use] @ array[j] for j in train_id]))
                        test_samples.append(weights[subject.subject][:,PC_use] @ array[test_id[0]])
                    joined = np.concatenate(train_samples)
                    events = [np.arange(len(train_samples[0])),
                              np.arange(len(train_samples[0]), len(joined))]
                    augmented, true_trials = DataAugmentation(joined, events, data_aug_method, n_tr-1)
                    for condition in range(2) :
                        values = augmented[condition*(n_tr-1):(condition+1)*(n_tr-1)]
                        Train_all[condition*(n_tr-1):(condition+1)*(n_tr-1)] += values[rng.permutation(n_tr-1)]
                    Test_all += np.stack(test_samples)
                # Apply the same affine projection to every trial and time point.
                Train_all -= projection_offset
                Test_all -= projection_offset
                # Verify raw-space mapping against the fixed fitted feature-space projection.
                expected = _project(fold['test'][modality], model, modality)[:,PC_use]
                actual = Test_all.mean(axis=0) if method_pls=='mean' else Test_all.reshape(-1)
                if not np.allclose(actual, expected, rtol=1e-4, atol=1e-4*max(1.,np.max(abs(expected)))) :
                    raise ValueError(f'Trial projection disagrees with fitted {method} mapping.')
                if save and i == 0:
                    folder = out_path / modality / band / 'fullmodel'
                    filename = (f'{method}_{method_pls}_{data_aug_method}_{PC_use}_'
                                f'{model_name}full{l_crop}_iteration0_time_courses.npz')
                    _save_network_time_courses(
                        folder / filename, Train_all, Test_all, trials.times,
                        trials.conditions, time_ids, method, PC_use, seed)
                Train_transformed = Train_all[:,time_ids]
                Test_transformed = Test_all[:,time_ids]
                y_train = np.repeat([0,1], n_tr-1)
                y_test = np.array([0,1])

                saved = stored[modality]
                saved['y_test'].extend(y_test.tolist())
                saved['pca_weight'].append(model[modality+'_weights'][:,PC_use].copy())
                # NORMAL DECODING AND CONDITIONAL SHUFFLED-CLASSIFIER CONTROL
                for j in range(-1, iter_perm) :
                    labels = y_train if j==-1 else rng.permutation(y_train)
                    model_clf = clone(clf)
                    model_clf.fit(Train_transformed, labels)
                    y_pred = model_clf.predict(Test_transformed)
                    suffix = '' if j == -1 else '_sh'
                    saved['y_pred'+suffix].extend(y_pred.tolist())
                    if model_name == 'LR' :
                        saved['entropy'+suffix].append(log_loss(y_test, model_clf.predict_proba(Test_transformed)[:,1]))
                        saved['weight'+suffix].append(model_clf.coef_.copy())
                    elif model_name in ['SVC_linear', 'SVC_rbf'] :
                        saved['entropy'+suffix].append(hinge_loss([-1,1], model_clf.decision_function(Test_transformed)))
                        saved['weight'+suffix].append(model_clf.coef_.copy() if model_name == 'SVC_linear' else None)
                    elif model_name == 'RandomForest' :
                        saved['entropy'+suffix].append(float(model_clf.oob_score_))
                        saved['weight'+suffix].append(model_clf.feature_importances_.copy())
                    if j >= 0 :
                        # Conditional classifier null: projection is not refitted after shuffling labels.
                        saved['pca_weight_sh'].append(model[modality+'_weights'][:,PC_use].copy())
        print(f'Decoding iteration {i+1}/{iteration}',flush=True)

    summaries = {}
    for modality in ['ieeg','meg'] :
        sumsum = pd.DataFrame()
        sumsum['band'] = [band]
        sumsum['method'] = [method]
        sumsum['method_pca'] = [method_pls]
        sumsum['data_aug_method'] = [data_aug_method]
        sumsum['iter'] = [iteration]
        sumsum['iter_perm'] = [iter_perm]
        for key in ['y_pred','y_pred_sh','y_test','entropy','entropy_sh',
                    'weight','weight_sh','pca_weight','pca_weight_sh'] :
            sumsum[key] = [stored[modality][key]]
        # Legacy column name; settings are fixed, not selected by GridSearchCV.
        sumsum['best_param'] = [clf.get_params()]
        summaries[modality] = sumsum
        if save :
            folder = out_path / modality / band / 'fullmodel'
            folder.mkdir(parents=True, exist_ok=True)
            l_crop = ('_crp'+str(crop_arg['t_id_min'])+'_'+str(crop_arg['t_id_max'])+'_'
                      if crop_arg['crop'] else '')
            # Avoid numpy's default truncation of large weight arrays in CSV cells.
            with np.printoptions(threshold=np.inf) :
                sumsum.to_csv(folder / f'{method}_{method_pls}_{data_aug_method}_{PC_use}_{model_name}full{l_crop}.csv')
    return summaries


def mainTS(pc_use=0, model='LR', method_pls='mean', data_aug_method='mean',
           bs_decoding=False, method='PLSSVD', **kwargs) :
    # Same mainTS entry-point style; data come from the shared MEG/iEEG trial cache.
    crop_arg = {'crop':False, 't_id_min':None, 't_id_max':None}
    if bs_decoding :
        from vs2_1_plssvd_eval_utils import load_trial_cache
        time = load_trial_cache(kwargs.get('cache_dir',CACHE_DIR)).times
        crop_arg = {'crop':True, 't_id_min':0, 't_id_max':int(np.searchsorted(time,-0.5,side='right'))}
    return decodingTS(method_pls, data_aug_method, PC_use=pc_use,
                      model_name=model, crop_arg=crop_arg, method=method, **kwargs)


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Decode conditions using training-only MEG/iEEG PCA or PLSSVD.')
    parser.add_argument('--method', type=str.upper, choices=['PCA', 'PLSSVD'], default='PLSSVD',
                        help='Projection method (default: PLSSVD); PCA is fitted separately per modality.')
    parser.add_argument('--band',default='broadband',help='Output band label; must match cache contents.')
    parser.add_argument('--pc_use',type=int,default=0)
    parser.add_argument('--model',choices=['LR','SVC_linear','SVC_rbf','RandomForest'],default='LR')
    parser.add_argument('--method_pls',choices=['mean','concat'],default='mean',
                        help='Condition representation for either method: average or stack conditions.')
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
        block_scaling=args.block_scaling,seed=args.seed,band=args.band,method=args.method)
