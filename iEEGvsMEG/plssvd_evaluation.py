"""Post-fit PLSSVD metrics from compact sufficient statistics, without raw trials.

Prediction uses training-only ridge calibration, recomputed for the selected k.
No metric tables or feature-sized prediction maps are saved by the fit job.
"""
import numpy as np
import pandas as pd


def collect_statistics(fold, scores, model):
    """Capture data-dependent statistics once, in bounded feature chunks.

    For prediction, B = S_train.T @ X_train and D = S_part.T @ X_part.
    Save B @ B.T and B @ D.T instead of the feature-sized B or coefficients.
    X is centered by the TRAIN feature mean. Crosscovariance denominators use
    partition centering, matching the original evaluation definition.
    """
    from cov_models_utils import _gram
    parts = ('train', 'test')
    output = {f'{p}_{m}': scores[p][m] for p in parts for m in ('ieeg', 'meg')}
    k = model['k_max']
    for modality in ('ieeg', 'meg'):
        other = 'meg' if modality == 'ieeg' else 'ieeg'
        bb = np.zeros((k,k))
        bd = {p: np.zeros((k,k)) for p in parts}
        energy = {p: 0. for p in parts}
        offset = 0
        for train, test in zip(fold['train'][modality].blocks(), fold['test'][modality].blocks()):
            for start in range(0, train.shape[1], 1024):
                stop = min(start+1024, train.shape[1])
                mean = model[modality+'_mean'][offset+start:offset+stop]
                values = {p: np.asarray(x[:,start:stop],float)-mean
                          for p,x in [('train',train),('test',test)]}
                b = scores['train'][other].T @ values['train']
                bb += b @ b.T
                for part in parts:
                    d = scores[part][other].T @ values[part]
                    bd[part] += b @ d.T
                    energy[part] += np.sum(values[part]**2)
            offset += train.shape[1]
        output[modality+'_prediction_bb'] = bb
        output[modality+'_weight_gram'] = model[modality+'_weights'].T @ model[modality+'_weights']
        output[modality+'_scale'] = model[modality+'_scale']
        for part in parts:
            output[f'{part}_{modality}_prediction_bd'] = bd[part]
            output[f'{part}_{modality}_baseline_energy'] = energy[part]
    for part in parts:
        n = len(scores[part]['ieeg'])
        gx = _gram(fold[part]['ieeg']) * model['ieeg_scale']**2
        gy = _gram(fold[part]['meg']) * model['meg_scale']**2
        output[part+'_total_crosscov_energy'] = np.sum(gx*gy)/(n-1)**2
    return output


def selected_count(requested, maximum):
    k = maximum if requested is None else requested
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= maximum:
        raise ValueError(f'n_components must be an integer between 1 and the fitted maximum {maximum}.')
    return int(k)


def evaluate_statistics(saved, k, ridge):
    """Exact selected-k metrics; ridge penalty also changes with selected k."""
    from plssvd_eval_utils import _r
    scores = {p: {m: saved[f'{p}_{m}'][:,:k] for m in ('ieeg','meg')} for p in ('train','test')}
    evaluations, covariances = {}, {}
    for part in ('train','test'):
        x, y = scores[part]['ieeg'], scores[part]['meg']
        covariance = (x-x.mean(0)).T @ (y-y.mean(0))/(len(x)-1)
        metrics = {}
        for modality in ('ieeg','meg'):
            other = 'meg' if modality == 'ieeg' else 'ieeg'
            baseline = float(saved[f'{part}_{modality}_baseline_energy'])
            own = scores[part][modality]/float(saved[modality+'_scale'])
            gram = own.T @ own
            weight_gram = saved[modality+'_weight_gram'][:k,:k]
            retained = 2*np.trace(gram)-np.sum(gram*weight_gram)
            metrics[modality+'_reconstruction_fraction'] = retained/baseline if baseline>0 else np.nan
            train = scores['train'][other]
            g = train.T @ train
            penalty = max(ridge*np.trace(g)/k, np.finfo(float).eps)
            inverse = np.linalg.solve(g+penalty*np.eye(k), np.eye(k))
            bb = saved[modality+'_prediction_bb'][:k,:k]
            bd = saved[f'{part}_{modality}_prediction_bd'][:k,:k]
            test_gram = scores[part][other].T @ scores[part][other]
            improvement = 2*np.trace(inverse@bd)-np.trace(test_gram@inverse@bb@inverse.T)
            metrics['predict_'+modality+'_q2'] = improvement/baseline if baseline>0 else np.nan
        total = float(saved[part+'_total_crosscov_energy'])
        captured = float(np.sum(covariance**2))
        metrics.update(crosscov_energy_fraction=float(np.clip(captured/total,0,1)) if total>0 else np.nan,
                       paired_crosscov_energy_fraction=float(np.clip(np.sum(np.diag(covariance)**2)/total,0,1)) if total>0 else np.nan,
                       mean_paired_covariance=float(np.trace(covariance)/k), total_crosscov_energy=total,
                       captured_crosscov_energy=captured, mean_r=float(np.mean(_r(x,y))))
        evaluations[part], covariances[part] = metrics, covariance
    return scores, evaluations, covariances


def evaluate_iteration(root, options, n_components=None):
    """Evaluate completed compact fit artifacts; large weights stay on disk."""
    from plssvd_eval_utils import _r, _across_split_stability
    k = selected_count(n_components, options['n_components'])
    metrics, summaries, components, all_scores = [], [], [], []
    for fold in range(options['repeats']):
        with np.load(root/f'scores_{fold:03d}.npz', allow_pickle=False) as saved:
            scores, values, covariances = evaluate_statistics(saved, k, options['ridge'])
        all_scores.append(scores)
        for part in ('train','test'):
            metrics.append(dict(repeat=fold, partition=part, n_components=k, **values[part]))
        train_r, test_r = (_r(scores[p]['ieeg'],scores[p]['meg']) for p in ('train','test'))
        denominator = values['train']['mean_paired_covariance']
        summaries.append(dict(repeat=fold,n_components=k,train_mean_r=np.mean(train_r),test_mean_r=np.mean(test_r),
            predict_ieeg_q2=values['test']['predict_ieeg_q2'],predict_meg_q2=values['test']['predict_meg_q2'],
            paired_covariance_retention=values['test']['mean_paired_covariance']/denominator if denominator>0 else np.nan,
            **{p+'_'+key:value for p in values for key,value in values[p].items() if key!='mean_r'}))
        for pc in range(k):
            components.append(dict(repeat=fold,component=pc+1,train_r=train_r[pc],test_r=test_r[pc],
                                   train_covariance=covariances['train'][pc,pc],test_covariance=covariances['test'][pc,pc]))
    stability, pairs = _across_split_stability(all_scores,k)
    table = pd.DataFrame(metrics)
    summary = table.melt(id_vars=['repeat','partition','n_components'],var_name='metric',value_name='value').groupby(
        ['partition','metric'],sort=False).value.agg(['count','mean','std','median','min','max']).reset_index()
    with np.load(root.parent/'trial_axes.npz') as axes:
        times, conditions = axes['times'],tuple(axes['conditions'].tolist())
    split = pd.read_csv(root/'split_audit.csv.gz')
    return dict(summary=pd.DataFrame(summaries),components=pd.DataFrame(components),fold_metrics=table,
        metric_summary=summary,fold_stability=stability,fold_component_pairs=pairs,
        primary_scores=all_scores[0],split_audit=split,
        participants=split[['repeat','modality','subject']].drop_duplicates(),
        times=times,conditions=conditions,validation_options=dict(options,n_components=k),
        fit_options=options,n_components_evaluated=k,n_components_fitted=options['n_components'],
        output_dir=root,null_tests=pd.DataFrame(),null_distributions={},trial_counts=pd.DataFrame(),
        electrode_metadata=pd.DataFrame(),run_config={},artifacts={p.name:p for p in root.iterdir() if p.is_file()})
