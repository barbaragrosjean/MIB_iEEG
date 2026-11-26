"""Stability of PCA time courses and weights across saved CSV associations.

All reported metrics are mean absolute Pearson correlations after one-to-one
component assignment. Domains are matched independently; weight rows retain iEEG electrode order.
"""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from vs1_utils_coverage_matching import Dataset, fit_block_pca, normalize_dataset


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(value)


def matched_correlation(a, b, k):
    """Match first k score columns; report unsupported ranks, never silently lower k."""
    k = _integer(k, 'k')
    if a.ndim != 2 or b.ndim != 2 or len(a) != len(b):
        raise ValueError('Scores need identical observation rows.')
    if min(a.shape[1], b.shape[1]) < k:
        return dict(correlation=np.nan, status='skipped_rank'), []
    x, y = a[:, :k]-a[:, :k].mean(0), b[:, :k]-b[:, :k].mean(0)
    denominator = np.outer(np.linalg.norm(x, axis=0), np.linalg.norm(y, axis=0))
    if not np.isfinite(denominator).all() or np.any(denominator == 0):
        return dict(correlation=np.nan, status='undefined_correlation'), []
    r = np.clip(x.T@y/denominator, -1, 1)
    ia, ib = linear_sum_assignment(-np.abs(r))
    pairs = [dict(component_a=int(i+1), component_b=int(j+1), signed_r=float(r[i, j]),
                  abs_r=float(abs(r[i, j])), sign_b=-1 if r[i, j] < 0 else 1) for i, j in zip(ia, ib)]
    return dict(correlation=float(np.mean(np.abs(r[ia, ib]))), status='ok'), pairs


def _context(reference):
    if reference.name != 'iEEG' or reference.source_data is None:
        raise ValueError('Use the iEEG reference returned by load_dataset.')
    s = reference.source_data
    owners = np.asarray(s['electrode_subjects'], str)
    ieeg_ids = list(dict.fromkeys(owners))
    meg_ids = list(map(str, s['meg_subjects']))
    if len(set(meg_ids)) != len(meg_ids):
        raise ValueError('MEG participant IDs must be unique.')
    return s, owners, ieeg_ids, meg_ids


def _ieeg_dataset(reference, subjects):
    s, owners, _, _ = _context(reference)
    rows = np.flatnonzero(np.isin(owners, subjects))
    return Dataset('iEEG', [s['ieeg'][:, rows, :]], s['electrode_metadata'].iloc[rows].reset_index(drop=True),
                   np.arange(len(rows)), reference.condition_mode, source_data=s), rows


def _scores(dataset, k, max_gram_gib):
    return fit_block_pca(dataset, k, max_gram_gib=max_gram_gib).scores


def _record(metrics, pairs, a, b, k, **labels):
    summary, assignment = matched_correlation(a, b, k)
    metrics.append(dict(**labels, k=k, **summary))
    pairs.extend(dict(**labels, k=k, **pair) for pair in assignment)


def _open_output(output_dir, config):
    if output_dir is None: return None
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Choose a new output directory or load the completed run.')
    (out/'config.json').write_text(json.dumps(config, indent=2))
    return out


def _finish(out, config, metrics, pairs, participants, scores, **extra):
    result = dict(config=config, metrics=pd.DataFrame(metrics), component_pairs=pd.DataFrame(pairs),
                  participants=pd.DataFrame(participants), **extra)
    if out is not None:
        for name, table in result.items():
            if isinstance(table, pd.DataFrame): table.to_csv(out/f'{name}.csv', index=False)
        np.savez_compressed(out/'scores.npz', **scores)
        (out/'COMPLETE.json').write_text(json.dumps({'tables': [n for n, v in result.items() if isinstance(v, pd.DataFrame)]}))
    return result


def load_stability_results(output_dir, expected_experiment=None):
    out = Path(output_dir)
    if not (out/'COMPLETE.json').exists(): raise ValueError('Analysis has not completed.')
    config = json.loads((out/'config.json').read_text())
    if expected_experiment is not None and config['experiment'] != expected_experiment:
        raise ValueError('This directory belongs to the other experiment.')
    result = dict(config=config)
    for name in json.loads((out/'COMPLETE.json').read_text())['tables']:
        try: result[name] = pd.read_csv(out/f'{name}.csv')
        except pd.errors.EmptyDataError: result[name] = pd.DataFrame()
    return result


def run_pairing_stability(reference, *, n_subjects=None, repeats=None, n_components=3, seed=2026,
                          baseline_pairing=None, max_gram_gib=2., output_dir=None,
                          baseline_datasets=None, saved_alternatives=None):
    """Compare an explicit baseline with saved CSV associations; never rematch.

    saved_alternatives yields (pairing_id, constructed datasets). All datasets
    must already contain the fixed iEEG electrode slots and their CSV sources.
    """
    from vs0_utils_pairing import active_pairing_id
    if baseline_datasets is None or saved_alternatives is None:
        raise ValueError('Stability requires constructed datasets from saved CSV pairings; use the managed runner.')
    k=_integer(n_components,'n_components')
    s,owners,ieeg_ids,meg_ids=_context(reference)
    if n_subjects is not None and n_subjects!=len(ieeg_ids):
        raise ValueError('Stability retains the full fixed iEEG cohort.')
    config=dict(experiment='pairing',n_subjects=len(ieeg_ids),n_components=k,seed=int(seed),
        baseline_pairing=baseline_datasets['paired_coverage'].pairing,baseline_id=active_pairing_id(),
        condition_mode=reference.condition_mode,block_scaling=reference.block_scaling,
        scope='Sensitivity across saved participant associations; no new associations or source matches',
        preprocessing=s.get('load_config',{}))
    out=_open_output(output_dir,config)
    anchor_fit=fit_block_pca(reference,k,max_gram_gib=max_gram_gib)
    anchor={'timecourses':anchor_fit.scores,'weights':anchor_fit.weights}
    saved={f'fixed_ieeg_{key}':value for key,value in anchor.items()}
    metrics,pairs,participants,mappings,assignments=[],[],[],[],[]
    baseline_scores={};resampled={name:[] for name in ('paired_coverage','random_control')}
    comparison_ids=[]
    def runs():
        yield -1,config['baseline_id'],baseline_datasets
        for repeat,(name,datasets) in enumerate(saved_alternatives):
            comparison_ids.append(name)
            yield repeat,name,datasets
    for repeat,pairing_id,datasets in runs():
        current=datasets['iEEG']
        if not current.metadata[['subject','channel_index']].reset_index(drop=True).equals(reference.metadata[['subject','channel_index']].reset_index(drop=True)):
            raise ValueError('Saved associations must retain identical iEEG electrode rows.')
        pairing=datasets['paired_coverage'].pairing
        assignments.extend(dict(repeat=repeat,pairing_id=pairing_id,ieeg_subject=a,meg_subject=b) for a,b in pairing.items())
        participants.extend(dict(repeat=repeat,pairing_id=pairing_id,modality=m,subject=v)
                            for m,roster in [('ieeg',list(pairing)),('meg',list(pairing.values()))] for v in roster)
        audit=datasets['paired_coverage'].matching
        for row in audit.to_dict('records'):mappings.append(dict(row,repeat=repeat,pairing_id=pairing_id))
        for kind in resampled:
            ds=datasets[kind]
            fitted=fit_block_pca(ds,k,max_gram_gib=max_gram_gib)
            values={'timecourses':fitted.scores,'weights':fitted.weights}
            labels=dict(modality='meg',dataset=kind,n_subjects=len(pairing),n_features=ds.n_features,repeat=repeat,repeat_other=-1)
            for domain,value in values.items():
                saved[f'{kind}_r{repeat}_{domain}']=value
                _record(metrics,pairs,value,anchor[domain],k,**labels,domain=domain,
                        comparison='cross_modal',reference_dataset='fixed_iEEG')
                if repeat>=0:
                    _record(metrics,pairs,value,baseline_scores[kind][domain],k,**labels,domain=domain,
                            comparison='to_baseline_pairing',reference_dataset=kind)
            if repeat==-1:baseline_scores[kind]=values
            else:resampled[kind].append(values)
    config.update(repeats=len(comparison_ids),comparison_pairings=comparison_ids)
    for kind,values in resampled.items():
        for a,b in combinations(range(len(values)),2):
            for domain in ('timecourses','weights'):
                _record(metrics,pairs,values[a][domain],values[b][domain],k,modality='meg',dataset=kind,
                        n_subjects=len(ieeg_ids),n_features=reference.n_features,repeat=a,repeat_other=b,
                        domain=domain,comparison='between_pairings',reference_dataset=kind)
    cross=pd.DataFrame(metrics).query("comparison == 'cross_modal'")
    differences=cross.pivot(index=['domain','repeat'],columns='dataset',values='correlation')
    differences['paired_minus_control']=differences.paired_coverage-differences.random_control
    if out is not None:
        s['electrode_metadata'].to_csv(out/'ieeg_features.csv',index=False)
        (out/'config.json').write_text(json.dumps(config,indent=2))
    return _finish(out,config,metrics,pairs,participants,saved,assignments=pd.DataFrame(assignments),
        source_mapping=pd.DataFrame(mappings),paired_control_differences=differences.reset_index())


def compare_to_full_concatenated(stability, output_dir, full_pca):
    """Compare saved shuffled PCs with full MEG PCs, sampling matching sources for weights."""
    if full_pca.dataset.name != 'full_concatenated':
        raise ValueError('Use the full_concatenated PCA result.')
    metadata = full_pca.dataset.metadata
    # Full dataset blocks follow the MEG subject order; each block contains all sources.
    subjects = metadata['meg_subject'].astype(str).to_numpy()
    offsets = {subject: int(np.flatnonzero(subjects == subject)[0])
               for subject in dict.fromkeys(subjects)}
    metrics, pairs = [], []
    k = stability['config']['n_components']
    with np.load(Path(output_dir) / 'scores.npz') as scores:
        for repeat, mapping in stability['source_mapping'].groupby('repeat'):
            if repeat < 0:
                continue
            mapping = mapping.sort_values('electrode_index')
            starts = mapping.meg_subject.astype(str).map(offsets)
            if starts.isna().any():
                # Different saved pairings may subsample different MEG participants.
                # Temporal scores share time rows; spatial weights lack a valid map.
                for name in ('paired_coverage','random_control'):
                    _record(metrics,pairs,scores[f'{name}_r{repeat}_timecourses'],full_pca.scores,k,
                            dataset=name,domain='timecourses',repeat=int(repeat),repeat_other=-1,
                            comparison='to_full_concatenated',reference_dataset='full_concatenated')
                continue
            for name, source_column in [('paired_coverage', 'source_index'),
                                        ('random_control', 'random_source_index')]:
                indices = starts.to_numpy(dtype=int) + mapping[source_column].to_numpy(dtype=int)
                for domain in ('timecourses', 'weights'):
                    reference = full_pca.scores if domain == 'timecourses' else full_pca.weights[indices]
                    _record(metrics, pairs, scores[f'{name}_r{repeat}_{domain}'], reference, k,
                            dataset=name, domain=domain, repeat=int(repeat), repeat_other=-1,
                            comparison='to_full_concatenated', reference_dataset='full_concatenated')
    return pd.DataFrame(pairs)
