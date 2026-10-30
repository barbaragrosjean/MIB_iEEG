"""Fixed-cohort random-assignment stability of PCA time courses and weights.

All reported metrics are mean absolute Pearson correlations after one-to-one
component assignment. Domains are matched independently; weight rows retain iEEG electrode order.
"""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from vs1_coverage_matching_utils import Dataset, fit_block_pca, normalize_dataset


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


def run_pairing_stability(reference, *, n_subjects=None, repeats=20, n_components=3, seed=2026,
                          baseline_pairing=None, max_gram_gib=2., output_dir=None):
    """Freeze participant identities/counts, vary only MEG-to-iEEG assignment.

    A single random source-index permutation PER MEG SUBJECT is fixed across all
    assignments. Applying it to the anatomical indices creates the random control,
    preserving source duplication without drawing fresh random locations per run.
    The original/baseline assignment is compared with the randomized assignments;
    their distribution measures dependence on pairing, not biological identity.
    """
    k = _integer(n_components, 'n_components'); repeats = _integer(repeats, 'repeats', 2)
    s, owners, ieeg_ids, meg_ids = _context(reference)
    roster_seq, assignment_seq, control_seq = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(roster_seq)
    if baseline_pairing is not None:
        baseline = {str(a): str(b) for a, b in baseline_pairing.items()}
        count = len(baseline)
        if not count or not set(baseline) <= set(ieeg_ids) or not set(baseline.values()) <= set(meg_ids) or len(set(baseline.values())) != count:
            raise ValueError('baseline_pairing must contain valid, distinct participants.')
        if n_subjects is not None and _integer(n_subjects, 'n_subjects') != count:
            raise ValueError('n_subjects must agree with baseline_pairing.')
        fixed_ieeg = [v for v in ieeg_ids if v in baseline]
        fixed_meg = [v for v in meg_ids if v in baseline.values()]
    else:
        count = min(len(ieeg_ids), len(meg_ids)) if n_subjects is None else _integer(n_subjects, 'n_subjects')
        if count > min(len(ieeg_ids), len(meg_ids)): raise ValueError('Fixed cohort exceeds available participants.')
        fixed_ieeg = [ieeg_ids[i] for i in sorted(rng.choice(len(ieeg_ids), count, replace=False))]
        fixed_meg = [meg_ids[i] for i in sorted(rng.choice(len(meg_ids), count, replace=False))]
        baseline = dict(zip(fixed_ieeg, rng.permutation(fixed_meg).tolist()))
    if count < 2: raise ValueError('Pairing sensitivity requires at least two subjects per modality.')
    anchor_dataset, electrode_rows = _ieeg_dataset(reference, fixed_ieeg)
    config = dict(experiment='pairing', n_subjects=count, repeats=repeats, n_components=k, seed=int(seed),
        ieeg_subjects=fixed_ieeg, meg_subjects=fixed_meg, baseline_pairing=baseline,
        condition_mode=reference.condition_mode, metric='one-to-one matched mean absolute Pearson r',
        random_control='one fixed source-index permutation per MEG participant; no new source draws per assignment',
        block_scaling=reference.block_scaling, weight_correspondence='fixed iEEG electrode slots',
        preprocessing=s.get('load_config', {}), scope='fixed-cohort assignment sensitivity, not validation of subject identity')
    out = _open_output(output_dir, config)
    normalize_dataset(anchor_dataset, reference.block_scaling)
    anchor_fit = fit_block_pca(anchor_dataset, k, max_gram_gib=max_gram_gib)
    anchor = {'timecourses': anchor_fit.scores, 'weights': anchor_fit.weights}
    saved = {f'fixed_ieeg_{key}': value for key, value in anchor.items()}
    indices = {v: meg_ids.index(v) for v in fixed_meg}
    nearest = {v: cKDTree(s['meg_positions'][indices[v]]).query(s['electrode_positions'][electrode_rows])[1] for v in fixed_meg}
    control_rng = np.random.default_rng(control_seq)
    permutations = {v: control_rng.permutation(s['meg'][indices[v]].shape[1]) for v in fixed_meg}
    if out is not None:
        np.savez_compressed(out/'control_source_permutations.npz', **{f'subject_{i}': permutations[v] for i, v in enumerate(fixed_meg)})
        s['electrode_metadata'].iloc[electrode_rows].to_csv(out/'ieeg_features.csv', index=False)
    metrics, pairs, participants, mappings, assignments = [], [], [], [], []
    for m, roster in [('ieeg', fixed_ieeg), ('meg', fixed_meg)]:
        participants.extend(dict(modality=m, subject=v) for v in roster)
    baseline_scores, resampled = {}, {v: [] for v in ('paired_coverage', 'random_control')}
    rng = np.random.default_rng(assignment_seq)
    for repeat in range(-1, repeats):
        pairing = baseline if repeat == -1 else dict(zip(fixed_ieeg, rng.permutation(fixed_meg).tolist()))
        assignments.extend(dict(repeat=repeat, ieeg_subject=a, meg_subject=b) for a, b in pairing.items())
        shape = (s['ieeg'].shape[0], len(electrode_rows), s['ieeg'].shape[-1])
        arrays = {kind: np.empty(shape, dtype=float) for kind in resampled}
        for ieeg_id, meg_id in pairing.items():
            rows = np.flatnonzero(owners[electrode_rows] == ieeg_id)
            ix = nearest[meg_id][rows]; rx = permutations[meg_id][ix]
            a = s['meg'][indices[meg_id]]
            arrays['paired_coverage'][:, rows, :] = a[:, ix, :]
            arrays['random_control'][:, rows, :] = a[:, rx, :]
            mappings.extend(dict(repeat=repeat, electrode_index=int(electrode_rows[row]),
                ieeg_subject=ieeg_id, meg_subject=meg_id, source_index=int(i), random_source_index=int(j))
                for row, i, j in zip(rows, ix, rx))
        for kind, a in arrays.items():
            dataset = Dataset(kind, [a], pd.DataFrame(), None, reference.condition_mode, source_data=s)
            normalize_dataset(dataset, reference.block_scaling)
            fitted = fit_block_pca(dataset, k, max_gram_gib=max_gram_gib)
            values = {'timecourses': fitted.scores, 'weights': fitted.weights}
            labels = dict(modality='meg', dataset=kind, n_subjects=count, n_features=dataset.n_features,
                          repeat=repeat, repeat_other=-1)
            for domain, value in values.items():
                saved[f'{kind}_r{repeat}_{domain}'] = value
                _record(metrics, pairs, value, anchor[domain], k, **labels, domain=domain,
                        comparison='cross_modal', reference_dataset='fixed_iEEG')
                if repeat >= 0:
                    _record(metrics, pairs, value, baseline_scores[kind][domain], k, **labels,
                            domain=domain, comparison='to_baseline_pairing', reference_dataset=kind)
            if repeat == -1: baseline_scores[kind] = values
            else: resampled[kind].append(values)
        if repeat >= 0: print(f'Pairing repetition {repeat+1}/{repeats}', flush=True)
    for kind, values in resampled.items():
        for a, b in combinations(range(repeats), 2):
            for domain in ('timecourses', 'weights'):
                _record(metrics, pairs, values[a][domain], values[b][domain], k, modality='meg', dataset=kind,
                        n_subjects=count, n_features=len(electrode_rows), repeat=a, repeat_other=b,
                        domain=domain, comparison='between_pairings', reference_dataset=kind)
    cross = pd.DataFrame(metrics).query("comparison == 'cross_modal'")
    differences = cross.pivot(index=['domain', 'repeat'], columns='dataset', values='correlation')
    differences['paired_minus_control'] = differences.paired_coverage-differences.random_control
    return _finish(out, config, metrics, pairs, participants, saved,
                   assignments=pd.DataFrame(assignments), source_mapping=pd.DataFrame(mappings),
                   paired_control_differences=differences.reset_index())


