"""Separate subject-count and fixed-cohort pairing sensitivity analyses.

All reported metrics are mean absolute Pearson correlations after one-to-one
component assignment. These are descriptive comparisons of averaged time courses.
"""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from coverage_matching_utils import Dataset, fit_block_pca

GROUP_KINDS = ('full_average', 'full_concatenated', 'coverage_average')


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


def _group_dataset(reference, indices, kind):
    """No participant assignment; even one MEG participant is supported."""
    s = reference.source_data
    arrays = [s['meg'][i] for i in indices]
    if kind == 'full_average':
        if any(a.shape != arrays[0].shape for a in arrays):
            raise ValueError('full_average requires equal MEG source/condition/time counts.')
        mean = np.zeros_like(arrays[0], dtype=np.result_type(np.float32, *[a.dtype for a in arrays]))
        for a in arrays: mean += a/len(arrays)
        arrays = [mean]
    elif kind == 'coverage_average':
        coords = s['electrode_positions']
        mean = np.zeros((arrays[0].shape[0], len(coords), arrays[0].shape[-1]), dtype=float)
        for index, a in zip(indices, arrays):
            _, ix = cKDTree(s['meg_positions'][index]).query(coords)
            mean += a[:, ix, :]/len(arrays)
        arrays = [mean]
    elif kind != 'full_concatenated':
        raise ValueError(f'Unsupported group composition: {kind}')
    return Dataset(kind, arrays, pd.DataFrame(), None, reference.condition_mode, source_data=s)


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


def run_subject_count_stability(reference, *, subject_counts=(5, 10, 20, 30), repeats=20,
                                n_components=3, seed=2026, meg_kinds=GROUP_KINDS,
                                max_gram_gib=2., output_dir=None):
    """Vary actual cohort size separately in MEG and iEEG, without subject pairing.

    Native features are kept: iEEG and concatenated MEG feature counts naturally
    grow with subject count. iEEG channels remain in their original order.
    Stability compares each subset with its full-cohort PCA AND all pairs of
    resampled fits at the same count. MEG subsets are compared with fixed full
    iEEG PCA; iEEG subsets are compared with fixed full MEG PCA for each kind.
    Subsets overlap; none of these correlations is a held-out validation score.
    """
    k = _integer(n_components, 'n_components'); repeats = _integer(repeats, 'repeats', 2)
    counts = sorted(set(_integer(n, 'subject_counts') for n in subject_counts))
    kinds = tuple(meg_kinds)
    if not counts or not kinds or len(set(kinds)) != len(kinds) or not set(kinds) <= set(GROUP_KINDS):
        raise ValueError('Use positive subject counts and unique group MEG compositions, without paired/control.')
    s, owners, ieeg_ids, meg_ids = _context(reference)
    available = {m: [n for n in counts if n <= total] for m, total in [('ieeg', len(ieeg_ids)), ('meg', len(meg_ids))]}
    if not any(available.values()): raise ValueError('No requested counts fit the available cohorts.')
    availability = pd.DataFrame([
        dict(modality=m, n_subjects=n, available_subjects=total,
             status='included' if n <= total else 'skipped_insufficient_subjects')
        for m, total in [('ieeg', len(ieeg_ids)), ('meg', len(meg_ids))]
        for n in counts])
    config = dict(experiment='subject_count', subject_counts=counts, repeats=repeats, n_components=k,
        seed=int(seed), meg_kinds=list(kinds), ieeg_subjects=ieeg_ids, meg_subjects=meg_ids,
        condition_mode=reference.condition_mode, metric='one-to-one matched mean absolute Pearson r',
        features='all native features; fixed full electrode target for MEG coverage_average',
        preprocessing=s.get('load_config', {}), scope='descriptive resampling; overlapping subsets; no confidence intervals')
    out = _open_output(output_dir, config)
    full_ieeg = _scores(reference, k, max_gram_gib)
    full_meg = {kind: _scores(_group_dataset(reference, range(len(meg_ids)), kind), k, max_gram_gib) for kind in kinds}
    saved = {'full_ieeg': full_ieeg, **{'full_'+kind: scores for kind, scores in full_meg.items()}}
    metrics, pairs, participants, fitted = [], [], [], {}
    for repeat, seq in enumerate(np.random.SeedSequence(seed).spawn(repeats)):
        for modality, child in zip(('ieeg', 'meg'), seq.spawn(2)):
            ids = ieeg_ids if modality == 'ieeg' else meg_ids
            order = np.random.default_rng(child).permutation(len(ids))
            for count in available[modality]:
                chosen = sorted(order[:count]); selected_ids = [ids[i] for i in chosen]
                participants.extend(dict(repeat=repeat, modality=modality, n_subjects=count, subject=v) for v in selected_ids)
                datasets = {'iEEG': _ieeg_dataset(reference, selected_ids)[0]} if modality == 'ieeg' else {
                    kind: _group_dataset(reference, chosen, kind) for kind in kinds}
                for name, dataset in datasets.items():
                    score = _scores(dataset, k, max_gram_gib)
                    fitted.setdefault((modality, name, count), []).append(score)
                    saved[f'{modality}_{name}_n{count}_r{repeat}'] = score
                    labels = dict(modality=modality, dataset=name, n_subjects=count, n_features=dataset.n_features,
                                  repeat=repeat, repeat_other=-1)
                    _record(metrics, pairs, score, full_ieeg if modality == 'ieeg' else full_meg[name], k,
                            **labels, comparison='to_full_cohort', reference_dataset=name)
                    targets = full_meg if modality == 'ieeg' else {'iEEG': full_ieeg}
                    for target, other in targets.items():
                        _record(metrics, pairs, score, other, k, **labels,
                                comparison='cross_modal', reference_dataset=target)
        print(f'Subject-count repetition {repeat+1}/{repeats}', flush=True)
    for (modality, name, count), values in fitted.items():
        for a, b in combinations(range(repeats), 2):
            _record(metrics, pairs, values[a], values[b], k, modality=modality, dataset=name,
                    n_subjects=count, n_features=np.nan, repeat=a, repeat_other=b,
                    comparison='between_resamples', reference_dataset=name)
    return _finish(out, config, metrics, pairs, participants, saved, availability=availability)


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
        baseline = dict(zip(fixed_ieeg, fixed_meg))
    if count < 2: raise ValueError('Pairing sensitivity requires at least two subjects per modality.')
    anchor_dataset, electrode_rows = _ieeg_dataset(reference, fixed_ieeg)
    config = dict(experiment='pairing', n_subjects=count, repeats=repeats, n_components=k, seed=int(seed),
        ieeg_subjects=fixed_ieeg, meg_subjects=fixed_meg, baseline_pairing=baseline,
        condition_mode=reference.condition_mode, metric='one-to-one matched mean absolute Pearson r',
        random_control='one fixed source-index permutation per MEG participant; no new source draws per assignment',
        preprocessing=s.get('load_config', {}), scope='fixed-cohort assignment sensitivity, not validation of subject identity')
    out = _open_output(output_dir, config)
    anchor = _scores(anchor_dataset, k, max_gram_gib)
    saved = {'fixed_ieeg': anchor}
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
            scores = _scores(dataset, k, max_gram_gib)
            saved[f'{kind}_r{repeat}'] = scores
            labels = dict(modality='meg', dataset=kind, n_subjects=count, n_features=dataset.n_features,
                          repeat=repeat, repeat_other=-1)
            _record(metrics, pairs, scores, anchor, k, **labels, comparison='cross_modal', reference_dataset='fixed_iEEG')
            if repeat == -1: baseline_scores[kind] = scores
            else:
                resampled[kind].append(scores)
                _record(metrics, pairs, scores, baseline_scores[kind], k, **labels,
                        comparison='to_baseline_pairing', reference_dataset=kind)
        if repeat >= 0: print(f'Pairing repetition {repeat+1}/{repeats}', flush=True)
    for kind, values in resampled.items():
        for a, b in combinations(range(repeats), 2):
            _record(metrics, pairs, values[a], values[b], k, modality='meg', dataset=kind,
                    n_subjects=count, n_features=len(electrode_rows), repeat=a, repeat_other=b,
                    comparison='between_pairings', reference_dataset=kind)
    cross = pd.DataFrame(metrics).query("comparison == 'cross_modal'")
    differences = cross.pivot(index='repeat', columns='dataset', values='correlation')
    differences['paired_minus_control'] = differences.paired_coverage-differences.random_control
    return _finish(out, config, metrics, pairs, participants, saved,
                   assignments=pd.DataFrame(assignments), source_mapping=pd.DataFrame(mappings),
                   paired_control_differences=differences.reset_index())


def plot_subject_count_stability(result):
    import matplotlib.pyplot as plt
    table = result['metrics']; figures = []
    for modality in ('meg', 'ieeg'):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        for ax, comparison, title in zip(axes, ['to_full_cohort', 'between_resamples', 'cross_modal'],
                ['PCA vs full-cohort PCA', 'PCA stability between resamples', 'Cross-modal PCA correspondence']):
            data = table[(table.modality == modality) & (table.comparison == comparison) & (table.status == 'ok')]
            for (name, target), rows in data.groupby(['dataset', 'reference_dataset'], sort=False):
                summary = rows.groupby('n_subjects').correlation.agg(['median', 'min', 'max'])
                label = f'{name} vs {target}' if comparison == 'cross_modal' else name
                ax.plot(summary.index, summary['median'], 'o-', label=label)
                ax.fill_between(summary.index, summary['min'], summary['max'], alpha=.15)
            ax.set(title=title, xlabel=f'Number of selected {modality.upper()} subjects',
                   ylabel='Mean matched |Pearson r|', ylim=(-.02, 1.02), xticks=result['config']['subject_counts'])
            if len(data): ax.legend(fontsize=8)
            else: ax.text(.5, .5, 'No supported results', ha='center', transform=ax.transAxes)
        fig.suptitle(f'{modality.upper()}: first {result["config"]["n_components"]} PCs; median/range, not confidence intervals')
        figures.append(fig)
    return figures


def plot_pairing_stability(result):
    import matplotlib.pyplot as plt
    table = result['metrics']; fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for ax, comparison, title in zip(axes, ['to_baseline_pairing', 'between_pairings', 'cross_modal'],
            ['PCA vs baseline assignment', 'PCA stability between assignments', 'PCA correspondence with fixed iEEG']):
        for index, kind in enumerate(('paired_coverage', 'random_control')):
            rows = table[(table.dataset == kind) & (table.comparison == comparison) & (table.repeat >= 0) & (table.status == 'ok')]
            values = rows.correlation.to_numpy()
            if len(values):
                ax.boxplot([values], positions=[index], widths=.45, showfliers=False)
                ax.scatter(index+np.linspace(-.13, .13, len(values)), values, s=12, alpha=.4)
            if comparison == 'cross_modal':
                baseline = table[(table.dataset == kind) & (table.comparison == comparison) & (table.repeat == -1)]
                if len(baseline): ax.plot(index, baseline.correlation.iloc[0], 'r*', markersize=12,
                                          label='Baseline assignment' if index == 0 else None)
        ax.set(title=title, xticks=[0, 1], xticklabels=['Paired coverage', 'Random control'],
               ylabel='Mean matched |Pearson r|', ylim=(-.02, 1.02))
        if comparison == 'cross_modal': ax.legend(fontsize=8)
    fig.suptitle(f'Fixed {result["config"]["n_subjects"]} subjects per modality; assignments vary, source randomization fixed')
    return [fig]
