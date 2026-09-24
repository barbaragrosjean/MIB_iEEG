"""Legacy mixed MEG coverage and composition sensitivity analysis.

The coverage notebook now uses coverage_stability.py to separate actual
subject-count effects from fixed-cohort assignment effects.

The iEEG reference is fixed. This operates on condition averages and is
descriptive, not a trial-held-out test or a population confidence interval.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from coverage_matching_utils import (Dataset, SETUP_NAMES, construct_five_datasets,
                                     fit_block_pca, compare_pca_to_ieeg)


def _positive_ints(values, name):
    values = list(values)
    if not values or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1 for v in values):
        raise ValueError(f'{name} must contain positive integers.')
    return sorted(set(map(int, values)))


def _select_features(dataset, indices):
    """Subset without materializing full participant concatenation."""
    indices = np.sort(np.asarray(indices, int))
    arrays = []
    offset = 0
    for array in dataset.arrays:
        local = indices[(indices >= offset) & (indices < offset+array.shape[1])]-offset
        if len(local):
            arrays.append(array[:, local, :])
        offset += array.shape[1]
    return Dataset(dataset.name, arrays, dataset.metadata.iloc[indices].reset_index(drop=True),
                   None, dataset.condition_mode, source_data=dataset.source_data)


def run_coverage_sampling(reference, *, participant_counts=None, feature_counts=None,
                          repeats=20, dimensions=(1, 2, 3, 5, 10), seed=2026,
                          kinds=SETUP_NAMES, max_gram_gib=2., output_dir=None):
    """Cross MEG pool size and feature budget using nested random subsets.

    participant_counts: MEG pool sizes, at least the number of iEEG participants
        because pairing is without replacement. Pair-based setups always use
        at most one MEG participant per iEEG participant; their pool-size effect
        is NOT an effect of averaging more participants.
    feature_counts: common feature budgets across setups. None in this list
        means native counts (different across setups). At a numeric budget,
        paired/coverage/random use the same sampled electrode slots; full
        setups use uniformly sampled native features. The fixed iEEG reference
        always uses every electrode. This tests MEG sampling, not iEEG sampling.

    Each repetition shares participant order, pairings, matched/random source
    draws and nested feature subsets across budgets. No k is selected here.
    Rank-deficient fits produce explicit skipped rows rather than silent k changes.
    """
    if reference.name != 'iEEG' or reference.source_data is None:
        raise ValueError('Use the iEEG reference returned by load_dataset.')
    dimensions = _positive_ints(dimensions, 'dimensions')
    repeats = _positive_ints([repeats], 'repeats')[0]
    kinds = tuple(kinds)
    if not kinds or len(set(kinds)) != len(kinds) or not set(kinds) <= set(SETUP_NAMES):
        raise ValueError('Choose unique known MEG kinds.')
    source = reference.source_data
    subjects = list(map(str, source['meg_subjects']))
    owners = np.asarray(source['electrode_subjects'], str)
    n_ieeg = len(set(owners)); n_meg = len(subjects); n_electrodes = len(owners)
    counts = _positive_ints(participant_counts if participant_counts is not None else [n_meg], 'participant_counts')
    if min(counts) < n_ieeg or max(counts) > n_meg:
        raise ValueError(f'Participant pool sizes must be between {n_ieeg} and {n_meg}.')
    budgets = list(feature_counts) if feature_counts is not None else [None]
    numeric = [v for v in budgets if v is not None]
    numeric = _positive_ints(numeric, 'feature_counts') if numeric else []
    budgets = ([None] if None in budgets else []) + numeric
    if not budgets:
        raise ValueError('Provide at least one feature budget.')
    limits = []
    if set(kinds) & set(SETUP_NAMES[2:]): limits.append(n_electrodes)
    if 'full_average' in kinds: limits.append(source['meg'][0].shape[1])
    if 'full_concatenated' in kinds:
        limits.append(sum(sorted(a.shape[1] for a in source['meg'])[:min(counts)]))
    if numeric and max(numeric) > min(limits):
        raise ValueError(f'Common feature budgets cannot exceed {min(limits)} for these setups.')
    if 'full_average' in kinds and any(a.shape != source['meg'][0].shape for a in source['meg']):
        raise ValueError('full_average requires the same MEG source count, condition count and time-sample count.')
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        if (out/'config.json').exists():
            raise FileExistsError('Choose a new coverage output directory; existing runs are not overwritten.')
    config = dict(participant_counts=counts, feature_counts=budgets, repeats=repeats,
                  dimensions=dimensions, seed=int(seed), kinds=list(kinds),
                  condition_mode=reference.condition_mode, fixed_ieeg_features=reference.n_features,
                  fixed_ieeg_participants=n_ieeg, meg_subjects=subjects,
                  preprocessing=source.get('load_config', {}),
                  scope='descriptive repeated sampling of MEG; fixed full iEEG reference; no confidence intervals')
    if out is not None:
        (out/'config.json').write_text(json.dumps(config, indent=2))
        reference.metadata.to_csv(out/'ieeg_features.csv', index=False)
        np.save(out/'times.npy', source['times'])
    anchor = fit_block_pca(reference, max(dimensions), max_gram_gib=max_gram_gib)
    rows, pairs, audits, selections, coverage_maps, diagnostics = [], [], [], [], [], []
    for repeat, seq in enumerate(np.random.SeedSequence(seed).spawn(repeats)):
        pool_seq, match_seq, feature_seq = seq.spawn(3)
        order = np.random.default_rng(pool_seq).permutation(n_meg)
        matching_seed = int(np.random.default_rng(match_seq).integers(2**31-1))
        # Reinitialize per pool: feature priorities remain coordinated across budgets.
        for count in counts:
            chosen = sorted(order[:count])
            ds, audit, pairing = construct_five_datasets(
                [source['meg'][i] for i in chosen], [source['meg_positions'][i] for i in chosen],
                [subjects[i] for i in chosen], source['electrode_positions'], owners,
                seed=matching_seed, condition_mode=reference.condition_mode, kinds=kinds)
            rng = np.random.default_rng(feature_seq)
            electrode_order = rng.permutation(n_electrodes)
            feature_orders = {kind: (electrode_order if kind in SETUP_NAMES[2:] else rng.permutation(data.n_features))
                              for kind, data in ds.items()}
            audits.append(audit.assign(repeat=repeat, participant_pool_count=count, matching_seed=matching_seed))
            group_maps = []
            for i in chosen:
                distances, nearest = cKDTree(source['meg_positions'][i]).query(source['electrode_positions'])
                group_maps.append(pd.DataFrame(dict(electrode_index=np.arange(n_electrodes),
                    meg_subject=subjects[i], source_index=nearest, distance_mm=distances)))
            group_map = pd.concat(group_maps, ignore_index=True)
            coverage_maps.append(group_map.assign(repeat=repeat, participant_pool_count=count))
            for kind, data in ds.items():
                data.source_data = source
                for budget in budgets:
                    ix = np.arange(data.n_features) if budget is None else np.sort(feature_orders[kind][:budget])
                    selected = data if budget is None else _select_features(data, ix)
                    budget_label = 'native' if budget is None else str(budget)
                    base = dict(repeat=repeat, participant_pool_count=count, feature_budget=budget_label,
                                dataset=kind, n_features=selected.n_features, matching_seed=matching_seed)
                    # Only sampled coverage slots contribute in paired/control configurations.
                    if kind in ('paired_coverage', 'random_control'):
                        actual_subjects = sorted(set(audit.iloc[ix].meg_subject))
                    elif kind == 'full_concatenated':
                        actual_subjects = sorted(set(selected.metadata.meg_subject.astype(str)))
                    else:
                        actual_subjects = [subjects[i] for i in chosen]
                    base['n_contributing_participants'] = len(actual_subjects)
                    base['contributing_participants'] = json.dumps(actual_subjects)
                    if kind in SETUP_NAMES[2:]:
                        if kind == 'coverage_average':
                            mapping = group_map[group_map.electrode_index.isin(ix)]
                        else:
                            columns = ['meg_subject', 'source_index', 'distance_mm']
                            mapping = audit.iloc[ix][columns] if kind == 'paired_coverage' else (
                                audit.iloc[ix][['meg_subject', 'random_source_index', 'random_distance_mm']]
                                .rename(columns={'random_source_index': 'source_index', 'random_distance_mm': 'distance_mm'}))
                        unique = len(mapping[['meg_subject', 'source_index']].drop_duplicates())
                        diagnostics.append(dict(**base, n_source_slots=len(mapping), n_unique_subject_sources=unique,
                            duplicate_fraction=1-unique/len(mapping), mean_distance_mm=float(mapping.distance_mm.mean()),
                            median_distance_mm=float(mapping.distance_mm.median()),
                            max_distance_mm=float(mapping.distance_mm.max()),
                            anatomical_correspondence=kind != 'random_control'))
                    # Native full datasets can contain millions of features: record
                    # contiguous block ranges instead of millions of audit rows.
                    if budget is None and kind in SETUP_NAMES[:2]:
                        offset = 0
                        for array in data.arrays:
                            selections.append(dict(**base, selection='all_in_block',
                                feature_start=offset, feature_stop=offset+array.shape[1],
                                meg_subject=str(data.metadata.iloc[offset].meg_subject)))
                            offset += array.shape[1]
                    else:
                        for i in ix:
                            meta = data.metadata.iloc[i]
                            selections.append(dict(**base, selection='individual', native_feature_index=int(i),
                                electrode_index=int(i) if kind in SETUP_NAMES[2:] else -1,
                                meg_subject=str(meta.meg_subject), source_index=int(meta.source_index)))
                    fitted = fit_block_pca(selected, max(dimensions), max_gram_gib=max_gram_gib)
                    for k in dimensions:
                        row = dict(**base, k=k, ieeg_pca_variance_fraction=float(anchor.explained_variance_ratio[:k].sum())
                                   if k <= anchor.scores.shape[1] else np.nan,
                                   meg_pca_variance_fraction=float(fitted.explained_variance_ratio[:k].sum())
                                   if k <= fitted.scores.shape[1] else np.nan)
                        if k > min(anchor.scores.shape[1], fitted.scores.shape[1]):
                            rows.append(dict(**row, status='skipped_rank', ieeg_variance_captured=np.nan,
                                             subspace_overlap=np.nan, matched_abs_r=np.nan))
                            continue
                        metrics, matched = compare_pca_to_ieeg({'iEEG': anchor, kind: fitted}, n_components=[k], plot=False)
                        rows.append(dict(**row, status='ok', **metrics.drop(columns=['dataset', 'k', 'rank']).iloc[0].to_dict()))
                        pairs.extend(dict(**base, **r) for r in matched.drop(columns='dataset').to_dict('records'))
            print(f'Coverage repetition {repeat+1}/{repeats}, MEG pool {count}', flush=True)
    metrics = pd.DataFrame(rows)
    deltas = []
    keys = ['repeat', 'participant_pool_count', 'feature_budget', 'k']
    if {'paired_coverage', 'random_control'} <= set(kinds):
        left = metrics.query("dataset == 'paired_coverage' and status == 'ok'")
        right = metrics.query("dataset == 'random_control' and status == 'ok'")
        joined = left.merge(right, on=keys, suffixes=('_paired', '_random'), validate='one_to_one')
        for row in joined.to_dict('records'):
            deltas.append({**{key: row[key] for key in keys}, **{
                metric+'_delta': row[metric+'_paired']-row[metric+'_random']
                for metric in ['ieeg_variance_captured', 'subspace_overlap', 'matched_abs_r']}})
    result = dict(metrics=metrics, component_pairs=pd.DataFrame(pairs),
                  matching=pd.concat(audits, ignore_index=True), selections=pd.DataFrame(selections),
                  coverage_mapping=pd.concat(coverage_maps, ignore_index=True),
                  matching_summary=pd.DataFrame(diagnostics),
                  paired_control_deltas=pd.DataFrame(deltas, columns=keys+[
                      m+'_delta' for m in ['ieeg_variance_captured', 'subspace_overlap', 'matched_abs_r']]))
    if out is not None:
        for name, table in result.items(): table.to_csv(out/f'{name}.csv', index=False)
        (out/'COMPLETE.json').write_text(json.dumps(dict(repeats=repeats)))
    return result


def load_coverage_sampling(output_dir):
    """Load a complete exported run without loading recordings or rerunning PCA."""
    root = Path(output_dir)
    if not (root/'COMPLETE.json').exists():
        raise ValueError('Coverage run is incomplete; no COMPLETE.json marker.')
    config = json.loads((root/'config.json').read_text())
    tables = {}
    for name in ('metrics', 'component_pairs', 'matching', 'selections', 'coverage_mapping',
                 'matching_summary', 'paired_control_deltas'):
        try:
            tables[name] = pd.read_csv(root/f'{name}.csv', dtype={'feature_budget': str})
        except pd.errors.EmptyDataError:
            tables[name] = pd.DataFrame()
    return config, tables


def plot_coverage_sampling(result, k):
    """Median/range across coordinated repetitions; ranges are NOT CIs."""
    import matplotlib.pyplot as plt
    data = result['metrics']
    data = data[(data.k == k) & (data.status == 'ok')]
    if data.empty: raise ValueError('No supported results for this k.')
    figures = []
    for budget, table in data.groupby('feature_budget', sort=False):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for ax, metric in zip(axes, ['ieeg_variance_captured', 'subspace_overlap', 'matched_abs_r']):
            for name, group in table.groupby('dataset', sort=False):
                summary = group.groupby('participant_pool_count')[metric].agg(['median', 'min', 'max'])
                x = summary.index.to_numpy()
                ax.plot(x, summary['median'], 'o-', label=name)
                ax.fill_between(x, summary['min'], summary['max'], alpha=.15)
            ax.set(xlabel='MEG participant pool size', ylabel=metric, ylim=(-.02, 1.02))
        axes[-1].legend(fontsize=8)
        fig.suptitle(f'k={k}, feature budget={budget}; median/range, fixed iEEG reference')
        figures.append(fig)
    # Feature-count effects at a fixed participant pool; native counts cannot
    # share a common x coordinate across setups and are deliberately omitted.
    numeric = data[data.feature_budget.astype(str) != 'native'].copy()
    if not numeric.empty:
        numeric['feature_count'] = numeric.feature_budget.astype(int)
        for count, table in numeric.groupby('participant_pool_count', sort=False):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            for ax, metric in zip(axes, ['ieeg_variance_captured', 'subspace_overlap', 'matched_abs_r']):
                for name, group in table.groupby('dataset', sort=False):
                    summary = group.groupby('feature_count')[metric].agg(['median', 'min', 'max'])
                    x = summary.index.to_numpy()
                    ax.plot(x, summary['median'], 'o-', label=name)
                    ax.fill_between(x, summary['min'], summary['max'], alpha=.15)
                ax.set(xlabel='Sampled MEG feature count', ylabel=metric, ylim=(-.02, 1.02))
            axes[-1].legend(fontsize=8)
            fig.suptitle(f'k={k}, MEG pool={count}; median/range, fixed iEEG reference')
            figures.append(fig)
    return figures
