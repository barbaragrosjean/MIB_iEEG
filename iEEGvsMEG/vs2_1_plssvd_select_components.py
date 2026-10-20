#!/usr/bin/env python3
"""Select a stable leading PLSSVD prefix against a selected null; never refit models."""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


NULL_TYPES = ('time_cirular_shift', 'time_block', 'time_point', 'space', 'phase')


def _null_name(perm, perm_type='phase'):
    if perm not in ('ieeg', 'meg', 'both'):
        raise ValueError('perm must be ieeg, meg or both.')
    if perm_type == 'time_circular_shift':
        perm_type = 'time_cirular_shift'
    if perm_type not in NULL_TYPES:
        raise ValueError(f'perm_type must be one of {NULL_TYPES}.')
    return perm+'__'+perm_type


def _selection_folder(runs_dir, n_components, perm, perm_type='phase'):
    # Preserve the existing both-phase output location.
    root = Path(runs_dir)/'component_selection'
    name = _null_name(perm, perm_type)
    return (root if name == 'both__phase' else root/name)/f'k_{n_components:03d}'


def select_components(components, n_components=10, null_quantile=.95, min_support=.90, perm='both', perm_type='phase'):
    """Input: matched iteration fold means, native component order, signed covariance."""
    if n_components < 1 or not 0 < null_quantile < 1 or not 0 < min_support <= 1:
        raise ValueError('Require positive component count and thresholds in (0,1), with support <=1.')
    name = _null_name(perm, perm_type)
    frame = components.query("partition == 'test'")
    matrices = {}
    for label in ('Unpermuted', 'Permuted'):
        subset = frame[frame.run == label]
        matrix = subset.pivot(index='iteration', columns='component', values='covariance')
        matrix = matrix.reindex(columns=range(1, n_components+1)).sort_index()
        if matrix.empty or not np.isfinite(matrix.to_numpy()).all():
            raise ValueError(f'{label}: missing or nonfinite held-out component covariances.')
        matrices[label] = matrix
    observed, null = matrices['Unpermuted'], matrices['Permuted']
    if not observed.index.equals(null.index):
        raise ValueError('Baseline and null iteration IDs must match.')
    if len(observed) < 2:
        raise ValueError('At least two matched iterations are required; 100 are recommended.')
    threshold = np.quantile(null.to_numpy(), null_quantile, axis=0)
    effective = np.maximum(threshold, 0.)
    support = (observed.to_numpy() > effective).mean(axis=0)
    passed = support >= min_support
    retained = np.logical_and.accumulate(passed)
    table = pd.DataFrame(dict(component=np.arange(1,n_components+1),
        null_threshold=threshold, effective_threshold=effective, support=support,
        passes=passed, retained=retained,
        observed_median=np.median(observed,axis=0),
        observed_low=np.quantile(observed,.05,axis=0),
        observed_high=np.quantile(observed,.95,axis=0)))
    return table, dict(selected_k=int(retained.sum()), n_components=n_components,
        n_iterations=len(observed), iterations=[int(i) for i in observed.index],
        null_quantile=null_quantile, min_support=min_support, permutation=name,
        boundary_supported=bool(retained.all()),
        interpretation='Descriptive robustness selection; not a significance test or independent test performance.')


def prepare_selection(runs_dir, n_components=10, null_quantile=.95, min_support=.90,
                      recompute=False, recompute_evaluation=False, perm='both', perm_type='phase'):
    from vs2_1_plssvd_figures import load_comparison
    runs_dir = Path(runs_dir)
    name = _null_name(perm, perm_type)
    if recompute_evaluation:
        from vs2_1_plssvd_postprocess import evaluate_run, prepare_comparison
        for run_name in ('none', name):
            evaluate_run(runs_dir/run_name, [n_components])
        prepare_comparison(runs_dir, name, n_components)
    data = load_comparison(runs_dir, _null_name(perm, perm_type), n_components)
    folder = _selection_folder(runs_dir, n_components, perm, perm_type)
    settings = dict(source_signature=data['metadata']['signature'],
                    n_components=n_components, null_quantile=null_quantile, min_support=min_support)
    manifest = folder/'selection.json'
    if manifest.exists() and not recompute:
        saved = json.loads(manifest.read_text())
        if any(saved.get(key) != value for key,value in settings.items()):
            raise ValueError('Selection settings or source results changed. Use --recompute.')
        return load_selection(runs_dir, n_components, perm=perm, perm_type=perm_type)
    table, summary = select_components(data['components'], n_components, null_quantile, min_support, perm=perm, perm_type=perm_type)
    folder.mkdir(parents=True, exist_ok=True)
    # Write the manifest last so interrupted output is not treated as complete.
    if manifest.exists():
        manifest.unlink()
    table.to_csv(folder/'selection.csv', index=False)
    data['components'].to_csv(folder/'components.csv', index=False)
    data['metrics'].to_csv(folder/'metrics.csv', index=False)
    manifest.write_text(json.dumps(dict(summary, **settings), indent=2))
    return load_selection(runs_dir, n_components, perm=perm, perm_type=perm_type)


def load_selection(runs_dir, n_components=10, perm='both', perm_type='phase'):
    """Read selection for perm (modality) and perm_type (null transformation).

    Supported types: time_cirular_shift, time_block, time_point, space, phase.
    Defaults preserve historical phase selections. No evaluation or fitting is
    performed; prepare_selection must first create results for the chosen null.
    """
    _null_name(perm, perm_type)
    folder = _selection_folder(runs_dir, n_components, perm, perm_type)
    summary = json.loads((folder/'selection.json').read_text())
    if summary['permutation'] != _null_name(perm, perm_type):
        raise ValueError('Saved selection uses a different null model.')
    # Reject stale results using the standard prepared-comparison provenance check.
    from vs2_1_plssvd_figures import load_comparison
    current = load_comparison(runs_dir, _null_name(perm, perm_type), n_components)
    if current['metadata']['signature'] != summary['source_signature']:
        raise ValueError('Selection is stale. Rerun plssvd_select_components.py --recompute.')
    return dict(folder=folder, summary=summary, **{name:pd.read_csv(folder/(name+'.csv'))
                for name in ('selection','components','metrics')})


def plot_selection(result):
    import matplotlib.pyplot as plt
    table, summary = result['selection'], result['summary']
    fig, axes = plt.subplots(2,2,figsize=(13,9),layout='constrained')
    x = table.component.to_numpy()
    ax = axes[0,0]
    ax.fill_between(x,table.observed_low,table.observed_high,alpha=.2,label='Baseline 5–95% split range')
    ax.plot(x,table.observed_median,'o-',label='Baseline median')
    ax.plot(x,table.effective_threshold,'o--',label=f"{summary['permutation']} {summary['null_quantile']:.0%} threshold (floor 0)")
    ax.set(title='Selection: signed held-out covariance',xlabel='Component rank',ylabel='Covariance');ax.legend(fontsize=8)
    ax=axes[0,1]
    ax.bar(x,table.support,color=['tab:blue' if keep else 'gray' for keep in table.retained])
    ax.axhline(summary['min_support'],color='black',linestyle='--')
    ax.set(title='Selection: fraction of iterations above threshold',xlabel='Component rank',ylabel='Support',ylim=(0,1.05))
    frame=result['components'].query("partition == 'test'")
    ax=axes[1,0]
    for label,color in [('Unpermuted','tab:blue'),('Permuted','tab:orange')]:
        stats=frame[frame.run==label].groupby('component').r
        ranks=stats.median().index
        ax.plot(ranks,stats.median(),'o-',color=color,label=label)
        ax.fill_between(ranks,stats.quantile(.05),stats.quantile(.95),color=color,alpha=.15)
    ax.set(title='Diagnostic: signed correlation (median, 5–95% range)',xlabel='Component rank',ylabel='Pearson r');ax.legend()
    ax=axes[1,1]
    frame=result['metrics'].query("partition == 'test'")
    for j,modality in enumerate(('ieeg','meg')):
        for offset,label,color in [(-.15,'Unpermuted','tab:blue'),(.15,'Permuted','tab:orange')]:
            values=frame.loc[frame.run==label,modality+'_reconstruction_fraction'].dropna()
            if len(values):
                lo,median,hi=np.quantile(values,[.05,.5,.95])
                ax.errorbar(j+offset,median,yerr=[[median-lo],[hi-median]],fmt='o',color=color,
                            label=label if j==0 else None,capsize=4)
    ax.set(xticks=[0,1],xticklabels=['iEEG','MEG'],ylabel='Reconstruction fraction',
           title=f"Diagnostic: reconstruction at evaluated k={summary['n_components']}")
    ax.legend()
    suffix=' (all evaluated ranks supported; evaluate more to locate the boundary)' if summary['boundary_supported'] else ''
    fig.suptitle(f"Stable excess covariance: retain k={summary['selected_k']}{suffix}\n"
                 f"{summary['n_iterations']} matched iterations · {summary['permutation']} · descriptive selection, not a p-value",fontsize=11)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir',type=Path,default=Path(__file__).resolve().parent/f'out/plssvd_eval_equal_var/paired_coverage')
    parser.add_argument('--perm', choices=['ieeg','meg','both'], default='both')
    parser.add_argument('--perm-type', choices=[*NULL_TYPES, 'time_circular_shift'], default='phase')
    parser.add_argument('--components',type=int,default=10)
    parser.add_argument('--null-quantile',type=float,default=.95)
    parser.add_argument('--min-support',type=float,default=.90)
    parser.add_argument('--recompute',action='store_true',help='Recompute selection tables from prepared comparisons.')
    parser.add_argument('--recompute-evaluation',action='store_true',help='Refresh evaluation/comparison from existing baseline and selected null fits; reuse valid cached evaluations. Never refit surrogates.')
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    result=prepare_selection(args.runs_dir,args.components,args.null_quantile,args.min_support,
                             args.recompute or args.recompute_evaluation,args.recompute_evaluation,perm=args.perm,perm_type=args.perm_type)
    fig=plot_selection(result)
    for ext in ('png','pdf'):fig.savefig(result['folder']/('selection.'+ext),dpi=180)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(json.dumps(result['summary'],indent=2))
    print(f"Saved to {result['folder']}")


if __name__=='__main__':main()
