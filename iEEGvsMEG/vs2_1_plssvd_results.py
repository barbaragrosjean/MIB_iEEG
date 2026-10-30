"""Aggregate every selected fold–iteration result; no fitting or plotting.

python vs2_1_plssvd_results.py --run-dir RUN --components 10 --iterations 20
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from vs2_1_plssvd_postprocess import snapshot_path, Moments, _fingerprint
from vs2_1_plssvd_eval_utils import _saved_scores
from vs2_0_compare_models import correlation_matrix


def selected_iterations(available, n_iterations=None):
    ids = sorted(available)
    if n_iterations is not None:
        if n_iterations < 1 or n_iterations > len(ids):
            raise ValueError(f'Request 1–{len(ids)} completed iterations, or None for all.')
        ids = ids[:n_iterations]
    return ids


def summary_directory(run_dir, k, n_iterations=None):
    count = 'all' if n_iterations is None else str(n_iterations)
    return Path(run_dir)/'summaries'/f'k_{k:03d}'/f'iterations_{count}'


def _source(run_dir, k, n_iterations):
    snapshot = snapshot_path(run_dir, k)
    metadata = json.loads((snapshot/'manifest.json').read_text())
    ids = selected_iterations(metadata['iterations'], n_iterations)
    paths = [snapshot/'manifest.json', snapshot/'fold_metrics.csv']
    for iteration in ids:
        for fold in range(metadata['options']['repeats']):
            child = Path(run_dir)/f'iteration_{iteration:03d}'
            paths.extend([child/f'scores_{fold:03d}.npz', child/f'model_{fold:03d}.npz'])
    return snapshot, metadata, ids, _fingerprint(paths)


def aggregate_evaluation(run_dir, k, n_iterations=None):
    snapshot, metadata, ids, fingerprint = _source(run_dir, k, n_iterations)
    metrics = pd.read_csv(snapshot/'fold_metrics.csv')
    metrics = metrics[metrics.iteration.isin(ids)].copy()
    moments = {}
    def add(key, values):
        moments.setdefault(key, Moments()).add(values)
    for iteration in ids:
        for fold in range(metadata['options']['repeats']):
            with _saved_scores(Path(run_dir)/f'iteration_{iteration:03d}', fold) as saved:
                scores = {p: {m: saved[f'{p}_{m}'][:, :k] for m in ('ieeg','meg')} for p in ('train','test')}
            train = scores['train']['ieeg']
            signs = np.sign(train[np.argmax(abs(train), axis=0), np.arange(k)])
            signs[signs == 0] = 1
            for part in ('train','test'):
                x, y = (scores[part][m]*signs for m in ('ieeg','meg'))
                add(part+'_covariance', (x-x.mean(0)).T@(y-y.mean(0))/(len(x)-1))
                r = correlation_matrix(x, y)
                add(part+'_correlation_absolute', abs(r))
                mask = (metrics.iteration == iteration) & (metrics['repeat'] == fold) & (metrics.partition == part)
                metrics.loc[mask, 'mean_abs_r'] = np.nanmean(abs(np.diag(r)))
                for modality in ('ieeg','meg'):
                    sd = scores['train'][modality].std(0)
                    values = scores[part][modality]*signs / np.where(sd > 0, sd, 1.)
                    add(f'{part}_{modality}_normalized', values)
    arrays = {}
    for key, moment in moments.items():
        for suffix, value in zip(('mean','std','count'), moment.arrays()): arrays[key+'_'+suffix] = value
    with np.load(snapshot/'arrays.npz') as saved:
        arrays.update(times=saved['times'], conditions=saved['conditions'])
    metadata = dict(metadata, iterations=ids, aggregation='all selected fold-iteration results',
                    summary_fingerprint=fingerprint, summary_version=1)
    return dict(metadata=metadata, arrays=arrays, fold_metrics=metrics)


def prepare_summary(run_dir, k, n_iterations=None):
    result = aggregate_evaluation(run_dir, k, n_iterations)
    out = summary_directory(run_dir, k, n_iterations)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'arrays.npz', **result['arrays'])
    result['fold_metrics'].to_csv(out/'fold_metrics.csv', index=False)
    (out/'manifest.json').write_text(json.dumps(result['metadata'], indent=2))
    return out


def load_summary(run_dir, k, n_iterations=None, *, meg_kind=None, block_scaling=None):
    out = summary_directory(run_dir, k, n_iterations)
    metadata = json.loads((out/'manifest.json').read_text())
    _, _, ids, fingerprint = _source(run_dir, k, n_iterations)
    if metadata['iterations'] != ids or metadata.get('summary_fingerprint') != fingerprint:
        raise ValueError('Summary is stale; rerun vs2_1_plssvd_results.py.')
    for key, expected in [('meg_kind', meg_kind), ('block_scaling', block_scaling)]:
        if expected is not None and metadata['options'][key] != expected:
            raise ValueError(f'Saved {key} differs from the notebook configuration.')
    with np.load(out/'arrays.npz') as saved: arrays = dict(saved)
    return dict(metadata=metadata, arrays=arrays, fold_metrics=pd.read_csv(out/'fold_metrics.csv'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--components', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=None)
    args = parser.parse_args()
    print(prepare_summary(args.run_dir, args.components, args.iterations))


if __name__ == '__main__': main()
