"""Run any analysis under saved (-1) or independently sampled subject assignments.

python vs0_run_pairings.py --view full --n-pairings 100 -- python ANALYSIS.py ...
The outer assignment stream is independent of evaluation/trial/permutation streams.
"""
from pathlib import Path
import argparse
import json
import hashlib
import os
import re
import shlex
import subprocess
import sys
from contextlib import contextmanager


def activate(view='matched', n_pairings=100, seed=2026, *, display_only=False):
    """Select a display context; computational commands always run the full batch."""
    if view not in ('matched', 'full'):
        raise ValueError("PAIRING_VIEW must be 'matched' or 'full'.")
    if not isinstance(n_pairings, int) or n_pairings < 1:
        raise ValueError('N_PAIRINGS must be a positive integer.')
    os.environ['SUBJECT_PAIRING_ITERATION'] = '-1' if display_only or view == 'matched' else '0'
    os.environ['SUBJECT_PAIRING_SEED'] = str(seed)
    os.environ['SUBJECT_PAIRING_VIEW'] = view
    os.environ['SUBJECT_PAIRING_COUNT'] = str(n_pairings)
    return view


def command(command, view='matched', n_pairings=100, seed=2026, scope='all'):
    return shlex.join(['python', '-u', str(Path(__file__).resolve()),
                       '--view', view, '--scope', scope, '--n-pairings', str(n_pairings), '--seed', str(seed),
                       '--', *map(str, command)])


def _replace(value, iteration):
    if isinstance(value, dict):
        return {k: _replace(v, iteration) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace(v, iteration) for v in value]
    if isinstance(value, str):
        return re.sub(r'(/subject_pairings/[^/]+/iteration_)-?\d+(?=/|$)',
                      lambda m: m[1] + str(iteration), value)
    return value


def run(command, view, n_pairings, seed, scope='all'):
    if scope not in ('all', 'matched', 'once'):
        raise ValueError('scope must be all, matched or once')
    ids = [-1, *range(n_pairings)] if scope == 'all' else [-1]
    for iteration in ids:
        env = dict(os.environ, SUBJECT_PAIRING_ITERATION=str(iteration), SUBJECT_PAIRING_SEED=str(seed),
                   SUBJECT_PAIRING_COUNT=str(n_pairings), SUBJECT_PAIRING_BATCH_CHILD='1')
        args = [_replace(arg, iteration) for arg in command]
        from vs0_load_and_config import analysis_output_root
        with pairing_context(iteration, seed):
            csv = args[args.index('--matched-pairs-csv')+1] if '--matched-pairs-csv' in args else None
            if csv is None and '--config' in command:
                initial = json.loads(Path(command[command.index('--config')+1]).read_text())
                csv = initial.get('data',initial.get('fit',{}).get('data',{})).get('matched_pairs_csv')
            namespace_root = analysis_output_root(Path(__file__).parent, matched_pairs_csv=csv)
        def routed(value):
            if '/subject_pairings/' in str(value):
                return _replace(str(value), iteration)
            path=Path(value)
            try:relative=path.resolve().relative_to(Path(__file__).parent/'out')
            except ValueError:relative=Path(path.name)
            return str(namespace_root/relative)
        for flag in ('--output-dir','--out-path','--run-dir','--runs-dir'):
            if flag in args:
                at=args.index(flag)+1;args[at]=routed(args[at])
        # Derive an independent configuration with paths for this assignment.
        if '--config' in args:
            at = args.index('--config') + 1
            original = Path(command[at])
            config = _replace(json.loads(original.read_text()), iteration)
            if 'output_dir' in config:
                config['output_dir'] = routed(config['output_dir'])
                config['n_pairings'] = n_pairings
                config['pairing_seed'] = seed
            target = Path(config.get('output_dir', namespace_root))/'batch_configs'/(original.stem + f'_pairing_{iteration}.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(config, indent=2))
            args[at] = str(target)
        # Successful command receipts allow an interrupted ensemble to resume.
        namespace = next((re.search(r'.*/subject_pairings/[^/]+/iteration_-?\d+(?=/|$)', arg)
                          for arg in args if '/subject_pairings/' in arg), None)
        receipt = None
        from vs0_load_and_config import behavior_provenance
        signature = dict(command=args, seed=seed, iteration=iteration, meg_behavior=behavior_provenance(),
                         config=config if '--config' in args else None,
                         code={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(__file__).parent.glob('*.py'))})
        if namespace is None:
            namespace = [str(namespace_root)]
        if namespace is not None:
            key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            receipt = Path(namespace[0])/'pairing_commands'/f'{key}.json'
            if receipt.exists() and '--recompute' not in args:
                print(f'Subject pairing {iteration}: command already completed.', flush=True)
                continue
        print(f'Subject pairing {iteration}: {shlex.join(args)}', flush=True)
        subprocess.run(args, env=env, check=True)
        if receipt is not None:
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps(signature, indent=2))


    # These metrics depend on variability across assignments, not within one fit.
    if scope == 'all' and any(Path(arg).name == 'vs3_0_weight_stability.py' for arg in command):
        from vs0_pairing_results import finalize_weight_batch
        output = command[command.index('--output-dir')+1] if '--output-dir' in command else str(namespace_root/'weight_stability'/'paired_coverage')
        roi = command[command.index('--roi-level')+1] if '--roi-level' in command else 'region'
        finalize_weight_batch(output,n_pairings,roi)


@contextmanager
def pairing_context(iteration, seed=2026):
    """Temporary assignment context; never modifies trial/permutation seeds."""
    keys = ('SUBJECT_PAIRING_ITERATION', 'SUBJECT_PAIRING_SEED')
    before = {k: os.environ.get(k) for k in keys}
    os.environ[keys[0]], os.environ[keys[1]] = str(iteration), str(seed)
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def assignment_config(config, iteration):
    """Resolve one saved assignment, independently of the notebook view."""
    return _replace(config, iteration)


def terminal_batch(script, scope='all'):
    """Terminal entry points run the complete batch unless called as a child."""
    if os.environ.get('SUBJECT_PAIRING_BATCH_CHILD') == '1' or '--help' in sys.argv or '-h' in sys.argv:
        return False
    # Notebook display contexts must never shrink a computational batch.
    if os.environ.get('SUBJECT_PAIRING_SINGLE_RUN') == '1':
        return False
    options = json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text()) if '--config' in sys.argv else {}
    run([sys.executable, str(Path(script).resolve()), *sys.argv[1:]], 'full',
        int(os.environ.get('SUBJECT_PAIRING_COUNT', options.get('n_pairings',100))),
        int(os.environ.get('SUBJECT_PAIRING_SEED', options.get('pairing_seed',2026))), scope)
    return True


def summarize(root, n_pairings, relative_csv, keys, metrics):
    """Equal weight per assignment; average internal folds before mean/sample SD.

    CSV paths must be the same relative path in every assignment. Explicit keys
    and metrics prevent participant IDs or component indices being averaged.
    Missing assignments fail instead of silently changing the denominator.
    """
    import pandas as pd
    frames = []
    for iteration in range(n_pairings):
        frame = pd.read_csv(Path(root)/f'iteration_{iteration}'/relative_csv)
        if keys:
            frame = frame.groupby(keys, dropna=False, as_index=False)[metrics].mean()
        else:
            frame = pd.DataFrame([frame[metrics].mean()])
        frame['pairing_iteration'] = iteration
        frames.append(frame)
    values = pd.concat(frames, ignore_index=True)
    if keys:
        summary = values.groupby(keys, dropna=False)[metrics].agg(['mean', 'std', 'count'])
    else:
        summary = values[metrics].agg(['mean', 'std', 'count']).T
    return values, summary


def plot_summary(values, metric, x, group=None):
    """Plot assignment means ± sample SD; assignment is the unit of variation."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    groups = values.groupby(group, dropna=False) if group else [(metric, values)]
    for label, frame in groups:
        table = frame.groupby(x)[metric].agg(['mean', 'std']).sort_index()
        ax.errorbar(table.index, table['mean'], yerr=table['std'], label=str(label), capsize=3)
    ax.set(xlabel=x, ylabel=metric, title='Random subject assignments: mean ± SD')
    ax.legend()
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--view', choices=['matched', 'full'], default='full')
    p.add_argument('--scope', choices=['all','matched','once'], default='all')
    p.add_argument('--n-pairings', type=int, default=100)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.n_pairings < 1:
        p.error('--n-pairings must be positive')
    cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not cmd:
        p.error('Provide the analysis command after --')
    run(cmd, a.view, a.n_pairings, a.seed, a.scope)



def format_command(args, scope='all'):
    """Notebook-aware cluster command, with a separate outer assignment loop."""
    view = os.environ.get('SUBJECT_PAIRING_VIEW')
    if view is None:
        return shlex.join(list(map(str, args)))
    return command(args, view, int(os.environ['SUBJECT_PAIRING_COUNT']),
                   int(os.environ.get('SUBJECT_PAIRING_SEED', 2026)), scope=scope)


def display_ensemble(root, n_pairings, patterns):
    """Display scalar metrics and mean ± SD curves from completed assignments.

    Each pattern must identify one table per assignment. Identical analysis keys
    are required; incomplete/mismatched collections are reported explicitly.
    Coefficients and component maps must be aligned separately before averaging.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    identifiers = {'k', 'component', 'dimension', 'n_components', 'n_channels',
                   'n_clusters', 'distance_bin', 'distance_mm', 'bin_center',
                   'bin_center_mm', 'radius_mm', 'percentile'}
    exclude = {'iteration', 'repeat', 'fold', 'pairing_iteration', 'association',
               'iteration_a', 'iteration_b', 'reference_iteration'}
    outputs = {}
    for pattern in patterns:
        frames = []
        try:
            for iteration in range(n_pairings):
                assignment_root = Path(root)/f'iteration_{iteration}'
                if '/CURRENT/' in pattern:
                    prefix, filename = pattern.split('/CURRENT/', 1)
                    pointer = assignment_root/prefix/'CURRENT.json'
                    snapshot = json.loads(pointer.read_text())['snapshot']
                    paths = [assignment_root/prefix/snapshot/filename]
                else:
                    paths = [p for p in assignment_root.glob(pattern)
                             if 'old' not in p.parts and '.previous_runs' not in p.parts]
                if len(paths) != 1:
                    raise ValueError(f'Pairing {iteration}: expected one {pattern}, found {len(paths)}.')
                parents = [paths[0].parent, *paths[0].parents]
                completed = any((parent/marker).is_file()
                                for parent in parents if parent == assignment_root or assignment_root in parent.parents
                                for marker in ('RUN_COMPLETE.json', 'COMPLETE.json', 'manifest.json'))
                if not completed:
                    raise ValueError(f'No completion marker for {paths[0]}')
                if paths[0].suffix == '.npz':
                    with np.load(paths[0]) as saved:
                        columns = {c: saved[c] for c in ('explained_variance', 'explained_variance_ratio') if c in saved}
                    frame = pd.DataFrame(columns)
                    frame['component'] = np.arange(1, len(frame)+1)
                else:
                    frame = pd.read_csv(paths[0])
                if frame.empty:
                    raise ValueError(f'Empty table: {paths[0]}')
                if not frames:
                    table_identifiers = set(identifiers)
                    if 'cdf' in frame or 'histogram_iterations' in pattern:
                        table_identifiers.add('magnitude')
                    if 'pairing_clustering' in pattern:
                        table_identifiers.discard('n_clusters')
                    keys = [c for c in frame if c not in exclude and
                            (c in table_identifiers or not pd.api.types.is_numeric_dtype(frame[c]))]
                    metrics = [c for c in frame.select_dtypes(include='number') if c not in set(keys) | exclude]
                    if not metrics:
                        raise ValueError(f'No scalar metrics in {pattern}.')
                # Average internal resampling first; assignments receive equal weight.
                if keys:
                    frame = frame.groupby(keys, dropna=False)[metrics].mean().reset_index()
                    if frames and set(map(tuple, frame[keys].fillna('<NA>').to_numpy())) != set(map(tuple, frames[0][keys].fillna('<NA>').to_numpy())):
                        raise ValueError(f'Analysis keys differ across assignments: {pattern}')
                else:
                    frame = pd.DataFrame([frame[metrics].mean()])
                frame['pairing_iteration'] = iteration
                frames.append(frame)
            values = pd.concat(frames, ignore_index=True)
            summary = (values.groupby(keys, dropna=False)[metrics].agg(['mean', 'std', 'count'])
                       if keys else values[metrics].agg(['mean', 'std', 'count']).T)
            print(pattern + f': {n_pairings} random assignments; SD is across assignments.')
            display(summary)
            outputs[pattern] = dict(values=values, summary=summary)
            # Scalar metric curves retain all analysis strata.
            x = next((c for c in keys if c in table_identifiers and pd.api.types.is_numeric_dtype(values[c])), None)
            strata = [c for c in keys if c != x]
            for metric in metrics:
                fig, ax = plt.subplots(figsize=(10, 4))
                groups = values.groupby(strata, dropna=False) if strata else [(metric, values)]
                for label, frame in groups:
                    if x:
                        table = frame.groupby(x)[metric].agg(['mean', 'std']).sort_index()
                        ax.errorbar(table.index, table['mean'], yerr=table['std'], label=str(label), capsize=3)
                    else:
                        # One scalar per stratum, averaged over assignments.
                        ax.errorbar(str(label), frame[metric].mean(), yerr=frame[metric].std(ddof=1), fmt='o', capsize=3)
                ax.set(xlabel=x or 'Analysis stratum', ylabel=metric,
                       title=f'{metric}: mean ± SD across {n_pairings} subject assignments')
                if x and strata:
                    ax.legend(fontsize=7)
                fig.tight_layout()
                plt.show()
        except (OSError, ValueError, KeyError) as exc:
            print(f'Ensemble summary unavailable: {exc}')
    return outputs


if __name__ == '__main__':
    main()
