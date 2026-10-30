"""Cluster entry point and result I/O for the coverage-matching notebook.

python vs1_run_coverage.py --config out/coverage/config.json --analysis all
"""
from pathlib import Path
import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
import numpy as np
from vs1_coverage_matching_utils import (
    SETUP_NAMES, load_dataset, fit_block_pca, PCAResult,
)

SCHEMA_VERSION = 1


def load_datasets(config):
    """Load once; construct all datasets with reproducible random associations."""
    options = dict(config['data'])
    # Use the verified iEEG epoch start as in the original notebook.
    if options.get('meg_tmin') is None and options.get('meg_times_file') is None:
        info = sorted(Path(options['ieeg_dir']).glob('*_info.json'))[0]
        options['meg_tmin'] = float(json.loads(info.read_text())['time_epoch'][0])
    reference = load_dataset('ieeg', condition_mode=config['condition_mode'],
                             block_scaling=config['block_scaling'], **options)
    datasets = {'iEEG': reference}
    datasets.update({name: load_dataset(name, reference=reference, seed=config['seed'])
                     for name in SETUP_NAMES})
    return datasets


def analysis_directory(config, analysis):
    payload = dict(config, schema_version=SCHEMA_VERSION)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return Path(config['output_dir']) / analysis / digest


def prepare_analysis(config, analysis, config_file, recompute=False):
    """Write the exact run configuration, print its command, and check completion."""
    config_file = Path(config_file)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(config, indent=2))
    command = ['python', '-u', str(Path(__file__).resolve()), '--config', str(config_file),
               '--analysis', analysis]
    if recompute:
        command.append('--recompute')
    print(shlex.join(command))
    out = analysis_directory(config, analysis)
    marker = out / 'RUN_COMPLETE.json'
    if not recompute and marker.exists():
        saved = json.loads(marker.read_text())
        if saved == dict(config, schema_version=SCHEMA_VERSION):
            print(f'Loading completed analysis: {out}')
            return out
    print('Run the command above, then set RECOMPUTE=False and rerun this cell.')
    return None


def load_pca_results(directory, datasets):
    results = {}
    for name, dataset in datasets.items():
        with np.load(Path(directory) / f'{name}.npz') as values:
            results[name] = PCAResult(**{key: values[key].copy() for key in values.files}, dataset=dataset)
    return results


def run_analysis(config, analysis, datasets, recompute=False):
    from vs1_coverage_information import compare_meg_information, save_information
    from vs1_elect_number_effect import run_elect_number_effect
    from vs1_coverage_stability import run_pairing_stability
    out = analysis_directory(config, analysis)
    if (out / 'RUN_COMPLETE.json').exists() and not recompute:
        print(f'Already complete: {out}', flush=True)
        return out
    if out.exists():
        # Preserve interrupted or explicitly superseded outputs, never mix runs.
        archive = out.parent / 'old'
        archive.mkdir(parents=True, exist_ok=True)
        out.rename(archive / f'{out.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}')
    out.mkdir(parents=True)
    k = config['n_components']
    memory = config['max_gram_gib']
    if analysis == 'manipulations':
        result = compare_meg_information(datasets, dimensions=list(range(1, k+1)), max_gram_gib=memory)
        save_information(result, out)
    elif analysis == 'channels':
        run_elect_number_effect(datasets['full_concatenated'], datasets['iEEG'],
            step=config['channel_step'], repeats=config['channel_repeats'], seed=config['seed'],
            n_components=k, output_dir=out, max_gram_gib=memory)
    elif analysis == 'pca':
        for name, dataset in datasets.items():
            result = fit_block_pca(dataset, k, max_gram_gib=memory)
            np.savez_compressed(out / f'{name}.npz', scores=result.scores, weights=result.weights,
                explained_variance=result.explained_variance,
                explained_variance_ratio=result.explained_variance_ratio, total_variance=result.total_variance)
    elif analysis == 'stability':
        run_pairing_stability(datasets['iEEG'], repeats=config['stability_repeats'],
            n_components=k, seed=config['seed'], baseline_pairing=datasets['paired_coverage'].pairing,
            output_dir=out, max_gram_gib=memory)
    else:
        raise ValueError(f'Unknown analysis: {analysis}')
    (out / 'RUN_COMPLETE.json').write_text(json.dumps(dict(config, schema_version=SCHEMA_VERSION), indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--analysis', choices=['all', 'manipulations', 'channels', 'pca', 'stability'], default='all')
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(__file__).resolve().parent
    sys.path.extend([str(root.parent), str(root.parent / 'LB')])
    datasets = load_datasets(config)
    for analysis in (['manipulations', 'channels', 'pca', 'stability'] if args.analysis == 'all' else [args.analysis]):
        run_analysis(config, analysis, datasets, args.recompute)


if __name__ == '__main__':
    main()
