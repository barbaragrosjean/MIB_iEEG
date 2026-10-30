"""Fit and save the three covariance models: python vs2_run_cov_models.py --config CONFIG."""
from pathlib import Path
from dataclasses import fields
import argparse
import json
from datetime import datetime, timezone
import sys
import numpy as np
from vs1_coverage_matching_utils import load_dataset
from vs1_run_coverage import analysis_directory, SCHEMA_VERSION
from vs2_0_cov_models_utils import CovModel, fit_cov_models, evaluate_cov_models
from vs2_0_compare_models import compare_cov_models_within


def load_selected_datasets(config):
    options = dict(config['data'])
    if options.get('meg_tmin') is None and options.get('meg_times_file') is None:
        info = sorted(Path(options['ieeg_dir']).glob('*_info.json'))[0]
        options['meg_tmin'] = float(json.loads(info.read_text())['time_epoch'][0])
    # Whole-modality normalization is applied once, by fit_cov_models.
    ieeg = load_dataset('ieeg', condition_mode=config['condition_mode'], block_scaling=False, **options)
    meg = load_dataset(config['meg_kind'], reference=ieeg, seed=config['seed'])
    return ieeg, meg


def save_models(models, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, model in models.items():
        values = {field.name: getattr(model, field.name) for field in fields(model)}
        arrays = {key: value for key, value in values.items() if isinstance(value, np.ndarray)}
        metadata[name] = {key: value for key, value in values.items() if key not in arrays}
        np.savez_compressed(directory / f'{name}.npz', **arrays)
    (directory / 'models.json').write_text(json.dumps(metadata, indent=2))


def load_models(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'models.json').read_text())
    models = {}
    for name, values in metadata.items():
        with np.load(directory / f'{name}.npz') as saved:
            models[name] = CovModel(**values, **dict(saved))
    return models


def run(config, recompute=False):
    out = analysis_directory(config, 'models')
    if (out/'RUN_COMPLETE.json').exists() and not recompute:
        return out
    if out.exists():
        archive = out.parent/'old'
        archive.mkdir(parents=True, exist_ok=True)
        out.rename(archive/f'{out.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}')
    ieeg, meg = load_selected_datasets(config)
    models = fit_cov_models(ieeg, meg, n_components=config['n_components'],
        block_scaling='equal_variance' if config['block_scaling'] else 'none',
        max_gram_gib=config['max_gram_gib'])
    save_models(models, out)
    evaluate_cov_models(models, plot=False).to_csv(out/'variance.csv', index=False)
    within = compare_cov_models_within(models, ieeg, meg, dimensions=range(1, config['n_components']+1))
    for name, table in within.items(): table.to_csv(out/f'{name}.csv', index=False)
    (out/'RUN_COMPLETE.json').write_text(json.dumps(dict(config, schema_version=SCHEMA_VERSION), indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.extend([str(root.parent), str(root.parent/'LB')])
    print(run(json.loads(args.config.read_text()), args.recompute))


if __name__ == '__main__':
    main()
