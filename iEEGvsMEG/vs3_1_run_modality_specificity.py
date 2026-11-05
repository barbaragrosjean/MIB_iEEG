"""Run modality-specificity analyses on the cluster; notebooks only load results.

python -u vs3_1_run_modality_specificity.py --config CONFIG --analysis weights
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import pickle
import shlex
import sys

import pandas as pd
from vs2_run_cov_models import load_selected_datasets
from vs2_0_cov_models_utils import fit_cov_models
from vs3_1_modality_specificity import shared_coordinate_regions, regional_channel_counts
from vs3_1_spatial_profiles import analyze_spatial_profiles

VERSION = 1


def result_directory(config, analysis):
    payload = dict(version=VERSION, fit=config['fit'])
    if analysis == 'pairing_profiles':
        payload['spatial_summary_version'] = 2  # Full distance range and pair-count controls
    if analysis == 'pairing_weights':
        payload['magnitude_summary_version'] = 2  # Bilateral ROIs and strict threshold counts
    if analysis.startswith('pairing_'):
        payload['pairing'] = config['pairing']
    if analysis != 'weights':
        payload['options'] = config['analyses'][analysis]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return Path(config['output_dir']) / analysis / digest


def load_result(config, analysis):
    """Load locally generated results; a completed run is required."""
    folder = result_directory(config, analysis)
    if not (folder / 'RUN_COMPLETE.json').exists():
        raise FileNotFoundError(f'Analysis has not completed: {folder}')
    # Only load result files produced by this project's runner.
    with (folder / 'result.pkl').open('rb') as stream:
        result = pickle.load(stream)
    if isinstance(result, dict) and 'provenance' in result:
        from vs3_0_spatial_reliability import _fingerprint
        if any(_fingerprint(p['path']) != p for p in result['provenance']):
            raise FileNotFoundError('Source fits changed; recompute the summary.')
    return result


def prepare_analysis(config, analysis, config_file, recompute=False):
    """Print a copyable command and load completed results without computing."""
    config_file = Path(config_file)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(config, indent=2))
    command = ['python', '-u', str(Path(__file__).resolve()), '--config', str(config_file),
               '--analysis', analysis]
    if recompute:
        command.append('--recompute')
    print('# Run analysis:', shlex.join(command), sep='\n')
    if recompute:
        print('# Recompute requested; rerun this cell with RECOMPUTE=False after the job completes.')
        return None
    try:
        result = load_result(config, analysis)
    except FileNotFoundError:
        print('# Results unavailable; run the command above, then rerun this cell.')
        return None
    print('# Loaded:', result_directory(config, analysis))
    return result


def compute_weights(fit):
    ieeg, meg = load_selected_datasets(fit)
    model = fit_cov_models(ieeg, meg, n_components=fit['n_components'],
        block_scaling='equal_variance' if fit['block_scaling'] else 'none',
        max_gram_gib=fit['max_gram_gib'])[fit['model']]
    datasets = {'iEEG': ieeg, 'MEG': meg}
    metadata = shared_coordinate_regions(datasets)
    return dict(weights={'iEEG': model.ieeg_weights, 'MEG': model.meg_weights},
                metadata=metadata, counts=regional_channel_counts(metadata))


def compute(config, analysis):
    if analysis == 'weights':
        return compute_weights(config['fit'])
    if analysis == 'pairing_weights':
        from vs3_1_pairing_specificity import ensure_fits, load_aligned, summarize_magnitudes
        signature = hashlib.sha256(json.dumps(dict(fit=config['fit'], pairing=config['pairing']), sort_keys=True).encode()).hexdigest()[:16]
        source = ensure_fits(config, Path(config['output_dir'])/'pairing_fits'/signature)
        aligned = load_aligned(source, config['fit']['model'], config['pairing']['iterations'])
        return summarize_magnitudes(aligned, **config['analyses'][analysis])
    if analysis == 'pairing_distributions':
        from vs3_1_weight_distributions import compare_weight_distributions
        return compare_weight_distributions(load_result(config, 'pairing_weights'),
                                            **config['analyses'][analysis])
    if analysis == 'pairing_clustering':
        from vs3_1_pairing_clustering import summarize_pairing_clustering
        return summarize_pairing_clustering(load_result(config, 'pairing_weights'),
            seed=config['fit']['seed'], checkpoint_dir=Path(config['output_dir'])/'clustering_checkpoints',
            **config['analyses'][analysis])
    if analysis == 'pairing_profiles':
        from vs3_1_pairing_specificity import summarize_profiles
        return summarize_profiles(load_result(config, 'pairing_weights'),
                                  config['analyses'][analysis], config['fit']['seed'])
    options = dict(config['analyses'][analysis])
    if analysis == 'profiles':
        # These curves require electrode-to-source correspondence in both modalities.
        fit = dict(config['fit'], meg_kind='paired_coverage',
                   n_components=options.pop('n_components'))
        ieeg, meg = load_selected_datasets(fit)
        model = fit_cov_models(ieeg, meg, n_components=fit['n_components'],
            block_scaling='equal_variance' if fit['block_scaling'] else 'none',
            max_gram_gib=fit['max_gram_gib'])[fit['model']]
        return analyze_spatial_profiles(
            {'iEEG': model.ieeg_weights, 'MEG': model.meg_weights},
            ieeg.metadata, meg.metadata, seed=fit['seed'], **options)
    base = load_result(config, 'weights')
    if analysis in ('gradients', 'gradient_comparison'):
        from vs3_1_gradients import analyze_gradients, compare_gradient_models
        models = options.pop('models', None)
        results = {m: analyze_gradients(w, base['metadata'][m], **options)
                   for m, w in base['weights'].items()}
        return compare_gradient_models(results, models=models) if models else results
    if analysis in ('radius_search', 'percent_search'):
        from vs3_1_gradient_radius_search import radius_search, select_radius_candidates
        results = {m: radius_search(w, base['metadata'][m], progress=print, **options)
                   for m, w in base['weights'].items()}
        candidates = {m: select_radius_candidates(r['grid']) for m, r in results.items()}
        return dict(results=results, candidates=candidates)
    raise ValueError(f'Unknown analysis: {analysis}')


def run(config, analysis, recompute=False):
    out = result_directory(config, analysis)
    if (out / 'RUN_COMPLETE.json').exists() and not recompute:
        try:
            load_result(config, analysis)
            return out
        except FileNotFoundError:
            pass
    if analysis in ('pairing_profiles', 'pairing_clustering', 'pairing_distributions'):
        run(config, 'pairing_weights')
    elif analysis not in ('weights', 'profiles', 'pairing_weights'):
        run(config, 'weights')
    result = compute(config, analysis)
    if out.exists():
        archive = out.parent / 'old'
        archive.mkdir(parents=True, exist_ok=True)
        out.rename(archive / f'{out.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}')
    out.mkdir(parents=True)
    with (out / 'result.pkl').open('wb') as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    # Save top-level tables in portable format as well.
    for name, value in (result.items() if isinstance(result, dict) else [('table', result)]):
        if isinstance(value, pd.DataFrame):
            value.to_csv(out / f'{name}.csv', index=False)
    (out / 'RUN_COMPLETE.json').write_text(json.dumps(dict(version=VERSION, config=config), indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--analysis', choices=['weights', 'profiles', 'gradients',
                        'gradient_comparison', 'radius_search', 'percent_search',
                        'pairing_weights', 'pairing_profiles', 'pairing_clustering', 'pairing_distributions'], required=True)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.extend([str(root.parent), str(root.parent / 'LB')])
    print(run(json.loads(args.config.read_text()), args.analysis, args.recompute))


if __name__ == '__main__':
    main()
