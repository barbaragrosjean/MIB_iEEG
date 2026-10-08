#!/usr/bin/env python3
"""Fit coherent phase surrogates and prepare baseline/null evaluation snapshots.

python -u plssvd_phase_null.py --perm all --components 5 10
Uses an existing complete out/trial_cache. Output layout matches plssvd_eval.py:
out/plssvd_eval/MEG_KIND/{none,meg__phase,ieeg__phase,both__phase}.
A missing baseline is fitted with identical settings. Completed runs are reused;
partial runs require a new --runs-dir (fits are not silently overwritten).
"""
from pathlib import Path
import argparse
import json
from dataclasses import asdict, replace


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--runs-dir', type=Path)
    parser.add_argument('--scratch-dir', type=Path)
    parser.add_argument('--meg-kind', choices=['full_concatenated','full_average','coverage_average','paired_coverage','random_control'], default='full_concatenated')
    parser.add_argument('--perm', choices=['meg','ieeg','both','all'], default='meg')
    parser.add_argument('--n-iterations', type=int, default=100)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--max-components', type=int, default=100)
    parser.add_argument('--components', type=int, nargs='+', default=[5], help='Prespecified component counts for evaluation.')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--split-unit', choices=['trial','group'], default='trial')
    parser.add_argument('--max-gram-gib', type=float, default=2.)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    from plssvd_eval_utils import ValidationOptions, load_trial_cache, fit_plssvd, validation_run_name, _checked_options
    from plssvd_postprocess import evaluate_run, prepare_comparison
    from plssvd_figures import plot_phase_null_comparison
    if not args.components or min(args.components) < 1 or max(args.components) > args.max_components:
        parser.error('--components must be between 1 and --max-components.')
    root = args.root.expanduser().resolve()
    cache = args.cache_dir or root/'out'/'trial_cache'
    runs = args.runs_dir or root/'out'/'plssvd_eval'/args.meg_kind
    options = _checked_options(ValidationOptions(n_components=args.max_components, repeats=args.n_splits,
        n_iterations=args.n_iterations, seed=args.seed, split_unit=args.split_unit, max_gram_gib=args.max_gram_gib))
    modes = ['meg','ieeg','both'] if args.perm == 'all' else [args.perm]
    trials = None
    for mode in [None] + modes:
        settings = replace(options, perm=mode, perm_type='phase' if mode else None)
        folder = runs/validation_run_name(settings.perm, settings.perm_type)
        if (folder/'validation_options.json').exists():
            saved = json.loads((folder/'validation_options.json').read_text())
            for key in ('n_components','repeats','n_iterations','seed','split_unit','block_scaling','ridge','perm','perm_type'):
                if saved.get(key) != asdict(settings)[key]:
                    raise ValueError(f'{folder}: incompatible {key}; match the existing run settings or use a new --runs-dir.')
            if saved.get('meg_kind') != args.meg_kind or not (folder/'COMPLETE.json').exists():
                raise ValueError(f'{folder}: incompatible or incomplete run; choose a new --runs-dir.')
            print(f'Reusing completed fit: {folder}', flush=True)
        else:
            if trials is None:
                trials = load_trial_cache(cache)
                manifest = json.loads((cache/'manifest.json').read_text())
                for modality in ('ieeg','meg'):
                    if not getattr(trials, modality) or {s.subject for s in getattr(trials, modality)} != set(map(str,manifest['config'][modality+'_subjects'])):
                        raise ValueError('Complete prepare_trial_cache before running phase surrogates.')
            fit_plssvd(trials, args.meg_kind, settings, folder, args.scratch_dir)
        evaluate_run(folder, args.components, cache, args.scratch_dir)
    for k in args.components:
        for mode in modes:
            prepare_comparison(runs, mode+'__phase', k)
        fig = plot_phase_null_comparison(runs, k, modes=modes)
        for extension in ('png','pdf'):
            fig.savefig(runs/f'phase_null_comparison_k{k}.{extension}', dpi=180, bbox_inches='tight')
        import matplotlib.pyplot as plt
        plt.close(fig)
    print(f'Phase results and comparisons saved to {runs}', flush=True)


if __name__ == '__main__':
    main()
