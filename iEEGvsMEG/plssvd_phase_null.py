#!/usr/bin/env python3
"""Fit coherent phase surrogates and prepare baseline/null evaluation snapshots.

python -u plssvd_phase_null.py --perm all --components 5 10
Uses an existing complete out/trial_cache. Output layout matches plssvd_eval.py:
out/plssvd_eval/MEG_KIND/{none,meg__phase,ieeg__phase,both__phase}.
A missing baseline is fitted with identical settings. Completed iterations are
reused, including from interrupted runs; incomplete iterations are excluded.
"""
from pathlib import Path
import argparse
import json
from dataclasses import asdict, replace


def reusable_iterations(folder, saved, meg_kind):
    """Use the same per-iteration completion rule as evaluate_run."""
    folder = Path(folder)
    if saved.get('meg_kind') != meg_kind:
        raise ValueError(f'{folder}: saved meg_kind={saved.get("meg_kind")!r}, '
                         f'requested {meg_kind!r}. Select the matching --meg-kind '
                         'or use a new --runs-dir.')
    if saved.get('schema_version') not in (4, 6):
        raise ValueError(f'{folder}: unsupported fit schema {saved.get("schema_version")!r}; '
                         'use a new --runs-dir to fit the current repeated-iteration format.')
    ids = [i for i in range(saved['n_iterations'])
           if (folder/f'iteration_{i:03d}'/'COMPLETE.json').is_file()]
    if not ids:
        raise ValueError(f'{folder}: no completed iterations were found. Complete the '
                         'original fit or use a new --runs-dir; existing files are preserved.')
    return ids


def resolve_max_components(runs, requested, components):
    """Use the baseline fit size unless the caller explicitly overrides it."""
    baseline = Path(runs)/'none'/'validation_options.json'
    saved = json.loads(baseline.read_text()) if baseline.exists() else None
    fitted = saved['n_components'] if saved is not None else 100
    resolved = fitted if requested is None else requested
    if saved is not None and resolved != fitted:
        raise ValueError(f'Existing baseline was fitted with {fitted} components, but '
                         f'--max-components is {resolved}. Omit --max-components to reuse '
                         'the baseline, or use a new --runs-dir for a different fit.')
    if not components or min(components) < 1 or max(components) > resolved:
        raise ValueError(f'--components must be between 1 and the fitted component count '
                         f'({resolved}). To evaluate more components, use a new --runs-dir '
                         'and a larger --max-components.')
    return resolved


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
    parser.add_argument('--max-components', type=int, default=None,
                        help='Components to fit: inherit the existing baseline, otherwise 100.')
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
    root = args.root.expanduser().resolve()
    cache = args.cache_dir or root/'out'/'trial_cache'
    runs = args.runs_dir or root/'out'/'plssvd_eval'/args.meg_kind
    try:
        args.max_components = resolve_max_components(runs, args.max_components, args.components)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'Fitted components: {args.max_components}; evaluation components: {args.components}', flush=True)
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
            ids = reusable_iterations(folder, saved, args.meg_kind)
            print(f'Reusing {len(ids)}/{saved["n_iterations"]} completed iterations: {folder}', flush=True)
            if len(ids) < saved['n_iterations']:
                print('Incomplete iterations are excluded; comparisons use shared completed '
                      'iteration IDs only. This command does not resume interrupted fits.', flush=True)
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
