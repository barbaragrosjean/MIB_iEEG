"""Analytical checks for the retained spatial profiles and offline result loading."""
import copy
import numpy as np
import pandas as pd
from vs3_1_spatial_profiles import analyze_spatial_profiles
from vs3_1_run_modality_specificity import prepare_analysis, result_directory


def test_profiles_match_direct_pairwise_statistics():
    rng = np.random.default_rng(8)
    n = 12
    metadata = pd.DataFrame(dict(x=-np.arange(1, n+1, dtype=float),
                                y=np.zeros(n), z=np.zeros(n), subject=['s1']*n))
    meg = metadata.assign(ieeg_subject='s1', meg_subject='m1', electrode_index=np.arange(n))
    weights = {m: rng.normal(size=(n, 3)) for m in ('iEEG', 'MEG')}
    options = dict(bin_edges=[0, 30], n_permutations=4, n_bootstrap=4, spatial_scope='pooled')
    result = analyze_spatial_profiles(weights, metadata, meg, **options)
    assert not {'patches', 'memberships'} & result.keys()
    a, b = np.triu_indices(n, 1)
    for modality, values in weights.items():
        norm = np.linalg.norm(values, axis=1)
        unit = np.abs(values) / norm[:, None]
        row = result['curves'].query('modality == @modality').iloc[0]
        np.testing.assert_allclose(row.similarity, (unit[a] * unit[b]).sum(axis=1).mean())
        np.testing.assert_allclose(row.magnitude_semivariance,
                                   ((norm[a]-norm[b])**2).mean() / (2*np.var(norm)))
        assert np.isnan(row.ci_low)  # No participant interval for one pooled block.
    # Whole-modality scaling and component sign flips cannot change these metrics.
    changed = {m: 7*w*np.array([-1, 1, -1]) for m, w in weights.items()}
    other = analyze_spatial_profiles(changed, metadata, meg, **options)
    np.testing.assert_allclose(result['curves'].select_dtypes('number'),
                               other['curves'].select_dtypes('number'), equal_nan=True)


def test_cache_separates_fit_and_analysis_settings(tmp_path):
    config = dict(output_dir=str(tmp_path), fit=dict(model='plssvd', block_scaling=True),
                  analyses=dict(profiles=dict(n_components=2)))
    other = copy.deepcopy(config)
    other['analyses']['profiles']['n_components'] = 3
    assert result_directory(config, 'weights') == result_directory(other, 'weights')
    assert result_directory(config, 'profiles') != result_directory(other, 'profiles')
    other['fit']['block_scaling'] = False
    assert result_directory(config, 'weights') != result_directory(other, 'weights')


def test_missing_results_only_print_command(tmp_path, capsys):
    config = dict(output_dir=str(tmp_path), fit={}, analyses={})
    assert prepare_analysis(config, 'weights', tmp_path/'config.json') is None
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith('#')
    assert lines[1].startswith('python -u ')
    assert not (result_directory(config, 'weights')/'RUN_COMPLETE.json').exists()
