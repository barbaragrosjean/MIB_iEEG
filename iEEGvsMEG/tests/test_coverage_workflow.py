"""Numerical checks for dataset scaling, assignment correspondence, and saved runs."""
import json
import pickle
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vs1_run_coverage import load_datasets, run_analysis, load_pca_results, prepare_analysis, analysis_directory
from vs1_coverage_matching_utils import fit_block_pca, normalize_dataset, variance_summary
from vs1_coverage_stability import load_stability_results


@pytest.fixture
def config(tmp_path):
    rng = np.random.default_rng(10)
    meg, ieeg = tmp_path/'meg', tmp_path/'ieeg'
    meg.mkdir(); ieeg.mkdir()
    xyz = rng.normal(size=(9, 3))*20
    metadata = []
    for i in range(3):
        subject = f's{i}'
        (meg/f'{subject}_source.p').write_bytes(pickle.dumps(rng.normal(size=(2, 9, 24))))
        pd.DataFrame(xyz/1000).to_csv(meg/f'{subject}_pos.csv', index=False)
        if i < 2:
            (ieeg/f'{subject}_epochs.p').write_bytes(pickle.dumps(rng.normal(size=(6, 3, 24))))
            (ieeg/f'{subject}_info.json').write_text(json.dumps(dict(event_id=[1,2,1,2,1,2],time_epoch=(np.arange(24)/250).tolist())))
            for j, pos in enumerate(xyz[i*3:i*3+3]):
                metadata.append(dict(subject=subject,channel_index=j,x=pos[0],y=pos[1],z=pos[2]))
    meta = tmp_path/'metadata.csv'; pd.DataFrame(metadata).to_csv(meta,index=False)
    return dict(data=dict(meg_dir=str(meg),ieeg_dir=str(ieeg),metadata_csv=str(meta),
                         meg_tmin=0.,sfreq=250.,conditions=[1,2]),
                output_dir=str(tmp_path/'out'),block_scaling=True,condition_mode='average',
                n_components=2,seed=12,channel_step=9,channel_repeats=2,stability_repeats=3,max_gram_gib=1.)


@pytest.mark.parametrize('mode', ['average', 'stack'])
def test_scaling_and_pca(config, mode):
    config['condition_mode'] = mode
    datasets = load_datasets(config)
    np.testing.assert_allclose(variance_summary(datasets).total_variance,1.,rtol=1e-6)
    for dataset in datasets.values():
        original = [a.copy() for a in dataset.arrays]
        fitted = fit_block_pca(dataset,2)
        x = np.concatenate(list(dataset.blocks()),axis=1).astype(float)
        x -= x.mean(0)
        u, singular, vt = np.linalg.svd(x,full_matrices=False)
        np.testing.assert_allclose(fitted.explained_variance,singular[:2]**2/(len(x)-1),rtol=1e-6)
        np.testing.assert_allclose(fitted.weights @ fitted.weights.T,vt[:2].T@vt[:2],atol=1e-6)
        normalize_dataset(dataset,False)
        unscaled = fit_block_pca(dataset,2)
        np.testing.assert_allclose(fitted.explained_variance_ratio,unscaled.explained_variance_ratio,rtol=1e-6)
        np.testing.assert_allclose(fitted.weights@fitted.weights.T,unscaled.weights@unscaled.weights.T,atol=1e-6)
        for before, after in zip(original,dataset.arrays): np.testing.assert_array_equal(before,after)


def test_all_stages_and_cache(config):
    datasets = load_datasets(config)
    original_ieeg = datasets['iEEG'].arrays[0].copy()
    for stage in ['manipulations','channels','pca','stability']:
        out = run_analysis(config,stage,datasets)
        assert prepare_analysis(config,stage,Path(config['output_dir'])/'config.json') == out
        assert prepare_analysis(config,stage,Path(config['output_dir'])/'config.json',True) is None
    fitted = load_pca_results(analysis_directory(config,'pca'),datasets)
    np.testing.assert_allclose(fitted['iEEG'].scores, fit_block_pca(datasets['iEEG'],2).scores)
    channels = pd.read_csv(analysis_directory(config,'channels')/'metrics.csv')
    np.testing.assert_allclose(channels.total_variance,1.,rtol=1e-6)
    assert 'pc2_variance' in channels and 'pc3_variance' not in channels
    out = analysis_directory(config,'stability')
    result = load_stability_results(out)
    assert set(result['metrics'].domain)=={'timecourses','weights'}
    np.testing.assert_array_equal(original_ieeg,datasets['iEEG'].arrays[0])
    with np.load(out/'scores.npz') as saved:
        np.testing.assert_allclose(saved['fixed_ieeg_weights'],fitted['iEEG'].weights)
        np.testing.assert_allclose(saved['fixed_ieeg_timecourses'],fitted['iEEG'].scores)
        # Reconstruct baseline spatial weights independently in electrode-slot order.
        mapping = result['source_mapping'].query('repeat == -1').sort_values('electrode_index')
        context = datasets['iEEG'].source_data
        matrix = np.stack([context['meg'][context['meg_subjects'].index(row.meg_subject)][:,row.source_index,:]
                           for row in mapping.itertuples()],axis=1).mean(0).T
        matrix -= matrix.mean(0)
        _, _, vt = np.linalg.svd(matrix,full_matrices=False)
        weights = saved['paired_coverage_r-1_weights']
        np.testing.assert_allclose(weights@weights.T,vt[:2].T@vt[:2],atol=1e-6)
    old_metrics = result['metrics']
    run_analysis(config,'stability',datasets,recompute=True)
    pd.testing.assert_frame_equal(old_metrics,load_stability_results(out)['metrics'])
    assert list((out.parent/'old').iterdir())
    changed = dict(config,block_scaling=False)
    assert analysis_directory(changed,'pca') != analysis_directory(config,'pca')
