"""Synthetic checks for covariance persistence and selected-iteration PLSSVD summaries."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vs2_1_plssvd_eval_utils import load_trial_cache, fit_plssvd, ValidationOptions, _saved_scores
from vs2_1_plssvd_postprocess import evaluate_run
from vs2_1_plssvd_results import prepare_summary, load_summary
from vs2_1_plssvd_fit_pca import run as run_pca, load_comparison
from vs2_1_plssvd_pca import fit_full_data_pca
from vs2_run_cov_models import save_models, load_models, load_selected_datasets
from vs2_0_cov_models_utils import fit_cov_models, evaluate_cov_models
from test_coverage_workflow import config


@pytest.fixture
def trial_cache(tmp_path):
    cache = tmp_path/'cache'; cache.mkdir()
    rng = np.random.default_rng(7)
    times = np.arange(24)/250
    xyz = rng.normal(size=(7,3))*20
    records = []
    for modality, count, channels in [('ieeg',2,3),('meg',3,7)]:
        for i in range(count):
            subject = f's{i}'
            files = []
            for condition in range(2):
                array = rng.normal(size=(9,channels,len(times)))
                array += np.sin(np.arange(len(times))*.4)[None,None,:]*rng.normal(size=(1,channels,1))
                filename = f'{modality}_{subject}_{condition}.npy'
                np.save(cache/filename,array); files.append(filename)
            metadata = pd.DataFrame(xyz[:channels],columns=['x','y','z'])
            metadata['subject'] = subject; metadata['channel_index'] = np.arange(channels)
            records.append(dict(modality=modality,subject=subject,files=files,positions=xyz[:channels].tolist(),
                metadata=metadata.to_dict(orient='list'),split_groups=[None,None],permutation_blocks=[['all']*9]*2))
    (cache/'manifest.json').write_text(json.dumps(dict(config=dict(conditions=[1,2]),times=times.tolist(),records=records)))
    return cache


def test_model_roundtrip(config,tmp_path):
    config['meg_kind']='paired_coverage'
    ieeg,meg=load_selected_datasets(config)
    models=fit_cov_models(ieeg,meg,2,block_scaling='equal_variance')
    save_models(models,tmp_path/'models')
    restored=load_models(tmp_path/'models')
    pd.testing.assert_frame_equal(evaluate_cov_models(models,plot=False),evaluate_cov_models(restored,plot=False))
    for name in models:
        np.testing.assert_array_equal(models[name].ieeg_weights,restored[name].ieeg_weights)
        np.testing.assert_array_equal(models[name].meg_scores,restored[name].meg_scores)


def test_all_fold_iteration_aggregation(trial_cache,tmp_path):
    root=tmp_path/'none'
    options=ValidationOptions(repeats=3,n_iterations=3,n_components=2,block_scaling='equal_variance')
    fit_plssvd(load_trial_cache(trial_cache),'paired_coverage',options,root)
    evaluate_run(root,[2],cache_dir=trial_cache)
    prepare_summary(root,2,n_iterations=2)
    result=load_summary(root,2,2,meg_kind='paired_coverage',block_scaling='equal_variance')
    assert result['metadata']['iterations']==[0,1]
    assert len(result['fold_metrics'])==2*3*2
    covariances=[];correlations=[];timecourses=[]
    for iteration in [0,1]:
        for fold in range(3):
            with _saved_scores(root/f'iteration_{iteration:03d}',fold) as saved:
                train=saved['train_ieeg'][:,:2]
                signs=np.sign(train[np.argmax(abs(train),axis=0),np.arange(2)]); signs[signs==0]=1
                x,y=saved['test_ieeg'][:,:2]*signs,saved['test_meg'][:,:2]*signs
                xc,yc=x-x.mean(0),y-y.mean(0)
                covariances.append(xc.T@yc/(len(x)-1))
                correlations.append(abs(np.corrcoef(x.T,y.T)[:2,2:]))
                timecourses.append(x/train.std(0))
    for key,values in [('test_covariance',covariances),('test_correlation_absolute',correlations),('test_ieeg_normalized',timecourses)]:
        np.testing.assert_allclose(result['arrays'][key+'_mean'],np.mean(values,axis=0),atol=1e-12)
        np.testing.assert_allclose(result['arrays'][key+'_std'],np.std(values,axis=0,ddof=1),atol=1e-12)
        np.testing.assert_array_equal(result['arrays'][key+'_count'],6)
    with pytest.raises(ValueError,match='scaling|block_scaling'):
        load_summary(root,2,2,block_scaling='none')
    # Full-data PCA follows the saved random association, with only the chosen iterations.
    path=run_pca(root,trial_cache,2,meg_kind='paired_coverage',block_scaling='equal_variance',n_iterations=2)
    assert path.exists()
    comparison=load_comparison(root,2,meg_kind='paired_coverage',block_scaling='equal_variance',n_iterations=2)
    assert comparison['mean'].shape==(2,2,3,2,2)
    np.testing.assert_array_equal(comparison['count'],2)
    assert comparison['iterations']==[0,1]


@pytest.mark.parametrize('kind',['full_average','full_concatenated','coverage_average','paired_coverage','random_control'])
def test_full_pca_selected_construction(trial_cache,kind):
    plain=fit_full_data_pca(trial_cache,2,meg_kind=kind,seed=35,block_scaling='none')
    scaled=fit_full_data_pca(trial_cache,2,meg_kind=kind,seed=35,block_scaling='equal_variance')
    assert scaled['meg_kind']==kind
    for modality in ['ieeg','meg']:
        d=scaled['singular_values'][modality]
        np.testing.assert_allclose(np.sum(d*d)/(len(scaled['times'])-1),1.,rtol=1e-6)
        a,b=plain['scores'][modality],scaled['scores'][modality]
        np.testing.assert_allclose(a/np.linalg.norm(a,axis=0),b/np.linalg.norm(b,axis=0),atol=1e-6)
