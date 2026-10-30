"""Controlled resampling and mapped-component summaries across MEG constructions."""
from pathlib import Path
from types import SimpleNamespace
import sys
import json
import numpy as np
import pandas as pd
import nibabel as nib
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vs3_0_compare_subspace import run_comparison, load_results
from vs3_0_spatial_reliability import fit_iterations, align_bundle, _load_bundle
from vs3_0_weight_stability import summarize_weights, load_weight_summary, roi_pair_correlations


@pytest.fixture
def stability_inputs(tmp_path):
    cache = tmp_path/'cache'; cache.mkdir()
    rng = np.random.default_rng(7)
    times = np.arange(24)/250
    xyz = rng.normal(size=(7,3))*20
    records = []
    for modality, count, channels in [('ieeg',2,3),('meg',3,7)]:
        for i in range(count):
            subject = f's{i}'; files = []
            for condition in range(2):
                values = rng.normal(size=(9,channels,len(times)))
                values += np.sin(np.arange(len(times))*.4)[None,None,:]*rng.normal(size=(1,channels,1))
                filename = f'{modality}_{subject}_{condition}.npy'
                np.save(cache/filename,values); files.append(filename)
            metadata = pd.DataFrame(xyz[:channels],columns=['x','y','z'])
            metadata['subject'] = subject; metadata['channel_index'] = np.arange(channels)
            records.append(dict(modality=modality,subject=subject,files=files,positions=xyz[:channels].tolist(),
                metadata=metadata.to_dict(orient='list'),split_groups=[None,None],permutation_blocks=[['all']*9]*2))
    manifest = dict(config=dict(conditions=[1,2]),times=times.tolist(),records=records)
    for m in ['ieeg','meg']:
        manifest['config'][m+'_subjects'] = [r['subject'] for r in manifest['records'] if r['modality']==m]
    (cache/'manifest.json').write_text(json.dumps(manifest))
    # Offline test atlas with two bilateral regions, covering all synthetic coordinates.
    data = np.ones((21,21,21),dtype=np.int16);data[:,11:,:]=2
    affine=np.diag([10.,10.,10.,1.]);affine[:3,3]=-100.
    atlas=tmp_path/'atlas.nii.gz';nib.save(nib.Nifti1Image(data,affine),atlas)
    labels=tmp_path/'labels.csv'
    pd.DataFrame(dict(atlas_id=[1,2],parcel=['A','B'],region=['A','B'],group=['A','B'])).to_csv(labels,index=False)
    return cache,atlas,labels


def stability_args(inputs,root,variation='pairing',kind='paired_coverage',models=('separate_pca',)):
    cache,atlas,labels=inputs
    return SimpleNamespace(cache_dir=cache,output_dir=root,variation=variation,weight_only=True,
        meg_kind=kind,models=list(models),iterations=3,components=2,seed=2026,split_unit='trial',
        block_scaling='equal_variance',max_gram_gib=1.,scratch_dir=None,ieeg_coords_dir=None,
        meg_coords_dir=None,ieeg_coords_unit='mm',meg_coords_unit='mm',atlas_image=atlas,
        atlas_labels=labels,pool_hemispheres=False)


@pytest.mark.parametrize('kind',['full_average','full_concatenated','coverage_average','paired_coverage','random_control'])
def test_fixed_ieeg_and_coverage_slots(stability_inputs,tmp_path,kind):
    root=tmp_path/kind
    fit_iterations(stability_args(stability_inputs,root,kind=kind))
    summaries=summarize_weights(root)
    anchor=_load_bundle(root/'iteration_000/separate_pca_A.npz')
    n=len(pd.read_csv(root/'iteration_000/ieeg_features.csv'))
    for i in range(3):
        current=_load_bundle(root/f'iteration_{i:03d}/separate_pca_A.npz')
        assert current['meg_weights'].shape==(n,2)
        np.testing.assert_allclose(current['ieeg_weights'],anchor['ieeg_weights'],atol=1e-12)
        np.testing.assert_allclose(current['ieeg_train_scores'],anchor['ieeg_train_scores'],atol=1e-12)
    temporal=summaries['timecourse_correlations'].query("modality == 'ieeg'")
    np.testing.assert_allclose(temporal.signed_r,1.)
    # Constant iEEG trajectories must not become apparent cross-modal covariation.
    assert summaries['roi_cross_correlations'].mean_r.isna().all()
    loaded=load_weight_summary(root,model='separate_pca',component=1,meg_kind=kind,
        block_scaling='equal_variance',n_components=2)
    assert loaded['config']['meg_kind']==kind
    assert set(loaded['timecourse_correlations'].component)=={1}


def test_controlled_streams_and_temporal_summary(stability_inputs,tmp_path):
    results={}
    for variation in ['trials','pairing','both']:
        root=tmp_path/variation
        fit_iterations(stability_args(stability_inputs,root,variation=variation,models=('plssvd','joint_pca')))
        results[variation]=summarize_weights(root)
    for variation in ['pairing','both']:
        a=_load_bundle(tmp_path/'trials/iteration_000/plssvd_A.npz')
        b=_load_bundle(tmp_path/variation/'iteration_000/plssvd_A.npz')
        np.testing.assert_array_equal(a['meg_weights'],b['meg_weights'])
    for i in range(3):
        folder=f'iteration_{i:03d}'
        assert (tmp_path/'trials'/folder/'splits.json').read_bytes()==(tmp_path/'both'/folder/'splits.json').read_bytes()
        assert (tmp_path/'pairing'/folder/'pairing.json').read_bytes()==(tmp_path/'both'/folder/'pairing.json').read_bytes()
    ref=_load_bundle(tmp_path/'trials/iteration_000/plssvd_A.npz')
    current,_=align_bundle(ref,_load_bundle(tmp_path/'trials/iteration_001/plssvd_A.npz'),'plssvd')
    expected=np.corrcoef(ref['meg_train_scores'][:,0],current['meg_train_scores'][:,0])[0,1]
    actual=results['trials']['timecourse_correlations'].query("model == 'plssvd' and modality == 'meg' and component == 1 and iteration_a == 0 and iteration_b == 1").signed_r.iloc[0]
    np.testing.assert_allclose(actual,expected)


def test_comparison_without_native_clouds(stability_inputs,tmp_path):
    cache,_,_=stability_inputs
    root=tmp_path/'comparison'
    run_comparison(cache,root,models=['separate_pca','plssvd'],dimensions=[1,2],repeats=3,cluster_counts=[2])
    config,tables=load_results(root)
    assert config['schema_version']==4
    assert set(tables['overlap'].model)=={'separate_pca','plssvd'}
    assert set(tables['overlap'].partition)=={'in_sample','train','test'}
    assert not list(root.glob('*cloud*'))
    assert 'cloud_metrics' not in tables


def test_signed_roi_pair_aggregation():
    trajectory=np.arange(5.)
    x=np.stack([trajectory,-trajectory],axis=1)[:,:,None]
    y=np.stack([trajectory,trajectory],axis=1)[:,:,None]
    rows=roi_pair_correlations(x,y,pd.DataFrame({'region':['A_L','A_R']}))
    assert len(rows)==1
    np.testing.assert_allclose(rows[0]['mean_r'],0.)
    np.testing.assert_allclose(rows[0]['sd_r'],np.std([1.,1.,-1.,-1.],ddof=1))
    assert rows[0]['n_pairs']==4
