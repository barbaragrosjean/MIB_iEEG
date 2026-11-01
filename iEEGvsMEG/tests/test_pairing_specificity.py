"""Pairing variability, component alignment, and saved-fit reuse regressions."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
from vs3_0_spatial_reliability import _fingerprint
from vs3_0_weight_stability import controlled_seeds
from vs3_1_pairing_specificity import (compatible_run, load_aligned, summarize_magnitudes,
                                     summarize_profiles, ensure_fits)
from vs3_1_run_modality_specificity import run, load_result


@pytest.fixture
def saved_pairing(tmp_path):
    rng = np.random.default_rng(32)
    root = tmp_path/'pairing'; root.mkdir()
    cache = tmp_path/'cache'; cache.mkdir()
    (cache/'manifest.json').write_text(json.dumps({'config': {'conditions': [1,2]}}))
    cfg = dict(schema_version=2, variation='pairing', meg_kind='paired_coverage',
               components=3, seed=2026, split_unit='trial', condition_mode='average',
               block_scaling='equal_variance', models=['plssvd'], iterations=3,
               provenance=[_fingerprint(cache/'manifest.json')])
    (root/'config.json').write_text(json.dumps(cfg))
    n=18
    anchor=pd.DataFrame(dict(x=-np.arange(1,n+1)*2.,y=np.zeros(n),z=np.zeros(n),
                             subject=np.repeat(['s1','s2','s3'],6),source_index=np.tile(np.arange(6),3),
                             region=np.tile(['A','B'],9)))
    reference = rng.normal(size=(40,3))
    for i in range(3):
        folder=root/f'iteration_{i:03d}';folder.mkdir()
        anchor.to_csv(folder/'ieeg_features.csv', index=False)
        meg=anchor.assign(ieeg_subject=anchor.subject, meg_subject='m'+anchor.subject,
                          electrode_index=np.arange(n))
        meg.to_csv(folder/'meg_features.csv', index=False)
        bundle={}
        for m in ['ieeg','meg']:
            bundle[m+'_weights']=rng.normal(size=(n,3))
            bundle[m+'_train_scores']=reference[:,[1,2,0]]*[-1,1,-1] if i else reference
        np.savez(folder/'plssvd_A.npz',**bundle)
        (folder/'splits.json').write_text(json.dumps({'A': {'fixed': [1,3,5]}}))
        a,b=controlled_seeds(2026,i,'pairing')
        (folder/'COMPLETE.json').write_text(json.dumps(dict(split_seed=a,pair_seed=b)))
    fit=dict(model='plssvd',n_components=3,seed=2026,block_scaling=True,condition_mode='average',
             data=dict(meg_scaling='channel_zscore',ieeg_multiplier=1000.,conditions=[1,2]))
    config=dict(output_dir=str(tmp_path/'results'),fit=fit,
                pairing=dict(existing_dir=str(root),cache_dir=str(cache),iterations=3,split_unit='trial'),
                analyses=dict(pairing_weights=dict(normalize=True,percentile=75,bins=8),
                    pairing_profiles=dict(n_components=2, bin_edges=[0,8,40],n_permutations=3,
                                          n_bootstrap=3,spatial_scope='within_participant')))
    return root, config


def test_reuse_and_reject_varying_trials(saved_pairing):
    root,cfg=saved_pairing
    assert compatible_run(root,cfg['fit'],cfg['pairing'])[0]
    assert ensure_fits(cfg,root/'unused')==root
    (root/'iteration_001/splits.json').write_text('{}')
    assert not compatible_run(root,cfg['fit'],cfg['pairing'])[0]


def test_thresholds_match_separate_iteration_calculation(saved_pairing):
    root,cfg=saved_pairing
    aligned=load_aligned(root,'plssvd',3)
    # Shared permutation/sign for coupled models, and expected inverse permutation.
    matches=aligned['component_matching']
    assert matches.query('iteration == 1 and modality == "iEEG"').original_component.tolist()==[3,1,2]
    result=summarize_magnitudes(aligned,normalize=True,percentile=75,bins=8)
    for modality,w in aligned['weights'].items():
        mag=np.abs((w-w.mean(axis=1,keepdims=True))/w.std(axis=1,keepdims=True))
        expected=np.percentile(mag,75,axis=1)
        actual=result['thresholds'].query('modality == @modality').pivot(index='iteration',columns='component',values='threshold')
        np.testing.assert_allclose(actual,expected)
        retained=np.where(mag>=expected[:,None,:],mag,0.)
        np.testing.assert_allclose(result['maps'][modality]['mean'],retained.mean(0))
        np.testing.assert_allclose(result['maps'][modality]['sd'],retained.std(0,ddof=1))


def test_two_distinct_variability_levels(saved_pairing):
    root,cfg=saved_pairing
    aligned=load_aligned(root,'plssvd',3)
    result=summarize_profiles(aligned,cfg['analyses']['pairing_profiles'],2026)
    raw=result['participant_curves']
    for row in result['iteration_curves'].itertuples():
        values=raw.query('iteration == @row.iteration and modality == @row.modality and distance_mm == @row.distance_mm').similarity
        np.testing.assert_allclose(row.similarity,values.mean(),equal_nan=True)
        np.testing.assert_allclose(row.similarity_subject_sd,values.std(ddof=1),equal_nan=True)
    for row in result['summary'].itertuples():
        values=result['iteration_curves'].query('modality == @row.modality and distance_mm == @row.distance_mm').similarity
        np.testing.assert_allclose(row.similarity_mean,values.mean(),equal_nan=True)
        np.testing.assert_allclose(row.similarity_association_sd,values.std(ddof=1),equal_nan=True)


def test_runner_exports_thresholds_and_invalidates_changed_source(saved_pairing):
    root,cfg=saved_pairing
    folder=run(cfg,'pairing_weights')
    assert (folder/'thresholds.csv').exists()
    assert len(pd.read_csv(folder/'thresholds.csv'))==18
    run(cfg,'pairing_profiles')
    assert not load_result(cfg,'pairing_profiles')['summary'].empty
    p=root/'iteration_001/splits.json';p.write_text('{}')
    with pytest.raises(FileNotFoundError,match='Source fits changed'):
        load_result(cfg,'pairing_weights')


def test_settings_mismatch_rejected(saved_pairing):
    root,cfg=saved_pairing
    changed=copy.deepcopy(cfg);changed['fit']['block_scaling']=False
    assert not compatible_run(root,changed['fit'],changed['pairing'])[0]
    changed=copy.deepcopy(cfg);changed['fit']['data']['ieeg_multiplier']=1.
    with pytest.raises(ValueError,match='requires'):
        ensure_fits(changed,root/'unused')


def test_fallback_fits_fixed_half_trials(tmp_path):
    import nibabel as nib
    rng=np.random.default_rng(91)
    cache=tmp_path/'cache';cache.mkdir()
    records=[]
    xyz=np.column_stack([-np.arange(10,22,2),np.zeros(6),np.zeros(6)])
    for modality in ['ieeg','meg']:
        for number in range(3):
            subject=f'{modality}{number}'
            files=[]
            for condition in [1,2]:
                name=f'{subject}_{condition}.npy'
                np.save(cache/name,rng.normal(size=(10,6,24)).astype('float32'))
                files.append(name)
            records.append(dict(modality=modality,subject=subject,files=files,
                positions=xyz.tolist(),metadata=dict(x=xyz[:,0].tolist(),y=xyz[:,1].tolist(),
                    z=xyz[:,2].tolist(),subject=[subject]*6),
                split_groups=[None,None],permutation_blocks=[['all']*10,['all']*10]))
    manifest=dict(config=dict(conditions=[1,2],ieeg_subjects=[f'ieeg{i}' for i in range(3)],
                              meg_subjects=[f'meg{i}' for i in range(3)]),
                  times=(np.arange(24)/250).tolist(),records=records)
    (cache/'manifest.json').write_text(json.dumps(manifest))
    affine=np.eye(4);affine[:3,3]=-40
    atlas=tmp_path/'atlas.nii.gz';nib.save(nib.Nifti1Image(np.ones((80,80,80),dtype=np.int16),affine),atlas)
    labels=tmp_path/'labels.csv';pd.DataFrame([dict(atlas_id=1,parcel='P',region='R',group='G')]).to_csv(labels,index=False)
    config=dict(output_dir=str(tmp_path/'result'),fit=dict(model='plssvd',n_components=2,
        seed=2026,block_scaling=True,condition_mode='average',max_gram_gib=2.,
        data=dict(meg_scaling='channel_zscore',ieeg_multiplier=1000.,conditions=[1,2])),
        pairing=dict(existing_dir=str(tmp_path/'missing'),cache_dir=str(cache),iterations=3,
                     split_unit='trial',atlas_image=str(atlas),atlas_labels=str(labels)))
    root=ensure_fits(config,tmp_path/'new_fits')
    assert compatible_run(root,config['fit'],config['pairing'])[0]
    split=json.loads((root/'iteration_000/splits.json').read_text())
    for i in range(3):
        assert json.loads((root/f'iteration_{i:03d}/splits.json').read_text())==split
    for modality in ['ieeg','meg']:
        for selected in split['A'][modality].values():
            assert all(len(condition)==5 for condition in selected['train'])
    aligned=load_aligned(root,'plssvd',3)
    assert aligned['weights']['MEG'].shape==(3,18,2)
    assert all(np.isfinite(w).all() for w in aligned['weights'].values())
