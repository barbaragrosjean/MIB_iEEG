import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from modality_specificity import spatial_weight_distance


def test_exact_pairs_subjects_and_sign_invariance():
    rng = np.random.default_rng(9)
    w = rng.normal(size=(8, 3))
    xyz = rng.normal(size=(8, 3))*10
    xyz[1] = xyz[0]  # distinct co-located features must be included
    meta = pd.DataFrame(xyz, columns=['x','y','z']).assign(subject=['a']*4+['b']*4)
    options = dict(bin_edges=[0,10,30,100], n_permutations=7, block_size=3)
    result = spatial_weight_distance(w, meta, **options)
    group = result['curves'].query("scope == 'group'")
    assert group.n_pairs.sum() == 28
    assert result['curves'].query("scope == 'subject'").n_pairs.sum() == 12
    for subject, frame in result['curves'].groupby('subject'):
        mask = np.ones(8, bool) if subject == 'all_channels' else meta.subject.eq(subject).to_numpy()
        distances, cosines = pdist(xyz[mask]), pdist(w[mask], metric='cosine')
        for row in frame.itertuples():
            selected = (distances >= row.distance_low_mm) & (distances < row.distance_high_mm)
            if selected.any():
                assert np.isclose(row.cosine_distance, cosines[selected].mean())
    w[:,1] *= -1
    flipped = spatial_weight_distance(w, meta, **options)
    np.testing.assert_allclose(result['curves'].cosine_distance, flipped['curves'].cosine_distance, equal_nan=True)
    np.testing.assert_allclose(result['null_curves'].cosine_distance, flipped['null_curves'].cosine_distance, equal_nan=True)
    assert (result['curves'].p_holm.dropna() >= result['curves'].p_two_sided.dropna()).all()


def test_coordinate_shuffle_equivalence_and_zero_profiles():
    w = np.array([[1.,0],[0,1],[-1,0],[1,1],[0,0]])
    meta = pd.DataFrame({'x':[0,2,10,20,30], 'y':0., 'z':0., 'subject':['a']*5})
    result = spatial_weight_distance(w, meta, bin_edges=[0,5,40], n_permutations=3, seed=2, block_size=2)
    assert result['channel_audit'].n_zero_norm.sum() == 1
    assert result['curves'].query("scope == 'group'").n_pairs.sum() == 6
    p = np.random.default_rng(2).permutation(4)
    # Permuting profiles by p equals permuting coordinates by inverse(p).
    distance = pdist(meta[['x','y','z']].to_numpy()[:4][np.argsort(p)])
    cosine = pdist(w[:4], metric='cosine')
    expected = [cosine[(distance>=lo)&(distance<hi)].mean() for lo,hi in [(0,5),(5,40)]]
    actual = result['null_curves'].query("scope == 'group' and permutation == 0").cosine_distance
    np.testing.assert_allclose(actual, expected)


def test_meg_metadata_uses_native_subject_and_preserves_rows():
    from modality_specificity import spatial_metadata
    meta = pd.DataFrame({'x':[0.,1.,2.,3.], 'y':0., 'z':0.,
                         'meg_subject':['m1','m1','m2','m2'],
                         'ieeg_subject':['i1','i2','i1','i2']}, index=[9,7,5,3])
    normalized = spatial_metadata(meta)
    assert normalized.subject.tolist() == ['m1','m1','m2','m2']
    assert normalized.index.tolist() == [9,7,5,3]
    assert 'subject' not in meta
    result = spatial_weight_distance(np.array([[1.,0],[0,1],[1,1],[-1,1]]),
                                    meta, bin_edges=[0,10], n_permutations=3)
    assert set(result['curves'].query("scope == 'subject'").subject) == {'m1','m2'}
    averaged = meta.drop(columns='ieeg_subject').assign(meg_subject='participant_average')
    assert spatial_metadata(averaged).subject.unique().tolist() == ['participant_average']
    try:
        spatial_metadata(meta.drop(columns=['meg_subject','ieeg_subject']))
    except ValueError as exc:
        assert 'requires subject' in str(exc)
    else:
        raise AssertionError('Missing ownership should not silently create a subject')


def test_semivariogram_formula_and_constant_component():
    from modality_specificity import component_semivariograms
    w = np.array([[0.,1.],[1,1],[2,1],[4,1]])
    meta = pd.DataFrame({'x':[0.,1,3,6], 'y':0., 'z':0., 'subject':['a']*4})
    result = component_semivariograms(w,meta,bin_edges=[0,2,10],n_permutations=3,block_size=2)
    frame = result['curves'].query("scope == 'group' and component == 1")
    d = pdist(meta[['x','y','z']])
    v = .5*pdist(w[:,:1],metric='sqeuclidean')/np.var(w[:,0])
    np.testing.assert_allclose(frame.semivariance,[v[d<2].mean(),v[d>=2].mean()])
    assert result['curves'].query('component == 2').semivariance.isna().all()
    assert result['curves'].query('component == 2').p_two_sided.isna().all()


def test_group_clusters_centroid_and_cross_subject_connectivity():
    from modality_specificity import weight_clusters
    # Three adjacent positive sources, isolated high negative, plus background.
    x = np.array([-20.,-19,-18,20,40,50,60,70,80,90,100,110,120,130,140,150])
    w = np.zeros((16,1)); w[:3,0]=[8,9,10]; w[3,0]=-10
    meta = pd.DataFrame({'x':x,'y':0.,'z':0.,'meg_subject':'m1'})
    result = weight_clusters(w,meta,radius_mm=2,min_sources=3)
    assert len(result['clusters']) == 1
    row = result['clusters'].iloc[0]
    assert row.n_sources == 3 and row.sign == 1 and row.hemisphere == 'left'
    assert np.isclose(row.centroid_x, np.average(x[:3],weights=[8,9,10]))
    assert row.representative_row == 1
    meta.loc[2,'meg_subject']='m2'
    pooled = weight_clusters(w,meta,radius_mm=2,min_sources=3)
    assert len(pooled['clusters']) == 1
    assert pooled['clusters'].iloc[0].n_subjects == 2
    assert pooled['clusters'].iloc[0].scope == 'group'
    assert np.isclose(pooled['clusters'].iloc[0].centroid_x, row.centroid_x)
    assert set(pooled['memberships'].subject) == {'m1', 'm2'}
    assert len(pooled['thresholds']) == 1
    assert np.isclose(pooled['thresholds'].iloc[0].threshold, np.abs(w).mean()+np.abs(w).std())
