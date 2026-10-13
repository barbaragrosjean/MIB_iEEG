"""Full-input cross-covariance energy dimensionality; no PLS refitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import eigh, svdvals
from plssvd_diagnostics import _audit


def crosscov_singular_values(gx, gy):
    """Nonzero singular spectrum of X.T @ Y / (T-1), from centered time Grams.

    X=Ux Dx Vx.T, Y=Uy Dy Vy.T: use Dx Ux.T Uy Dy without
    forming the potentially enormous feature-by-feature cross-covariance.
    """
    if gx.shape != gy.shape or gx.ndim != 2 or gx.shape[0] != gx.shape[1] or len(gx)<2:
        raise ValueError('Expected matching square temporal Gram matrices with T >= 2.')
    factors=[]
    for gram in (gx,gy):
        if not np.isfinite(gram).all():raise ValueError('Nonfinite Gram matrix.')
        values, vectors=eigh((gram+gram.T)/2, check_finite=False)
        tol=max(float(values[-1]),0.)*np.finfo(float).eps*len(gram)
        if values[0] < -max(tol*100,np.finfo(float).tiny):
            raise ValueError('Gram matrix is not positive semidefinite.')
        keep=values>tol
        factors.append(vectors[:,keep]*np.sqrt(values[keep]))
    if not factors[0].shape[1] or not factors[1].shape[1]:return np.array([])
    return svdvals(factors[0].T@factors[1])/(len(gx)-1)


def spectrum_metrics(s):
    s=np.asarray(s,dtype=float)
    if np.any(s<0) or not np.isfinite(s).all():raise ValueError('Invalid singular values.')
    if s.size==0 or s.max()==0:return np.nan,0.
    energy=(s/s.max())**2
    fractions=energy/energy.sum()
    return float(1/np.sum(fractions**2)),float(s.max()**2*energy.sum())


def _recover(root, iteration, trials, scratch_dir=None):
    from plssvd_eval_utils import ValidationOptions, _temporary_fold, _project, _saved_scores
    from cov_models_utils import _gram
    child = root/f'iteration_{iteration:03d}'
    config = json.loads((root/'validation_options.json').read_text())
    child_config = json.loads((child/'validation_options.json').read_text())
    options = ValidationOptions(**{k:v for k,v in child_config.items() if k in ValidationOptions.__dataclass_fields__})
    matching_seq, _, permutation_seq = np.random.SeedSequence(config['seed']).spawn(3)
    matching_seed = int(np.random.default_rng(matching_seq).integers(2**31-1))
    perm_seed = int(np.random.default_rng(permutation_seq.spawn(iteration+1)[iteration]).integers(2**31-1))
    audit = _audit(root,iteration)
    for modality in ('ieeg','meg'):
        if set(audit.loc[audit.modality==modality,'subject']) != {s.subject for s in getattr(trials,modality)}:
            raise ValueError('Trial cache cohort differs from the saved run.')
    saved_axes = root/'trial_axes.npz'
    if not saved_axes.exists(): saved_axes = child/'trial_axes.npz'
    with np.load(saved_axes) as axes:
        if not np.array_equal(axes['times'],trials.times) or tuple(axes['conditions']) != tuple(trials.conditions):
            raise ValueError('Trial cache axes differ from the saved run.')
    for fold in range(config['repeats']):
        sidecar=child/f'crosscov_spectrum_{fold:03d}.npz'
        if sidecar.exists():continue
        indices={}
        for modality in ('ieeg','meg'):
            indices[modality]={}
            for subject in getattr(trials,modality):
                rows=audit[(audit.repeat==fold)&(audit.modality==modality)&(audit.subject==subject.subject)]
                indices[modality][subject.subject]={part:[
                    rows.loc[(rows.partition==part)&(rows.condition==condition),'trial_index'].to_numpy(int)
                    for condition in trials.conditions] for part in ('train','test')}
        spectra={}
        with _temporary_fold(trials,config['meg_kind'],indices,matching_seed,scratch_dir,
                             condition_mode='average',permutation_options=options,permutation_seed=perm_seed) as prepared:
            datasets,_,_,_=prepared
            with np.load(child/f'model_{fold:03d}.npz') as saved_model:
                model = {key:saved_model[key] for key in ('ieeg_mean','meg_mean','ieeg_scale','meg_scale')}
                model['k_max'] = min(3,int(saved_model['k_max']))
                for modality in ('ieeg','meg'):
                    model[modality+'_weights'] = saved_model[modality+'_weights'][:,:model['k_max']].copy()
            with _saved_scores(child,fold) as saved_scores:
                for part in ('train','test'):
                    for modality in ('ieeg','meg'):
                        expected=saved_scores[f'{part}_{modality}'][:,:model['k_max']]
                        actual=_project(datasets[part][modality],model,modality)
                        if not np.allclose(actual,expected,rtol=1e-4,atol=1e-5*max(1.,np.max(np.abs(expected)))):
                            raise ValueError('Reconstructed scores differ from saved scores; check the original trial cache and settings.')
                for part in ('train','test'):
                    grams = [_gram(datasets[part][m]) * float(model[m+'_scale'])**2
                             for m in ('ieeg','meg')]
                    spectra[part] = crosscov_singular_values(*grams)
        from uuid import uuid4
        pending = sidecar.with_name(sidecar.name + '.' + uuid4().hex + '.tmp')
        try:
            with pending.open('wb') as stream:
                np.savez_compressed(stream, **spectra)
            pending.replace(sidecar)
        finally:
            if pending.exists(): pending.unlink()
        print(f'Cross-covariance spectrum: iteration {iteration}, fold {fold}', flush=True)


def crosscov_dimensionality(run_dir, iterations, cache_dir, scratch_dir=None):
    """Recover missing spectra using original trials; reuse complete sidecars.

    Delete crosscov_spectrum_*.npz if original fits/cache are intentionally changed.
    """
    from plssvd_eval_utils import load_trial_cache
    root=Path(run_dir)
    config=json.loads((root/'validation_options.json').read_text())
    trials=None;rows=[]
    for iteration in iterations:
        child=root/f'iteration_{iteration:03d}'
        if any(not (child/f'crosscov_spectrum_{f:03d}.npz').exists() for f in range(config['repeats'])):
            if trials is None:trials=load_trial_cache(cache_dir)
            _recover(root,iteration,trials,scratch_dir)
        for fold in range(config['repeats']):
            with np.load(child/f'crosscov_spectrum_{fold:03d}.npz') as saved:
                for partition in ('train','test'):
                    ed,energy=spectrum_metrics(saved[partition])
                    rows.append(dict(iteration=iteration,fold=fold,partition=partition,
                                     shared_ED=ed,total_crosscov_energy=energy))
    return pd.DataFrame(rows)
