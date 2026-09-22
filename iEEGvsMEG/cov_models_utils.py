"""Separate PCA, joint PCA and exact PLSSVD for aligned MEG/iEEG datasets.

Matrices have observations (time or condition x time) in rows. Computation uses
observation Gram matrices, avoiding full channel covariance/cross-covariance
matrices and a dense concatenation of all MEG participants.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh, svd
from scipy.optimize import linear_sum_assignment
from pca_comparison import compare_pca_to_ieeg


@dataclass
class CovModel:
    name: str
    ieeg_scores: np.ndarray
    meg_scores: np.ndarray
    ieeg_weights: np.ndarray
    meg_weights: np.ndarray
    gram_ieeg: np.ndarray
    gram_meg: np.ndarray
    times: np.ndarray
    condition_mode: str
    conditions: list
    scales: tuple
    joint_scores: np.ndarray = None
    joint_basis: np.ndarray = None
    singular_values: np.ndarray = None


def _chunks(dataset, width=1024):
    for block in dataset.blocks():
        for start in range(0, block.shape[1], width):
            x = np.array(block[:, start:start+width], dtype=float, copy=True)
            if not np.isfinite(x).all():
                raise ValueError('Signals must be finite.')
            x -= x.mean(axis=0)
            yield x


def _gram(dataset):
    g = np.zeros((dataset.n_observations, dataset.n_observations))
    for x in _chunks(dataset):
        g += x @ x.T
    if np.trace(g) <= 0:
        raise ValueError(f'{dataset.name}: no variance.')
    return g


def _spectrum(g, n_features):
    values, u = eigh(g, check_finite=False)
    values, u = values[::-1], u[:, ::-1]
    keep = values > values[0] * np.finfo(float).eps * max(g.shape[0], n_features)
    return u[:, keep], np.sqrt(values[keep])


def _weights(dataset, temporal_coefficients, scale):
    # W = (centred, scaled data).T @ temporal_coefficients; only p x k stored.
    out = np.empty((dataset.n_features, temporal_coefficients.shape[1]))
    offset = 0
    for x in _chunks(dataset):
        out[offset:offset+x.shape[1]] = (x.T @ temporal_coefficients) * scale
        offset += x.shape[1]
    return out


def fit_cov_models(ieeg, meg, n_components=10, block_scaling='none', max_gram_gib=2.):
    """Fit all three models with identical centring and modality scaling.

    'none' preserves the coverage notebook's preprocessing. 'equal_variance'
    rescales each WHOLE modality to total sample variance 1; it does not z-score
    individual channels. Independent PCA uses separate axes. Joint PCA uses
    [X Y]; its modality scores are X Wx and Y Wy, whose sum is the joint score.
    PLSSVD is the SVD of X.T Y/(n-1), computed exactly through full nonzero
    sample-space factors, without an initial top-k PCA approximation.
    """
    if n_components < 1 or int(n_components) != n_components:
        raise ValueError('n_components must be a positive integer.')
    if ieeg.n_observations != meg.n_observations or ieeg.condition_mode != meg.condition_mode:
        raise ValueError('Align observations and condition modes before fitting.')
    a, b = ieeg.source_data, meg.source_data
    if a is None or b is None or not np.array_equal(a['times'], b['times']):
        raise ValueError('Datasets need matching time metadata from load_dataset.')
    if a.get('load_config', {}).get('conditions') != b.get('load_config', {}).get('conditions'):
        raise ValueError('Condition orders differ.')
    n = ieeg.n_observations
    if n < 2 or 12 * n*n*8 / 2**30 > max_gram_gib:
        raise MemoryError('Sample-space working arrays exceed budget or too few observations.')
    gx, gy = _gram(ieeg), _gram(meg)
    if block_scaling == 'none':
        sx = sy = 1.
    elif block_scaling == 'equal_variance':
        sx, sy = np.sqrt((n-1)/np.trace(gx)), np.sqrt((n-1)/np.trace(gy))
        gx *= sx*sx
        gy *= sy*sy
    else:
        raise ValueError("block_scaling must be 'none' or 'equal_variance'.")
    ux, dx = _spectrum(gx, ieeg.n_features)
    uy, dy = _spectrum(gy, meg.n_features)
    k = int(n_components)
    if min(len(dx), len(dy)) < k:
        raise ValueError(f'Only {len(dx)} iEEG and {len(dy)} MEG nonzero dimensions; reduce n_components.')
    common = dict(gram_ieeg=gx, gram_meg=gy, times=np.asarray(a['times']),
                  conditions=a.get('load_config', {}).get('conditions', list(range(1, ieeg.arrays[0].shape[0]+1))),
                  condition_mode=ieeg.condition_mode, scales=(sx, sy))
    independent = CovModel(
        name='separate_pca', ieeg_scores=ux[:, :k]*dx[:k], meg_scores=uy[:, :k]*dy[:k],
        ieeg_weights=_weights(ieeg, ux[:, :k]/dx[:k], sx),
        meg_weights=_weights(meg, uy[:, :k]/dy[:k], sy), **common)
    uj, dj = _spectrum(gx+gy, ieeg.n_features+meg.n_features)
    coeff = uj[:, :k]/dj[:k]
    joint = CovModel(
        name='joint_pca', ieeg_scores=gx @ coeff, meg_scores=gy @ coeff,
        ieeg_weights=_weights(ieeg, coeff, sx), meg_weights=_weights(meg, coeff, sy),
        joint_scores=uj[:, :k]*dj[:k], joint_basis=uj[:, :k], **common)
    # X = Ux Dx Vx.T, Y = Uy Dy Vy.T; SVD of the small matrix gives exact
    # cross-covariance singular vectors lifted by Vx and Vy.
    reduced_cross = dx[:, None] * (ux.T @ uy) * dy[None, :] / (n-1)
    left, singular, right_t = svd(reduced_cross, full_matrices=False, check_finite=False)
    tol = singular[0]*np.finfo(float).eps*max(reduced_cross.shape)
    if singular[k-1] <= tol:
        raise ValueError('Too few nonzero cross-covariance dimensions for PLSSVD; reduce n_components.')
    left, right = left[:, :k], right_t[:k].T
    pls = CovModel(
        name='plssvd', ieeg_scores=(ux*dx) @ left, meg_scores=(uy*dy) @ right,
        ieeg_weights=_weights(ieeg, (ux/dx) @ left, sx),
        meg_weights=_weights(meg, (uy/dy) @ right, sy),
        singular_values=singular[:k], **common)
    return {m.name: m for m in (independent, joint, pls)}


def _orthogonal_score_projection(scores, weights):
    """X Q for an orthonormal basis Q of span(W), given scores X W."""
    _, singular, vt = svd(weights, full_matrices=False, check_finite=False)
    keep = singular > singular[0]*np.finfo(float).eps*max(weights.shape)
    return (scores @ vt[keep].T)/singular[keep]


def evaluate_cov_models(models, plot=True):
    """Return cumulative reconstruction EV and retained cross-covariance energy.

    Within-modality EV = 1 - ||X-Xhat||²/||X||², separately normalised for X/Y.
    Separate PCA/PLSSVD: Xhat = (X W) W.T (orthonormal W).
    Joint PCA: Xhat = Z Wx.T, Yhat = Z Wy.T, with Z=XWx+YWy; equivalently
    projection of both datasets onto the joint PCA temporal basis.

    Shared cross-covariance energy fraction = ||Qx.T C Qy||F²/||C||F²,
    C=X.T Y/(n-1), Qx/Qy orthonormal bases of retained FEATURE weight spaces.
    This is not a fraction of shared biological variance. For PLSSVD it equals
    sum(retained singular_values²)/sum(all singular_values²).
    """
    rows = []
    for name, m in models.items():
        n = len(m.ieeg_scores)
        tx, ty = np.trace(m.gram_ieeg), np.trace(m.gram_meg)
        cross_energy = float(np.sum(m.gram_ieeg*m.gram_meg))/(n-1)**2
        for k in range(1, m.ieeg_scores.shape[1]+1):
            x, y = m.ieeg_scores[:, :k], m.meg_scores[:, :k]
            if m.joint_basis is not None:
                q = m.joint_basis[:, :k]
                evx = np.sum(q*(m.gram_ieeg @ q))/tx
                evy = np.sum(q*(m.gram_meg @ q))/ty
            else:
                evx, evy = np.sum(x*x)/tx, np.sum(y*y)/ty
            ox = _orthogonal_score_projection(x, m.ieeg_weights[:, :k])
            oy = _orthogonal_score_projection(y, m.meg_weights[:, :k])
            captured_cross = np.sum((ox.T @ oy/(n-1))**2)
            vx, vy = np.var(x[:, -1], ddof=1), np.var(y[:, -1], ddof=1)
            cov = float(x[:, -1] @ y[:, -1]/(n-1))
            rows.append(dict(model=name, k=k, ieeg_variance_explained=float(evx),
                meg_variance_explained=float(evy),
                shared_crosscov_fraction=float(np.clip(captured_cross/cross_energy, 0, 1)) if cross_energy > 0 else np.nan,
                ieeg_total_variance=tx/(n-1), meg_total_variance=ty/(n-1),
                ieeg_score_variance=vx, meg_score_variance=vy, score_crosscovariance=cov,
                sum_score_variance=vx+vy+2*cov,
                combined_reconstruction_fraction=float((evx*tx+evy*ty)/(tx+ty))))
    table = pd.DataFrame(rows)
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        for ax, metric, title in zip(axes,
            ['ieeg_variance_explained','meg_variance_explained','shared_crosscov_fraction'],
            ['iEEG reconstruction variance explained','MEG reconstruction variance explained','Cross-covariance energy captured']):
            for name, group in table.groupby('model', sort=False):
                ax.plot(group.k, group[metric], 'o-', label=name)
            ax.set(xlabel='Retained components', ylabel='Fraction', title=title, ylim=(-.02,1.02))
            ax.grid(alpha=.2)
        axes[-1].legend(fontsize=8)
        plt.show(); plt.close(fig)
    return table


def plot_model_timecourses(model, n_components=10, standardize=True):
    """Plot modality-specific native scores; joint model also shows their sum.

    Unit-SD scaling is for display only and does not alter model evaluation.
    No per-modality sign flips: joint contributions retain their relative sign.
    """
    k = min(n_components, model.ieeg_scores.shape[1])
    fig, axes = plt.subplots(int(np.ceil(k/2)), 2, figsize=(15, 2.4*int(np.ceil(k/2))), squeeze=False, constrained_layout=True)
    for pc, ax in enumerate(axes.flat):
        if pc >= k:
            ax.set_visible(False); continue
        traces = [('iEEG', model.ieeg_scores[:, pc], 'navy'), ('MEG', model.meg_scores[:, pc], 'darkorange')]
        if model.joint_scores is not None:
            traces.append(('Joint sum', model.joint_scores[:, pc], 'black'))
        for label, values, color in traces:
            if standardize:
                values = values / max(values.std(), np.finfo(float).tiny)
            parts = values.reshape(-1, len(model.times))
            for j, part in enumerate(parts):
                ax.plot(model.times, part, color=color, ls=['-', '--', ':', '-.'][j % 4],
                        label=label if len(parts)==1 else f'{label}, condition {model.conditions[j]}')
        ax.set(title=f'Component {pc+1}', xlabel='Time (s)', ylabel='Score / SD' if standardize else 'Score')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(model.name)
    plt.show(); plt.close(fig)
    return fig


def plot_model_covariances(model, n_components=10, common_scale=False):
    """Small native-score covariance blocks (not the huge channel matrices)."""
    k = min(n_components, model.ieeg_scores.shape[1])
    x, y = model.ieeg_scores[:, :k], model.meg_scores[:, :k]
    matrices = [x.T@x/(len(x)-1), y.T@y/(len(x)-1), x.T@y/(len(x)-1)]
    limit = max(float(np.abs(c).max()) for c in matrices) or 1.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, matrix, title in zip(axes, matrices, ['iEEG covariance', 'MEG covariance', 'Cross-covariance']):
        panel_limit = limit if common_scale else (float(np.abs(matrix).max()) or 1.)
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-panel_limit, vmax=panel_limit)
        ax.set(xlabel='MEG component' if title != 'iEEG covariance' else 'iEEG component',
               ylabel='MEG component' if title == 'MEG covariance' else 'iEEG component', title=title, xticks=range(k), xticklabels=range(1,k+1), yticks=range(k), yticklabels=range(1,k+1))
        fig.colorbar(im, ax=ax, label='Score covariance')
    fig.suptitle(model.name)
    plt.show(); plt.close(fig)
    # The summary notebook's componentwise variance/covariance decomposition.
    fig2, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for values, label in [(np.diag(matrices[0]), 'Var(iEEG score)'),
                          (np.diag(matrices[1]), 'Var(MEG score)'),
                          (2*np.diag(matrices[2]), '2 Cov(iEEG, MEG)')]:
        ax.plot(range(1,k+1), values, 'o-', label=label)
    ax.plot(range(1,k+1), np.var(x+y, axis=0, ddof=1), 'k--', label='Var(sum of scores)')
    ax.set(title=model.name, xlabel='Component', ylabel='Native score variance / covariance')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    plt.show(); plt.close(fig2)
    return fig, fig2


def _pearson(x, y):
    x, y = x-x.mean(0), y-y.mean(0)
    den = np.outer(np.linalg.norm(x,axis=0), np.linalg.norm(y,axis=0))
    return np.divide(x.T@y, den, out=np.full(den.shape,np.nan), where=den>0).clip(-1,1)


def compare_cov_models(models, n_components=(1,2,3,5,10), plot=True):
    """Compare to one fixed iEEG-PCA reference, and native component matching.

    Reuses coverage notebook metrics on each model's MEG-only scores versus
    separate iEEG PCA. Also reports native paired and optimally matched Pearson
    correlations. Joint summed scores are NEVER used as both modalities.
    Rankings are descriptive, in sample; PLS/joint PCA optimise using both data.
    """
    anchor = models['separate_pca'].ieeg_scores
    reference = SimpleNamespace(scores=anchor)
    scores = {'iEEG': reference, **{name:SimpleNamespace(scores=m.meg_scores) for name,m in models.items()}}
    table, anchor_pairs = compare_pca_to_ieeg(scores, n_components=n_components, plot=plot)
    table = table.rename(columns={'dataset':'model', 'rank':'rank_ieeg_reference'})
    rows, pairs = [], []
    for row in table.itertuples():
        m, k = models[row.model], row.k
        r = _pearson(m.ieeg_scores[:, :k], m.meg_scores[:, :k])
        ix, iy = linear_sum_assignment(-np.nan_to_num(np.abs(r),nan=0))
        rows.append(dict(model=row.model, k=k,
            native_paired_abs_r=float(np.abs(np.diag(r)).mean()),
            native_matched_abs_r=float(np.abs(r[ix,iy]).mean())))
        for i,j in zip(ix,iy):
            pairs.append(dict(model=row.model,k=k,ieeg_component=int(i+1),meg_component=int(j+1),
                              signed_r=float(r[i,j]),abs_r=float(abs(r[i,j]))))
    table = table.merge(pd.DataFrame(rows), on=['model','k'])
    table['rank_native_matching'] = table.groupby('k').native_matched_abs_r.rank(ascending=False,method='min').astype(int)
    ev = evaluate_cov_models(models, plot=False)
    table = table.merge(ev[['model','k','ieeg_variance_explained','meg_variance_explained','shared_crosscov_fraction']],on=['model','k'])
    if plot:
        fig, axes = plt.subplots(1, len(models), figsize=(5*len(models),4), squeeze=False, constrained_layout=True)
        display_k = min(3, min(m.ieeg_scores.shape[1] for m in models.values()))
        for ax,(name,m) in zip(axes.flat,models.items()):
            r = _pearson(m.ieeg_scores[:,:display_k],m.meg_scores[:,:display_k])
            im=ax.imshow(r,vmin=-1,vmax=1,cmap='RdBu_r')
            ax.set(title=name,xlabel='MEG component',ylabel='iEEG component',
                   xticks=range(display_k),xticklabels=range(1,display_k+1),
                   yticks=range(display_k),yticklabels=range(1,display_k+1))
            for i in range(display_k):
                for j in range(display_k):ax.text(j,i,f'{r[i,j]:.2f}',ha='center',va='center',fontsize=9)
            fig.colorbar(im,ax=ax,label='Pearson r')
        plt.show(); plt.close(fig)
    return table, pd.DataFrame(pairs), anchor_pairs
