"""Compatibility plotting helpers; editable plotting code lives in the covariance notebook."""
import numpy as np
import matplotlib.pyplot as plt
from vs2_0_cov_models_utils import _pearson

def plot_variance_summary(table):
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
                          (2*abs(np.diag(matrices[2])), '2 |Cov(iEEG, MEG)|')]:
        ax.plot(range(1,k+1), values, 'o-', label=label)
    ax.set(title=model.name, xlabel='Component', ylabel='Native score variance / covariance')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    plt.show(); plt.close(fig2)
    return fig, fig2


def plot_native_correlations(models):
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models),4), squeeze=False, constrained_layout=True)
    display_k = min(5, min(m.ieeg_scores.shape[1] for m in models.values()))
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


