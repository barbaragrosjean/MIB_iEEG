"""Optional plotting helpers. Notebook cells may customize returned figures."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting
from vs1_coverage_information import COMPARISONS
from vs0_load_and_config import coordinates_mm

MODALITY_COLORS = {'iEEG': '#D55E00', 'MEG': '#0072B2'}
SETUP_NAMES = ('full_average', 'full_concatenated', 'coverage_average', 'paired_coverage', 'random_control')
SETUP_COLORS = dict.fromkeys(SETUP_NAMES, MODALITY_COLORS['MEG'])
SETUP_LINESTYLES = dict(zip(SETUP_NAMES, ('-', '--', ':', '-.', (0, (3, 1, 1, 1, 1, 1)))))


def apply_style():
    plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .25,
                         'axes.spines.top': False, 'axes.spines.right': False})

def plot_pca_comparison(table):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, metric in zip(axes, ('ieeg_variance_captured', 'subspace_overlap', 'matched_abs_r')):
        for name, group in table.groupby('dataset', sort=False):
            ax.plot(group.k, group[metric], 'o-', color=SETUP_COLORS.get(name), label=name)
        ax.set(xlabel='Number of PCs', ylabel=metric, ylim=(0, 1.02)); ax.grid(True)
    axes[-1].legend(fontsize=8)
    return fig

def plot_pca(result, label, times, condition_mode='average', condition_labels=('1', '2')):
    k = result.scores.shape[1]
    fig, axes = plt.subplots(int(np.ceil(k / 2)), 2, figsize=(15, 2.2 * int(np.ceil(k / 2))), squeeze=False, constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        if i >= k:
            ax.set_visible(False)
            continue
        if condition_mode == 'average':
            ax.plot(times, result.scores[:, i], color=MODALITY_COLORS['iEEG' if label == 'iEEG' else 'MEG'])
        else:
            traces = result.scores[:, i].reshape(-1, len(times))
            for c, trace in enumerate(traces):
                ax.plot(times, trace, label=str(condition_labels[c]), color=MODALITY_COLORS['iEEG' if label == 'iEEG' else 'MEG'], linestyle=['-', '--', ':', '-.'][c % 4])
            ax.legend(fontsize=8)
        ax.set(title=f'PC {i+1} ({100 * result.explained_variance_ratio[i]:.1f}%)', xlabel='Time (s)', ylabel='Score')
        ax.grid(alpha=.2)
    fig.suptitle(label)
    return fig


def plot_explained_variance(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    for name, r in results.items():
        x = np.arange(1, len(r.explained_variance_ratio) + 1)
        axes[0].plot(x, 100 * r.explained_variance_ratio, '.-', label=name, color=MODALITY_COLORS['iEEG' if name == 'iEEG' else 'MEG'])
        axes[1].plot(x, 100 * np.cumsum(r.explained_variance_ratio), '.-', label=name, color=MODALITY_COLORS['iEEG' if name == 'iEEG' else 'MEG'])
    axes[0].set(xlabel='PC', ylabel='Explained variance (%)')
    axes[1].set(xlabel='Number of PCs', ylabel='Cumulative explained variance (%)', ylim=(0, 101))
    axes[1].legend(fontsize=8)
    return fig


def show_coverage(dataset, max_points=20000):
    """Show MNI channel/source positions on a glass brain; skip large sets."""
    pos = dataset.metadata[['x', 'y', 'z']].drop_duplicates().to_numpy()
    if len(pos) > max_points:
        pos = pos[np.linspace(0, len(pos)-1, max_points, dtype=int)]
    fig = plt.figure(figsize=(11, 3.5))
    plotting.plot_markers(
        np.ones(len(pos)), pos, node_size=8, node_cmap='Reds',
        node_vmin=0, node_vmax=1, node_threshold=None,
        figure=fig, colorbar=False,
        title=f'{dataset.name}: {dataset.n_features:,} features / {len(pos):,} locations',
    )
    for ax in fig.axes: ax.grid(True, alpha=.25)
    plt.show()
    plt.close(fig)
    return fig


def plot_pca_timecourses(result):
    """Show all retained score time courses and individual/cumulative EVR."""
    data = result.dataset
    if data is None or data.source_data is None:
        raise ValueError('Use compute_pca on a dataset from load_dataset.')
    conditions = data.source_data.get('load_config', {}).get(
        'conditions', list(range(1, data.arrays[0].shape[0] + 1)))
    temporal = plot_pca(result, data.name, data.source_data['times'],
                        data.condition_mode, conditions)
    plt.show()
    plt.close(temporal)
    variance = plot_explained_variance({data.name: result})
    plt.show()
    plt.close(variance)
    return temporal, variance


def _brain_weights(values, coordinates, absolute):
    """Validate one weight per MNI-mm coordinate without modifying inputs."""
    xyz = coordinates_mm(coordinates, 'mm')
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(xyz) or not len(values):
        raise ValueError('Provide one weight per coordinate and at least one point.')
    if not np.isfinite(values).all():
        raise ValueError('Weights must be finite.')
    return (np.abs(values) if absolute else values), xyz


def plot_voxel_weights(values, coordinates, *, voxel_size=20., size_base=2.,
                       size_scale=5., absolute=True, cmap=None, title=None,
                       figure=None, axes=None, show=True):
    """Mean weights in MNI-mm voxels, with marker area reflecting point count.

    Voxels are half-open cubes on a grid anchored at MNI (0, 0, 0), including
    points exactly on grid boundaries. Nodes sit at the mean coordinate of
    their contributing points. Marker size = size_base + size_scale * count.
    With absolute=True, magnitudes are taken BEFORE averaging, as in the old
    notebook. Each input row counts once (features for concatenated MEG).
    Returns (figure, table) with voxel indices, mean coordinates/weight, count,
    and marker size. No thresholding or changes to original weights are made.
    """
    values, xyz = _brain_weights(values, coordinates, absolute)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('voxel_size must be positive millimetres.')
    if not np.isfinite([size_base, size_scale]).all() or min(size_base, size_scale) < 0:
        raise ValueError('Marker size parameters must be finite and nonnegative.')
    table = pd.DataFrame(xyz, columns=['x', 'y', 'z'])
    table[['voxel_x', 'voxel_y', 'voxel_z']] = np.floor(xyz / voxel_size).astype(np.int64)
    table['weight'] = values
    grouped = table.groupby(['voxel_x', 'voxel_y', 'voxel_z'], sort=True).agg(
        x=('x', 'mean'), y=('y', 'mean'), z=('z', 'mean'),
        weight=('weight', 'mean'), count=('weight', 'size')).reset_index()
    grouped['node_size'] = size_base + size_scale * grouped['count']
    limit = float(np.max(np.abs(grouped.weight))) or 1.
    fig = figure if figure is not None else (axes.figure if axes is not None else plt.figure(figsize=(12, 3.5)))
    plotting.plot_markers(
        grouped.weight.to_numpy(), grouped[['x', 'y', 'z']].to_numpy(),
        node_size=grouped.node_size.to_numpy(), node_cmap=cmap or ('Reds' if absolute else 'RdBu_r'),
        node_vmin=0 if absolute else -limit, node_vmax=limit, node_threshold=None,
        figure=fig, axes=axes, colorbar=True, title=title or f'Mean weights in {voxel_size:g} mm voxels',
    )
    for ax in fig.axes: ax.grid(True, alpha=.25)
    if show:
        plt.show()
    return fig, grouped


def plot_glasser_weights(values, coordinates, *, sigma=4., radius_sigma=3.,
                         min_support=.05, absolute=True, template=None, meshes=None,
                         view='lateral', threshold=None, cmap=None, title=None,
                         figure=None, axes=None, show=True, vmin=None, vmax=None, hemisphere=None):
    """Gaussian-weighted electrode maps on the HCP inflated cortical surfaces.

    Reproduces OLD/LB_Summary.ipynb's 'Glasser' surface visualization: this is
    continuous interpolation on HCP meshes, NOT a Glasser parcel average.
    Coordinates and sigma are MNI mm. Kernels are truncated at radius_sigma
    standard deviations for bounded work/memory; unsupported voxels are masked
    using min_support. Both hemispheres share one colour scale. Optional vmin
    and vmax override colour limits (not data or masking). Missing limits use
    the original defaults: 0 to max magnitude for absolute maps, otherwise
    -max magnitude to +max magnitude. Explicit limits may be asymmetric.

    Optional meshes has pial_left/right, inflated_left/right and sulc_left/right
    attributes (defaults to hcp_utils.mesh); template defaults to MNI152 2 mm.
    Supply local meshes/template to avoid any external data requirements.
    hemisphere=None plots both; 'left' or 'right' plots only that surface.
    Returns (figure, dict) containing image, support mask and hemisphere textures.
    """
    import nibabel as nib
    from nilearn import datasets, surface

    if hemisphere not in (None, 'left', 'right'):
        raise ValueError('hemisphere must be None, left or right.')
    hemispheres = ('left', 'right') if hemisphere is None else (hemisphere,)
    values, xyz = _brain_weights(values, coordinates, absolute)
    if not np.isfinite([sigma, radius_sigma, min_support]).all() or min(sigma, radius_sigma, min_support) <= 0:
        raise ValueError('sigma, radius_sigma and min_support must be positive finite values.')
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError('threshold must be nonnegative or None.')
    limit = float(np.max(np.abs(values))) or 1.
    color_min = (0. if absolute else -limit) if vmin is None else float(vmin)
    color_max = limit if vmax is None else float(vmax)
    if not np.isfinite([color_min, color_max]).all() or color_min >= color_max:
        raise ValueError('Colour limits must be finite with vmin < vmax (including automatic defaults).')
    if meshes is None:
        try:
            import hcp_utils as hcp
        except ImportError as exc:
            raise ImportError('The Glasser/HCP plot requires hcp-utils; install it or pass meshes= explicitly.') from exc
        meshes = hcp.mesh
    if template is None:
        template = datasets.load_mni152_template(resolution=2)
    elif isinstance(template, (str, Path)):
        template = nib.load(str(template))
    shape = template.shape
    if len(shape) != 3:
        raise ValueError('template must be a 3D MNI image.')
    inverse = np.linalg.inv(template.affine)
    radius = sigma * radius_sigma
    extent = np.linalg.norm(inverse[:3, :3], axis=1) * radius
    numerator = np.zeros(shape, dtype=float)
    denominator = np.zeros(shape, dtype=float)
    # Visit only voxels near each electrode, avoiding a whole-brain distance
    # array per electrode and any electrodes-by-voxels dense matrix.
    for coordinate, value in zip(xyz, values):
        center = nib.affines.apply_affine(inverse, coordinate)
        lower = np.maximum(np.floor(center-extent).astype(int), 0)
        upper = np.minimum(np.ceil(center+extent).astype(int)+1, shape)
        if np.any(lower >= upper):
            continue
        indices = np.stack(np.meshgrid(*[np.arange(lo, hi) for lo, hi in zip(lower, upper)], indexing='ij'), axis=-1)
        world = nib.affines.apply_affine(template.affine, indices)
        distance2 = np.sum((world-coordinate)**2, axis=-1)
        kernel = np.exp(-distance2/(2*sigma*sigma))
        kernel[distance2 > radius*radius] = 0
        section = tuple(slice(lo, hi) for lo, hi in zip(lower, upper))
        numerator[section] += kernel * value
        denominator[section] += kernel
    supported = denominator >= min_support
    if not supported.any():
        raise ValueError('No supported template voxels; check MNI coordinates, template and min_support.')
    volume = np.divide(numerator, denominator, out=np.zeros(shape), where=supported)
    image = nib.Nifti1Image(volume, template.affine)
    mask = nib.Nifti1Image(supported.astype(np.uint8), template.affine)
    textures = {hemi: surface.vol_to_surf(image, getattr(meshes, f'pial_{hemi}'), mask_img=mask)
                for hemi in hemispheres}
    limit = float(np.max(np.abs(values))) or 1.
    if axes is None:
        fig = figure if figure is not None else plt.figure(figsize=(12, 5))
        axes = [fig.add_subplot(1, len(hemispheres), i+1, projection='3d') for i in range(len(hemispheres))]
    else:
        axes = np.asarray(axes, dtype=object).ravel()
        if len(axes) != len(hemispheres):
            raise ValueError('Provide one 3D axis per selected hemisphere.')
        fig = figure if figure is not None else axes[0].figure
    for ax, hemi in zip(axes, hemispheres):
        texture = textures[hemi]
        if not np.isfinite(texture).any():
            ax.set_axis_off()
            ax.set_title(f'{hemi}: no supported cortical surface')
            continue
        plotting.plot_surf_stat_map(
            getattr(meshes, f'inflated_{hemi}'), texture, hemi=hemi, view=view,
            bg_map=getattr(meshes, f'sulc_{hemi}'), axes=ax, figure=fig,
            cmap=cmap or ('Reds' if absolute else 'RdBu_r'),
            vmin=color_min, vmax=color_max,
            symmetric_cbar=(not absolute and vmin is None and vmax is None), threshold=threshold, colorbar=True,
            title=f'{title or "Electrode weights"}: {hemi}',
        )
    for ax in fig.axes: ax.grid(True, alpha=.25)
    if show:
        plt.show()
    return fig, dict(image=image, support_mask=mask, textures=textures)


def plot_pca_weights(result, n_components=3, *, plot_type='glass', absolute=True,
                     show=True, **plot_kwargs):
    """Plot PCA weights using 'glass' (default), 'voxel', or 'glasser'.

    Examples::

        plot_pca_weights(result, plot_type='voxel', voxel_size=20)
        plot_pca_weights(result, plot_type='glasser', sigma=4, absolute=True)

    Extra options are forwarded to the selected helper. Absolute values are the
    default, applied before spatial averaging. Coordinates
    are MNI mm. Voxel counts include every feature row, including co-located
    participant features. The PCA weights themselves are never modified.
    Returns a Matplotlib figure, preserving the original return type.
    """
    if result.dataset is None:
        raise ValueError('Use compute_pca to attach source locations.')
    if not isinstance(n_components, (int, np.integer)) or n_components < 1:
        raise ValueError('n_components must be a positive integer.')
    if plot_type not in ('glass', 'voxel', 'glasser'):
        raise ValueError("plot_type must be 'glass', 'voxel', or 'glasser'.")
    if plot_type == 'glass' and plot_kwargs:
        raise TypeError('Extra plot options apply to voxel or glasser plots.')
    k = min(n_components, result.weights.shape[1])
    if k < 1:
        raise ValueError('No PCA components to plot.')
    xyz = result.dataset.metadata[['x', 'y', 'z']].to_numpy()
    if plot_type == 'glasser':
        fig, axes = plt.subplots(k, 2, figsize=(12, 4.5*k), squeeze=False, subplot_kw={'projection': '3d'})
    else:
        fig, axes = plt.subplots(k, 1, figsize=(12, 3.3*k), squeeze=False)
    for pc in range(k):
        title = f'{result.dataset.name}: PC{pc+1}'
        values = result.weights[:, pc]
        if plot_type == 'voxel':
            plot_voxel_weights(values, xyz, absolute=absolute, figure=fig, axes=axes[pc, 0],
                               title=title, show=False, **plot_kwargs)
        elif plot_type == 'glasser':
            plot_glasser_weights(values, xyz, absolute=absolute, figure=fig, axes=axes[pc],
                                 title=title, show=False, **plot_kwargs)
        else:
            values, coordinates = _brain_weights(values, xyz, absolute)
            table = pd.DataFrame(coordinates, columns=['x', 'y', 'z'])
            table['weight'] = values
            grouped = table.groupby(['x', 'y', 'z'], sort=False).weight.mean().reset_index()
            limit = float(np.max(np.abs(grouped.weight))) or 1.
            note = ' (mean at co-located features)' if len(grouped) < len(table) else ''
            plotting.plot_markers(
                grouped.weight.to_numpy(), grouped[['x', 'y', 'z']].to_numpy(),
                node_size=7, node_cmap='Reds' if absolute else 'RdBu_r',
                node_vmin=0 if absolute else -limit, node_vmax=limit,
                node_threshold=None, colorbar=True, figure=fig, axes=axes[pc, 0], title=title+note,
            )
    for ax in fig.axes: ax.grid(True, alpha=.25)
    if show:
        plt.show()
        plt.close(fig)
    return fig


def _correlation_plot(matrix, name, title, plot):
    frame = pd.DataFrame(matrix,
                         index=[f'MEG PC{i+1}' for i in range(matrix.shape[0])],
                         columns=[f'iEEG PC{i+1}' for i in range(matrix.shape[1])])
    if plot:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        sns.heatmap(matrix,
                    vmin=-1,
                    vmax=1,
                    cmap='RdBu_r',
                    annot=True, 
                    ax=ax, 
                    cbar_kws={'label': 'Signed Spearman r'},
                    fmt='.2f')
        ax.set(title=f'{name}: {title}', xlabel='iEEG PC', ylabel='MEG PC',
               xticks=range(matrix.shape[1]), xticklabels=range(1, matrix.shape[1]+1),
               yticks=range(matrix.shape[0]), yticklabels=range(1, matrix.shape[0]+1))
        plt.show()
        plt.close(fig)
    return frame

def plot_meg_information(result):
    """One column per manipulation: geometry, normalized spectra, dimensionality."""
    import matplotlib.pyplot as plt
    if result['config'].get('schema_version') != 2:
        raise ValueError('Recompute direct manipulation comparisons into a new output folder.')
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), layout='constrained')
    for col, (manipulation, a, b, title) in enumerate(COMPARISONS):
        frame = result['metrics'].query('manipulation == @manipulation')
        ax = axes[0, col]
        for key, label, style in [('temporal_overlap', 'Symmetric temporal overlap', 'o-'),
            ('A_variance_captured_by_B', 'A variance captured by B', 's--'),
            ('B_variance_captured_by_A', 'B variance captured by A', '^:')]:
            ax.plot(frame.k, frame[key], style, label=label)
        ax.set(title=f'{title}\nA: {a}\nB: {b}', xlabel='Number of PCs (k)', ylabel='Fraction', ylim=(0, 1.02))
        ax.legend(fontsize=8); ax.grid(alpha=.2)
        for name, label in [(a, 'A'), (b, 'B')]:
            spectrum = result['spectra'].query('dataset == @name')
            spectrum = spectrum[spectrum.component <= max(result['config']['dimensions'])]
            axes[1, col].plot(spectrum.component, spectrum.cumulative_variance, 'o-', label=f'{label}: {name}')
        axes[1, col].set(xlabel='Number of PCs', ylabel='Cumulative explained variance', ylim=(0, 1.02))
        axes[1, col].legend(fontsize=8); axes[1, col].grid(alpha=.2)
        vals = result['summary'].set_index('dataset').loc[[a, b], 'effective_dimensionality']
        axes[2, col].bar(['A', 'B'], vals, color=['tab:blue', 'tab:orange'])
        axes[2, col].set(ylabel='Participation ratio', title='Effective dimensionality · full spectrum',
                         ylim=(0, result['summary'].effective_dimensionality.max()*1.15))
        for j, value in enumerate(vals):
            axes[2, col].text(j, value, f'{value:.2f}', ha='center', va='bottom')
    fig.suptitle('MEG construction comparisons · descriptive, in sample · no ground-truth construction')
    return fig


def plot_elect_number_effect(result):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,5),layout='constrained')
    for ax,key,title in zip(axes.flat,
        ['total_variance','subspace_overlap','matched_abs_r'],
        ['MEG total variance (all PCs)','Temporal subspace overlap with retained PCs iEEG','Matched component correlation retained PCs iEEG']):
        
        g=result['metrics'].groupby('n_channels')[key].agg(['median','min','max']).sort_index()
        
        ax.plot(g.index,g['median'],'o-',markersize=3, color=MODALITY_COLORS['MEG'])
        
        if result['config']['repeats']>1:
            ax.fill_between(g.index,g['min'],g['max'],alpha=.2)
        
        ax.set(title=title, xlabel='Number of MEG channels')
        if key == 'total_variance':
            ax.set_ylabel('Total variance (sum across channels)')
            ax.set_ylim(bottom=0)  # Absolute variance is not bounded by one.
        else:
            ax.set(ylabel='Similarity', ylim=(0, 1.02))
        ax.grid(alpha=.2)
    fig.suptitle('Channel-count effect: full-concatenated MEG vs fixed retained-PC iEEG reference')
    return fig

