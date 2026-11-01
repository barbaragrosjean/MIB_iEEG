"""Automatic cluster-count selection across fixed-trial subject associations."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from vs3_1_weight_clustering import select_cluster_count


def summarize_pairing_clustering(aligned, *, component_counts, methods, seed=2026,
                                use_thresholded=False, k_min=2, k_max=10,
                                silhouette_sample_size=2000, checkpoint_dir=None):
    """Keep algorithm seed fixed; summarize successful association fits, with counts.

    Input weights retain fixed electrode coverage slots. Repeated MEG source
    matches remain repeated slots, as in the preceding pairing-weight analysis.
    No matching of categorical cluster IDs across associations is required.
    """
    signature = hashlib.sha256(json.dumps(dict(provenance=aligned['provenance'],
        component_counts=component_counts, methods=methods, seed=seed,
        use_thresholded=use_thresholded, k_min=k_min, k_max=k_max,
        silhouette_sample_size=silhouette_sample_size), sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(checkpoint_dir)/signature if checkpoint_dir else None
    if folder: folder.mkdir(parents=True, exist_ok=True)
    metadata = aligned['anchor'].copy()
    metadata['region_shared'] = metadata.region.fillna('Unassigned')
    rows, selections = [], []
    for modality, stack in aligned['weights'].items():
        for dimensions in component_counts:
            for method in methods:
                for iteration, weights in enumerate(stack):
                    key = dict(modality=modality, method=method, n_components=dimensions, iteration=iteration)
                    path = folder/f'{modality}_{method}_k{dimensions}_i{iteration:03d}.json' if folder else None
                    if path and path.exists():
                        saved = json.loads(path.read_text())
                    else:
                        try:
                            result = select_cluster_count(weights, metadata, method=method,
                                use_thresholded=use_thresholded, seed=seed, k_min=k_min, k_max=k_max,
                                silhouette_sample_size=silhouette_sample_size, n_components=dimensions)
                            table = result['cluster_selection']
                            best = table.loc[table.selected].iloc[0]
                            row = dict(**key, n_clusters=int(best.actual_k), silhouette=float(best.silhouette), status='ok')
                            candidates = table.assign(**key).to_dict('records')
                        except ValueError as exc:
                            if not str(exc).startswith('No valid cluster-count candidate'):
                                raise
                            row = dict(**key, n_clusters=None, silhouette=None, status=str(exc))
                            candidates = []
                        saved = dict(row=row, candidates=candidates)
                        if path:
                            temporary = path.with_suffix('.tmp')
                            temporary.write_text(json.dumps(saved))
                            temporary.replace(path)
                    rows.append(saved['row']);selections.extend(saved['candidates'])
                print(f'{modality}, {method}, {dimensions} components: {len(stack)} associations summarized', flush=True)
    iterations = pd.DataFrame(rows)
    for metric in ['n_clusters','silhouette']:
        iterations[metric] = pd.to_numeric(iterations[metric], errors='coerce')
    groups = ['modality','method','n_components']
    summary = iterations.groupby(groups)[['n_clusters','silhouette']].agg(['mean','std','count'])
    summary.columns = [f'{metric}_{stat}' for metric,stat in summary.columns]
    summary = summary.reset_index()
    summary['n_requested'] = len(next(iter(aligned['weights'].values())))
    return dict(iterations=iterations, summary=summary, candidates=pd.DataFrame(selections),
                provenance=aligned['provenance'], source_dir=aligned['source_dir'])
