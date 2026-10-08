#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

from plssvd_full_pca import fit_full_data_pca, correlate_full_pca, plot_full_pca_comparison
import numpy as np

MEG_KIND = 'full_concatenated'
N_COMPONENTS = 10  # An explicitly prepared component count (see evaluation command above)

ROOT = Path.cwd()
OUTPUT_DIR = ROOT / 'out' / 'plssvd_eval' / MEG_KIND / 'ieeg__time_point'
PCA_ABSOLUTE = True 


def main():
    full_pca = fit_full_data_pca(ROOT / 'out' / 'trial_cache', n_components=N_COMPONENTS)

    full_pca_result = correlate_full_pca(OUTPUT_DIR, full_pca, n_components=N_COMPONENTS, absolute=PCA_ABSOLUTE)
    fig = plot_full_pca_comparison(full_pca_result)

    pca_destination = OUTPUT_DIR / 'full_data_pca_comparison'
    pca_destination.mkdir(exist_ok=True)
    statistic = 'absolute' if PCA_ABSOLUTE else 'signed'
    fig.savefig(pca_destination / f'comparison_k{N_COMPONENTS}_{statistic}.png', dpi=180, bbox_inches='tight')
    
    np.savez_compressed(pca_destination / f'correlations_k{N_COMPONENTS}_{statistic}.npz',
        mean=full_pca_result['mean'], std=full_pca_result['std'], count=full_pca_result['count'],
        iterations=full_pca_result['iterations'])


if __name__=='__main__':main()
