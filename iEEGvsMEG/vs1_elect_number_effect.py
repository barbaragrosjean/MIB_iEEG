"""Temporal PCA recovery under nested subsampling of full-concatenated MEG."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from vs1_coverage_matching_utils import fit_block_pca
from vs1_coverage_information import reference_fraction


CHANNEL_SWEEP_SCHEMA_VERSION = 4


class _SelectedChannels:
    """Read selected feature blocks without copying a full concatenated dataset."""
    def __init__(self, dataset, selected):
        self.original = dataset
        self.selected = np.sort(selected)
        self.name = 'full_concatenated_subset'
        self.n_features = len(selected)
        self.n_observations = dataset.n_observations

    def blocks(self):
        offset = 0
        for block in self.original.blocks():
            ix = self.selected[(self.selected >= offset) & (self.selected < offset+block.shape[1])] - offset
            for start in range(0, len(ix), 1024):
                yield block[:, ix[start:start+1024]]
            offset += block.shape[1]


def run_elect_number_effect(meg, *, step=100, repeats=1, seed=2026,
                           output_dir='out/elect_number_effect', max_gram_gib=2., n_components=5):
    """Compare the same number of temporal PCs in each subset and full MEG.

    Subsets follow nested prefixes of a random source order, with no subject
    balancing. The sweep includes the full reference as an all-source endpoint.
    """
    for name, value in [('step', step), ('repeats', repeats), ('n_components', n_components)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    step, repeats, n_components = map(int, (step, repeats, n_components))
    if meg.name != 'full_concatenated':
        raise ValueError('Supply the full_concatenated MEG dataset.')
    if meg.n_features < n_components:
        raise ValueError('Full MEG has fewer sources than requested components.')
    counts = sorted({*range(step, min(20000, meg.n_features) + 1, step), meg.n_features})
    if counts[0] < n_components:
        raise ValueError('The smallest subset must have at least n_components sources.')

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use a new output directory to avoid mixing runs.')
    config = dict(
        schema_version=CHANNEL_SWEEP_SCHEMA_VERSION, max_subset_channels=20000,
        n_components=n_components, step=step, repeats=repeats, seed=seed,
        channel_counts=counts, n_full_channels=meg.n_features, condition_mode=meg.condition_mode,
        sampling='nested uniform channel addition, independent seeded paths',
        scope='descriptive in-sample recovery of full-concatenated MEG temporal PCs',
    )

    def temporal_pca(dataset):
        fit = fit_block_pca(dataset, n_components, max_gram_gib=max_gram_gib)
        rank = fit.scores.shape[1]
        if rank != n_components:
            raise ValueError(f'{dataset.name} ({dataset.n_features} sources): '
                             f'only {rank} PCs available, {n_components} required.')
        return dict(basis=fit.scores / np.linalg.norm(fit.scores, axis=0),
                    eigenvalues=fit.explained_variance, rank=rank)

    reference = temporal_pca(meg)
    metrics = []
    for repeat, sequence in enumerate(np.random.SeedSequence(seed).spawn(repeats)):
        order = np.random.default_rng(sequence).permutation(meg.n_features)
        for count in counts:
            subset = (reference if count == meg.n_features else
                      temporal_pca(_SelectedChannels(meg, order[:count])))
            cross = reference['basis'].T @ subset['basis']
            metrics.append(dict(
                repeat=repeat, n_channels=count,
                variance_fraction_retained=reference_fraction(reference, subset, n_components),
                subspace_overlap=float(np.clip(np.sum(cross**2) / n_components, 0, 1)),
            ))
            print(f'Channel sweep {repeat+1}/{repeats}: {count}/{meg.n_features} channels', flush=True)
        pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    (out / 'COMPLETE.json').write_text(json.dumps(config))
    return load_elect_number_effect(out)


def load_elect_number_effect(output_dir):
    out = Path(output_dir)
    if not (out / 'COMPLETE.json').exists():
        raise FileNotFoundError('Channel sweep is incomplete.')
    config = json.loads((out / 'config.json').read_text())
    if config['schema_version'] != CHANNEL_SWEEP_SCHEMA_VERSION:
        raise ValueError('Recompute the channel sweep for the full-MEG reference.')
    return dict(config=config, metrics=pd.read_csv(out / 'metrics.csv'))
