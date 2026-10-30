"""Random channel-count sensitivity of full-concatenated MEG PCA.

Call run_elect_number_effect from coverage_matching.ipynb with loaded datasets.
Each seeded repetition adds random channels in fixed steps up to 20,000.
The fixed iEEG reference and MEG representation/preprocessing never change.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from vs1_coverage_matching_utils import fit_block_pca, compare_pca_to_ieeg


class _SelectedChannels:
    """Read selected feature blocks without copying a full concatenated dataset."""
    def __init__(self, dataset, selected):
        self.original = dataset
        self.selected = np.sort(selected)
        self.name = 'full_concatenated_subset'
        self.n_features = len(selected)
        self.n_observations = dataset.n_observations
        self.condition_mode = dataset.condition_mode
        self.source_data = dataset.source_data
        self.scale = 1.
        if dataset.block_scaling:
            variance = sum(np.var(x, axis=0, ddof=1).sum() for x in self.blocks())
            self.scale = float(1 / np.sqrt(variance))

    def blocks(self):
        offset = 0
        for block in self.original.blocks():
            ix = self.selected[(self.selected >= offset) & (self.selected < offset+block.shape[1])] - offset
            for start in range(0, len(ix), 1024):
                yield block[:, ix[start:start+1024]] * self.scale
            offset += block.shape[1]


def _channel_counts(n_features, step):
    counts=list(range(step,min(20000,n_features)+1,step))
    if not counts:
        raise ValueError('step exceeds the available channel count or the 20,000-channel cap.')
    return counts


def run_elect_number_effect(meg, ieeg, *, step=100, repeats=1, seed=2026,
                           output_dir='out/elect_number_effect', max_gram_gib=2., n_components=5):
    """Fit the requested PCs at step, 2*step, ... up to min(20,000, available channels).

    Only complete steps are evaluated. Each repetition uses nested prefixes
    of a random permutation of all channels; no subject balancing is applied.
    Variance fractions use the selected channels as their denominator.
    """
    for name,value in [('step',step),('repeats',repeats)]:
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if meg.name != 'full_concatenated':
        raise ValueError('Supply the full_concatenated MEG dataset.')
    if min(meg.n_features,ieeg.n_features) < n_components:
        raise ValueError('Datasets have fewer features than requested components.')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError('Use a new output directory to avoid mixing runs.')
    counts = _channel_counts(meg.n_features, step)
    if counts[0] < n_components:
        raise ValueError('Channel step must be at least n_components.')
    config=dict(schema_version=3,max_channels=20000,n_components=n_components,step=step,repeats=repeats,seed=seed,channel_counts=counts,
                n_full_channels=meg.n_features,condition_mode=meg.condition_mode,
                sampling='nested uniform channel addition, independent seeded paths',
                scope='descriptive in-sample sensitivity; fixed full iEEG PCA reference')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    reference=fit_block_pca(ieeg,n_components=n_components,max_gram_gib=max_gram_gib);reference.dataset=ieeg
    if reference.scores.shape[1]!=n_components: raise ValueError('iEEG rank is below n_components.')
    np.savez_compressed(out/'ieeg_reference.npz',scores=reference.scores,
                        explained_variance=reference.explained_variance,
                        explained_variance_ratio=reference.explained_variance_ratio)
    if len(meg.metadata)==meg.n_features:
        meg.metadata.reset_index(drop=True).rename_axis('channel_index').to_csv(out/'channel_metadata.csv')
    metrics=[];pairs=[];full_fit=None
    for repeat,sequence in enumerate(np.random.SeedSequence(seed).spawn(repeats)):
        order=np.random.default_rng(sequence).permutation(meg.n_features)
        # Selected channels at count n are order[:n]; each step adds the next batch.
        np.save(out/f'channel_order_{repeat:03d}.npy',order)
        for count in counts:
            selected=_SelectedChannels(meg,order[:count])
            if count==meg.n_features and full_fit is not None: fitted=full_fit
            else:
                fitted=fit_block_pca(selected,n_components=n_components,max_gram_gib=max_gram_gib)
                fitted.dataset=selected
                if count==meg.n_features:full_fit=fitted
            if fitted.scores.shape[1]!=n_components: raise ValueError(f'Only {fitted.scores.shape[1]} PCs at {count} channels; {n_components} required.')
            table,matched=compare_pca_to_ieeg({'iEEG':reference,'MEG':fitted},n_components=(n_components,),plot=False)
            row=dict(repeat=repeat,n_channels=count,n_removed=meg.n_features-count,
                     total_variance=fitted.total_variance,
                     variance_retained=float(fitted.explained_variance.sum()),
                     variance_fraction_retained=float(fitted.explained_variance_ratio.sum()))
            row.update({key:float(table.iloc[0][key]) for key in ('ieeg_variance_captured','subspace_overlap','matched_abs_r')})
            for pc in range(n_components):
                row[f'pc{pc+1}_variance']=float(fitted.explained_variance[pc])
                row[f'pc{pc+1}_variance_ratio']=float(fitted.explained_variance_ratio[pc])
            metrics.append(row)
            matched=matched.assign(repeat=repeat,n_channels=count)
            pairs.extend(matched.to_dict('records'))
            print(f'Channel sweep {repeat+1}/{repeats}: {count}/{meg.n_features} channels',flush=True)
        pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(pairs).to_csv(out/'component_pairs.csv',index=False)
    (out/'COMPLETE.json').write_text(json.dumps(config))
    return load_elect_number_effect(out)


def load_elect_number_effect(output_dir):
    out=Path(output_dir)
    if not (out/'COMPLETE.json').exists(): raise FileNotFoundError('Channel sweep is incomplete.')
    return dict(config=json.loads((out/'config.json').read_text()),metrics=pd.read_csv(out/'metrics.csv'),
                component_pairs=pd.read_csv(out/'component_pairs.csv'))
