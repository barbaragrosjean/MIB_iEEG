"""Random channel-removal sensitivity of full-concatenated MEG PCA.

Call run_elect_number_effect from coverage_matching.ipynb with loaded datasets.
Each seeded repetition removes another random batch of 100 remaining channels.
The fixed iEEG reference and MEG representation/preprocessing never change.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from coverage_matching_utils import fit_block_pca, compare_pca_to_ieeg


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

    def blocks(self):
        offset = 0
        for block in self.original.blocks():
            ix = self.selected[(self.selected >= offset) & (self.selected < offset+block.shape[1])] - offset
            for start in range(0, len(ix), 1024):
                yield block[:, ix[start:start+1024]]
            offset += block.shape[1]


def run_elect_number_effect(meg, ieeg, *, step=100, repeats=1, seed=2026,
                           output_dir='out/elect_number_effect', max_gram_gib=2.):
    """Fit five PCs at full count, then N-100, N-200, ... while >=5 remain.

    Uniform sampling is across concatenated channels, not balanced by subject.
    Within each repetition subsets are nested. Independent permutations across
    repetitions quantify sampling variability. Variance fractions use the
    remaining MEG channels as denominator; total variance is saved separately.
    """
    for name,value in [('step',step),('repeats',repeats)]:
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if meg.name != 'full_concatenated':
        raise ValueError('Supply the full_concatenated MEG dataset.')
    if min(meg.n_features,ieeg.n_features) < 5:
        raise ValueError('Both datasets require at least five features.')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError('Use a new output directory to avoid mixing runs.')
    counts=list(range(meg.n_features,4,-step))
    config=dict(n_components=5,step=step,repeats=repeats,seed=seed,channel_counts=counts,
                n_full_channels=meg.n_features,condition_mode=meg.condition_mode,
                sampling='nested uniform channel removal, independent seeded paths',
                scope='descriptive in-sample sensitivity; fixed full iEEG PCA reference')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    reference=fit_block_pca(ieeg,n_components=5,max_gram_gib=max_gram_gib);reference.dataset=ieeg
    if reference.scores.shape[1]!=5: raise ValueError('iEEG rank is below five.')
    np.savez_compressed(out/'ieeg_reference.npz',scores=reference.scores,
                        explained_variance=reference.explained_variance,
                        explained_variance_ratio=reference.explained_variance_ratio)
    if len(meg.metadata)==meg.n_features:
        meg.metadata.reset_index(drop=True).rename_axis('channel_index').to_csv(out/'channel_metadata.csv')
    metrics=[];pairs=[];full_fit=None
    for repeat,sequence in enumerate(np.random.SeedSequence(seed).spawn(repeats)):
        order=np.random.default_rng(sequence).permutation(meg.n_features)
        # Remaining channels at count n are order[:n]; the tail is removed first.
        np.save(out/f'channel_order_{repeat:03d}.npy',order)
        for count in counts:
            selected=_SelectedChannels(meg,order[:count])
            if count==meg.n_features and full_fit is not None: fitted=full_fit
            else:
                fitted=fit_block_pca(selected,n_components=5,max_gram_gib=max_gram_gib)
                fitted.dataset=selected
                if count==meg.n_features:full_fit=fitted
            if fitted.scores.shape[1]!=5: raise ValueError(f'Only {fitted.scores.shape[1]} PCs at {count} channels; five required.')
            table,matched=compare_pca_to_ieeg({'iEEG':reference,'MEG':fitted},n_components=(5,),plot=False)
            row=dict(repeat=repeat,n_channels=count,n_removed=meg.n_features-count,
                     total_variance=fitted.total_variance,
                     variance_first5=float(fitted.explained_variance.sum()),
                     variance_fraction_first5=float(fitted.explained_variance_ratio.sum()))
            row.update({key:float(table.iloc[0][key]) for key in ('ieeg_variance_captured','subspace_overlap','matched_abs_r')})
            for pc in range(5):
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


def plot_elect_number_effect(result):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,key,title in zip(axes.flat,
        ['variance_fraction_first5','ieeg_variance_captured','subspace_overlap','matched_abs_r'],
        ['MEG variance fraction in first 5 PCs','Retained iEEG-PC variance captured',
         'Temporal subspace overlap','Matched component correlation (mean |r|)']):
        g=result['metrics'].groupby('n_channels')[key].agg(['median','min','max']).sort_index()
        ax.plot(g.index,g['median'],'o-',markersize=3)
        if result['config']['repeats']>1:ax.fill_between(g.index,g['min'],g['max'],alpha=.2)
        ax.set(title=title,xlabel='Remaining MEG channels',ylabel='Fraction / similarity',ylim=(0,1.02))
        ax.grid(alpha=.2)
    fig.suptitle('Channel-count effect: full-concatenated MEG vs fixed 5-PC iEEG reference')
    fig.supxlabel('Channels increase left → right. Nested random removal; shading = range across sampling paths, not confidence intervals.',fontsize=9)
    return fig
