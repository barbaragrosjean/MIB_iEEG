"""Random channel-count sensitivity of full-concatenated MEG PCA.

Call run_elect_number_effect from coverage_matching.ipynb with loaded datasets.
Each seeded repetition adds random channels in fixed steps up to 20,000.
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


def _channel_counts(n_features, step):
    if step < 5:
        raise ValueError('step must be at least 5 for five-component PCA.')
    counts=list(range(step,min(20000,n_features)+1,step))
    if not counts:
        raise ValueError('step exceeds the available channel count or the 20,000-channel cap.')
    return counts


def run_elect_number_effect(meg, ieeg, *, step=100, repeats=1, seed=2026,
                           output_dir='out/elect_number_effect', max_gram_gib=2.):
    """Fit five PCs at step, 2*step, ... up to min(20,000, available channels).

    Only complete steps are evaluated. Each repetition uses nested prefixes
    of a random permutation of all channels; no subject balancing is applied.
    Variance fractions use the selected channels as their denominator.
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
    counts = _channel_counts(meg.n_features, step)
    config=dict(schema_version=2,max_channels=20000,n_components=5,step=step,repeats=repeats,seed=seed,channel_counts=counts,
                n_full_channels=meg.n_features,condition_mode=meg.condition_mode,
                sampling='nested uniform channel addition, independent seeded paths',
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
        # Selected channels at count n are order[:n]; each step adds the next batch.
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
    fig.supxlabel('Channels increase left → right. Nested random addition (new runs capped at 20,000); shading = range across sampling paths, not confidence intervals.',fontsize=9)
    return fig


def visualize_elect_number_effect(output_dir, *, save=True):
    """Read completed result tables and plot; never loads recordings or runs PCA."""
    result = load_elect_number_effect(output_dir)
    fig = plot_elect_number_effect(result)
    if save:
        for extension in ('png','pdf'):
            fig.savefig(Path(output_dir)/f'channel_number_effect.{extension}',dpi=180,bbox_inches='tight')
    return result, fig


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Run five-PC MEG channel-count sensitivity up to 20,000 channels on a cluster.')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--meg-dir',type=Path)
    parser.add_argument('--ieeg-dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--metadata-csv',type=Path,help='Optional electrode metadata; otherwise use LB src.setting.GetInfo.')
    parser.add_argument('--project-path',type=Path,help='Anatomical project path for GetInfo.')
    parser.add_argument('--ieeg-coordinate-unit',choices=['m','mm'],default='mm')
    parser.add_argument('--meg-coordinate-unit',choices=['m','mm'],default='m')
    parser.add_argument('--meg-times-file',type=Path)
    parser.add_argument('--meg-tmin',type=float,help='Defaults to first iEEG epoch time.')
    parser.add_argument('--sfreq',type=float,default=250.)
    parser.add_argument('--conditions',nargs='+',type=int,default=[1,2])
    parser.add_argument('--condition-mode',choices=['average','stack'],default='average')
    parser.add_argument('--step',type=int,default=100)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--max-gram-gib',type=float,default=2.)
    parser.add_argument('--plot-only',action='store_true',help='Load saved results without loading recordings or fitting PCA.')
    args=parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=args.root.expanduser().resolve()
    out=args.output_dir or root/'out'/'elect_number_effect'
    if not args.plot_only:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Choose a new --output-dir or use --plot-only for completed results.')
        for directory in (root,root.parent,root.parent/'LB'):sys.path.insert(0,str(directory))
        from coverage_matching_utils import load_dataset
        meg_dir=args.meg_dir or root/'MEG'/'dataMEG'
        ieeg_dir=args.ieeg_dir or root/'ieeg_shortWOBS_fs250'
        subjects=sorted(p.name.removesuffix('_epochs.p') for p in ieeg_dir.glob('*_epochs.p'))
        if not subjects:raise FileNotFoundError(f'No iEEG epochs in {ieeg_dir}.')
        tmin=args.meg_tmin
        if tmin is None and args.meg_times_file is None:
            info=json.loads((ieeg_dir/f'{subjects[0]}_info.json').read_text())
            tmin=float(info['time_epoch'][0])
        project=args.project_path
        if args.metadata_csv is None and project is None:
            from src.setting import PROJECT_PATH
            project=PROJECT_PATH
        ieeg=load_dataset('ieeg',meg_dir=meg_dir,ieeg_dir=ieeg_dir,metadata_csv=args.metadata_csv,
                          project_path=project,ieeg_subjects=subjects,
                          ieeg_coordinate_unit=args.ieeg_coordinate_unit,meg_coordinate_unit=args.meg_coordinate_unit,
                          meg_times_file=args.meg_times_file,meg_tmin=tmin,sfreq=args.sfreq,
                          conditions=tuple(args.conditions),condition_mode=args.condition_mode)
        meg=load_dataset('full_concatenated',reference=ieeg)
        print(f'Loaded {meg.n_features} MEG channels; {ieeg.n_features} iEEG electrodes.',flush=True)
        run_elect_number_effect(meg,ieeg,step=args.step,repeats=args.repeats,seed=args.seed,
                               output_dir=out,max_gram_gib=args.max_gram_gib)
        (out/'run_config.json').write_text(json.dumps(vars(args),default=str,indent=2))
    _,fig=visualize_elect_number_effect(out)
    plt.close(fig)
    print(f'Results and figures: {out.resolve()}',flush=True)


if __name__ == '__main__':
    main()
