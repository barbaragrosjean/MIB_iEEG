"""Configuration-driven analyses over all 22 explicit participant pairings.

Run commands from any directory. Relative config paths resolve against the config
file; {ROOT} resolves to iEEGvsMEG. --dry-run validates the plan without raw data.
"""
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
import argparse
import hashlib
import importlib
import json
import shutil
import sys

from vs0_utils_pairing import PAIRING_IDS, pairing_paths, pairing_context, digest_file

ROOT = Path(__file__).resolve().parent
COVERAGE = {'vs1_manipulation':'manipulations', 'vs1_channels':'channels',
            'vs1_pca':'pca', 'vs1_stability':'stability'}
SPECIFICITY = {'vs3_weights':'weights','vs3_pairing_weights':'pairing_weights',
    'vs3_spatial_profiles':'pairing_profiles','vs3_pairing_clustering':'pairing_clustering',
    'vs3_cross_modal_profiles':'pairing_cross_modal','vs3_gradients':'gradients',
    'vs3_radius_search':'radius_search','vs3_percent_search':'percent_search',
    'vs3_gradient_comparison':'gradient_comparison'}
ANALYSES = ('vs1_construct', *COVERAGE, 'vs1_channel_space', 'vs1_roi_shell_maps',
    'vs2_cov_models','vs2_plssvd_fit','vs2_plssvd_evaluation','vs2_plssvd_pca',
    'vs2_plssvd_permutation','vs2_plssvd_selection','vs3_subspace','vs3_weight_stability',
    'vs3_weights','vs3_pairing_weights','vs3_spatial_profiles','vs3_weight_clustering',
    'vs3_pairing_clustering','vs3_cross_modal_profiles','vs3_gradients',
    'vs3_radius_search','vs3_percent_search','vs3_gradient_comparison')


def read_config(path):
    path=Path(path).resolve()
    def expand(v):
        if isinstance(v,str):return v.replace('{ROOT}',str(ROOT)).replace('{OUT}',str(path.parent.parent))
        if isinstance(v,list):return [expand(x) for x in v]
        if isinstance(v,dict):return {k:expand(x) for k,x in v.items()}
        return v
    config=expand(json.loads(path.read_text()))
    if config.get('analysis') not in (*ANALYSES,'vs0_pairs'):raise ValueError(f'{path}: unknown analysis')
    if config.get('schema_version') != 1:raise ValueError(f'{path}: unsupported config schema')
    return config


def config_for(out_root, analysis):
    return Path(out_root)/analysis/'config.json'


def parameters(out_root, analysis):
    return read_config(config_for(out_root,analysis))['parameters']


def output_path(out_root, analysis, pairing):
    if analysis not in ANALYSES or pairing not in PAIRING_IDS:raise ValueError('Unknown analysis/pairing')
    return Path(out_root)/analysis/pairing


def data_options(out_root):
    return parameters(out_root,'vs1_construct')['data'].copy()


def legacy_config(out_root, analysis, pairing):
    p=parameters(out_root,analysis).copy()
    p.pop('display',None)
    p.update(data=data_options(out_root), output_dir=str(Path(out_root)/analysis),
             _managed_output=str(output_path(out_root,analysis,pairing)))
    return p


def specificity_config(out_root, pairing):
    fit=parameters(out_root,'vs3_weights').copy()
    fit.pop('display',None);fit['data']=data_options(out_root)
    options={}
    for name,key in SPECIFICITY.items():
        if name!='vs3_weights':options[key]={k:v for k,v in parameters(out_root,name).items() if k not in ('pairing','display')}
    return dict(fit=fit, analyses=options, pairing=parameters(out_root,'vs3_pairing_weights')['pairing'],
                output_dir=str(Path(out_root)/'vs3_weights'), _managed_root=str(Path(out_root)),
                _pairing_id=pairing)


def required_analyses(analysis):
    if analysis=='vs1_construct':return []
    if analysis in (*COVERAGE,'vs1_channel_space','vs2_cov_models','vs2_plssvd_fit','vs3_subspace','vs3_weight_stability','vs3_weights'):
        return ['vs1_construct']
    if analysis=='vs1_roi_shell_maps':return ['vs1_channel_space']
    if analysis=='vs2_plssvd_evaluation':return ['vs2_plssvd_fit']
    if analysis=='vs2_plssvd_pca':return ['vs2_plssvd_evaluation']
    if analysis=='vs2_plssvd_permutation':return ['vs2_plssvd_evaluation']
    if analysis=='vs2_plssvd_selection':return ['vs2_plssvd_permutation']
    if analysis=='vs3_pairing_weights':return ['vs1_construct']
    if analysis in ('vs3_spatial_profiles','vs3_pairing_clustering','vs3_cross_modal_profiles'):return ['vs3_pairing_weights']
    return ['vs3_weights']


def plan(config_path, pairings=None):
    path=Path(config_path).resolve();cfg=read_config(path)
    selected=pairings or cfg.get('pairings','all')
    selected=list(PAIRING_IDS) if selected=='all' or selected==['all'] else ([selected] if isinstance(selected,str) else selected)
    if len(selected)!=len(set(selected)) or not set(selected)<=set(PAIRING_IDS):raise ValueError('Invalid pairing selection')
    return [dict(analysis=cfg['analysis'],pairing=p,output=str(output_path(path.parent.parent,cfg['analysis'],p)),
                 dependencies=required_analyses(cfg['analysis'])) for p in selected]


def source_inputs(out_root, analysis, pairing):
    """Track files by content for small metadata and by signature for recordings."""
    p=dict(parameters(out_root,'vs0_pairs'),**{k:v for k,v in data_options(out_root).items() if k in ('ieeg_dir','meg_dir','project_path')});dem,elect=pairing_paths(Path(out_root)/'vs0_pairs',pairing)
    import pandas as pd
    pairs=pd.read_csv(dem,dtype=str)
    paths=[Path(p['ieeg_dem_csv']),Path(p['meg_dem_xlsx'])]
    for subject in pairs.ieeg_subject:
        paths += [Path(p['ieeg_dir'])/f'{subject}{suffix}' for suffix in ('_epochs.p','_info.json','_coords.csv')]
        paths += [Path(p['project_path'])/'misc/electrodes'/f'{subject}_electrodes.csv']
    for subject in pairs.meg_subject:paths.append(Path(p['meg_dir'])/f'{subject}_norm0_abs_0.mat')
    result=[]
    for path in paths:
        stat=path.stat();r=dict(path=str(path.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        if path.suffix in ('.csv','.json','.xlsx'):r['sha256']=digest_file(path)
        result.append(r)
    return result


def signature(out_root, analysis, pairing):
    relevant={'vs0_pairs','vs1_construct',analysis}
    pending=[analysis]
    while pending:
        for parent in required_analyses(pending.pop()):
            if parent not in relevant:relevant.add(parent);pending.append(parent)
    if analysis in SPECIFICITY:relevant.add('vs3_weights')
    configs={name:digest_file(config_for(out_root,name)) for name in sorted(relevant)}
    code={p.name:digest_file(p) for p in ROOT.glob('*.py')}
    pair_root=Path(out_root)/'vs0_pairs'
    pair_manifest=json.loads((pair_root/'manifest.json').read_text())
    if pair_manifest.get('status')!='complete':raise ValueError('Pair creation is incomplete.')
    if pair_manifest['config']!=read_config(pair_root/'config.json'):
        raise ValueError('Pairing configuration changed; regenerate pairs explicitly.')
    for name,expected in pair_manifest['inputs'].items():
        if digest_file(name)!=expected:raise ValueError(f'Eligibility input changed: {name}; regenerate pairs.')
    pairs={p.name:digest_file(p) for p in pairing_paths(pair_root,pairing)}
    if any(pair_manifest['outputs'].get(name)!=value for name,value in pairs.items()):
        raise ValueError('Pairing CSV changed after publication; regenerate pairs explicitly.')
    parents={name:digest_file(output_path(out_root,name,pairing)/'manifest.json') for name in required_analyses(analysis)}
    payload=dict(configs=configs,code=code,pairs=pairs,dependencies=parents,
                 inputs=source_inputs(out_root,analysis,pairing))
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload


def validate_result(out_root, analysis, pairing):
    path=output_path(out_root,analysis,pairing);manifest=json.loads((path/'manifest.json').read_text())
    expected,_=signature(out_root,analysis,pairing)
    if manifest.get('status')!='complete' or manifest['signature']!=expected:
        raise ValueError(f'Stale results: {path}. Run its config with --recompute.')
    for record in manifest['outputs']:
        file=path/record['path']
        if not file.is_file() or file.stat().st_size!=record['size']:
            raise ValueError(f'Missing or incomplete result: {file}')
    return path


def ensure_trial_cache(out_root):
    """One union cache for all pairings; never truncate it to one selected cohort."""
    from vs0_utils_pairing import without_pairing
    from vs0_utils_load_and_config import GetInfo,ieeg_getinfo_coordinates_m
    from vs2_1_utils_plssvd_eval import prepare_trial_cache
    import pandas as pd
    data=data_options(out_root);cache=Path(out_root)/'vs1_construct/shared/trials'
    # Union of all selected MEG participants; all pairing files must already exist.
    union=set();ieeg=None
    for pairing in PAIRING_IDS:
        dem,_=pairing_paths(Path(out_root)/'vs0_pairs',pairing)
        pairs=pd.read_csv(dem,dtype={'ieeg_subject':str,'meg_subject':str})
        subjects=pairs.ieeg_subject.tolist()
        if ieeg is None:ieeg=subjects
        if subjects!=ieeg:raise ValueError('All 22 pairings must retain the same ordered iEEG cohort.')
        union.update(pairs.meg_subject)
    with without_pairing():
        coords,_,names,owners,regions=GetInfo(ieeg,project_path=data['project_path'],data_path=data['ieeg_dir'])
        meta=pd.DataFrame(ieeg_getinfo_coordinates_m(coords),columns=['x','y','z'])
        meta['channel']=names;meta['subject']=owners;meta['region']=regions
        meta['channel_index']=meta.groupby('subject',sort=False).cumcount()
        return prepare_trial_cache(data['meg_dir'],data['ieeg_dir'],cache,sorted(union),ieeg,meta,
            conditions=data['conditions'],ieeg_coordinate_unit='m',meg_coordinate_unit=data['meg_coordinate_unit'])


def construct(out_root,pairing,destination):
    import numpy as np
    from vs1_run_coverage import load_datasets
    from vs1_utils_coverage_matching import variance_summary
    cfg=legacy_config(out_root,'vs1_construct',pairing)
    export_trials=cfg.pop('export_trials',True)
    datasets=load_datasets(cfg);source=datasets['iEEG'].source_data
    destination.mkdir(parents=True,exist_ok=True)
    shared=Path(out_root)/'vs1_construct/shared';shared.mkdir(exist_ok=True)
    # Fixed iEEG means are shared; construction/scaling views remain pairing-specific.
    target=shared/'ieeg.npy'
    if target.exists():
        if not np.array_equal(np.load(target,mmap_mode='r'),source['ieeg']):
            raise ValueError('Shared iEEG preprocessing changed; choose a new output root or clear its shared cache explicitly.')
    else:np.save(target,source['ieeg'])
    np.save(shared/'times.npy',source['times'])
    source['electrode_metadata'].to_csv(shared/'ieeg_metadata.csv',index=False)
    source['trial_counts'].to_csv(shared/'ieeg_trial_counts.csv',index=False)
    variance_summary(datasets).to_csv(destination/'variance.csv',index=False)
    for name,ds in datasets.items():
        ds.metadata.to_csv(destination/f'{name}_metadata.csv',index=False)
        np.savez_compressed(destination/f'{name}_mapping.npz',electrode_to_feature=ds.electrode_to_feature,scale=ds.scale)
        if ds.matching is not None:ds.matching.to_csv(destination/f'{name}_matching.csv',index=False)
    if export_trials:ensure_trial_cache(out_root)


@contextmanager
def cli_arguments(arguments):
    previous=sys.argv
    sys.argv=[previous[0],*map(str,arguments)]
    try:yield
    finally:sys.argv=previous


def execute(out_root,analysis,pairing,destination):
    p=parameters(out_root,analysis);cache=Path(out_root)/'vs1_construct/shared/trials'
    if analysis=='vs1_construct':
        with file_lock(Path(out_root)/'vs1_construct/.shared.lock'):
            return construct(out_root,pairing,destination)
    if analysis in COVERAGE:
        import vs1_run_coverage as m
        cfg=legacy_config(out_root,analysis,pairing)
        return m.run_analysis(cfg,COVERAGE[analysis],m.load_datasets(cfg))
    if analysis=='vs1_channel_space':
        from vs1_run_channel_space import run
        return run(legacy_config(out_root,analysis,pairing))
    if analysis=='vs1_roi_shell_maps':
        import numpy as np
        import pandas as pd
        from vs1_run_channel_space import load_result
        from vs1_1_utils_channel_space import roi_shell_source_maps
        from vs0_utils_load_and_config import meg_average,coordinates_mm
        result=load_result(legacy_config(out_root,'vs1_channel_space',pairing))
        data=data_options(out_root)
        positions=[coordinates_mm(meg_average(Path(data['meg_dir'])/f'{s}_norm0_abs_0.mat',data['average_cache_dir'])[1],data['meg_coordinate_unit'])
                   for s in result['correlations'].meg_subject.unique()]
        destination.mkdir(parents=True,exist_ok=True)
        roi_shell_source_maps(result,np.concatenate(positions)).to_csv(destination/'roi_shell_source_maps.csv',index=False)
        return
    if analysis=='vs2_cov_models':
        from vs2_run_cov_models import run
        return run(legacy_config(out_root,analysis,pairing))
    if analysis=='vs2_plssvd_fit':
        from vs2_1_utils_plssvd_eval import load_trial_cache,ValidationOptions,fit_plssvd,validation_run_name
        trials=load_trial_cache(cache)
        for variant in p['variants']:
            name=validation_run_name(variant['perm'],variant['perm_type']);folder=destination/name
            if (folder/'COMPLETE.json').exists():continue
            options=ValidationOptions(**p['options'],**variant)
            fit_plssvd(trials,p['meg_kind'],options,output_dir=folder,scratch_dir=p.get('scratch_dir'),resume=True)
        return
    runs=output_path(out_root,'vs2_plssvd_fit',pairing)
    if analysis=='vs2_plssvd_evaluation':
        from vs2_1_run_plssvd_postprocess import evaluate_run
        from vs2_1_run_plssvd_results import prepare_summary
        for run in sorted(runs.iterdir()):
            if (run/'validation_options.json').exists():
                evaluate_run(run,p['components'],cache)
                for k in p['components']:prepare_summary(run,k,p['n_iterations'])
        return
    if analysis=='vs2_plssvd_pca':
        from vs2_1_run_plssvd_fit_pca import run
        return run(runs/'none',cache,**p)
    if analysis=='vs2_plssvd_permutation':
        from vs2_1_run_plssvd_postprocess import prepare_comparison
        from vs2_1_utils_plssvd_eval import validation_run_name
        for mode in p['modes']:
            for k in p['components']:prepare_comparison(runs,validation_run_name(mode,p['perm_type']),k)
        return
    if analysis=='vs2_plssvd_selection':
        from vs2_1_run_plssvd_select_components import prepare_selection
        return prepare_selection(runs,**p)
    if analysis=='vs3_subspace':
        from vs3_0_run_compare_subspace import run_comparison
        return run_comparison(cache,destination,**p)
    if analysis=='vs3_weight_stability':
        from vs3_0_run_spatial_reliability import fit_iterations
        from vs3_0_run_weight_stability import summarize_weights
        common=dict(cache_dir=cache,meg_kind='paired_coverage',models=['separate_pca','plssvd'],
            components=5,iterations=100,seed=2026,split_unit='trial',block_scaling='equal_variance',
            max_gram_gib=2.,weight_only=True,pool_hemispheres=False,scratch_dir=None,
            ieeg_coords_dir=None,meg_coords_dir=None,ieeg_coords_unit='mm',meg_coords_unit='mm',
            atlas_image=None,atlas_labels=None)
        common.update({k:v for k,v in p.items() if k not in ('conditions','roi_level')})
        for condition in p['conditions']:
            args=SimpleNamespace(**common,variation=condition,output_dir=destination/condition)
            fit_iterations(args);summarize_weights(args.output_dir,p['roi_level'])
        return
    if analysis in SPECIFICITY:
        from vs3_1_run_modality_specificity import run
        cfg=specificity_config(out_root,pairing)
        return run(cfg,SPECIFICITY[analysis])
    if analysis=='vs3_weight_clustering':
        from vs0_utils_result_io import load_result
        from vs3_1_run_weight_clustering import select_cluster_count,cluster_weights
        base=load_result(output_path(out_root,'vs3_weights',pairing))
        destination.mkdir(parents=True,exist_ok=True)
        for dimensions in p['component_counts']:
            for method in p['methods']:
                for modality,w in base['weights'].items():
                    options={k:v for k,v in p.items() if k not in ('methods','component_counts','n_clusters')}
                    if p['n_clusters']=='auto':result=select_cluster_count(w,base['metadata'][modality],method=method,n_components=dimensions,**options)
                    else:result=cluster_weights(w,base['metadata'][modality],method=method,n_components=dimensions,n_clusters=p['n_clusters'],**{k:v for k,v in options.items() if k not in ('k_min','k_max')})
                    folder=destination/method/f'k_{dimensions:03d}'/modality;folder.mkdir(parents=True,exist_ok=True)
                    for name,table in result.items():table.to_csv(folder/f'{name}.csv',index=False)
        return
    raise ValueError(analysis)


def _run_analysis(config_path,pairing,*,recompute=False):
    path=Path(config_path).resolve();cfg=read_config(path);analysis=cfg['analysis'];out_root=path.parent.parent
    if path.parent.name!=analysis:raise ValueError('Config must reside in its named analysis folder.')
    for parent in required_analyses(analysis):
        run_analysis(config_for(out_root,parent),pairing)
    expected,payload=signature(out_root,analysis,pairing)
    destination=output_path(out_root,analysis,pairing)
    pending_state=path.parent/f'.{pairing}.pending.json'
    if pending_state.exists() and json.loads(pending_state.read_text())['signature']!=expected and not recompute:
        raise ValueError('Interrupted run inputs/settings changed; use --recompute explicitly.')
    if (destination/'manifest.json').exists():
        if not recompute:return validate_result(out_root,analysis,pairing)
        archive=destination.parent/'archive';archive.mkdir(exist_ok=True)
        destination.rename(archive/f'{pairing}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}')
    elif destination.exists() and any(destination.iterdir()):
        # Fit routines checkpoint internally. Other computations need a clean destination.
        if recompute or analysis!='vs2_plssvd_fit':
            archive=destination.parent/'archive';archive.mkdir(exist_ok=True)
            destination.rename(archive/f'{pairing}_interrupted_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}')
    pending_state.write_text(json.dumps(dict(signature=expected,configuration=cfg)))
    with pairing_context(out_root/'vs0_pairs',pairing):
        execute(out_root,analysis,pairing,destination)
    destination.mkdir(parents=True,exist_ok=True)
    files=[dict(path=str(p.relative_to(destination)),size=p.stat().st_size) for p in sorted(destination.rglob('*')) if p.is_file() and p != destination/'manifest.json']
    manifest=dict(status='complete',analysis=analysis,pairing=pairing,signature=expected,
                  configuration=cfg,provenance=payload,outputs=files)
    pending=destination/'manifest.json.tmp';pending.write_text(json.dumps(manifest,indent=2));pending.replace(destination/'manifest.json')
    pending_state.unlink(missing_ok=True)
    update_summary_index(out_root,analysis)
    return destination


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--pairing',nargs='+',default=None)
    parser.add_argument('--recompute',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    jobs=plan(args.config,args.pairing)
    if args.dry_run:print(json.dumps(jobs,indent=2));return
    for job in jobs:
        print(f"Running {job['analysis']} / {job['pairing']}",flush=True)
        print(run_analysis(args.config,job['pairing'],recompute=args.recompute),flush=True)




def dispatch_managed():
    """Allow existing entry-point filenames to accept the new analysis configs."""
    if '--config' not in sys.argv:return False
    i=sys.argv.index('--config')
    if i+1>=len(sys.argv):return False
    path=Path(sys.argv[i+1])
    if not path.is_file():return False
    cfg=json.loads(path.read_text())
    if cfg.get('schema_version')!=1 or cfg.get('analysis') not in ANALYSES:return False
    args=sys.argv[1:]
    # Legacy coverage commands may repeat an analysis already fixed by the config.
    if '--analysis' in args:
        j=args.index('--analysis');args=args[:j]+args[j+2:]
    main(args)
    return True


def update_summary_index(out_root,analysis):
    """Index completed pairings without duplicating large result tables/arrays."""
    root=Path(out_root)/analysis;folder=root/'summary';folder.mkdir(exist_ok=True)
    records=[]
    for pairing in PAIRING_IDS:
        path=root/pairing/'manifest.json'
        if path.exists():
            manifest=json.loads(path.read_text())
            if manifest.get('status')=='complete':
                records.append(dict(pairing=pairing,path=str(path.resolve()),signature=manifest['signature']))
    data=dict(analysis=analysis,pairings=records,random_pairings=[r['pairing'] for r in records if r['pairing'].startswith('random_')],
              interpretation='Random pairings reuse participants; they are not independent cohort samples.')
    (folder/'manifest.json').write_text(json.dumps(data,indent=2))
    return folder




@contextmanager
def file_lock(path):
    """Serialize same-job writers and shared preparation across cluster processes."""
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def run_analysis(config_path,pairing,*,recompute=False):
    path=Path(config_path).resolve()
    if pairing not in PAIRING_IDS:raise ValueError('Unknown pairing')
    with file_lock(path.parent/f'.{pairing}.lock'):
        return _run_analysis(path,pairing,recompute=recompute)


if __name__=='__main__':main()
