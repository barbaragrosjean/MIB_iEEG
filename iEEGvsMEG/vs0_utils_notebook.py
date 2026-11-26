"""Read-only configuration/result adapters; no figure definitions or config writes."""
from pathlib import Path
import json
from contextlib import contextmanager
from vs0_run_analysis import (ROOT,read_config,parameters,config_for,legacy_config,
    specificity_config,validate_result,output_path,COVERAGE)
from vs0_utils_pairing import pairing_context


def settings(analysis, pairing='match', out_root=None):
    root=Path(out_root or ROOT/'out')
    cfg=read_config(config_for(root,analysis))
    return cfg['parameters'].copy()


def show_parameters(analysis,pairing='match',out_root=None):
    import pandas as pd
    from IPython.display import display
    root=Path(out_root or ROOT/'out');cfg=read_config(config_for(root,analysis))
    print('Pairing:',pairing)
    print(cfg['command'])
    display(pd.json_normalize(cfg['parameters'],sep='.').T.rename(columns={0:'value'}))


def result_path(analysis,pairing='match',out_root=None):
    return validate_result(Path(out_root or ROOT/'out'),analysis,pairing)


def coverage_data(pairing='match',out_root=None):
    from vs1_run_coverage import load_datasets
    root=Path(out_root or ROOT/'out')
    result_path('vs1_construct',pairing,root)
    cfg=legacy_config(root,'vs1_construct',pairing)
    with pairing_context(root/'vs0_pairs',pairing):return load_datasets(cfg)


def coverage_result(analysis,pairing='match',out_root=None):
    root=Path(out_root or ROOT/'out');show_parameters(analysis,pairing,root)
    try:return result_path(analysis,pairing,root)
    except (FileNotFoundError,ValueError) as exc:
        print(exc);return None


def channel_space(pairing='match',out_root=None):
    from vs1_run_channel_space import load_result
    root=Path(out_root or ROOT/'out');show_parameters('vs1_channel_space',pairing,root)
    cfg=legacy_config(root,'vs1_channel_space',pairing)
    try:
        result_path('vs1_channel_space',pairing,root)
        return cfg,load_result(cfg)
    except (FileNotFoundError,ValueError) as exc:
        print(exc);return cfg,None


def covariance(pairing='match',out_root=None):
    from vs2_run_cov_models import load_models
    root=Path(out_root or ROOT/'out');show_parameters('vs2_cov_models',pairing,root)
    cfg=legacy_config(root,'vs2_cov_models',pairing)
    try:
        folder=result_path('vs2_cov_models',pairing,root)
        return cfg,load_models(folder)
    except (FileNotFoundError,ValueError) as exc:
        print(exc);return cfg,None


def specificity(analysis,pairing='match',out_root=None):
    from vs3_1_run_modality_specificity import load_result
    from vs0_run_analysis import SPECIFICITY
    root=Path(out_root or ROOT/'out');show_parameters(analysis,pairing,root)
    cfg=specificity_config(root,pairing)
    try:
        result_path(analysis,pairing,root)
        return load_result(cfg,SPECIFICITY[analysis])
    except (FileNotFoundError,ValueError) as exc:
        print(exc);return None
