"""Cluster runner for channel-space ERP comparisons: --config CONFIG [--recompute]."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shlex
import sys
import numpy as np
import pandas as pd
from vs0_load_and_config import load_project_data
from vs1_1_channel_space import analyze_channel_space

VERSION = 1
TABLES = ('metadata', 'correlations', 'electrode_summary', 'roi_subjects', 'roi_summary', 'scaling', 'shell_table')


def result_directory(config):
    payload = dict(config, version=VERSION)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return Path(config['output_dir'])/digest


def load_result(config):
    folder = result_directory(config)
    marker = json.loads((folder/'RUN_COMPLETE.json').read_text())
    for entry in marker['input_files']:
        path = Path(entry['path'])
        if path.stat().st_mtime_ns != entry['mtime_ns'] or path.stat().st_size != entry['size']:
            raise FileNotFoundError('Input files changed; recompute channel-space results.')
    result = {name: pd.read_csv(folder/f'{name}.csv', dtype={'subject':str, 'ieeg_subject':str, 'meg_subject':str}) for name in TABLES}
    with np.load(folder/'erps.npz') as saved:
        result['arrays'] = dict(saved)
    return result


def prepare_analysis(config, config_file, recompute=False):
    path = Path(config_file); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    command = ['python','-u',str(Path(__file__).resolve()),'--config',str(path)]
    if recompute: command.append('--recompute')
    print('# Run on the cluster:', shlex.join(command), sep='\n')
    if not recompute:
        try:
            return load_result(config)
        except FileNotFoundError:
            print('# Results unavailable. Run the command, then reload this cell.')
    else:
        print('# After the job completes, set RECOMPUTE=False and reload.')
    return None


def run(config, recompute=False):
    folder = result_directory(config)
    if not recompute and (folder/'RUN_COMPLETE.json').exists():
        try:
            load_result(config)
            return folder
        except FileNotFoundError:
            pass
    options = dict(config['data'])
    if options.get('meg_tmin') is None and options.get('meg_times_file') is None:
        info = sorted(Path(options['ieeg_dir']).glob('*_info.json'))[0]
        options['meg_tmin'] = float(json.loads(info.read_text())['time_epoch'][0])
    data = load_project_data(**options)
    roi_column = config['roi_column']
    if roi_column not in data['electrode_metadata']:
        from vs0_region_labels import get_region_label
        data['electrode_metadata'][roi_column] = get_region_label(data['electrode_metadata'][['x','y','z']]/1000.)
    result = analyze_channel_space(data, shell_edges_mm=config['shell_edges_mm'],
        block_scaling=config['block_scaling'], time_window=config['time_window'],
        roi_column=roi_column, pool_hemispheres=config['pool_hemispheres'])
    if folder.exists():
        archive = folder.parent/'old';archive.mkdir(exist_ok=True)
        folder.rename(archive/f'{folder.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}')
    folder.mkdir(parents=True)
    for name in TABLES: result[name].to_csv(folder/f'{name}.csv', index=False)
    np.savez_compressed(folder/'erps.npz', **result['arrays'])
    data['trial_counts'].to_csv(folder/'ieeg_trial_counts.csv', index=False)
    paths = []
    for subject in data['ieeg_subjects']:
        paths.extend(Path(options['ieeg_dir'])/f'{subject}{suffix}' for suffix in ['_epochs.p','_info.json'])
    for subject in data['meg_subjects']:
        paths.extend(Path(options['meg_dir'])/f'{subject}{suffix}' for suffix in ['_source.p','_pos.csv'])
    paths.extend(Path(options[name]) for name in ['metadata_csv','meg_times_file'] if options.get(name))
    files = [dict(path=str(path.resolve()),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns) for path in paths]
    (folder/'RUN_COMPLETE.json').write_text(json.dumps(dict(version=VERSION,config=config,input_files=files),indent=2))
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--recompute',action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.extend([str(root.parent),str(root.parent/'LB')])
    print(run(json.loads(args.config.read_text()),args.recompute))


if __name__ == '__main__': main()
