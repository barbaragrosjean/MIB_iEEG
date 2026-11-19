"""Prepare reusable condition averages; loading never reads raw recording arrays.

python vs0_prepare_data.py --config CONFIG.json
The configuration may contain a top-level ``data`` mapping, as in the notebooks.
Trial-held-out validation continues to use its independent trial cache.
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import io
import json
import tempfile
import time

import numpy as np
import pandas as pd
import vs0_load_and_config as raw

# Increment when the saved representation or preprocessing implementation changes.
SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / 'out' / 'prepared_data'
_RAW_SIGNATURE = inspect.signature(raw.load_project_data)
ARRAY_NAMES = ('times', 'electrode_positions', 'electrode_subjects')


def _signature(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def data_options(config):
    """Use the same preparation settings in notebooks and analysis runners."""
    options = dict(config['data'])
    options.setdefault('selection_seed', config.get('seed', 2026))
    return options


def _request(options, metadata_dependencies=(), preparation_revision='1'):
    bound = _RAW_SIGNATURE.bind(**options)
    bound.apply_defaults()
    options = dict(bound.arguments)
    ieeg_dir = Path(options['ieeg_dir'])
    ieeg = options['ieeg_subjects']
    if ieeg is None:
        ieeg = sorted(p.name.removesuffix('_epochs.p') for p in ieeg_dir.glob('*_epochs.p'))
    meg, ieeg, files, pairing = raw.select_subjects(
        options['meg_dir'], ieeg, use_matched_pairs=options['use_matched_pairs'],
        matched_pairs_csv=options['matched_pairs_csv'], seed=options['selection_seed'],
        meg_subjects=options['meg_subjects'])
    if not ieeg or not meg:
        raise FileNotFoundError('No participants selected for preparation.')
    # Resolve the legacy pickle time-axis default identically for prepare and load.
    if options['meg_tmin'] is None and options['meg_times_file'] is None:
        info = json.loads((ieeg_dir / f'{ieeg[0]}_info.json').read_text())
        options['meg_tmin'] = float(info['time_epoch'][0])
    paths = [ieeg_dir / f'{s}{suffix}' for s in ieeg for suffix in ('_epochs.p', '_info.json')]
    for path in files.values():
        path = Path(path)
        paths.append(path)
        if path.suffix != '.mat':
            paths.append(path.with_name(path.name.removesuffix('_source.p') + '_pos.csv'))
    for name in ('metadata_csv', 'meg_times_file'):
        if options[name] is not None:
            paths.append(Path(options[name]))
    paths.extend(Path(p) for p in metadata_dependencies)
    # GetInfo can depend on external metadata. Record its code/project and allow
    # callers to declare those additional files through metadata_dependencies.
    if options['metadata_csv'] is None and options['electrode_metadata'] is None:
        from src.setting import GetInfo, PROJECT_PATH
        if options['project_path'] is None:
            options['project_path'] = str(PROJECT_PATH)
        source = inspect.getsourcefile(GetInfo)
        if source is not None:
            paths.append(Path(source))
    signatures = [_signature(p) for p in sorted(set(map(lambda p: str(Path(p).resolve()), paths)))]
    # Participant preparation is independent of CSV row order and assignment.
    # A different cohort/file or preprocessing setting still gets a separate cache.
    excluded = {'meg_subjects', 'ieeg_subjects', 'matched_pairs_csv',
                'use_matched_pairs', 'selection_seed'}
    settings = {k: v for k, v in options.items() if k not in excluded}
    if settings['electrode_metadata'] is not None:
        frame = settings['electrode_metadata']
        settings['electrode_metadata'] = dict(
            sha256=hashlib.sha256(frame.to_json(orient='table', index=False).encode()).hexdigest())
    payload = dict(schema_version=SCHEMA_VERSION, revision=str(preparation_revision),
                   meg_behavior=raw.behavior_provenance(),
                   settings=settings, source_files=signatures,
                   meg_files=sorted((s, str(Path(files[s]).resolve())) for s in meg),
                   ieeg_subjects=sorted(ieeg))
    # JSON round-trip normalizes Path/tuple/scalar representations for comparison.
    payload = json.loads(json.dumps(payload, default=str, sort_keys=True))
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    options.update(meg_subjects=meg, ieeg_subjects=ieeg)
    return key, payload, options, files, pairing


def _cache_path(prepared_cache_dir, key):
    return Path(prepared_cache_dir or DEFAULT_CACHE_DIR).expanduser().resolve() / key


def prepare_data(*, prepared_cache_dir=None, metadata_dependencies=(),
                 preparation_revision='1', **options):
    """Prepare and persist participant arrays, or reuse a compatible cache.

    No PCA, block normalization, or spatial matching is performed here. Large
    MATLAB recordings are read only on a cache miss. Changes to known input
    file size/mtime or settings select a new cache. Declare external GetInfo
    metadata files in metadata_dependencies, or change preparation_revision.
    """
    key, payload, effective, _, _ = _request(options, metadata_dependencies, preparation_revision)
    dest = _cache_path(prepared_cache_dir, key)
    if (dest / 'manifest.json').is_file():
        _manifest(dest, payload)
        print(f'Prepared data already available: {dest}', flush=True)
        return dest
    if dest.exists():
        raise FileExistsError(f'Incomplete cache at {dest}; use a new preparation_revision.')
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print('Preparing condition averages from raw recordings (first run only).', flush=True)
    inputs = raw.load_project_data(**effective)
    # Publish only after every array/table has been saved; interrupted preparation
    # leaves an unpublished staging folder and cannot be loaded as complete.
    stage = Path(tempfile.mkdtemp(prefix='.preparing-', dir=dest.parent))
    arrays = {}
    def save(name, value):
        value = np.asarray(value)
        np.save(stage / f'{name}.npy', value, allow_pickle=False)
        arrays[name] = dict(shape=list(value.shape), dtype=str(value.dtype))
    for name in ARRAY_NAMES:
        save(name, inputs[name])
    for i, (array, pos) in enumerate(zip(inputs['meg'], inputs['meg_positions'])):
        save(f'meg_{i:04d}', array)
        save(f'meg_positions_{i:04d}', pos)
    for i, subject in enumerate(inputs['ieeg_subjects']):
        rows = np.flatnonzero(inputs['electrode_subjects'] == subject)
        save(f'ieeg_{i:04d}', inputs['ieeg'][:, rows, :])
    from vs0_meg_behavior import meg_behavior_performance
    meg_behavior_performance().to_csv(stage / 'meg_behavior_eligibility.csv', index=False)
    for name in ('electrode_metadata', 'trial_counts'):
        (stage / f'{name}.json').write_text(inputs[name].to_json(orient='table', index=False))
    # Confirm that raw files were not modified while preparation was running.
    if any(_signature(v['path']) != v for v in payload['source_files']):
        raise RuntimeError('A source file changed during preparation; staged cache was not published.')
    manifest = dict(request=payload, arrays=arrays, meg_subjects=inputs['meg_subjects'],
                    ieeg_subjects=inputs['ieeg_subjects'], load_config=inputs['load_config'],
                    preparation_seconds=time.perf_counter()-started)
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    try:
        stage.rename(dest)
    except OSError:
        if not (dest / 'manifest.json').exists():
            raise
        # Another preparation process may have published the same request.
        _manifest(dest, payload)
    print(f'Prepared data saved: {dest} ({manifest["preparation_seconds"]:.1f} s)', flush=True)
    return dest


def _manifest(folder, payload):
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['request'] != payload:
        raise ValueError('Prepared-data manifest does not match the requested inputs/settings.')
    for table in ('electrode_metadata.json', 'trial_counts.json'):
        if not (folder / table).is_file():
            raise FileNotFoundError(f'Incomplete prepared cache: {table}')
    for name in manifest['arrays']:
        if not (folder / f'{name}.npy').is_file():
            raise FileNotFoundError(f'Incomplete prepared cache: {name}.npy')
    return manifest


def load_prepared_data(*, prepared_cache_dir=None, metadata_dependencies=(),
                       preparation_revision='1', **options):
    """Read prepared arrays only; never prepare data or read raw signal arrays.

    Raw recording signatures and small selection/time metadata are checked.
    MEG arrays are read-only memory maps; iEEG participant blocks are joined in
    the current requested order. CSV assignment is applied at load time.
    """
    key, payload, effective, files, pairing = _request(options, metadata_dependencies, preparation_revision)
    folder = _cache_path(prepared_cache_dir, key)
    if not (folder / 'manifest.json').is_file():
        raise FileNotFoundError(
            f'No compatible prepared data at {folder}. Run prepare_data() with the same data options first.')
    manifest = _manifest(folder, payload)
    def array(name):
        value = np.load(folder / f'{name}.npy', mmap_mode='r', allow_pickle=False)
        spec = manifest['arrays'][name]
        if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype']:
            raise ValueError(f'Prepared array metadata mismatch: {name}')
        return value
    meg_ids, ieeg_ids = effective['meg_subjects'], effective['ieeg_subjects']
    meg_lookup = {s: i for i, s in enumerate(manifest['meg_subjects'])}
    ieeg_lookup = {s: i for i, s in enumerate(manifest['ieeg_subjects'])}
    owners = array('electrode_subjects')
    rows = np.concatenate([np.flatnonzero(owners == s) for s in ieeg_ids])
    metadata = pd.read_json(io.StringIO((folder / 'electrode_metadata.json').read_text()), orient='table')
    counts = pd.read_json(io.StringIO((folder / 'trial_counts.json').read_text()), orient='table')
    counts = pd.concat([counts.loc[counts.subject == s] for s in ieeg_ids], ignore_index=True)
    config = dict(manifest['load_config'], pairing=pairing, meg_files=files,
                  selection_seed=effective['selection_seed'],
                  use_matched_pairs=raw.pairing_settings(effective['use_matched_pairs'], effective['matched_pairs_csv'])[0],
                  matched_pairs_csv=str(raw.pairing_settings(effective['use_matched_pairs'], effective['matched_pairs_csv'])[1]))
    return dict(meg=[array(f'meg_{meg_lookup[s]:04d}') for s in meg_ids],
                meg_positions=[array(f'meg_positions_{meg_lookup[s]:04d}') for s in meg_ids],
                meg_subjects=meg_ids, ieeg_subjects=ieeg_ids, pairing=pairing, meg_files=files,
                ieeg=np.concatenate([array(f'ieeg_{ieeg_lookup[s]:04d}') for s in ieeg_ids], axis=1),
                electrode_subjects=np.asarray(owners[rows]), electrode_positions=array('electrode_positions')[rows],
                electrode_metadata=metadata.iloc[rows].reset_index(drop=True), times=array('times'),
                trial_counts=counts, load_config=config, prepared_data_dir=str(folder))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    print(prepare_data(**(data_options(config) if 'data' in config else config)))


if __name__ == '__main__':
    main()
