"""Explicit outer participant assignments, shared by average and trial loaders."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import json
from functools import lru_cache

_CONTEXT = ContextVar('analysis_pairing', default=None)
PAIRING_IDS = ('match', 'order', *(f'random_{i:03d}' for i in range(1, 21)))


@lru_cache(maxsize=8192)
def _digest_cached(path, size, mtime_ns):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_file(path):
    path=Path(path).resolve();stat=path.stat()
    return _digest_cached(str(path),stat.st_size,stat.st_mtime_ns)


def pairing_paths(directory, pairing_id):
    if pairing_id not in PAIRING_IDS:
        raise ValueError(f'Unknown pairing: {pairing_id}')
    prefix = {'match': 'match', 'order': 'order'}.get(pairing_id, pairing_id)
    return tuple(Path(directory)/f'{prefix}_pairing_{suffix}.csv' for suffix in ('dem', 'elect'))


def active_pairing():
    context = _CONTEXT.get()
    return None if context is None else context['dem']


def active_electrodes():
    context = _CONTEXT.get()
    return None if context is None else context['elect']


def active_pairing_id():
    context = _CONTEXT.get()
    return None if context is None else context['id']


@contextmanager
def pairing_context(directory, pairing_id):
    import pandas as pd
    from vs0_utils_load_and_config import read_matched_pairs
    dem_path, elect_path = pairing_paths(directory, pairing_id)
    dem = read_matched_pairs(dem_path)
    elect = pd.read_csv(elect_path, dtype={'ieeg_subject': str, 'meg_subject': str, 'ieeg_electrode': str})
    required = {'ieeg_subject', 'meg_subject', 'channel_index', 'source_index',
                'ieeg_x', 'ieeg_y', 'ieeg_z', 'meg_x', 'meg_y', 'meg_z', 'distance_mm'}
    if not required <= set(elect) or elect.empty:
        raise ValueError(f'{elect_path}: incomplete electrode correspondence.')
    if elect.duplicated(['ieeg_subject', 'channel_index']).any():
        raise ValueError('Duplicate electrode identity in pairing CSV.')
    mapping = dict(zip(dem.ieeg_subject, dem.meg_subject))
    if set(elect.ieeg_subject) != set(mapping) or not elect.meg_subject.eq(elect.ieeg_subject.map(mapping)).all():
        raise ValueError('Electrode and demographic pairings disagree.')
    token = _CONTEXT.set(dict(id=pairing_id, dem=dem, elect=elect))
    try:
        yield dem
    finally:
        _CONTEXT.reset(token)


def fixed_mapping(subjects, meg_subjects):
    dem = active_pairing()
    if dem is None: return None
    mapping = dict(zip(dem.ieeg_subject, dem.meg_subject))
    # Some stability calculations intentionally use a subset of the fixed cohort.
    if not set(subjects) <= set(mapping) or not set(mapping[s] for s in subjects) <= set(meg_subjects):
        raise ValueError('Requested participants do not belong to the selected pairing.')
    return {s: mapping[s] for s in subjects}


def explicit_source_indices(subject, n_channels, positions_mm, electrode_positions_mm):
    import numpy as np
    frame = active_electrodes()
    if frame is None: return None
    rows = frame.loc[frame.ieeg_subject.eq(subject)].sort_values('channel_index')
    if rows.channel_index.tolist() != list(range(n_channels)):
        raise ValueError(f'{subject}: pairing CSV does not cover saved epoch channels in order.')
    indices = rows.source_index.to_numpy(dtype=int)
    if (indices < 0).any() or (indices >= len(positions_mm)).any():
        raise ValueError('Source index outside recording.')
    if not np.allclose(rows[['ieeg_x', 'ieeg_y', 'ieeg_z']], electrode_positions_mm, atol=1e-6, rtol=0):
        raise ValueError('iEEG coordinates differ from the pairing correspondence.')
    if not np.allclose(rows[['meg_x', 'meg_y', 'meg_z']], np.asarray(positions_mm)[indices], atol=1e-6, rtol=0):
        raise ValueError('MEG coordinates differ from the pairing correspondence.')
    return indices


def selected_cache_manifest(manifest):
    """Return a read-only subset view; never rewrite the shared cache manifest."""
    dem = active_pairing()
    if dem is None: return manifest
    selected = {'ieeg': dem.ieeg_subject.tolist(), 'meg': dem.meg_subject.tolist()}
    result = dict(manifest, config=dict(manifest['config']))
    result['records'] = [r for r in manifest['records'] if str(r['subject']) in selected[r['modality']]]
    for modality, subjects in selected.items():
        records = {str(r['subject']): r for r in result['records'] if r['modality'] == modality}
        if set(records) != set(subjects): raise ValueError(f'Trial cache is missing selected {modality} participants.')
        result['config'][modality+'_subjects'] = subjects
    result['records'] = [next(r for r in result['records'] if r['modality']==m and str(r['subject'])==s)
                         for m in ('ieeg', 'meg') for s in selected[m]]
    return result


def plssvd_output(run_dir, analysis, *parts):
    """Resolve derived artifacts alongside, never inside, the fit analysis."""
    run = Path(run_dir)
    for parent in (run, *run.parents):
        if parent.name == 'vs2_plssvd_fit':
            relative = run.relative_to(parent)
            if len(relative.parts) < 1: raise ValueError('Missing pairing ID in PLSSVD path.')
            pairing = relative.parts[0]
            # Fit-run variants retain their names (none, meg_time_point_shared, ...).
            variant = relative.parts[1:] or ()
            return parent.parent/analysis/pairing/Path(*variant, *map(str,parts))
    return None


@contextmanager
def without_pairing():
    token = _CONTEXT.set(None)
    try: yield
    finally: _CONTEXT.reset(token)
