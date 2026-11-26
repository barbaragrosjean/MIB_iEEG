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


def saved_source_maps(subjects, owners, electrode_positions, meg_subjects, meg_positions,
                      *, need_coverage=False):
    """Read authoritative CSV indices; never perform a spatial search."""
    import numpy as np
    import pandas as pd
    frame=active_electrodes()
    if frame is None:
        raise ValueError('Load an explicit pairing CSV with pairing_context before constructing MEG datasets.')
    ordered=[]
    for subject in subjects:
        rows=frame.loc[frame.ieeg_subject.eq(subject)].sort_values('channel_index')
        selected=np.flatnonzero(np.asarray(owners)==subject)
        if rows.channel_index.tolist()!=list(range(len(selected))):
            raise ValueError(f'{subject}: saved correspondence does not cover every electrode.')
        ordered.append(rows)
    rows=pd.concat(ordered,ignore_index=True)
    if not np.array_equal(rows.ieeg_subject.to_numpy(),owners):
        raise ValueError('Saved electrode order differs from recording order.')
    if not np.allclose(rows[['ieeg_x','ieeg_y','ieeg_z']],electrode_positions,atol=1e-6,rtol=0):
        raise ValueError('Saved electrode coordinates differ from recordings.')
    if 'random_source_index' not in rows:
        raise ValueError('Electrode CSV lacks saved control indices. Regenerate vs0_pairs with --overwrite.')
    coverage=None
    if need_coverage:
        if 'coverage_source_indices' not in rows:
            raise ValueError('Electrode CSV lacks coverage indices. Regenerate vs0_pairs with --overwrite.')
        coverage=[json.loads(v) for v in rows.coverage_source_indices]
    nearest=[]
    for subject,pos in zip(meg_subjects,meg_positions):
        assigned=rows.meg_subject.eq(subject).to_numpy()
        indices=np.full(len(rows),-1,dtype=int)
        if coverage is not None:
            for i,entry in enumerate(coverage):
                if subject not in entry:raise ValueError(f'Missing saved coverage source for {subject}.')
                value=entry[subject]
                if not isinstance(value,int):raise ValueError('Source indices must be integers.')
                indices[i]=value
        saved=rows.loc[assigned,'source_index'].to_numpy(float)
        if not np.isfinite(saved).all() or not np.equal(saved,np.floor(saved)).all():
            raise ValueError('Assigned source indices must be finite integers.')
        if coverage is not None and not np.array_equal(indices[assigned],saved.astype(int)):
            raise ValueError('Assigned and coverage source indices disagree.')
        indices[assigned]=saved.astype(int)
        required=np.ones(len(rows),bool) if need_coverage else assigned
        if np.any(indices[required]<0) or np.any(indices[required]>=len(pos)):
            raise ValueError('Saved source index outside recording.')
        if not np.allclose(rows.loc[assigned,['meg_x','meg_y','meg_z']],np.asarray(pos)[indices[assigned]],atol=1e-6,rtol=0):
            raise ValueError('Saved MEG coordinates differ from recordings.')
        nearest.append(indices)
    return rows,nearest
