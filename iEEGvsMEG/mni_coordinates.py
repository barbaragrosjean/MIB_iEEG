"""Load ordered per-subject MNI coordinates without participant alignment."""
from pathlib import Path
import numpy as np
import pandas as pd
from coverage_matching_utils import coordinates_mm


def read_mni_coordinates(path, n_features, unit='mm'):
    """Accept named x/y/z columns or exactly three headerless numeric columns.

    CSV rows must already match the cached signal-channel order. No reordering
    or magnitude-based correction is applied to declared MNI coordinates.
    """
    path=Path(path)
    table=pd.read_csv(path)
    lower={str(c).strip().lower():c for c in table.columns}
    if all(c in lower for c in ('x','y','z')):
        values=table[[lower[c] for c in ('x','y','z')]].to_numpy(float)
    else:
        table=pd.read_csv(path,header=None)
        if table.shape[1]!=3:
            raise ValueError(f'{path}: use x,y,z headers or exactly three numeric columns.')
        try:values=table.to_numpy(float)
        except ValueError as exc:raise ValueError(f'{path}: unrecognized coordinate columns; use x,y,z headers.') from exc
    if values.shape!=(n_features,3):
        raise ValueError(f'{path}: {len(values)} coordinate rows, expected {n_features} cached channels. '
                         'Use the same channel selection/order as the trial cache.')
    return coordinates_mm(values,unit)


def update_trial_coordinates(trials, *, ieeg_dir=None, meg_dir=None, ieeg_unit='mm', meg_unit='mm'):
    """Replace in-memory positions only; never modify trials/cache or align rows."""
    prepared=[]
    for modality,directory,unit,suffix in [('ieeg',ieeg_dir,ieeg_unit,'_coords.csv'),('meg',meg_dir,meg_unit,'_pos.csv')]:
        if directory is None:continue
        for subject in getattr(trials,modality):
            path=Path(directory)/f'{subject.subject}{suffix}'
            coords=read_mni_coordinates(path,subject.data[0].shape[1],unit)
            prepared.append((modality,subject,path,unit,coords))
    rows=[]
    for modality,subject,path,unit,coords in prepared:
        subject.positions=coords
        subject.metadata=subject.metadata.copy()
        subject.metadata[['x','y','z']]=coords
        rows.append(dict(modality=modality,subject=subject.subject,path=str(path.resolve()),input_unit=unit,n_features=len(coords)))
    return pd.DataFrame(rows,columns=['modality','subject','path','input_unit','n_features'])
