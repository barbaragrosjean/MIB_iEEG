"""Create demographic, ordered, and 20 reproducible random participant associations."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
from vs0_utils_load_and_config import (select_subject, GetInfo, ieeg_getinfo_coordinates_m,
                                coordinates_mm, load_meg_mat, match_sources)
from vs0_utils_pairing import PAIRING_IDS, pairing_paths, digest_file


def demographic_match(ieeg, meg, *, seed=42, age_weight=.9, sex_weight=1.):
    """Notebook's greedy nearest available neighbor; musical expertise omitted."""
    scale = float(pd.concat([ieeg.age, meg.age]).std(ddof=1))
    if not np.isfinite(scale) or scale <= 0: scale = 1.
    age = abs(ieeg.age.to_numpy()[:, None] - meg.age.to_numpy()[None, :])
    cost = np.sqrt(age_weight*(age/scale)**2 + sex_weight*(
        ieeg.sex.to_numpy()[:, None] != meg.sex.to_numpy()[None, :]))
    available, selected = np.ones(len(meg), bool), np.empty(len(ieeg), int)
    order = np.random.default_rng(seed).permutation(len(ieeg))
    for i in order:
        selected[i] = int(np.argmin(np.where(available, cost[i], np.inf)))
        available[selected[i]] = False
    return selected, cost[np.arange(len(ieeg)), selected]


def create_pairs(config_path, *, overwrite=False):
    from vs0_run_analysis import read_config
    cfg = read_config(config_path)
    out = Path(config_path).resolve().parent
    p = cfg['parameters']
    targets = [path for name in PAIRING_IDS for path in pairing_paths(out, name)]
    if any(path.exists() for path in targets) and not overwrite:
        raise FileExistsError('Pairing CSVs already exist; use --overwrite explicitly to regenerate them.')
    if int(p.get('n_random', 20)) != 20: raise ValueError('This workflow requires exactly 20 random pairings.')
    ieeg, meg, audit = select_subject(**{k:p[k] for k in (
        'ieeg_dem_csv','meg_dem_xlsx','ieeg_dir','meg_dir','behavior_dir','project_path','meg_sheet')})
    coord, _, names, owners, regions = GetInfo(ieeg.subject.tolist(), project_path=p['project_path'], data_path=p['ieeg_dir'])
    xyz = coordinates_mm(ieeg_getinfo_coordinates_m(coord), 'm')
    metadata = pd.DataFrame(dict(ieeg_subject=owners, ieeg_electrode=names, ieeg_roi=regions))
    metadata['channel_index'] = metadata.groupby('ieeg_subject', sort=False).cumcount()
    metadata[['ieeg_x','ieeg_y','ieeg_z']] = xyz
    for subject, rows in metadata.groupby('ieeg_subject', sort=False):
        import pickle
        with (Path(p['ieeg_dir'])/f'{subject}_epochs.p').open('rb') as stream:
            epochs = np.asarray(pickle.load(stream))
        if epochs.ndim != 3 or epochs.shape[1] != len(rows):
            raise ValueError(f'{subject}: electrode metadata does not match saved channels.')
    # Read source positions only; all pairings share one per-participant copy in memory.
    positions, labels = {}, {}
    from vs0_utils_region_labels import get_region_label
    for subject in meg.subject:
        mat = load_meg_mat(Path(p['meg_dir'])/f'{subject}_norm0_abs_0.mat')
        pos = coordinates_mm(mat['pos_brainsources_MNI8'], p['meg_coordinate_unit'])
        positions[subject] = pos
        labels[subject] = pd.Series(get_region_label(pd.DataFrame(pos/1000., columns=['x','y','z']))).fillna('Unassigned').tolist()
        del mat
    selected, costs = demographic_match(ieeg, meg, seed=p['seed'], **p['matching_weights'])
    rngs = np.random.SeedSequence(p['seed']).spawn(21)
    selections = {'match': selected, 'order': np.arange(len(ieeg))}
    for i in range(1, 21):
        selections[f'random_{i:03d}'] = np.random.default_rng(rngs[i-1]).choice(len(meg), len(ieeg), replace=False)
    # Build everything before replacing files, so invalid input cannot publish partial pairs.
    outputs = []
    for pairing_id in PAIRING_IDS:
        dem = pd.concat([ieeg.add_prefix('ieeg_'), meg.iloc[selections[pairing_id]].reset_index(drop=True).add_prefix('meg_')], axis=1)
        dem['pairing_id'] = pairing_id
        dem['age_gap_years'] = abs(dem.ieeg_age - dem.meg_age)
        dem['sex_match'] = dem.ieeg_sex.eq(dem.meg_sex)
        if pairing_id == 'match': dem['matching_distance'] = costs
        frames = []
        for row in dem.itertuples():
            frame = metadata.loc[metadata.ieeg_subject.eq(row.ieeg_subject)].copy()
            tie_seed = int.from_bytes(hashlib.sha256(f"{p['seed']}:{pairing_id}:{row.ieeg_subject}".encode()).digest()[:8], 'little')
            indices, distances = match_sources(frame[['ieeg_x','ieeg_y','ieeg_z']], positions[row.meg_subject], seed=tie_seed)
            frame['meg_subject'] = row.meg_subject
            frame['source_index'] = indices
            frame[['meg_x','meg_y','meg_z']] = positions[row.meg_subject][indices]
            frame['meg_roi'] = np.asarray(labels[row.meg_subject])[indices]
            frame['distance_mm'] = distances
            frame['pairing_id'] = pairing_id
            frames.append(frame)
        outputs.append((pairing_id, dem, pd.concat(frames, ignore_index=True)))
    for pairing_id, dem, elect in outputs:
        for path, table in zip(pairing_paths(out, pairing_id), (dem,elect)):
            temp = path.with_suffix('.csv.tmp'); table.to_csv(temp,index=False);temp.replace(path)
    audit.to_csv(out/'selection_audit.csv', index=False)
    inputs = [Path(p['ieeg_dem_csv']),Path(p['meg_dem_xlsx'])]
    inputs += sorted(Path(p['behavior_dir']).glob('*.xlsx')) + sorted(Path(p['behavior_dir']).glob('*.csv'))
    records = {str(path): digest_file(path) for path in inputs}
    (out/'manifest.json').write_text(json.dumps(dict(status='complete',config=cfg,inputs=records,
        code={path.name:digest_file(path) for path in (Path(__file__),Path(__file__).with_name('vs0_utils_load_and_config.py'),Path(__file__).with_name('vs0_utils_meg_behavior.py'))},
        pairings=list(PAIRING_IDS),outputs={path.name:digest_file(path) for path in targets}),indent=2))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parent/'out/vs0_pairs/config.json')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args();print(create_pairs(args.config,overwrite=args.overwrite))


if __name__=='__main__': main()
