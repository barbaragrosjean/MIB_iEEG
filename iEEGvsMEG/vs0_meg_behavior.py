"""Shared MEG eligibility from pooled old/new recognition trials in Blocks 3–6."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import os
import re

import pandas as pd

BEHAVIOR_DIR = Path(__file__).resolve().parent / 'MEG' / 'Blocks3456_csv'
MIN_ACCURACY = 0.5


def behavior_directory():
    return Path(os.environ.get('MEG_BEHAVIOR_DIR', BEHAVIOR_DIR)).expanduser().resolve()


def meg_subject_key(subject):
    """Map questionnaire IDs and SUBJ_0001 recording IDs to the same identity."""
    value = str(subject).strip()
    match = re.fullmatch(r'(?:subj[_ -]?)?(\d+)(?:\.0+)?', value, flags=re.I)
    if match is None:
        raise ValueError(f'Cannot map MEG participant {subject!r} to behavioral files.')
    return str(int(match[1]))


def _behavior_files(directory):
    files = []
    for path in sorted(Path(directory).glob('*.csv')):
        match = re.fullmatch(r'Subj_(\d+)_Block_([3456])\.csv', path.name, flags=re.I)
        if match:
            files.append((path, str(int(match[1])), int(match[2])))
    if not files:
        raise FileNotFoundError(f'No Blocks 3–6 behavioral CSVs found in {directory}.')
    return files


def behavior_provenance():
    files = _behavior_files(behavior_directory())
    signatures = [(str(p), hashlib.sha256(p.read_bytes()).hexdigest()) for p, _, _ in files]
    payload = dict(min_accuracy=MIN_ACCURACY, blocks=[3, 4, 5, 6],
                   scoring='pooled old=1,new=2; missing/invalid responses incorrect', files=signatures)
    payload['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


@lru_cache(maxsize=8)
def _performance(signatures):
    rows = []
    for name, _, _ in signatures:
        path = Path(name)
        match = re.fullmatch(r'Subj_(\d+)_Block_([3456])\.csv', path.name, flags=re.I)
        subject = str(int(match[1]))
        table = pd.read_csv(path, sep=None, engine='python', dtype=str)
        table.columns = table.columns.str.strip()
        if not {'trial', 'response'} <= set(table):
            raise ValueError(f'{path}: required columns are trial and response.')
        stimulus = table['trial'].fillna('').str.lower()
        old = stimulus.str.contains('old', regex=False)
        new = stimulus.str.contains('new', regex=False)
        if (old & new).any():
            raise ValueError(f'{path}: ambiguous stimulus contains both old and new.')
        selected = old | new
        if 'subj' in table and any(meg_subject_key(s) != subject for s in table.loc[selected, 'subj'].dropna()):
            raise ValueError(f'{path}: participant IDs differ from the filename.')
        response = pd.to_numeric(table['response'], errors='coerce')
        correct = ((old & response.eq(1)) | (new & response.eq(2))) & selected
        rows.append(dict(subject=subject, block=int(match[2]), n_trials=int(selected.sum()),
                         n_correct=int(correct.sum()), n_old=int(old.sum()), n_new=int(new.sum()),
                         n_ignored=int((~selected).sum())))
    blocks = pd.DataFrame(rows)
    summary = blocks.groupby('subject', sort=True)[['n_trials','n_correct','n_old','n_new','n_ignored']].sum()
    if (summary.n_trials == 0).any():
        raise ValueError('No scorable old/new trials for participants: ' + ', '.join(summary.index[summary.n_trials == 0]))
    summary['n_blocks'] = blocks.groupby('subject').block.nunique()
    if (summary.n_blocks != 4).any():
        raise ValueError('Missing behavioral blocks for participants: ' + ', '.join(summary.index[summary.n_blocks != 4]))
    summary['accuracy'] = summary.n_correct / summary.n_trials
    summary['eligible'] = summary.accuracy >= MIN_ACCURACY
    return summary.reset_index()


def meg_behavior_performance():
    """Return an audit table; accuracy weights trials equally across all blocks."""
    files = _behavior_files(behavior_directory())
    signatures = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p, _, _ in files)
    return _performance(signatures).copy()


def eligible_meg_subjects(subjects):
    """Filter a cohort without changing its IDs/order; missing behavior is an error."""
    subjects = list(map(str, subjects))
    if not subjects:
        return []
    performance = meg_behavior_performance().set_index('subject')
    keys = [meg_subject_key(s) for s in subjects]
    missing = sorted(set(keys) - set(performance.index))
    if missing:
        raise ValueError('Missing MEG behavioral data for participants: ' + ', '.join(missing))
    return [subject for subject, key in zip(subjects, keys) if bool(performance.loc[key, 'eligible'])]


def filter_meg_demographics(frame, subject_column='subject'):
    """Apply the same eligibility rule before demographic matching."""
    keep = set(eligible_meg_subjects(frame[subject_column]))
    return frame.loc[frame[subject_column].astype(str).isin(keep)].copy()
