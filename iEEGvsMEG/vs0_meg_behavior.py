"""Shared MEG eligibility from pooled old/new trials in available Blocks 3–6."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import os
import warnings
import re
import posixpath
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

BEHAVIOR_DIR = Path(__file__).resolve().parent / 'MEG' / 'Final_xlsx'
MIN_ACCURACY = 0.5


def behavior_directory():
    return Path(os.environ.get('MEG_BEHAVIOR_DIR', BEHAVIOR_DIR)).expanduser().resolve()


def meg_subject_key(subject):
    """Map questionnaire IDs and SUBJ_0001 recording IDs to the same identity."""
    value = str(subject).strip()
    match = re.fullmatch(r'(?:subj[_ -]?)?(\d+)(?:bis\d*)?(?:\.0+)?', value, flags=re.I)
    if match is None:
        raise ValueError(f'Cannot map MEG participant {subject!r} to behavioral files.')
    return str(int(match[1]))


def _behavior_file_identity(path):
    return re.fullmatch(r'Subj_(\d+)(?:bis\d*)?_Block_([3456])\.(?:xlsx|csv)', path.name, flags=re.I)


def _behavior_files(directory):
    files = []
    seen = set()
    for path in sorted(Path(directory).iterdir()):
        if path.name.startswith(('._', '~$')):
            continue
        match = _behavior_file_identity(path)
        if match:
            subject, block = str(int(match[1])), int(match[2])
            if (subject, block) in seen:
                raise ValueError(f'Duplicate behavioral files for participant {subject}, Block {block}; select one session/format.')
            seen.add((subject, block))
            files.append((path, subject, block))
    if not files:
        raise FileNotFoundError(f'No Blocks 3–6 behavioral XLSX/CSV files found in {directory}.')
    return files


def _read_behavior_table(path):
    """Read trial tables, including Strict OOXML XLSX exports unsupported by openpyxl.

    Only stored cell values are read. Original workbooks are never modified.
    """
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path, sep=None, engine='python', dtype=str)
    with zipfile.ZipFile(path) as book:
        strings = []
        if 'xl/sharedStrings.xml' in book.namelist():
            for node in ET.fromstring(book.read('xl/sharedStrings.xml')):
                strings.append(''.join(t.text or '' for t in node.findall('.//{*}t')))
        workbook = ET.fromstring(book.read('xl/workbook.xml'))
        sheet = workbook.find('.//{*}sheet')
        if sheet is None:
            raise ValueError(f'{path}: no worksheet found.')
        relation = next((v for k, v in sheet.attrib.items() if k.endswith('}id')), None)
        relationships = ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
        target = next((node.get('Target') for node in relationships if node.get('Id') == relation), None)
        if target is None:
            raise ValueError(f'{path}: cannot resolve the first worksheet.')
        worksheet = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl', target))
        rows = []
        for row in ET.fromstring(book.read(worksheet)).findall('.//{*}sheetData/{*}row'):
            cells = {}
            for cell in row:
                letters = re.match(r'[A-Z]+', cell.get('r', ''))
                if letters is None:
                    raise ValueError(f'{path}: worksheet cell has no column reference.')
                column = 0
                for letter in letters[0]:
                    column = column * 26 + ord(letter) - ord('A') + 1
                value = cell.find('{*}v')
                value = value.text if value is not None else None
                if cell.get('t') == 's' and value is not None:
                    value = strings[int(value)]
                elif cell.get('t') == 'inlineStr':
                    value = ''.join(t.text or '' for t in cell.findall('.//{*}t'))
                cells[column-1] = value
            if any(value is not None for value in cells.values()):
                rows.append(cells)
    if not rows:
        raise ValueError(f'{path}: empty worksheet.')
    width = max(rows[0]) + 1
    headers = [str(rows[0].get(i, '')).strip() for i in range(width)]
    return pd.DataFrame([[row.get(i) for i in range(width)] for row in rows[1:]], columns=headers)


def behavior_provenance():
    files = _behavior_files(behavior_directory())
    signatures = [(p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p, _, _ in files]
    payload = dict(min_accuracy=MIN_ACCURACY, blocks=[3, 4, 5, 6], reader_version=3, missing_behavior='exclude',
                   scoring='pooled old=1,new=2; missing/invalid responses incorrect', files=signatures)
    payload['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


@lru_cache(maxsize=8)
def _performance(signatures):
    rows = []
    for name, _, _ in signatures:
        path = Path(name)
        match = _behavior_file_identity(path)
        subject = str(int(match[1]))
        table = _read_behavior_table(path)
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
    summary['n_blocks'] = blocks.groupby('subject').block.nunique()
    summary['accuracy'] = summary.n_correct / summary.n_trials.where(summary.n_trials > 0)
    summary['eligible'] = summary.accuracy >= MIN_ACCURACY
    return summary.reset_index()


def meg_behavior_performance():
    """Return an audit table; accuracy weights trials equally across all blocks."""
    files = _behavior_files(behavior_directory())
    signatures = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p, _, _ in files)
    return _performance(signatures).copy()


def eligible_meg_subjects(subjects):
    """Keep assessed participants with accuracy >=50%; exclude missing performance."""
    subjects = list(map(str, subjects))
    if not subjects:
        return []
    performance = meg_behavior_performance().set_index('subject')
    keys = [meg_subject_key(s) for s in subjects]
    missing = sorted(set(keys) - set(performance.index))
    if missing:
        warnings.warn('Excluded MEG participants with missing behavioral performance: '
                      + ', '.join(missing), stacklevel=2)
    return [subject for subject, key in zip(subjects, keys) if key in performance.index and bool(performance.loc[key, 'eligible'])]


def filter_meg_demographics(frame, subject_column='subject'):
    """Apply the same eligibility rule before demographic matching."""
    keep = set(eligible_meg_subjects(frame[subject_column]))
    return frame.loc[frame[subject_column].astype(str).isin(keep)].copy()


def meg_behavior_coverage(subjects=None):
    """Audit missing subjects/blocks against an explicit roster or inferred ID range.

    With no roster, gaps between 1 and the largest observed numeric ID are
    reported as possible missing subjects, not confirmed missing recordings.
    """
    files = _behavior_files(behavior_directory())
    observed = {}
    for _, subject, block in files:
        observed.setdefault(subject, set()).add(block)
    if subjects is None:
        expected = [str(i) for i in range(1, max(map(int, observed)) + 1)]
        basis = 'inferred_id_range'
    else:
        expected = list(dict.fromkeys(meg_subject_key(s) for s in subjects))
        basis = 'provided_roster'
    return pd.DataFrame([dict(subject=subject, has_behavior=subject in observed,
                              available_blocks=','.join(map(str, sorted(observed.get(subject, set())))),
                              missing_blocks=','.join(map(str, sorted({3,4,5,6} - observed.get(subject, set())))),
                              reference=basis) for subject in expected])
