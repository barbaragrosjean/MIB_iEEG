"""MEG behavior reader and condition-specific eligibility for Blocks 3–6."""
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


def meg_behavior_performance(directory=None):
    """Require >=50% in EACH old/new condition and >=24 trials in total.

    Actual within-condition denominators are used. Missing/invalid responses
    count as incorrect. Blocks 3–6 are pooled within each condition.
    """
    rows = []
    for path, subject, block in _behavior_files(directory or behavior_directory()):
        table = _read_behavior_table(path)
        table.columns = table.columns.str.strip()
        if not {'trial', 'response'} <= set(table):
            raise ValueError(f'{path}: required columns are trial and response.')
        stimulus = table.trial.fillna('').astype(str).str.lower()
        old, new = stimulus.str.contains('old', regex=False), stimulus.str.contains('new', regex=False)
        if (old & new).any():
            raise ValueError(f'{path}: ambiguous old/new stimulus labels.')
        if 'subj' in table and any(meg_subject_key(x) != subject for x in table.loc[old | new, 'subj'].dropna()):
            raise ValueError(f'{path}: subject ID differs from filename.')
        response = pd.to_numeric(table.response, errors='coerce')
        rows.append(dict(subject=subject, block=block, n_old=int(old.sum()), n_new=int(new.sum()),
                         correct_old=int((old & response.eq(1)).sum()),
                         correct_new=int((new & response.eq(2)).sum())))
    frame = pd.DataFrame(rows).groupby('subject', sort=True)[
        ['n_old', 'n_new', 'correct_old', 'correct_new']].sum()
    frame['accuracy_old'] = frame.correct_old / frame.n_old.where(frame.n_old > 0)
    frame['accuracy_new'] = frame.correct_new / frame.n_new.where(frame.n_new > 0)
    frame['n_trials'] = frame.n_old + frame.n_new
    frame['accuracy'] = (frame.correct_old + frame.correct_new) / frame.n_trials.where(frame.n_trials > 0)
    frame['eligible'] = ((frame.accuracy_old >= .5) & (frame.accuracy_new >= .5) & (frame.n_trials >= 24))
    return frame.reset_index()


def filter_meg_demographics(frame, subject_column='subject'):
    eligible = set(meg_behavior_performance().query('eligible').subject)
    return frame.loc[frame[subject_column].map(meg_subject_key).isin(eligible)].copy()
