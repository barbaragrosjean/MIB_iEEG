import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from plssvd_select_components import NULL_TYPES, load_selection, prepare_selection


class SelectionNullTypesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = dict(
            metadata={'signature': 'test-signature'},
            components=pd.DataFrame([
                dict(partition='test', run=run, iteration=i, component=c,
                     covariance=value, r=value)
                for i in range(3)
                for run, values in [('Unpermuted', [2., .1]), ('Permuted', [.5, .2])]
                for c, value in enumerate(values, 1)
            ]),
            metrics=pd.DataFrame({'partition': ['test'], 'value': [1.]}),
        )
        self.loader = Mock(return_value=self.data)
        mock_module = SimpleNamespace(load_comparison=self.loader)
        self.patch = patch.dict('sys.modules', plssvd_figures=mock_module)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_all_nulls_and_modalities_round_trip_without_collision(self):
        folders = set()
        for perm in ('ieeg', 'meg', 'both'):
            for null in NULL_TYPES:
                with self.subTest(perm=perm, null=null):
                    result = prepare_selection(self.root, 2, perm=perm, perm_type=null)
                    loaded = load_selection(self.root, 2, perm=perm, perm_type=null)
                    name = f'{perm}__{null}'
                    self.loader.assert_called_with(self.root, name, 2)
                    self.assertEqual(loaded['summary']['permutation'], name)
                    self.assertEqual(loaded['summary']['selected_k'], 1)
                    pd.testing.assert_frame_equal(result['selection'], loaded['selection'])
                    folders.add(loaded['folder'])
        self.assertEqual(len(folders), 15)

    def test_phase_default_preserves_existing_path(self):
        result = prepare_selection(self.root, 2)
        self.assertEqual(result['folder'], self.root/'component_selection'/'k_002')
        self.assertEqual(load_selection(self.root, 2)['summary']['permutation'], 'both__phase')

    def test_correctly_spelled_alias_uses_canonical_results(self):
        prepare_selection(self.root, 2, perm_type='time_cirular_shift')
        loaded = load_selection(self.root, 2, perm_type='time_circular_shift')
        self.assertEqual(loaded['summary']['permutation'], 'both__time_cirular_shift')

    def test_invalid_null_rejected_before_reading_files(self):
        with self.assertRaisesRegex(ValueError, 'perm_type'):
            load_selection(self.root, perm_type='unknown')

    def test_wrong_null_manifest_rejected(self):
        result = prepare_selection(self.root, 2, perm_type='space')
        manifest = result['folder']/'selection.json'
        summary = json.loads(manifest.read_text())
        summary['permutation'] = 'both__phase'
        manifest.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'different null'):
            load_selection(self.root, 2, perm_type='space')

    def test_stale_source_rejected(self):
        prepare_selection(self.root, 2, perm_type='time_block')
        self.data['metadata']['signature'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'stale'):
            load_selection(self.root, 2, perm_type='time_block')


if __name__ == '__main__':
    unittest.main()
