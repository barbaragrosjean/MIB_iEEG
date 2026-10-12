import json
from pathlib import Path
import tempfile
import unittest

from plssvd_eval import prepare_output_directory


class OutputDirectoryTests(unittest.TestCase):
    def test_upgrade_preserves_old_run_and_removes_stale_active_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'both__time_point'
            prepare_output_directory(root, 10)
            (root / 'validation_options.json').write_text(json.dumps({'n_components': 10}))
            (root / 'iteration_000').mkdir()
            (root / 'iteration_000' / 'COMPLETE.json').write_text('{}')
            (root / 'evaluation').mkdir()
            (root / 'evaluation' / 'old.csv').write_text('old metrics')
            for k in (5, 10):
                with self.assertRaisesRegex(FileExistsError, '10 components already exists'):
                    prepare_output_directory(root, k)
                self.assertTrue((root / 'iteration_000' / 'COMPLETE.json').exists())
            prepare_output_directory(root, 100)
            self.assertEqual(list(root.iterdir()), [])
            backups = list((Path(tmp) / '.previous_runs').iterdir())
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / 'evaluation' / 'old.csv').read_text(), 'old metrics')
            self.assertTrue((backups[0] / 'iteration_000' / 'COMPLETE.json').exists())

    def test_unknown_or_invalid_count_does_not_replace_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / 'model_000.npz'
            marker.write_bytes(b'existing fit')
            for config in (None, '{}', 'invalid json', '{"n_components": null}'):
                if config is not None:
                    (root / 'validation_options.json').write_text(config)
                with self.assertRaisesRegex(FileExistsError, 'cannot be verified'):
                    prepare_output_directory(root, 100)
                self.assertEqual(marker.read_bytes(), b'existing fit')
            with self.assertRaises(ValueError):
                prepare_output_directory(root, 0)


if __name__ == '__main__':
    unittest.main()
