"""Round-trip and saved-results notebook tests using synthetic trial data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from test_compare_subspace import make_cache
from plssvd_eval_utils import (ValidationOptions, load_trial_cache, validate_plssvd,
                               load_plssvd_results, plot_plssvd_validation)


class ResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.cache = cls.root/'cache'; cls.cache.mkdir(); make_cache(cls.cache)
        cls.out = cls.root/'results'
        cls.original = validate_plssvd(load_trial_cache(cls.cache), 'paired_coverage',
            ValidationOptions(repeats=1, candidates=(1, 2), n_null=2, subject_fraction=1.), cls.out)

    @classmethod
    def tearDownClass(cls):
        plt.close('all')
        cls.temp.cleanup()

    def test_roundtrip_and_plotting(self):
        result = load_plssvd_results(self.out)
        for name in ('summary', 'components', 'selection', 'split_audit', 'participants', 'null_tests'):
            # Empty split-group CSV values become NaN when read from disk.
            self.assertEqual(result[name].shape, self.original[name].shape)
        np.testing.assert_allclose(result['summary'].test_mean_r, self.original['summary'].test_mean_r)
        for part in self.original['primary_scores']:
            for modality in ('ieeg', 'meg'):
                np.testing.assert_array_equal(result['primary_scores'][part][modality],
                                              self.original['primary_scores'][part][modality])
        np.testing.assert_array_equal(result['times'], self.original['times'])
        self.assertEqual(result['conditions'], self.original['conditions'])
        for name in self.original['null_distributions']:
            np.testing.assert_allclose(result['null_distributions'][name], self.original['null_distributions'][name])
        plot_plssvd_validation(result, output_dir=self.root/'plots', show=False)
        self.assertTrue((self.root/'plots'/'primary_null_tests.png').is_file())

    def test_no_null_run_ignores_stale_null_outputs(self):
        path = self.out/'validation_options.json'; original = path.read_text()
        options = json.loads(original); options['n_null'] = 0
        try:
            path.write_text(json.dumps(options))
            result = load_plssvd_results(self.out)
            self.assertTrue(result['null_tests'].empty)
            self.assertEqual(result['null_distributions'], {})
        finally:
            path.write_text(original)

    def test_legacy_axes_fallback_and_missing_files(self):
        axes = self.out/'trial_axes.npz'; temporary = self.out/'axes.saved.npz'
        axes.rename(temporary)
        try:
            with self.assertRaisesRegex(FileNotFoundError, 'trial_axes'):
                load_plssvd_results(self.out)
            result = load_plssvd_results(self.out, cache_dir=self.cache)
            np.testing.assert_array_equal(result['times'], self.original['times'])
        finally:
            temporary.rename(axes)
        with self.assertRaisesRegex(FileNotFoundError, 'Incomplete PLSSVD'):
            load_plssvd_results(self.root/'missing')

    def test_notebook_saved_mode_never_computes_or_loads_raw_data(self):
        notebook = json.loads(Path('plssvd_eval.ipynb').read_text())
        namespace = {'display': lambda *args, **kwargs: None}
        with patch('plssvd_eval_utils.prepare_trial_cache') as prepare, \
             patch('plssvd_eval_utils.load_trial_cache') as load, \
             patch('plssvd_eval_utils.validate_plssvd') as validate:
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source']).replace('from IPython.display import display', '')
                exec(compile(source, 'plssvd_eval.ipynb', 'exec'), namespace)
                if 'RUN_ANALYSIS = False' in source:
                    namespace['OUTPUT_DIR'] = self.out
            prepare.assert_not_called(); load.assert_not_called(); validate.assert_not_called()
        self.assertEqual(namespace['MEG_KIND'], 'paired_coverage')
        self.assertIn('result', namespace)


if __name__ == '__main__':
    unittest.main()
