"""Exercise the inline decoding figures using current CSV/NPZ output fixtures."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class DecodingNotebookTests(unittest.TestCase):
    def run_notebook(self, complete=True, cropped=False):
        notebook = json.loads((ROOT / 'vs4_1_decoding.ipynb').read_text())
        config = json.loads((ROOT / 'out/vs4_decoding/config.json').read_text())
        config['pairings'] = 'order'
        config['parameters']['bs_decoding'] = cropped
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            base = out / 'vs4_decoding'
            base.mkdir()
            (base / 'config.json').write_text(json.dumps(config))
            times = np.linspace(-1, 1, 12)
            ids = np.arange(3) if cropped else np.arange(12)
            truth = np.tile([0, 1], 2)
            shuffled = np.array([[[0, 1], [1, 0], [0, 0]], [[1, 1], [0, 1], [1, 0]]])
            methods = ['PCA', 'PLSSVD'] if complete else ['PLSSVD']
            models = ['LR', 'RandomForest', 'SVC_linear'] if complete else ['LR']
            for method in methods:
                for model in models:
                    for pc in range(8) if complete else [0]:
                        active = (method, model, pc) == ('PLSSVD', 'LR', 0)
                        run = base / 'order' if active else base / 'archive' / f'order_{method}_{model}_{pc}'
                        run.mkdir(parents=True)
                        manifest = dict(status='complete', pairing='order', configuration=config,
                            provenance=dict(pairs={'dem': 'same'}, trial_cache={'manifest_sha256': 'same'}))
                        (run / 'manifest.json').write_text(json.dumps(manifest))
                        for modality in ['ieeg', 'meg']:
                            folder = run / modality / 'broadband' / 'fullmodel'
                            folder.mkdir(parents=True)
                            stem = f'{method}_mean_mean_{pc}_{model}full' + ('_crp0_3_' if cropped else '')
                            weights = np.arange(2*len(ids), dtype=float).reshape(2, len(ids))/10
                            control = np.arange(6*len(ids), dtype=float).reshape(2, 3, len(ids))/100
                            row = dict(band='broadband', method=method, method_pca='mean', data_aug_method='mean',
                                iter=2, iter_perm=3, y_test=repr(truth.tolist()), y_pred=repr(truth.tolist()),
                                y_pred_sh=repr(shuffled.tolist()), weight=repr(weights.tolist()),
                                weight_sh=repr(control.tolist()), pca_weight='unused')
                            pd.DataFrame([row]).to_csv(folder / (stem+'.csv'), index=False)
                            np.savez(folder / (stem+'_iteration0_time_courses.npz'), times=times,
                                conditions=np.array(['a', 'b']), pc_use=pc, method=method,
                                decoding_time_ids=ids, train_condition_mean=np.array([times, -times]))
                            # Legacy pickles must never be read.
                            (folder / (stem+'_cleaned.pkl')).write_bytes(b'not a pickle')
            namespace = {'__name__': '__main__'}
            sizes = [(20, 10), (15, 20), (10, 3), (20, 5), (8, 4)]
            with patch.dict(os.environ, {'MIB_PROJECT_ROOT': str(ROOT), 'MIB_OUTPUT_ROOT': str(out)}), contextlib.redirect_stdout(io.StringIO()):
                for i, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] != 'code':
                        continue
                    exec(compile(''.join(cell['source']), f'notebook cell {i}', 'exec'), namespace)
                    if i >= 4:
                        figure = plt.gcf()
                        self.assertTrue(np.array_equal(figure.get_size_inches(), sizes[i-4]))
                        figure.canvas.draw()
                        if i == 4:
                            self.assertEqual(len(figure.axes), 4)
                            self.assertEqual(len(figure.axes[0].lines[0].get_xdata()), len(ids))
                            self.assertEqual(len(figure.axes[2].lines[0].get_xdata()), len(times))
                        plt.close('all')
                self.assertEqual(namespace['PAIRING_ID'], 'order')
                record, result = namespace['load_decoding']('ieeg', 'PLSSVD', 0, 'LR')
                expected = shuffled.transpose(0, 2, 1).reshape(4, 3)
                np.testing.assert_array_equal(result['y_pred_sh'], expected)
                self.assertEqual(float(result['acc_sh']), np.mean(expected == truth[:, None]))
                np.testing.assert_allclose(result['weight_sh_95'], np.percentile(control.reshape(6, -1), 95, axis=0))
                np.testing.assert_array_equal(namespace['numbers']('[np.int64(1), np.float64(-0.25)]'), [1, -.25])
                self.assertLessEqual(len(namespace['_RESULT_CACHE']), 2)
                self.assertTrue(record['csv'].with_name(record['csv'].stem+'_cleaned.npz').is_file())
                namespace['_RESULT_CACHE'].clear()
                _, cached = namespace['load_decoding']('ieeg', 'PLSSVD', 0, 'LR')
                np.testing.assert_array_equal(cached['y_pred_sh'], expected)
                if complete:
                    self.assertEqual(len(namespace['RESULTS']), 96)
                    self.assertFalse(namespace['_MISSING_RESULTS'])
                else:
                    self.assertEqual(len(namespace['RESULTS']), 2)
                    self.assertTrue(namespace['_MISSING_RESULTS'])

    def test_all_comparison_figures_with_archived_runs(self):
        self.run_notebook()

    def test_partial_results_with_cropped_time_axis(self):
        self.run_notebook(complete=False, cropped=True)


if __name__ == '__main__':
    unittest.main()
