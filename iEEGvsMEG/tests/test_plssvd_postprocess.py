import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from test_plssvd_iterations import synthetic_trials
from plssvd_eval_utils import fit_plssvd, ValidationOptions, load_plssvd_results
from plssvd_postprocess import evaluate_run, prepare_comparison, publish_available, snapshot_path
from plssvd_figures import (load_evaluation,plot_first_iteration,plot_evaluation_summary,
    plot_prepared_permutations,load_comparison,plot_prepared_diagnostics,plot_prepared_pca)


class PipelineTests(unittest.TestCase):
    def test_resumption_and_plot_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,perm,kind in [('none',None,None),('meg__time_point','meg','time_point')]:
                fit_plssvd(synthetic_trials(),'full_concatenated',
                    ValidationOptions(n_components=2,n_iterations=2,perm=perm,perm_type=kind),root/name)
                evaluate_run(root/name,[1,2])
            prepare_comparison(root,'meg__time_point',1)
            expected=load_plssvd_results(root/'none',n_components=1)
            actual=load_evaluation(root/'none',1)
            np.testing.assert_allclose(actual['iteration_metrics'].mean_r,expected['iteration_metrics'].mean_r)
            first=snapshot_path(root/'none',1)
            # Identical commands must not repeat fold evaluation or PCA.
            with patch('plssvd_postprocess._common_fold',side_effect=AssertionError('repeated common computation')), \
                 patch('plssvd_postprocess._metric_fold',side_effect=AssertionError('repeated metric computation')):
                evaluate_run(root/'none',[1,2])
            self.assertEqual(snapshot_path(root/'none',1),first)
            # Plotting is portable: move the evaluation directories away from all fit files.
            import shutil
            portable=root/'portable'
            for name in ('none','meg__time_point'):
                shutil.copytree(root/name/'evaluation',portable/name/'evaluation')
            with patch('plssvd_eval_utils._fit',side_effect=AssertionError('fit in notebook')), \
                 patch('plssvd_eval_utils.load_trial_cache',side_effect=AssertionError('raw data in notebook')), \
                 patch('plssvd_eval_utils._saved_scores',side_effect=AssertionError('fit scores in notebook')), \
                 patch('plssvd_postprocess.evaluate_run',side_effect=AssertionError('evaluation in notebook')):
                data=load_evaluation(portable/'none',1)
                comparison=load_comparison(portable,'meg__time_point',1)
                plot_first_iteration(data)
                plt.close(plot_evaluation_summary(data))
                plt.close(plot_prepared_permutations(portable,1))
                for name,fig in plot_prepared_diagnostics(comparison).items():
                    fig.savefig(f'/tmp/plssvd_prepared_{name}.png');plt.close(fig)
                plt.close(plot_prepared_pca(comparison,comparison=True))
                plt.close(plot_prepared_pca(data))
                with self.assertRaisesRegex(FileNotFoundError,'No prepared evaluation'):
                    load_evaluation(portable/'none',3)
            # New preparation invalidates a comparison until it is republished.
            (root/'none'/'iteration_001'/'COMPLETE.json').unlink()
            publish_available(root/'none',[1])
            partial=load_evaluation(root/'none',1)
            self.assertEqual(partial['metadata']['iterations'],[0])
            self.assertTrue(np.isnan(partial['arrays']['test_correlation_absolute_std']).all())
            with self.assertRaisesRegex(ValueError,'stale'):load_comparison(root,'meg__time_point',1)
            prepare_comparison(root,'meg__time_point',1)
            self.assertEqual(load_comparison(root,'meg__time_point',1)['metadata']['iterations'],[0])

    def test_interrupted_evaluation_keeps_fold_checkpoints(self):
        import plssvd_postprocess as module
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'none'
            fit_plssvd(synthetic_trials(),'full_concatenated',ValidationOptions(n_components=2,n_iterations=2),root)
            original=module._metric_fold;calls=[]
            def stop(*args,**kwargs):
                calls.append(1)
                if len(calls)==7:raise MemoryError('simulated allocation error')
                return original(*args,**kwargs)
            with patch.object(module,'_metric_fold',side_effect=stop):
                with self.assertRaisesRegex(MemoryError,'simulated'):evaluate_run(root,[1])
            self.assertEqual(load_evaluation(root,1)['metadata']['iterations'],[0])
            completed=root/'evaluation'/'k_001'/'iteration_001'/'fold_000.npz'
            before=completed.stat().st_mtime_ns
            evaluate_run(root,[1])
            self.assertEqual(completed.stat().st_mtime_ns,before)
            self.assertEqual(load_evaluation(root,1)['metadata']['iterations'],[0,1])
            # Adding a count reuses all-rank component/PCA correlations.
            with patch.object(module,'_common_fold',side_effect=AssertionError('repeated correlations')):
                evaluate_run(root,[2])
            # Recovering optional PCA later must not repeat already cached metrics.
            scores=root/'iteration_000'/'scores_000.npz'
            with np.load(scores) as z:values=dict(z)
            np.savez_compressed(scores,**{key:value for key,value in values.items() if not key.endswith('_pca')})
            evaluate_run(root,[1])
            self.assertFalse(load_evaluation(root,1)['metadata']['pca_available'])
            cache=Path(directory)/'trial_cache';cache.mkdir();(cache/'manifest.json').write_text('{}')
            with patch('plssvd_eval_utils.load_trial_cache',return_value=synthetic_trials()), \
                 patch.object(module,'_metric_fold',side_effect=AssertionError('unnecessary metric recomputation')):
                evaluate_run(root,[1],cache_dir=cache)
            self.assertTrue(load_evaluation(root,1)['metadata']['pca_available'])


if __name__=='__main__':unittest.main()
