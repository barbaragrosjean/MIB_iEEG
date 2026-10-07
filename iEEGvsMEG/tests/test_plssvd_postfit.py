"""Compare post-fit sufficient-statistic metrics with direct feature-space evaluation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coverage_matching_utils import Dataset
from plssvd_evaluation import collect_statistics, evaluate_statistics
from plssvd_eval_utils import (ValidationOptions, _fit, _project, _predictor,
    _evaluate_fixed_model, fit_plssvd, load_plssvd_results, plot_iteration_summary,
    plot_permutation_comparison, plot_test_score_correlations)
from test_plssvd_iterations import synthetic_trials


class PostFitTests(unittest.TestCase):
    def test_matches_direct_metrics_at_each_k(self):
        rng = np.random.default_rng(17)
        fold = {p:{} for p in ('train','test')}
        # Multiple feature blocks, nonzero test means, unequal modality dimensions.
        for m,widths in [('ieeg',[4,3]),('meg',[5,4])]:
            for part in fold:
                arrays=[rng.normal(size=(2,w,32))+(2 if part=='test' else .3) for w in widths]
                fold[part][m]=Dataset(m,arrays,pd.DataFrame(index=range(sum(widths))),np.arange(4),'average')
        for scaling in ('none','equal_variance'):
            options=ValidationOptions(n_components=5,block_scaling=scaling)
            maximum=_fit(fold['train'],5,options)
            scores={p:{m:_project(ds,maximum,m) for m,ds in modalities.items()} for p,modalities in fold.items()}
            saved=collect_statistics(fold,scores,maximum)
            for k in (1,2,5):
                model=_fit(fold['train'],k,options)
                direct={p:{m:_project(ds,model,m) for m,ds in modalities.items()} for p,modalities in fold.items()}
                predictors={m:_predictor(direct['train']['meg' if m=='ieeg' else 'ieeg'],
                            fold['train'][m],model[m+'_mean'],options.ridge) for m in ('ieeg','meg')}
                _,actual,cov=evaluate_statistics(saved,k,options.ridge)
                for part in fold:
                    expected,expected_cov=_evaluate_fixed_model(fold[part],direct[part],model,predictors)
                    np.testing.assert_allclose(cov[part],expected_cov,rtol=1e-9,atol=1e-9)
                    for metric,value in expected.items():
                        self.assertAlmostEqual(actual[part][metric],value,places=9,msg=f'{scaling}/{k}/{part}/{metric}')

    def test_legacy_output_stays_readable(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            new=root/'new'
            fit_plssvd(synthetic_trials(),'full_concatenated',
                       ValidationOptions(n_components=2,n_iterations=1),new)
            result=load_plssvd_results(new)
            legacy=root/'legacy';child=legacy/'iteration_000';child.mkdir(parents=True)
            outer=dict(result['fit_options'],schema_version=4)
            inner=dict(result['validation_options'],schema_version=3)
            (legacy/'validation_options.json').write_text(json.dumps(outer))
            (child/'validation_options.json').write_text(json.dumps(inner))
            (child/'COMPLETE.json').write_text('{}')
            for name in ('summary','components','fold_metrics','metric_summary','fold_stability',
                         'fold_component_pairs','split_audit','participants'):
                result[name].to_csv(child/f'{name}.csv',index=False)
            np.savez(child/'trial_axes.npz',times=result['times'],conditions=result['conditions'])
            for fold in range(5):
                with np.load(new/'iteration_000'/f'scores_{fold:03d}.npz') as scores:
                    np.savez(child/f'model_{fold:03d}.npz',
                             **{f'{p}_{m}':scores[f'{p}_{m}'] for p in ('train','test') for m in ('ieeg','meg')})
            loaded=load_plssvd_results(legacy,n_components=2)
            np.testing.assert_allclose(loaded['iteration_metrics'].mean_r,result['iteration_metrics'].mean_r)
            with self.assertRaisesRegex(ValueError,'Legacy fits lack sufficient statistics'):
                load_plssvd_results(legacy,n_components=1)

    def test_fit_once_evaluate_many_without_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'none'
            # Fitting must not call old metric/predictor functions or the loader.
            with patch('plssvd_eval_utils._evaluate_fixed_model',side_effect=AssertionError('evaluated during fit')), \
                 patch('plssvd_eval_utils._predictor',side_effect=AssertionError('prediction map saved')), \
                 patch('plssvd_eval_utils.load_plssvd_results',side_effect=AssertionError('evaluated during fit')):
                fit_plssvd(synthetic_trials(),'full_concatenated',
                           ValidationOptions(n_components=3,n_iterations=2),root)
            self.assertFalse((root/'iteration_metrics.csv').exists())
            self.assertFalse((root/'iteration_000'/'fold_metrics.csv').exists())
            with np.load(root/'iteration_000'/'model_000.npz') as model:
                self.assertNotIn('predict_ieeg',model.files)
                self.assertNotIn('test_ieeg',model.files)
            with patch('plssvd_eval_utils._fit',side_effect=AssertionError('refit')), \
                 patch('plssvd_eval_utils._temporary_fold',side_effect=AssertionError('raw trials loaded')):
                for k in (1,2,3):
                    result=load_plssvd_results(root,n_components=k)
                    self.assertEqual(result['n_components_fitted'],3)
                    self.assertEqual(result['n_components_evaluated'],k)
                    self.assertEqual(result['primary_scores']['test']['ieeg'].shape[1],k)
                    self.assertEqual(result['components'].component.max(),k)
                    self.assertTrue((result['iteration_fold_metrics'].n_components==k).all())
                plt.close(plot_iteration_summary(root,n_components=2))
                fig=plot_permutation_comparison(root.parent,n_components=2);plt.close(fig)
                fig,_=plot_test_score_correlations(root/'iteration_000',n_components=2);plt.close(fig)
                for bad in (0,4,True,1.5):
                    with self.assertRaises(ValueError):load_plssvd_results(root,n_components=bad)
            # Still usable if interrupted after the first iteration.
            (root/'COMPLETE.json').unlink()
            (root/'iteration_001'/'COMPLETE.json').unlink()
            result=load_plssvd_results(root,n_components=1)
            self.assertEqual(result['n_iterations_loaded'],1)
            self.assertTrue(result['iteration_metric_summary']['std'].isna().all())

if __name__=='__main__':unittest.main()
