"""Regression tests for temporary fold cleanup with open memory maps."""
import errno
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from plssvd_eval_utils import _temporary_fold, load_trial_cache, _split_subject
from test_compare_subspace import make_cache


class CleanupTests(unittest.TestCase):
    def test_real_fold_maps_closed_before_removal_and_cache_stays_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); cache=root/'cache'; cache.mkdir(); make_cache(cache)
            trials=load_trial_cache(cache); rng=np.random.default_rng(1)
            indices={m:{s.subject:_split_subject(s,rng,'trial') for s in getattr(trials,m)}
                     for m in ('ieeg','meg')}
            scratch=root/'scratch'; handles=[]; original_remove=shutil.rmtree
            original_open=np.lib.format.open_memmap
            def track(*args,**kwargs):
                array=original_open(*args,**kwargs);handles.append(array);return array
            def checked_remove(path,*args,**kwargs):
                self.assertTrue(handles)
                self.assertTrue(all(a._mmap.closed for a in handles))
                return original_remove(path,*args,**kwargs)
            with patch('numpy.lib.format.open_memmap',side_effect=track), \
                 patch('plssvd_eval_utils.shutil.rmtree',side_effect=checked_remove):
                with _temporary_fold(trials,'full_concatenated',indices,1,scratch) as (fold,*_):
                    self.assertTrue(np.isfinite(next(fold['train']['ieeg'].blocks())).all())
                    self.assertTrue(all(not a._mmap.closed for a in handles))
            self.assertEqual(list(scratch.iterdir()),[])
            self.assertFalse(trials.ieeg[0].data[0]._mmap.closed)
            self.assertTrue(np.isfinite(trials.ieeg[0].data[0][0]).all())

    def test_directory_not_empty_warns_without_aborting(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('plssvd_eval_utils._build_fold',return_value='fold'), \
                 patch('plssvd_eval_utils.shutil.rmtree',side_effect=OSError(errno.ENOTEMPTY,'Directory not empty')):
                with self.assertWarnsRegex(RuntimeWarning,'Only temporary files remain'):
                    with _temporary_fold(None,None,None,1,root) as fold:
                        self.assertEqual(fold,'fold')
            # Parent TemporaryDirectory cleans the intentionally retained test directory.

    def test_cleanup_error_does_not_replace_analysis_error(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('plssvd_eval_utils._build_fold',return_value='fold'), \
                 patch('plssvd_eval_utils.shutil.rmtree',side_effect=OSError(errno.ENOTEMPTY,'Directory not empty')):
                with self.assertWarns(RuntimeWarning):
                    with self.assertRaisesRegex(ValueError,'original fitting error'):
                        with _temporary_fold(None,None,None,1,root):
                            raise ValueError('original fitting error')

    def test_partial_build_failure_closes_created_maps(self):
        handles=[]
        def failed_build(trials,kind,indices,root,seed,mmap_handles):
            array=np.lib.format.open_memmap(Path(root)/'partial.npy',mode='w+',shape=(3,3))
            mmap_handles.append(array);handles.append(array)
            raise ValueError('build failed')
        with tempfile.TemporaryDirectory() as root:
            with patch('plssvd_eval_utils._build_fold',side_effect=failed_build):
                with self.assertRaisesRegex(ValueError,'build failed'):
                    with _temporary_fold(None,None,None,1,root):
                        self.fail('should not yield')
            self.assertTrue(handles[0]._mmap.closed)
            self.assertEqual(list(Path(root).iterdir()),[])


if __name__=='__main__':
    unittest.main()
