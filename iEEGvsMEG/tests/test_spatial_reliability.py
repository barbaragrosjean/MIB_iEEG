import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd

from plssvd_eval_utils import TrialData, TrialSubject
from spatial_reliability import (align_bundle, annotate_coordinates, fit_iterations,
    iteration_splits, parcel_correspondence, plot_results, subject_profiles, summarize)


class SpatialReliabilityTests(unittest.TestCase):
    def test_independent_halves_and_heldout_quarters(self):
        s = TrialSubject('s', (1, 2), [np.zeros((16, 2, 10))]*2,
                         np.zeros((2, 3)), pd.DataFrame(), [None]*2, [None]*2)
        trials = TrialData([s], [s], np.arange(10), (1, 2))
        result = iteration_splits(trials, 12)
        for m in ('ieeg', 'meg'):
            a, b = result['A'][m]['s'], result['B'][m]['s']
            for c in range(2):
                self.assertFalse(set(a['train'][c]) & set(b['train'][c]))
                self.assertEqual(set(a['test'][c]), set(b['train'][c]))
                self.assertFalse(set(a['test_a'][c]) & set(a['test_b'][c]))
                self.assertEqual(set(a['train'][c]) | set(a['test'][c]), set(range(16)))

    def test_temporal_alignment_restores_sign_and_order(self):
        rng = np.random.default_rng(2)
        reference = {}
        for m in ('ieeg', 'meg'):
            reference[m+'_train_scores'] = rng.normal(size=(30, 3))
            reference[m+'_weights'] = rng.normal(size=(8, 3))
        changed = {key: value[:, [2, 0, 1]]*[-1, 1, -1] for key, value in reference.items()}
        for model in ('separate_pca', 'plssvd', 'joint_pca'):
            aligned, _ = align_bundle(reference, changed, model)
            for key in reference:
                np.testing.assert_allclose(reference[key], aligned[key])

    def test_participant_balance_duplicate_collapse_and_missingness(self):
        meta = pd.DataFrame(dict(subject=['a', 'a', 'a', 'b'], source_index=[0, 0, 1, 0],
                                 parcel=['P', 'P', 'P', 'P']))
        values = np.array([[2], [-2], [10], [20.]])
        profiles = subject_profiles(values, meta, 'parcel')
        self.assertEqual(profiles.set_index('subject').loc['a', 'c1'], 6.)
        self.assertEqual(profiles.c1.mean(), 13.)
        self.assertEqual(profiles.n_unique_sources.tolist(), [2, 1])
        # Region Q is absent, not synthesized as zero.
        self.assertEqual(set(profiles.parcel), {'P'})

    def test_bootstrap_known_correspondence_and_insufficient_coverage(self):
        frame = pd.DataFrame([dict(subject=s, parcel=p, c1=float(i+1))
                              for s in ('a', 'b', 'c') for i, p in enumerate('PQRS')])
        rows, _ = parcel_correspondence({'ieeg': frame, 'meg': frame}, 'parcel', 1, 100, 4)
        self.assertAlmostEqual(rows[0]['spearman_r'], 1.)
        self.assertAlmostEqual(rows[0]['ci_low'], 1.)
        self.assertEqual(rows[0]['n_valid_bootstrap'], 100)
        rows, _ = parcel_correspondence({'ieeg': frame, 'meg': frame}, 'parcel', 1, 100, 4, min_subjects=4)
        self.assertEqual(rows[0]['n_common_parcels'], 0)
        self.assertTrue(np.isnan(rows[0]['spearman_r']))

    def test_end_to_end_cache_resume_summary_and_plots(self):
        rng = np.random.default_rng(20)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cache = root/'cache'; cache.mkdir()
            atlas = np.zeros((10, 3, 3), dtype=np.int16)
            for i in range(1, 9):
                atlas[i, 1, 1] = i
            atlas_path = root/'atlas.nii.gz'
            nib.save(nib.Nifti1Image(atlas, np.eye(4)), atlas_path)
            labels_path = root/'labels.csv'
            pd.DataFrame(dict(atlas_id=range(1, 9), parcel=[f'P{i}' for i in range(8)],
                              region=[f'R{i//2}' for i in range(8)],
                              group=[f'G{i//2}' for i in range(8)])).to_csv(labels_path, index=False)
            xyz = np.column_stack([np.arange(1, 9), np.ones(8), np.ones(8)])
            labels = annotate_coordinates(np.vstack([xyz, [-100, 0, 0]]), atlas_path, labels_path)
            self.assertTrue(labels.iloc[-1].isna().all())
            self.assertEqual(labels.iloc[0].parcel, 'P0_R')
            records = []
            subjects = {'ieeg': ['i1', 'i2'], 'meg': ['m1', 'm2', 'm3']}
            time = np.linspace(0, 2*np.pi, 25)
            scores = np.stack([np.sin(time), np.cos(2*time)])
            for modality in subjects:
                for subject in subjects[modality]:
                    signal = rng.normal(size=(8, 2)) @ scores
                    files = []
                    for c in (1, 2):
                        filename = f'{subject}_{c}.npy'
                        np.save(cache/filename, (signal[None]+rng.normal(scale=.2, size=(12, 8, 25))).astype('float32'))
                        files.append(filename)
                    meta = pd.DataFrame(xyz, columns=['x', 'y', 'z'])
                    meta['subject'] = subject
                    records.append(dict(modality=modality, subject=subject, files=files,
                        positions=xyz.tolist(), metadata=meta.to_dict('list'),
                        split_groups=[None, None], permutation_blocks=[['all']*12]*2))
            (cache/'manifest.json').write_text(json.dumps(dict(records=records, times=time.tolist(),
                config=dict(conditions=[1, 2], ieeg_subjects=subjects['ieeg'], meg_subjects=subjects['meg']))))
            args = SimpleNamespace(output_dir=root/'out', cache_dir=cache,
                iterations=3, components=2, seed=42, split_unit='trial', block_scaling='equal_variance',
                pool_hemispheres=False, ieeg_coords_dir=None, meg_coords_dir=None,
                ieeg_coords_unit='mm', meg_coords_unit='mm', atlas_image=atlas_path,
                atlas_labels=labels_path, scratch_dir=root/'scratch', max_gram_gib=1.)
            fit_iterations(args)
            saved = args.output_dir/'iteration_000'/'plssvd_A.npz'
            mtime = saved.stat().st_mtime_ns
            fit_iterations(args)
            self.assertEqual(saved.stat().st_mtime_ns, mtime)
            tables = summarize(args.output_dir, n_bootstrap=30, min_subjects=2)
            self.assertEqual(len(tables['across_iterations']), 3*3*2*2*2)
            self.assertEqual(set(tables['reliability'].measure), {'independent_half_weights',
                'independent_half_patterns', 'fixed_axes_A_heldout_quarters', 'fixed_axes_B_heldout_quarters'})
            self.assertEqual(set(tables['parcels'].level), {'parcel', 'region', 'group'})
            self.assertTrue((tables['parcels'].n_common_parcels > 0).all())
            pairings = [json.loads((args.output_dir/f'iteration_{i:03d}'/'pairing.json').read_text()) for i in range(3)]
            self.assertTrue(any(p != pairings[0] for p in pairings[1:]))
            import matplotlib.pyplot as plt
            for i, fig in enumerate(plot_results(args.output_dir)):
                fig.savefig(root/f'figure_{i}.png')
                plt.close(fig)
            args.seed = 43
            with self.assertRaisesRegex(ValueError, 'configuration/provenance changed'):
                fit_iterations(args)


if __name__ == '__main__':
    unittest.main()
