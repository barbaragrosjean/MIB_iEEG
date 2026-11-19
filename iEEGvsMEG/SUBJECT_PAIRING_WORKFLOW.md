# Subject-pairing analyses

In the coverage, covariance, PLSSVD evaluation, modality comparison, modality
specificity, and decoding notebooks, set:

```python
PAIRING_VIEW = 'matched'  # saved demographic assignment, ID -1
# PAIRING_VIEW = 'full'   # assignments 0 through N_PAIRINGS - 1
N_PAIRINGS = 100
PAIRING_SEED = 2026
```

Restart the kernel after changing these settings, and run the notebook from the
beginning. Run the commands it prints on the cluster, then reload the notebook.
The commands use `vs0_run_pairings.py` to run each assignment in a separate
subprocess. Successful commands are recorded so interrupted batches can resume.

The iEEG cohort and its order are fixed by `out/demographic_matching/matched_pairs.csv`.
ID -1 uses the CSV mapping and exact MEG paths. Each nonnegative ID independently
samples distinct MEG participants without replacement from the entire available
MEG directory. A participant may recur across assignments. Random assignments
are reproducible from the pairing seed and ID. The averaged-data and trial-data
workflows share the same assignment function and sorted MEG roster.

Outputs are stored under
`out/subject_pairings/<CSV-content-hash>/iteration_<ID>/`.
Each analysis retains its own trial folds, evaluation iterations, and surrogate
permutations inside that assignment directory. The outer assignment ID never
replaces a time-point permutation index. Use the same pairing seed and IDs for
unpermuted and temporally permuted runs.

## Preparing and evaluating trials

Use the new shared `out/trial_cache_full_meg` directory. The PLSSVD fitting command
exports every available MEG participant there before applying an assignment.
Existing restricted caches are rejected for random-assignment analyses. Prepare
this cache once before running trial-based modality comparisons or decoding.
The existing cluster data directories and metadata dependencies are still needed.

`N_FIT_ITERATIONS` in PLSSVD controls repeated trial-fold evaluations *within*
one subject assignment. `N_PAIRINGS` controls the outer subject-assignment loop.
`PERMUTATION_TYPE = 'time_point'` still controls temporal surrogates. These are
three independent controls. A matched assignment no longer forces
`N_FIT_ITERATIONS = 1`; its notebook default remains 1 and can be increased.

## Displaying the ensemble

The final ensemble panels report scalar metric curves and tables as mean ± sample
SD across random assignments, after averaging internal resampling within each
assignment. The saved match is excluded. Counts are shown for metrics with missing
values. A missing assignment or ambiguous/stale table prevents that panel from
being aggregated. The original matched plots are retained in matched view.

These are descriptive assignment SDs, not confidence intervals. Componentwise
scalar summaries retain native component rank. Signed coefficient maps and
component time courses are not averaged across independently fitted axes without
an explicit alignment. The full-view panels summarize the exported scalar tables;
the original anatomical maps remain available in matched view.

Spatial profile ensembles use common 10-mm distance bins in full view. Adjust the
shared edges in the notebook if a different range or resolution is required.
Decoding has separate controls for subject assignments, decoding resampling, and
label permutations; its ensemble panel averages decoding resamples within each
assignment before summarizing accuracy and absolute coefficient magnitude.

## MEG behavioral eligibility

All MEG selection uses `MEG/Blocks3456_csv` (override its location with the
`MEG_BEHAVIOR_DIR` environment variable when needed). Accuracy is the number of
correct responses divided by the number of old/new trials, pooled over available
Blocks 3–6. Stimuli containing `old` require response 1; those containing `new`
require response 2. Missing, invalid, or zero responses count as incorrect.
Other stimulus types and export footers are ignored. Participants with accuracy
below 0.50 are excluded; exactly 0.50 is retained. This is pooled trial accuracy,
not an average of block accuracies or balanced accuracy across stimulus classes.

The demographic notebook applies this rule inside `prepare_for_matching` before
selecting pairs. Raw/averaged preparation, MEG roster discovery, random pairing,
and existing trial-cache loading use the same rule. Unknown participant IDs or
missing behavioral files cause an explicit error rather than silently retaining
an unassessed participant. Numeric questionnaire IDs and `SUBJ_0001` recording
IDs are resolved to the same behavioral identity.

If an older saved demographic match contains an excluded MEG participant, that
pair is removed and a warning identifies it. Rerun demographic matching to retain
the full iEEG cohort by selecting replacements from eligible MEG participants.
Random analyses use the same retained iEEG cohort until that match is regenerated.

Preparation saves `meg_behavior_eligibility.csv` alongside its manifest, and the
matching notebook exports the same audit. Behavioral file contents and the rule
are included in cache/result provenance. Default trial caches now live in
`out/trial_cache_full_meg/<behavior-content-hash>/`; previous caches are preserved
and the new default is populated by rerunning the PLSSVD preparation command.
Subject-pairing output namespaces also incorporate the behavioral content hash.
