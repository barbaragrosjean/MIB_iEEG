# Prepared condition-average data

The coverage and covariance notebooks now have separate **Prepare condition averages** and **Load prepared data** cells. The first preparation still reads the original recordings. Subsequent runs reuse `out/prepared_data/<input-and-settings-hash>/`.

```python
from vs0_prepare_data import prepare_data, load_prepared_data, data_options

options = data_options(CONFIG)
prepare_data(**options)                  # Idempotent; performs preprocessing on a cache miss.
data = load_prepared_data(**options)     # Loads prepared arrays; never prepares recordings.
```

`load_dataset()` and the coverage/covariance dataset-loading helpers also use this cache. Calling them before preparation raises an error identifying the missing cache. Cluster analysis entry points perform an explicit preparation stage before loading. Preparation can also be run separately:

```sh
python vs0_prepare_data.py --config path/to/config.json
```

The cache stores participant-level condition averages, source/electrode positions, electrode metadata, participant IDs, time samples, trial-count information, and a manifest. MEG arrays load as read-only memory maps. Whole-dataset normalization remains after construction of each analysis setup. Coverage constructs all five MEG setups together, sharing one spatial-matching pass.

Known recording and metadata file sizes/modification times, the selected cohort/files, conditions, coordinate settings, and preprocessing options determine compatibility. Participant assignments and CSV row order are applied at loading and do not invalidate prepared recordings when the cohort/files are unchanged. Different cohorts require separate preparation. Keep the same `prepared_cache_dir` across analyses to reuse the cache.

When GetInfo uses additional external metadata, list those files in `CONFIG['data']['metadata_dependencies']` so changes invalidate the cache. Alternatively, change `CONFIG['data']['preparation_revision']` to request fresh preparation after untracked external changes. GetInfo code and its project path are included in the preparation identity.

This cache is **separate from `trial_cache`**. Trial-held-out validation continues to estimate standardization from training trials within each fold; these prepared averages do not replace individual trials.
