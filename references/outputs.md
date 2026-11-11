# Output Contract

All paths are relative to the requested report directory.

## `samples.csv`

One row per unique image path. Important groups:

- identity: `sample_id`, `path`, `rel_path`, `label`, `group_id`,
  `label_source`;
- quality: blur, brightness, contrast, spectral entropy, high-frequency
  ratio, image entropy, dimensions, hashes, quality flags;
- semantic: availability, centroid distance, confusion risk, cross-class
  margin, augmentation stability/instability, and raw component ranks;
- scoring: visual accessibility, descriptive visual complexity, configured
  and effective weights, contributions, `difficulty_scope`, coverage,
  difficulty score, band, and sample status;
- label statistics: `class_count`, `label_rarity_score`,
  `label_rarity_weight=0`, `label_rarity_contribution=0`, and
  `label_rarity_in_difficulty=False`;
- integrity: `anomaly_types`, duplicate group, canonical flag, statistical
  outlier metrics, and `near_duplicate_status`;
- split: `split`.

Paths and hashes are metadata only. Original files are never rewritten.

## `summary.json`

Includes:

- total, anomaly, scored, unscored, and semantic-available counts;
- class counts;
- difficulty-band counts;
- anomaly reason counts;
- split counts and diagnostics;
- semantic backend;
- semantic runtime: device, model, batch size, cache hits/misses, failures,
  and CUDA peak allocated/reserved memory;
- stage timings and total analysis/report time;
- the fully resolved analysis configuration;
- warnings and limitations;
- input and output paths;
- artifact list.

The writer replaces NaN/Infinity with JSON `null`.

## `report.html`

Standalone review page with:

- dataset counts;
- difficulty and split summaries;
- hardest scored sample table;
- anomaly table;
- hard and anomaly preview sheets;
- warnings and artifact links.

## `hard_samples.png`

Top scored non-anomalous samples ordered by descending difficulty score. Each
panel is captioned with sample ID, band, and anomaly field. When no scored
samples exist, the sheet contains an explicit empty-state message.

## `anomaly_samples.png`

Anomalous samples ordered by relative path, with the same empty-state
behavior.

## `split_manifest.csv`

Minimal split manifest:

`sample_id`, `path`, `rel_path`, `label`, `group_id`, `difficulty_band`,
`difficulty_score`, visual/semantic contributions, `difficulty_scope`,
`component_coverage`, `split`, `is_anomaly`, `anomaly_types`.

Anomalies use `split=excluded_anomaly` when exclusion is enabled.

## `split_summary.json`

Contains normalized ratios, seed, split mode, group count, counts, actual
scored ratios, class/difficulty distributions, group and duplicate leakage
counts, leakage-risk level/messages, excluded anomaly count, per-stratum
allocations, and tiny-stratum warnings.

## Cache

The cache directory is separate from report artifacts. Each image/content and
model-signature combination produces one NPZ file containing base and
augmented embeddings. Cached files are written atomically. A corrupt cache
entry is ignored and recomputed.
