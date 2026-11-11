# Configuration and Input Reference

The default YAML is `configs/default.yaml`. `load_config` deep-merges a user
file over packaged defaults.

The report output and resolved cache directories must be outside the dataset
tree. This prevents generated artifacts from modifying or overwriting input
data.

## Dataset Inputs

### Folder-per-Class

The first directory below the dataset root is the class label. Nested files
under that directory inherit the same class.

### CSV

```csv
path,label
cat/001.jpg,cat
dog/002.jpg,dog
```

Relative paths are resolved against the label file directory first, then the
dataset root. Empty labels are preserved. A path with multiple distinct labels
is collapsed to one conflict-marked sample.

Override columns:

```bash
--path-column image_path --label-column class_name
```

### JSON

Supported shapes:

```json
[
  {"path": "cat/001.jpg", "label": "cat"}
]
```

```json
{
  "cat/001.jpg": "cat",
  "dog/002.jpg": "dog"
}
```

```json
{
  "items": [
    {"path": "cat/001.jpg", "label": "cat"}
  ]
}
```

The top-level list keys `items`, `data`, and `samples` are accepted.

## Configuration Sections

### `project`

- `seed`: deterministic split and augmentation-view seed; it is included in
  the feature-cache signature.
- `version`: informational report version.

### `data`

- `format`: `auto`, `folder`, `csv`, or `json`.
- `path_column`, `label_column`, `group_column`: CSV/JSON column names.
- `extensions`: image suffixes scanned in folder mode.
- `ignore_dirs`: directory names below the dataset root to exclude.

### `quality`

- `blur_laplacian_min`
- `brightness_min`, `brightness_max`
- `contrast_min`
- `image_entropy_min`
- `spectral_entropy_min`

These absolute thresholds produce `quality_issue_flags`.

### `anomaly`

- `min_image_size`: minimum width and height.
- `robust_z_threshold`: per-label outlier threshold.
- `robust_min_group_size`: minimum label group size for robust statistics.
- `near_duplicate.enabled`
- `near_duplicate.hamming_threshold`
- `near_duplicate.max_samples`: skip near-duplicate comparison above this
  count; exact duplicates are still detected.

### `semantic`

- `enabled`
- `backend`: `auto`, `clip`, or `disabled`.
- `min_class_samples`: minimum class size for centroid distance.
- `clip.model_name`: Hugging Face model ID or local model directory.
- `clip.device`: `auto`, `cpu`, `cuda`, or a supported device string.
- `clip.batch_size`
- `clip.allow_download`: defaults to false.
- `clip.augmentations`: number of deterministic augmented views.
- `cache.enabled`
- `cache.directory`: relative paths resolve beside the report directory.

### `difficulty`

- `weights.visual`, `weights.semantic`: composite weights. `weights.label` is
  ignored and Label Rarity always has zero contribution.
- `visual_weights.blur`, `brightness`, `contrast`: visual accessibility.
- `visual_complexity_weights.image_entropy`, `spectral_entropy`,
  `high_frequency`: descriptive complexity only.
- `semantic_weights.outlier`, `confusion`, `instability`.
- `bands.quantiles`: two ascending quantiles.
- `bands.labels`: Easy, Medium, Hard names.

### `split`

- `ratios.train`, `ratios.val`, `ratios.test`.
- `exclude_anomalies`: default true.
- `ensure_val_test_when_possible`: assigns Val and Test to strata of size at
  least three.
- Non-empty `group_id` values switch to whole-group allocation automatically.

### `report`

- `preview_count`
- `preview_columns`
- `image_thumbnail_size`
- `title`

## CLI Overrides

```text
--config PATH
--format auto|folder|csv|json
--path-column NAME
--label-column NAME
--group-column NAME
--semantic-backend auto|clip|disabled
--no-semantic
--clip-model MODEL_OR_PATH
--allow-model-download
--device DEVICE
--seed INTEGER
--verbose
```

`--no-semantic` takes precedence over `--semantic-backend`.
