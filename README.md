# Dataset Difficulty Skill

A training-free, model-independent difficulty prior for image classification
datasets. The tool combines image accessibility measurements, CLIP embedding
geometry, independent class-distribution statistics, data-integrity checks,
and reproducible group-safe stratification.

It does not use downstream training weights or training losses. It does not
claim to predict model accuracy.

## Install

Target runtime: Python 3.11. The project accepts Python `>=3.11,<3.13`;
compatibility is retained for Python 3.12 verification environments.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

CLIP uses the Transformers CLIP implementation. Model resolution is local-only
by default. The first CLIP run requires either a local Transformers model
directory or explicit authorization for network/model download.

## Quick Start

Folder-per-class:

```bash
python scripts/analyze_dataset.py path/to/dataset --output path/to/reports
```

CSV labels:

```bash
python scripts/analyze_dataset.py path/to/images \
  --labels path/to/labels.csv \
  --format csv \
  --output path/to/reports
```

JSON labels:

```bash
python scripts/analyze_dataset.py path/to/images \
  --labels path/to/labels.json \
  --format json \
  --output path/to/reports
```

Optional leakage-safe groups:

```bash
python scripts/analyze_dataset.py path/to/images \
  --labels path/to/labels.csv \
  --format csv \
  --group-column group_id \
  --output path/to/reports
```

Rows sharing a non-empty `group_id` always enter the same split. Without a
group column, the original sample-level joint stratification remains active.

Offline report without CLIP:

```bash
python scripts/analyze_dataset.py path/to/dataset \
  --output path/to/reports \
  --no-semantic
```

Explicit local-only CLIP mode:

```bash
python scripts/analyze_dataset.py path/to/dataset \
  --output path/to/reports \
  --semantic-backend clip
```

Authorized model download:

```bash
python scripts/analyze_dataset.py path/to/dataset \
  --output path/to/reports \
  --semantic-backend clip \
  --allow-model-download
```

Installed console entrypoint:

```bash
dataset-difficulty path/to/dataset --output path/to/reports
```

## Example Dataset

Generate the deterministic toy dataset and run the offline example:

```bash
python examples/generate_example_dataset.py
python scripts/analyze_dataset.py examples/toy_dataset \
  --labels examples/toy_dataset/labels.csv \
  --format csv \
  --output examples/output \
  --no-semantic
```

The example intentionally includes a corrupt file and an exact duplicate so
the anomaly section is non-empty. Its semantic columns are null because the
example run is offline; this is a disclosed limitation, not fabricated data.

## Outputs

- `samples.csv`: raw metrics, semantic metrics, component scores, anomaly
  reasons, difficulty bands, group IDs, and split assignments.
- `summary.json`: counts, class distribution, anomaly reasons, configuration
  diagnostics, warnings, and limitations.
- `report.html`: standalone review page.
- `hard_samples.png`: hardest scored sample preview sheet.
- `anomaly_samples.png`: anomaly sample preview sheet.
- `split_manifest.csv`: Train/Val/Test/excluded manifest.
- `split_summary.json`: exact split counts and per-class/band diagnostics.

## Configuration

Copy or edit `configs/default.yaml`. All weights, quality thresholds,
robust-z settings, CLIP settings, difficulty quantiles, split ratios, and
report preview settings are configurable.

Important defaults:

- anomaly-relative robust z threshold: `3.5`;
- semantic min class size: `2`;
- difficulty weights: visual accessibility `0.30`, semantic `0.70`;
- label rarity: independent statistic with zero difficulty contribution;
- band quantiles: `0.33`, `0.67`;
- split ratios: Train `0.70`, Val `0.15`, Test `0.15`;
- CLIP download: disabled;
- feature cache: enabled.

See `references/configuration.md` for schemas and CLI overrides.

## Method Summary

Quality signals include Laplacian blur, normalized brightness and contrast,
FFT spectral entropy, high-frequency energy ratio, and 8-bit histogram
entropy. Absolute thresholds identify quality anomalies. Blur, exposure
extremes, and contrast form the bounded visual-accessibility component.

`visual_complexity_score` combines entropy and frequency ranks as a
descriptive statistic only. Those signals are not assumed to imply higher
task difficulty and do not affect the composite score.

CLIP signals include class-centroid distance, cross-class confusion margin,
and stability under deterministic augmentation. Embeddings are L2-normalized.
No vector entropy is treated as a probability distribution.

The final prior combines visual accessibility and semantic complexity, and
records configured weights, effective weights, contributions, and component
coverage. Label rarity is reported but contributes zero.

Hard anomalies such as corrupt, empty-label, conflicting-label, duplicate,
or undersized samples receive `difficulty_band=Anomaly` and are excluded from
the split by default. Statistical-only anomalies and low-quality but readable
images are soft anomalies: they keep their flags as `Statistically_Unusual`
or `Quality_Flagged`, participate in scoring/splitting, and remain visible in
anomaly audits. Set `split.include_statistical_only` or
`split.include_quality_outliers` to `false` to restore hard exclusion.

When CLIP is unavailable, `difficulty_scope=visual_only` and the report states
that the result is not a complete semantic difficulty assessment.

## Validation

```bash
python -m pytest -v
```

The suite covers folder/CSV/JSON loading, conflicting labels, quality metrics,
corrupt images, exact/near duplicates, robust statistical outliers,
content-addressed feature caching, semantic geometry, difficulty scoring,
joint and group-safe splitting, all report artifacts, visual-only fallback,
and the CLI end-to-end path.

## Real Validation

Real Transformers CLIP validation and CIFAR-10 results are recorded under
`validation/`:

- `validation/real_clip/`: CPU/GPU, cache-hit, and resume metadata.
- `validation/cifar10_test/`: complete official CIFAR-10 test-split report.
- `validation/before_after/`: V1/V2 score and band comparison.
- `validation/anomaly_audit/`: deterministic 50-image anomaly review.
- `validation/experiment_manifest.json`: source commit, config hash, model
  source/hash, command, and dataset scope.

The verified model is `openai/clip-vit-base-patch32` on an RTX 3060 Laptop
GPU. The CIFAR-10 experiment uses the official 10,000-image test split, not
the complete 60,000-image train+test dataset. It processed all 10,000 test
images with no CLIP extraction failures.

## Safety

Original images and labels are read-only. Generated report and cache files
are written only to the requested output and cache locations. Model downloads
require the explicit `--allow-model-download` flag.

The output and cache directories must be outside the dataset tree. The
analyzer rejects an output or cache path that would write generated files into
the original dataset.

Difficulty bands are relative to the analyzed dataset. They are not calibrated
probabilities and do not replace a validation set or training experiment.
