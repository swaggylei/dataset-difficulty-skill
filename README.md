# Dataset Difficulty Skill

A training-free, model-independent difficulty prior for image classification
datasets. The tool combines OpenCV image measurements, CLIP embedding
geometry, label accessibility, data-integrity checks, and reproducible joint
stratification.

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
  reasons, difficulty bands, and split assignments.
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
- difficulty weights: visual `0.20`, semantic `0.60`, label `0.20`;
- band quantiles: `0.33`, `0.67`;
- split ratios: Train `0.70`, Val `0.15`, Test `0.15`;
- CLIP download: disabled;
- feature cache: enabled.

See `references/configuration.md` for schemas and CLI overrides.

## Method Summary

Quality signals include Laplacian blur, normalized brightness and contrast,
FFT spectral entropy, high-frequency energy ratio, and 8-bit histogram
entropy. Absolute thresholds identify quality anomalies; dataset-relative
ranks contribute a bounded visual component.

CLIP signals include class-centroid distance, cross-class confusion margin,
and stability under deterministic augmentation. Embeddings are L2-normalized.
No vector entropy is treated as a probability distribution.

The final prior combines only available components and records component
coverage. Anomalies receive `difficulty_band=Anomaly` and are excluded from
the split by default.

## Validation

```bash
python -m pytest -v
```

The suite covers folder/CSV/JSON loading, conflicting labels, quality metrics,
corrupt images, exact/near duplicates, robust statistical outliers,
content-addressed feature caching, semantic geometry, difficulty scoring,
joint splitting, all report artifacts, no-semantic fallback, and the CLI
end-to-end path.

## Safety

Original images and labels are read-only. Generated report and cache files
are written only to the requested output and cache locations. Model downloads
require the explicit `--allow-model-download` flag.

The output and cache directories must be outside the dataset tree. The
analyzer rejects an output or cache path that would write generated files into
the original dataset.

Difficulty bands are relative to the analyzed dataset. They are not calibrated
probabilities and do not replace a validation set or training experiment.
