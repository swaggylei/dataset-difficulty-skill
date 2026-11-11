---
name: dataset-difficulty-skill
description: Analyze image classification datasets before training for visual quality issues, semantic outliers, class confusion, augmentation instability, corrupt or duplicate samples, relative difficulty bands, and stratified splits. Use for dataset audits and training-readiness checks; do not use for detection, segmentation, video, or claims about downstream accuracy.
metadata:
  short-description: Training-free image classification difficulty audit
---

# Dataset Difficulty Skill

Produce a reproducible, model-independent difficulty prior for an image
classification dataset. Use CLIP embeddings when available, but keep every
raw metric and component visible. Never present the final score as a predicted
accuracy or as a substitute for downstream validation.

## When to Use

Use this Skill when the user wants to:

- audit an image classification dataset before training;
- find corrupt, duplicate, unlabeled, conflicting-label, or statistical
  outliers;
- compare relative Easy/Medium/Hard difficulty;
- create class-and-difficulty-stratified Train/Val/Test manifests;
- obtain machine-readable and human-reviewable reports.

Do not use it for object detection, segmentation, video analysis, or
fine-tuning a classifier.

## Check Inputs First

1. Confirm the dataset directory exists.
2. Determine format: folder-per-class, CSV, or JSON.
3. For CSV/JSON, confirm the label file and relevant path/label columns.
4. If leakage-safe grouping is required, confirm the optional `group_id`
   column name.
5. Select an output directory outside the dataset tree.
6. Never modify or delete original images or labels.
7. Do not download CLIP weights unless the user explicitly authorizes it.

Supported folder layout:

```text
dataset/
  cat/
    image-1.jpg
  dog/
    image-2.png
```

CSV and JSON examples and column overrides are in
[references/configuration.md](references/configuration.md).

## Run the Analysis

The source tree is directly executable:

```bash
python scripts/analyze_dataset.py DATA_ROOT \
  --output OUTPUT_DIR \
  --config configs/default.yaml
```

With a label file:

```bash
python scripts/analyze_dataset.py DATA_ROOT \
  --labels labels.csv \
  --format csv \
  --output OUTPUT_DIR
```

With leakage-safe groups:

```bash
python scripts/analyze_dataset.py DATA_ROOT \
  --labels labels.csv \
  --format csv \
  --group-column group_id \
  --output OUTPUT_DIR
```

Offline analysis without CLIP:

```bash
python scripts/analyze_dataset.py DATA_ROOT \
  --output OUTPUT_DIR \
  --no-semantic
```

CLIP is local-only by default. If the user explicitly authorizes a model
download and network use:

```bash
python scripts/analyze_dataset.py DATA_ROOT \
  --output OUTPUT_DIR \
  --semantic-backend clip \
  --allow-model-download
```

Run `python scripts/analyze_dataset.py --help` for all overrides.

## Interpret in This Order

1. Read `summary.json` first.
2. Handle `corrupt_image`, `empty_label`, and `conflicting_label` as
   data-integrity findings.
3. Review duplicate groups and near duplicates.
4. Review quality outliers separately from difficulty.
5. Compare visual accessibility, descriptive visual complexity, raw semantic
   components, weights, and contribution values.
6. Interpret Easy/Medium/Hard only among non-anomalous scored samples.
7. Check split diagnostics and warnings before using the manifest.

Report the components that drive a result. For example: "Hard because the
sample is far from its class centroid and unstable under augmentation; its
image quality itself is normal." Do not say "the model will fail" or "this
sample has low accuracy."

If semantic fields are null, state that CLIP was unavailable or disabled.
Do not infer or invent embeddings. Do not treat poor quality as semantic
difficulty. Do not merge anomaly labels into Easy/Medium/Hard.

Label rarity is an independent dataset statistic with zero weight and zero
contribution. Image entropy and frequency complexity are descriptive visual
complexity, not evidence that a sample is harder. When
`difficulty_scope=visual_only`, explicitly warn that the prior lacks CLIP
semantic evidence.

Read `summary.json["semantic_runtime"]` and `summary.json["performance"]`
for actual device, batch size, cache hits/misses, failures, stage timings, and
CUDA peak memory. Do not report those fields unless they came from a real run.

Read:

- [references/metrics.md](references/metrics.md) for formulas and limits;
- [references/configuration.md](references/configuration.md) for every config
  section, input schema, and CLI override;
- [references/outputs.md](references/outputs.md) for artifact and column
  contracts.

## Verify

Use the project environment that has the required dependencies:

```bash
python -c "import cv2, numpy, pandas, sklearn, matplotlib, yaml, torch, transformers"
python -m pytest -v
```

For a real run, confirm that `samples.csv`, `summary.json`, `report.html`,
`hard_samples.png`, `anomaly_samples.png`, `split_manifest.csv`, and
`split_summary.json` exist. Verify that `summary.json` contains no `NaN` or
`Infinity` tokens.

## Safety and Scope

- All analysis writes go to the explicit output and cache directories.
- Output and cache paths must be outside the input dataset tree.
- Feature cache entries are keyed by image content hash and model signature.
- Missing metrics remain null; available components are reweighted and
  coverage is reported.
- The final score is a relative prior, not a calibrated probability.
- Class balance and difficulty stratification are approximate for tiny
  strata; report split warnings.
