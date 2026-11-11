# Verification Results

Date: 2026-11-11

## Environment

- Host Python: 3.12.7
- Test Python: `E:\Jupyter Notebook\python.exe` (CPython 3.12.7)
- OpenCV and related image dependencies supplied by the existing
  `E:\大学\AIGC\AGIQA-1K\.venv-baselines2` package directory
- pytest supplied by the existing `E:\Jupyter Notebook` Python
- Python 3.11 was not installed on this host; the project declares
  `requires-python = ">=3.11,<3.13"` and compatibility testing used 3.12.7

No packages were installed or modified during verification. The dependency
install approval was declined, so existing local environments were composed
through `PYTHONPATH`.

## Unit and End-to-End Tests

Command:

```powershell
$env:PYTHONPATH='E:\大学\AIGC\AGIQA-1K\.venv-baselines2\Lib\site-packages'
& 'E:\Jupyter Notebook\python.exe' -m pytest -q
```

Result:

```text
71 passed in 26.12s
```

Coverage includes:

- folder, CSV, JSON loading and conflicting labels;
- OpenCV quality metrics and unreadable/corrupt images;
- exact duplicates, near duplicates, tiny images, quality and robust-z
  outliers;
- resumable cache hits, malformed cache entries, disabled cache, zero
  augmentations, and seed-sensitive augmentation;
- CLIP-shaped normalized embeddings, centroid distance, class confusion,
  augmentation instability, and missing semantic values;
- hard-anomaly isolation, statistical-only soft anomalies, and quantile bands;
- zero Label Rarity contribution, explicit component weights/contributions,
  visual-only scope, and descriptive-only entropy/frequency complexity;
- reproducible class-and-difficulty joint splits, whole-group `group_id`
  splitting, leakage diagnostics, and tiny strata;
- empty datasets, JSON null sanitization, HTML/CSV/PNG reports;
- CLI subprocess execution, group-column input, config-driven semantic backend
  selection, transformer output compatibility, and PyTorch checkpoint loading;
- missing explicit group-column failure, prepare-only semantic preflight,
  usable-semantic availability, and explicit before-CSV comparison;
- CIFAR-10 Parquet export and V1/V2 metric comparison utilities.
- anomaly category audit and statistical-only soft-anomaly handling.

## Skill Validation

Command:

```powershell
python 'C:\Users\12089\.codex\skills\.system\skill-creator\scripts\quick_validate.py' .
```

Result:

```text
Skill is valid!
```

## Example Run

Command:

```powershell
python scripts/analyze_dataset.py examples/toy_dataset `
  --labels examples/toy_dataset/labels.csv `
  --format csv `
  --output examples/output `
  --no-semantic
```

Result summary:

```text
Total: 9
Anomalies: 6
Scored: 3
Semantic available: 0 (intentionally disabled offline)
Non-finite JSON tokens: none
```

The example output includes all required CSV, JSON, HTML, split, and preview
artifacts. The preview images were visually inspected for readable captions
and non-overlapping text.

## Real CLIP and CIFAR-10 Validation

Real validation used `openai/clip-vit-base-patch32`, 40 CIFAR-10 images for
CPU/GPU checks, and the complete 10,000-image CIFAR-10 test split.

```text
CPU first run: 62.75 s; 38 real embeddings; 0 failures
CPU cache hit: 38 hits; 0 misses; model not reloaded
GPU batch-32 first run: 52.44 s; peak VRAM 679.39 MiB allocated / 780 MiB reserved
GPU cache hit: 38 hits; 0 misses; model not reloaded
Partial-cache resume: 10 hits, 28 misses, 0 failures
CIFAR-10 full run: 10,000 images, 9,626 scored, 794 anomalies, 0 CLIP failures
CIFAR-10 full run: 282.12 s total; peak VRAM 773.45 MiB allocated / 914 MiB reserved
CIFAR-10 full-cache rerun: 9,206 hits, 0 misses, model not reloaded
```

Records and required artifacts:

- `validation/real_clip/`
- `validation/cifar10_test/`
- `validation/before_after.md`
- `validation/anomaly_audit/`
- `validation/experiment_manifest.json`

## Packaging

`setuptools.build_meta.build_wheel` parsed `pyproject.toml`, built the package
tree, included `default.yaml`, generated entry-point metadata, and staged the
wheel contents. The sandbox then denied writing the final temporary `.whl`
file under `.wheelhouse`, so final wheel creation could not be completed in
this environment. This is a filesystem permission limitation rather than a
package configuration error.

## Known Environment Limits

- The real model run used Transformers `5.20.1`, PyTorch `2.14.1+cu130`, CUDA
  on one RTX 3060 Laptop GPU, and Python 3.12.4.
- CPU and GPU first-run totals include model/device initialization; extraction
  seconds are recorded separately.
- CIFAR-10 images are 32x32; validation set `anomaly.min_image_size=32`.
- Near-duplicate detection was disabled for the 10,000-image validation run
  to avoid quadratic comparison. This is recorded in the config.
- Python 3.11 was unavailable; runtime verification used Python 3.12.7.
- Residual `.pytest_tmp`, `.wheel_tmp`, and `.wheelhouse` sandbox-temporary
  subdirectories have ACL-denied child files and are ignored by `.gitignore`;
  they contain no deliverable source.
