# Real Validation Assets

This directory contains evidence from real Transformers CLIP and CIFAR-10
runs. It does not contain simulated embeddings.

## Data and Model

- Dataset: CIFAR-10 official test split, 10,000 images, 1,000 per class.
- Dataset mirror: Hugging Face `uoft-cs/cifar10` Parquet test file.
- Model: `openai/clip-vit-base-patch32`, downloaded to
  `validation/.work/clip_model/` (ignored by Git).
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU.
- Environment: `validation/real_clip/environment.json`.

Raw CIFAR and CLIP model files are intentionally stored under
`validation/.work/`, which is ignored. `scripts/prepare_cifar10.py` and
`validation/real_clip/commands.md` document reproduction.

## Required CIFAR-10 Artifacts

`validation/cifar10_test/` contains:

- `samples.csv`
- `summary.json`
- `report.html`
- `hard_samples.png`
- `anomaly_samples.png`
- `split_manifest.csv`
- `split_summary.json`

`summary_first_run.json` preserves the first extraction run.
`summary_cache_hit.json` preserves the full-cache rerun.

## Interpretation

The 10,000-image run found 794 anomalies and scored 9,206 samples. Hard
samples visually concentrated around low contrast, small distant objects,
blur/scale ambiguity, and class-boundary cases. The top examples are
aviation/deer/bird/dog images with low visual accessibility and elevated
semantic geometry.

No CLIP extraction failed. The complete first run took 304.8 seconds; model
preparation was 50.4 seconds, embedding extraction was 95.1 seconds, and CUDA
peak memory was 773.5 MiB allocated / 914 MiB reserved.

## Reproduction Notes

- The Toronto CIFAR URL was too slow in this environment and the first
  torchvision download failed on TLS. The Hugging Face official dataset mirror
  was used instead.
- Transformers 5.20.1 required compatibility handling for PyTorch `.bin`
  weights and `BaseModelOutputWithPooling`.
- The Windows environment requires importing PyTorch before Pillow/Pandas to
  avoid a CUDA DLL initialization conflict; the CLI now enforces that order
  for semantic modes.
