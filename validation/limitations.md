# Remaining Limitations

- The score remains a training-free prior, not calibrated task difficulty or
  measured model error.
- Only one real CLIP backbone was validated:
  `openai/clip-vit-base-patch32`.
- GPU validation used one NVIDIA RTX 3060 Laptop GPU; results may differ on
  other GPUs or drivers.
- CIFAR-10 images are 32x32. Validation therefore used
  `anomaly.min_image_size=32`.
- Near-duplicate comparison was disabled for the 10,000-image run because the
  implementation is quadratic. Exact duplicates remain supported, and the
  skip is explicit in the config.
- Group-aware allocation is deterministic and leakage-safe but approximate for
  small or highly mixed groups.
- Class distributions in CIFAR-10 are balanced; stronger class-imbalance
  behavior still requires separate dataset validation.
- The Windows host required PyTorch-first import ordering because Pillow/Pandas
  imported before PyTorch triggered a CUDA DLL conflict. The CLI handles this,
  but direct library callers in a fresh process should import PyTorch first
  for semantic extraction on this host.
- Transformers 5.20.1 required explicit PyTorch `.bin` and output-object
  compatibility handling. Other versions may expose different APIs.
- Python 3.11 was unavailable on this host; automated tests ran on the Python
  3.12 environments documented in `TEST_RESULTS.md`.
- The model and CIFAR raw inputs are kept under ignored `validation/.work/`
  to avoid committing large weights and images.
