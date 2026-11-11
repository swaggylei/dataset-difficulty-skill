# Real CLIP Runtime Record

All values below come from real `openai/clip-vit-base-patch32` runs on CIFAR-10
images. CPU and GPU extraction used separate caches.

## CPU

- Input: 40 images, 10 classes, 4 images per class.
- Batch size: 8.
- Augmentations: 1.
- First run: 62.75 s total; 57.60 s model preparation; 3.02 s embedding
  extraction; 38 misses; 0 failures.
- Cache-hit run: 2.29 s total; 38 hits, 0 misses; model not reloaded.
- Partial-cache resume: 10 hits, 28 misses; model loaded only for missing
  features; prepare 49.40 s, extraction 2.12 s.

## GPU

- Device: NVIDIA GeForce RTX 3060 Laptop GPU.
- Batch size: 32.
- First run: 52.44 s total; 49.69 s model preparation; 0.66 s embedding
  extraction; 38 misses; 0 failures.
- Peak allocated VRAM: 679.39 MiB.
- Peak reserved VRAM: 780.00 MiB.
- Cache-hit run: 2.29 s total; 38 hits, 0 misses; model not reloaded.

The first-run totals include one-time model/device initialization. Extraction
seconds isolate the actual embedding loop.

## Full CIFAR-10

- Input: 10,000 test images; 9,206 non-anomalous samples embedded.
- Batch size: 64.
- First run: 304.82 s total; quality 78.79 s; semantic 146.05 s;
  model preparation 50.43 s; embedding extraction 95.08 s.
- Peak allocated VRAM: 773.45 MiB.
- Peak reserved VRAM: 914.00 MiB.
- Failures: 0.
- Full-cache rerun: 9,206 hits, 0 misses, model not loaded; 189.60 s total.

## Files

- `cpu_first_run.json`
- `cpu_cache_hit.json`
- `cpu_partial_cache_resume.json`
- `gpu_first_run.json`
- `gpu_cache_hit.json`
- `cifar10_first_run.json`
- `cifar10_cache_hit.json`
- `environment.json`
- `commands.md`
