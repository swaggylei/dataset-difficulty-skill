# Real CLIP Runtime Record

All values below come from real `openai/clip-vit-base-patch32` runs on CIFAR-10
images. CPU and GPU extraction used separate caches.

## CPU

- Input: 40 images, 10 classes, 4 images per class.
- Batch size: 8.
- Augmentations: 1.
- First run: 50.13 s total; 2.03 s embedding extraction; 38 misses;
  0 failures.
- Cache-hit run: 1.91 s total; 38 hits, 0 misses; model not reloaded.
- Partial-cache resume: 10 hits, 28 misses; model loaded only for missing
  features.

## GPU

- Device: NVIDIA GeForce RTX 3060 Laptop GPU.
- First run: 49.06 s total; 0.54 s embedding extraction; 38 misses;
  0 failures.
- Peak allocated VRAM: 611.61 MiB.
- Peak reserved VRAM: 656.00 MiB.
- Cache-hit run: 2.01 s total; 38 hits, 0 misses; model not reloaded.

The first-run totals include one-time model/device initialization. Extraction
seconds isolate the actual embedding loop.

## Full CIFAR-10

- Input: 10,000 test images; 9,206 non-anomalous samples embedded.
- Batch size: 64.
- First run: 290.12 s total; quality 74.68 s; semantic 139.18 s;
  embedding extraction 89.70 s.
- Peak allocated VRAM: 773.45 MiB.
- Peak reserved VRAM: 898.00 MiB.
- Failures: 0.
- Full-cache rerun: 9,206 hits, 0 misses, model not loaded; 187.40 s total.

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
