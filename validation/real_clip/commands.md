# Reproduction Commands

## Prepare CIFAR-10 from the Hugging Face Parquet Mirror

```powershell
curl.exe -L --ssl-no-revoke --fail --retry 3 `
  --output validation/.work/cifar10_test.parquet `
  "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"

python scripts/prepare_cifar10.py `
  --parquet validation/.work/cifar10_test.parquet `
  --output validation/.work/cifar10_test `
  --split test `
  --limit 10000
```

## Download Local CLIP Model

```powershell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('openai/clip-vit-base-patch32', local_dir='validation/.work/clip_model'))"
```

## CPU and GPU

```powershell
python scripts/analyze_dataset.py validation/.work/cifar10_test `
  --labels validation/.work/clip_small_labels.csv --format csv `
  --output validation/real_clip_cpu --config validation/config_clip_cpu.yaml `
  --semantic-backend clip --clip-model validation/.work/clip_model --device cpu

python scripts/analyze_dataset.py validation/.work/cifar10_test `
  --labels validation/.work/clip_small_labels.csv --format csv `
  --output validation/real_clip_gpu --config validation/config_clip_gpu.yaml `
  --semantic-backend clip --clip-model validation/.work/clip_model --device cuda
```

## Full CIFAR-10

```powershell
python scripts/analyze_dataset.py validation/.work/cifar10_test `
  --labels validation/.work/cifar10_test/labels.csv --format csv `
  --output validation/cifar10_test --config validation/config_cifar10_test.yaml `
  --semantic-backend clip --clip-model validation/.work/clip_model --device cuda
```

## V1/V2 Comparison

```powershell
python scripts/compare_v1_v2.py `
  --after-csv validation/cifar10_test/samples.csv `
  --output validation/before_after
```
