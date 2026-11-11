from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from dataset_difficulty.features import (
    ClipFeatureExtractor,
    FeatureCache,
    NullFeatureCache,
    clip_checkpoint_kwargs,
    extract_features,
)

from conftest import write_image


class FakeExtractor:
    signature = "fake-color-encoder-v1"

    def __init__(self) -> None:
        self.calls = 0

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        self.calls += len(images)
        vectors = []
        for image in images:
            array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
            red, green, blue = array.mean(axis=(0, 1))
            vectors.append([red, green, blue, array.mean(), 0.1])
        return np.asarray(vectors, dtype=np.float64)


def _samples(paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(len(paths))],
            "path": paths,
            "rel_path": [path.name for path in paths],
            "label": ["cat"] * len(paths),
        }
    )


def test_feature_extraction_normalizes_and_resumes_from_cache(
    tmp_path: Path,
) -> None:
    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    write_image(red, np.full((32, 32, 3), (0, 0, 255), dtype=np.uint8))
    write_image(blue, np.full((32, 32, 3), (255, 0, 0), dtype=np.uint8))
    samples = _samples([red, blue])
    cache = FeatureCache(tmp_path / "cache", "fake-v1")

    first_extractor = FakeExtractor()
    first = extract_features(
        samples,
        extractor=first_extractor,
        cache=cache,
        augmentations=2,
        seed=7,
    )
    second_extractor = FakeExtractor()
    second = extract_features(
        samples,
        extractor=second_extractor,
        cache=cache,
        augmentations=2,
        seed=7,
    )

    assert first.base.shape == (2, 5)
    assert first.augmented.shape == (2, 2, 5)
    assert first.available.all()
    assert np.allclose(np.linalg.norm(first.base, axis=1), 1.0)
    assert not first.cache_hits.any()
    assert second.cache_hits.all()
    assert first_extractor.calls > 0
    assert second_extractor.calls == 0
    assert list(cache.root.rglob("*.npz"))


def test_feature_extraction_keeps_failed_image_rows_nan(
    tmp_path: Path,
    sample_array: np.ndarray,
) -> None:
    good = tmp_path / "good.png"
    missing = tmp_path / "missing.png"
    write_image(good, sample_array)
    result = extract_features(
        _samples([good, missing]),
        extractor=FakeExtractor(),
        cache=FeatureCache(tmp_path / "cache", "fake-v1"),
        augmentations=1,
        seed=7,
    )

    assert result.available.tolist() == [True, False]
    assert np.isnan(result.base[1]).all()
    assert np.isnan(result.augmented[1]).all()
    assert result.errors[0] == ""
    assert result.errors[1]


def test_clip_extractor_defaults_to_local_model_resolution(tmp_path: Path) -> None:
    extractor = ClipFeatureExtractor(
        model_name=str(tmp_path / "local-clip"),
        device="cpu",
        allow_download=False,
        batch_size=2,
    )

    assert extractor.allow_download is False
    assert "local-clip" in extractor.signature


def test_null_feature_cache_disables_reuse(tmp_path: Path) -> None:
    image = tmp_path / "texture.png"
    write_image(image, np.arange(48 * 48 * 3).reshape(48, 48, 3).astype(np.uint8))
    samples = _samples([image])
    cache = NullFeatureCache("fake-v1")
    extractor = FakeExtractor()

    first = extract_features(samples, extractor, cache, augmentations=1, seed=1)
    second = extract_features(samples, extractor, cache, augmentations=1, seed=1)

    assert not first.cache_hits.any()
    assert not second.cache_hits.any()
    assert extractor.calls == 4


def test_malformed_cache_entry_is_ignored_and_recomputed(tmp_path: Path) -> None:
    image = tmp_path / "texture.png"
    write_image(image, np.arange(48 * 48 * 3).reshape(48, 48, 3).astype(np.uint8))
    content_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    cache = FeatureCache(tmp_path / "cache", "fake-v1")
    cache.path_for(content_hash).write_bytes(b"not-an-npz")
    extractor = FakeExtractor()

    result = extract_features(
        _samples([image]),
        extractor,
        cache,
        augmentations=1,
        seed=1,
    )

    assert not result.cache_hits.any()
    assert result.available.all()
    assert extractor.calls > 0


def test_zero_augmentations_keeps_stability_unavailable(tmp_path: Path) -> None:
    image = tmp_path / "texture.png"
    write_image(image, np.arange(48 * 48 * 3).reshape(48, 48, 3).astype(np.uint8))

    result = extract_features(
        _samples([image]),
        FakeExtractor(),
        NullFeatureCache("fake-v1"),
        augmentations=0,
        seed=1,
    )

    assert result.base.shape == (1, 5)
    assert result.augmented.shape == (1, 0, 5)
    assert result.available.all()


def test_seed_changes_deterministic_augmentation_views(tmp_path: Path) -> None:
    image = tmp_path / "texture.png"
    gradient = np.tile(np.arange(48, dtype=np.uint8)[None, :, None], (48, 1, 3))
    write_image(image, gradient)
    samples = _samples([image])

    first = extract_features(
        samples, FakeExtractor(), NullFeatureCache("fake-v1"), augmentations=1, seed=1
    )
    second = extract_features(
        samples, FakeExtractor(), NullFeatureCache("fake-v1"), augmentations=1, seed=2
    )

    assert not np.allclose(first.augmented, second.augmented)


def test_cache_load_ignores_bad_zipfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FeatureCache(tmp_path / "cache", "fake-v1")
    path = cache.path_for("abc")
    path.write_bytes(b"broken")

    def raise_bad_zip(*args, **kwargs):
        del args, kwargs
        raise zipfile.BadZipFile("broken")

    monkeypatch.setattr(np, "load", raise_bad_zip)

    assert cache.load("abc") is None


def test_clip_runtime_info_is_explicit_before_model_load(tmp_path: Path) -> None:
    extractor = ClipFeatureExtractor(
        model_name=str(tmp_path / "local-clip"),
        device="cpu",
        allow_download=False,
        batch_size=8,
    )

    info = extractor.runtime_info()

    assert info["requested_device"] == "cpu"
    assert info["device"] == "not_loaded"
    assert info["model_loaded"] is False
    assert info["batch_size"] == 8
    assert info["gpu_peak_allocated_mb"] is None


def test_clip_checkpoint_kwargs_support_pytorch_bin_local_model(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pytorch_model.bin").write_bytes(b"placeholder")

    kwargs = clip_checkpoint_kwargs(str(tmp_path))

    assert kwargs["use_safetensors"] is False


def test_clip_feature_tensor_accepts_transformers_output_objects() -> None:
    from types import SimpleNamespace

    import torch

    from dataset_difficulty.features import clip_feature_tensor

    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert clip_feature_tensor(expected) is expected
    assert clip_feature_tensor(SimpleNamespace(pooler_output=expected)) is expected
    assert clip_feature_tensor(SimpleNamespace(image_embeds=expected)) is expected
