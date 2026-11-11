from __future__ import annotations

import hashlib
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps


class FeatureExtractor(Protocol):
    signature: str

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...


@dataclass(slots=True)
class FeatureResult:
    base: np.ndarray
    augmented: np.ndarray
    available: np.ndarray
    cache_hits: np.ndarray
    errors: list[str]
    signature: str
    backend: str


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Feature encoder must return a two-dimensional array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = np.divide(
        array,
        norms,
        out=np.full_like(array, np.nan, dtype=np.float64),
        where=norms > np.finfo(float).eps,
    )
    return normalized


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


class FeatureCache:
    """One atomic NPZ entry per image content hash and model signature."""

    def __init__(self, cache_dir: str | Path, signature: str) -> None:
        self.signature = signature
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", signature)[:64].strip("-")
        self.root = Path(cache_dir) / f"{slug or 'features'}-{digest}"
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, content_hash: str) -> Path:
        return self.root / f"{content_hash}.npz"

    def load(self, content_hash: str) -> tuple[np.ndarray, np.ndarray] | None:
        path = self.path_for(content_hash)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                base = np.asarray(payload["base"], dtype=np.float64)
                augmented = np.asarray(payload["augmented"], dtype=np.float64)
            if base.ndim != 1 or augmented.ndim != 2:
                return None
            return base, augmented
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return None

    def save(
        self,
        content_hash: str,
        base: np.ndarray,
        augmented: np.ndarray,
    ) -> None:
        target = self.path_for(content_hash)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=self.root, prefix=".tmp-", suffix=".npz"
        )
        os.close(file_descriptor)
        try:
            np.savez(
                temporary_name,
                base=np.asarray(base, dtype=np.float64),
                augmented=np.asarray(augmented, dtype=np.float64),
                signature=np.asarray(self.signature),
            )
            os.replace(temporary_name, target)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)


class NullFeatureCache:
    """No-op cache used when semantic.cache.enabled is false."""

    def __init__(self, signature: str) -> None:
        self.signature = signature

    def load(self, content_hash: str) -> None:
        del content_hash
        return None

    def save(
        self,
        content_hash: str,
        base: np.ndarray,
        augmented: np.ndarray,
    ) -> None:
        del content_hash, base, augmented


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _augment_image(image: Image.Image, index: int) -> Image.Image:
    variant = index % 4
    if variant == 0:
        return ImageOps.mirror(image)
    if variant == 1:
        return ImageEnhance.Brightness(image).enhance(0.90)
    if variant == 2:
        return ImageEnhance.Contrast(image).enhance(1.10)
    return ImageEnhance.Color(image).enhance(0.90)


def _encode_safely(
    extractor: FeatureExtractor,
    images: Sequence[Image.Image],
) -> tuple[list[np.ndarray], list[str]]:
    if not images:
        return [], []
    try:
        vectors = _normalize_rows(extractor.encode_images(images))
        if len(vectors) != len(images):
            raise ValueError("Encoder returned a row count different from its input")
        return [vectors[index] for index in range(len(images))], [""] * len(images)
    except Exception as batch_error:
        vectors: list[np.ndarray] = []
        errors: list[str] = []
        for image in images:
            try:
                single = _normalize_rows(extractor.encode_images([image]))
                vectors.append(single[0])
                errors.append("")
            except Exception as single_error:
                vectors.append(np.asarray([], dtype=np.float64))
                errors.append(f"{type(single_error).__name__}: {single_error}")
        if not any(vector.size for vector in vectors):
            message = f"{type(batch_error).__name__}: {batch_error}"
            return vectors, [error or message for error in errors]
        return vectors, errors


def extract_features(
    samples: pd.DataFrame,
    extractor: FeatureExtractor,
    cache: FeatureCache,
    augmentations: int,
    seed: int,
    batch_size: int = 16,
) -> FeatureResult:
    """Extract base and deterministic augmented embeddings with resumable caching."""

    if augmentations < 0:
        raise ValueError("augmentations must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    count = len(samples)
    signatures = str(getattr(extractor, "signature", extractor.__class__.__name__))
    base_rows: dict[int, np.ndarray] = {}
    augmented_rows: dict[int, np.ndarray] = {}
    cache_hits = np.zeros(count, dtype=bool)
    errors = [""] * count
    content_hashes: list[str | None] = []

    for position, (_, row) in enumerate(samples.iterrows()):
        stored_hash = str(row.get("file_sha256", "") or "")
        content_hash = stored_hash or _file_sha256(Path(row["path"]))
        content_hashes.append(content_hash)
        if not content_hash:
            continue
        cached = cache.load(content_hash)
        if cached is None:
            continue
        base, augmented = cached
        if augmented.shape[0] != augmentations:
            continue
        base_rows[position] = base
        augmented_rows[position] = augmented
        cache_hits[position] = True

    missing_positions = [
        position for position in range(count) if not cache_hits[position]
    ]
    for batch_start in range(0, len(missing_positions), batch_size):
        batch_positions = missing_positions[batch_start : batch_start + batch_size]
        batch_images: list[Image.Image] = []
        valid_positions: list[int] = []
        for position in batch_positions:
            path = Path(samples.iloc[position]["path"])
            try:
                image = _load_rgb_image(path)
            except Exception as error:
                errors[position] = f"{type(error).__name__}: {error}"
                continue
            batch_images.append(image)
            valid_positions.append(position)
        if not batch_images:
            continue

        base_vectors, base_errors = _encode_safely(extractor, batch_images)
        for position, vector, error in zip(
            valid_positions, base_vectors, base_errors, strict=True
        ):
            if vector.size:
                base_rows[position] = vector
            if error:
                errors[position] = error

        per_sample_augmentations: dict[int, list[np.ndarray]] = {
            position: [] for position in valid_positions
        }
        for augmentation_index in range(augmentations):
            augmented_images = [
                _augment_image(
                    image,
                    augmentation_index + int(seed) + position,
                )
                for image, position in zip(batch_images, valid_positions, strict=True)
            ]
            augmented_vectors, augmented_errors = _encode_safely(
                extractor, augmented_images
            )
            for position, vector, error in zip(
                valid_positions,
                augmented_vectors,
                augmented_errors,
                strict=True,
            ):
                per_sample_augmentations[position].append(vector)
                if error and not errors[position]:
                    errors[position] = error

        for position in valid_positions:
            if position not in base_rows:
                continue
            augmented_for_sample = per_sample_augmentations[position]
            if augmentations == 0:
                augmented_rows[position] = np.empty(
                    (0, int(base_rows[position].size)), dtype=np.float64
                )
            elif len(augmented_for_sample) == augmentations:
                augmented_rows[position] = np.vstack(augmented_for_sample)
            else:
                continue
            content_hash = content_hashes[position]
            if content_hash and np.isfinite(base_rows[position]).all():
                cache.save(
                    content_hash,
                    base_rows[position],
                    augmented_rows[position],
                )

    dimensions = [
        int(array.shape[-1])
        for array in list(base_rows.values()) + list(augmented_rows.values())
        if array.size
    ]
    dimension = max(dimensions, default=0)
    base = np.full((count, dimension), np.nan, dtype=np.float64)
    augmented = np.full(
        (count, augmentations, dimension), np.nan, dtype=np.float64
    )
    for position, vector in base_rows.items():
        if vector.size == dimension:
            base[position] = vector
    for position, array in augmented_rows.items():
        if array.shape == (augmentations, dimension):
            augmented[position] = array
    available = np.isfinite(base).all(axis=1)
    return FeatureResult(
        base=base,
        augmented=augmented,
        available=available,
        cache_hits=cache_hits,
        errors=errors,
        signature=signatures,
        backend=extractor.__class__.__name__,
    )


class ClipFeatureExtractor:
    """Lazy Transformers CLIP image encoder with local-only default resolution."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        allow_download: bool = False,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.allow_download = allow_download
        self.batch_size = batch_size
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._runtime_device: str | None = None

    @property
    def signature(self) -> str:
        mode = "download-allowed" if self.allow_download else "local-only"
        return f"clip:{self.model_name}:{mode}:v1"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPImageProcessor, CLIPModel
        except ImportError as error:
            raise RuntimeError(
                "CLIP extraction requires torch and transformers"
            ) from error

        if self.device == "auto":
            runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            runtime_device = self.device
        local_only = not self.allow_download
        self._model = CLIPModel.from_pretrained(
            self.model_name,
            local_files_only=local_only,
        ).to(runtime_device)
        self._processor = CLIPImageProcessor.from_pretrained(
            self.model_name,
            local_files_only=local_only,
        )
        self._model.eval()
        self._torch = torch
        self._runtime_device = runtime_device

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float64)
        self._load()
        outputs: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = list(images[start : start + self.batch_size])
            inputs = self._processor(images=batch, return_tensors="pt")
            inputs = {
                key: value.to(self._runtime_device) for key, value in inputs.items()
            }
            with self._torch.inference_mode():
                features = self._model.get_image_features(**inputs)
            outputs.append(features.detach().cpu().numpy().astype(np.float64))
        return _normalize_rows(np.vstack(outputs))
