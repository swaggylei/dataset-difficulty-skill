from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _file_digest(path: Path) -> tuple[str | None, int | None]:
    if not path.exists() or not path.is_file():
        return None, None
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        return None, None
    return digest.hexdigest(), size


def _decode_image(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def _spectral_metrics(gray: np.ndarray) -> tuple[float, float]:
    centered = gray.astype(np.float64) - float(gray.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    total = float(power.sum())
    if total <= np.finfo(float).eps:
        return 0.0, 0.0
    probability = power / total
    positive = probability[probability > 0]
    entropy = float(-(positive * np.log2(positive)).sum())
    entropy /= float(np.log2(probability.size))

    height, width = gray.shape
    center_y, center_x = height // 2, width // 2
    radius = max(1.0, min(height, width) * 0.10)
    yy, xx = np.ogrid[:height, :width]
    center_mask = ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= radius**2
    high_frequency_ratio = float(power[~center_mask].sum() / total)
    return float(np.clip(entropy, 0.0, 1.0)), float(
        np.clip(high_frequency_ratio, 0.0, 1.0)
    )


def _histogram_entropy(gray: np.ndarray) -> float:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probability = histogram / max(1.0, float(gray.size))
    positive = probability[probability > 0]
    return float(-(positive * np.log2(positive)).sum())


def _average_hash(gray: np.ndarray) -> str:
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (resized >= resized.mean()).astype(np.uint8).ravel()
    packed = np.packbits(bits)
    return f"{int.from_bytes(packed.tobytes(), byteorder='big', signed=False):016x}"


def _quality_flags(metrics: dict[str, Any], config: dict[str, Any]) -> list[str]:
    quality = config.get("quality", {})
    flags: list[str] = []
    if metrics["blur_laplacian_var"] < float(quality.get("blur_laplacian_min", 35.0)):
        flags.append("blurred")
    if metrics["brightness"] < float(quality.get("brightness_min", 0.12)):
        flags.append("dark_image")
    if metrics["brightness"] > float(quality.get("brightness_max", 0.88)):
        flags.append("overexposed")
    if metrics["contrast"] < float(quality.get("contrast_min", 0.08)):
        flags.append("low_contrast")
    if metrics["image_entropy"] < float(quality.get("image_entropy_min", 2.0)):
        flags.append("low_image_entropy")
    if metrics["spectral_entropy"] < float(quality.get("spectral_entropy_min", 0.35)):
        flags.append("low_frequency_complexity")
    return flags


def analyze_image_quality(path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Extract interpretable quality and image-statistics signals."""

    image_path = Path(path)
    file_hash, file_size = _file_digest(image_path)
    image = _decode_image(image_path)
    if image is None:
        return {
            "decode_ok": False,
            "error": "Unable to decode image",
            "width": None,
            "height": None,
            "blur_laplacian_var": None,
            "brightness": None,
            "contrast": None,
            "spectral_entropy": None,
            "high_freq_energy_ratio": None,
            "image_entropy": None,
            "file_sha256": file_hash,
            "file_size": file_size,
            "average_hash": None,
            "quality_issue_flags": "",
        }

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    spectral_entropy, high_frequency_ratio = _spectral_metrics(gray)
    metrics: dict[str, Any] = {
        "decode_ok": True,
        "error": "",
        "width": int(width),
        "height": int(height),
        "blur_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "spectral_entropy": spectral_entropy,
        "high_freq_energy_ratio": high_frequency_ratio,
        "image_entropy": _histogram_entropy(gray),
        "file_sha256": file_hash,
        "file_size": file_size,
        "average_hash": _average_hash(gray),
    }
    metrics["quality_issue_flags"] = ";".join(_quality_flags(metrics, config))
    return metrics
