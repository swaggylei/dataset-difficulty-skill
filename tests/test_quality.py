from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset_difficulty.config import load_config
from dataset_difficulty.quality import analyze_image_quality

from conftest import write_image


def test_quality_metrics_separate_sharp_and_blurred_images(
    tmp_path: Path, sample_array: np.ndarray
) -> None:
    cv2 = pytest.importorskip("cv2")
    sharp_path = tmp_path / "sharp.png"
    blurred_path = tmp_path / "blurred.png"
    write_image(sharp_path, sample_array)
    write_image(blurred_path, cv2.GaussianBlur(sample_array, (21, 21), 8))

    sharp = analyze_image_quality(sharp_path, load_config())
    blurred = analyze_image_quality(blurred_path, load_config())

    assert sharp["decode_ok"] is True
    assert sharp["width"] == 96
    assert sharp["height"] == 96
    assert sharp["blur_laplacian_var"] > blurred["blur_laplacian_var"]
    assert 0.0 <= sharp["brightness"] <= 1.0
    assert 0.0 <= sharp["contrast"] <= 1.0
    assert 0.0 <= sharp["spectral_entropy"] <= 1.0
    assert 0.0 <= sharp["high_freq_energy_ratio"] <= 1.0
    assert 0.0 <= sharp["image_entropy"] <= 8.0
    assert sharp["file_sha256"] != blurred["file_sha256"]


def test_quality_flags_dark_low_contrast_image(tmp_path: Path) -> None:
    image = np.full((96, 96, 3), 10, dtype=np.uint8)
    path = tmp_path / "dark.png"
    write_image(path, image)

    result = analyze_image_quality(path, load_config())

    flags = set(result["quality_issue_flags"].split(";"))
    assert "dark_image" in flags
    assert "low_contrast" in flags
    assert "low_image_entropy" in flags


def test_corrupt_image_keeps_unavailable_metrics_empty(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")

    result = analyze_image_quality(path, load_config())

    assert result["decode_ok"] is False
    assert result["width"] is None
    assert result["height"] is None
    assert result["blur_laplacian_var"] is None
    assert result["error"]


def test_frequency_complexity_distinguishes_flat_and_textured_images(
    tmp_path: Path,
) -> None:
    flat_path = tmp_path / "flat.png"
    texture = np.indices((96, 96)).sum(axis=0).astype(np.uint8) * 8
    texture_path = tmp_path / "texture.png"
    write_image(flat_path, np.full((96, 96, 3), 127, dtype=np.uint8))
    write_image(texture_path, np.repeat(texture[:, :, None], 3, axis=2))

    flat = analyze_image_quality(flat_path, load_config())
    textured = analyze_image_quality(texture_path, load_config())

    assert textured["spectral_entropy"] > flat["spectral_entropy"]
    assert textured["high_freq_energy_ratio"] > flat["high_freq_energy_ratio"]


def test_unreadable_file_digest_does_not_abort_quality_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    path = tmp_path / "restricted.png"
    write_image(path, image)
    original_open = Path.open

    def restricted_open(self, *args, **kwargs):
        if self.resolve() == path.resolve():
            raise PermissionError("restricted")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", restricted_open)

    result = analyze_image_quality(path, load_config())

    assert result["decode_ok"] is True
    assert result["file_sha256"] is None
