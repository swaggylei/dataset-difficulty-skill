from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataset_difficulty.anomalies import detect_anomalies
from dataset_difficulty.config import load_config
from dataset_difficulty.quality import analyze_image_quality

from conftest import write_image


def _samples(paths: list[Path], labels: list[str] | None = None) -> pd.DataFrame:
    labels = labels or ["cat"] * len(paths)
    return pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(len(paths))],
            "path": paths,
            "rel_path": [path.name for path in paths],
            "label": labels,
            "label_source": ["folder"] * len(paths),
            "label_conflict": [False] * len(paths),
            "conflicting_labels": [""] * len(paths),
        }
    )


def _quality(paths: list[Path]) -> pd.DataFrame:
    config = load_config()
    rows = []
    for index, path in enumerate(paths):
        row = analyze_image_quality(path, config)
        row["sample_id"] = f"s{index}"
        rows.append(row)
    return pd.DataFrame(rows)


def test_detects_corrupt_empty_conflicting_and_tiny_samples(
    tmp_path: Path, sample_array: np.ndarray
) -> None:
    good = tmp_path / "good.png"
    corrupt = tmp_path / "corrupt.png"
    tiny = tmp_path / "tiny.png"
    write_image(good, sample_array)
    corrupt.write_bytes(b"broken")
    write_image(tiny, np.zeros((16, 16, 3), dtype=np.uint8))

    paths = [good, corrupt, tiny]
    samples = _samples(paths, labels=["cat", "", ""])
    samples.loc[2, "label_conflict"] = True
    samples.loc[2, "conflicting_labels"] = "cat;dog"

    result = detect_anomalies(samples, _quality(paths), load_config())
    by_path = result.set_index("rel_path")["anomaly_types"].to_dict()

    assert "corrupt_image" in by_path["corrupt.png"]
    assert "empty_label" in by_path["corrupt.png"]
    assert "conflicting_label" in by_path["tiny.png"]
    assert "insufficient_resolution" in by_path["tiny.png"]
    assert not result.loc[result["rel_path"] == "good.png", "is_anomaly"].iloc[0]


def test_detects_exact_and_near_duplicates(tmp_path: Path) -> None:
    base = np.tile(np.arange(96, dtype=np.uint8)[None, :, None], (96, 1, 3))
    first = tmp_path / "first.png"
    exact = tmp_path / "exact.png"
    near_image = base.copy()
    near_image[0, 0, 0] = 255 - near_image[0, 0, 0]
    near = tmp_path / "near.png"
    write_image(first, base)
    exact.write_bytes(first.read_bytes())
    write_image(near, near_image)

    paths = [first, exact, near]
    config = load_config(overrides={"anomaly": {"near_duplicate": {"hamming_threshold": 0}}})
    result = detect_anomalies(_samples(paths), _quality(paths), config)
    by_path = result.set_index("rel_path")["anomaly_types"].to_dict()

    assert "exact_duplicate" in by_path["first.png"]
    assert "exact_duplicate" in by_path["exact.png"]
    assert "near_duplicate" in by_path["near.png"]
    assert result["duplicate_group"].fillna("").ne("").all()


def test_detects_robust_statistical_outlier_within_label(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    paths = []
    for index in range(9):
        image = rng.integers(30, 220, size=(64, 64, 3), dtype=np.uint8)
        path = tmp_path / f"normal_{index}.png"
        write_image(path, image)
        paths.append(path)
    outlier = tmp_path / "outlier.png"
    write_image(outlier, np.full((64, 64, 3), 2, dtype=np.uint8))
    paths.append(outlier)
    config = load_config(
        overrides={
            "anomaly": {
                "robust_min_group_size": 8,
                "near_duplicate": {"enabled": False},
            }
        }
    )

    result = detect_anomalies(_samples(paths), _quality(paths), config)
    outlier_row = result.loc[result["rel_path"] == "outlier.png"].iloc[0]

    assert outlier_row["is_anomaly"]
    assert "statistical_outlier" in outlier_row["anomaly_types"]
    assert "quality_outlier" in outlier_row["anomaly_types"]


def test_near_duplicate_large_dataset_skip_is_explicit(tmp_path: Path) -> None:
    paths = [tmp_path / f"image_{index}.png" for index in range(3)]
    samples = _samples(paths)
    quality = pd.DataFrame(
        {
            "sample_id": samples["sample_id"],
            "decode_ok": [True, True, True],
            "width": [64, 64, 64],
            "height": [64, 64, 64],
            "blur_laplacian_var": [20.0, 30.0, 40.0],
            "brightness": [0.4, 0.5, 0.6],
            "contrast": [0.2, 0.2, 0.2],
            "spectral_entropy": [0.5, 0.5, 0.5],
            "high_freq_energy_ratio": [0.2, 0.2, 0.2],
            "image_entropy": [4.0, 4.0, 4.0],
            "file_sha256": ["a", "b", "c"],
            "average_hash": ["0000000000000000", "0000000000000001", "0000000000000002"],
            "quality_issue_flags": ["", "", ""],
        }
    )
    config = load_config(
        overrides={
            "anomaly": {
                "near_duplicate": {
                    "enabled": True,
                    "max_samples": 1,
                }
            }
        }
    )

    result = detect_anomalies(samples, quality, config)

    assert result["near_duplicate_status"].tolist() == ["skipped_large_dataset"] * 3
