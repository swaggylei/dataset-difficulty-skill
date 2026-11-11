from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dataset_difficulty.config import load_config
from dataset_difficulty.pipeline import AnalysisResult
from dataset_difficulty.reports import _preview_caption, write_reports

from conftest import write_image


def test_reports_write_required_artifacts_and_valid_json(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    rng = np.random.default_rng(5)
    write_image(image_a, rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    write_image(image_b, rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    samples = pd.DataFrame(
        {
            "sample_id": ["sample-a", "sample-b"],
            "path": [image_a, image_b],
            "rel_path": ["cat/a.png", "dog/b.png"],
            "label": ["cat", "dog"],
            "semantic_available": [True, False],
            "semantic_complexity": [0.8, np.nan],
            "difficulty_score": [0.9, np.nan],
            "difficulty_band": ["Hard", "Anomaly"],
            "sample_status": ["Scored", "Anomaly"],
            "is_anomaly": [False, True],
            "anomaly_types": ["", "corrupt_image"],
            "split": ["train", "excluded_anomaly"],
        }
    )
    output = tmp_path / "output"
    result = AnalysisResult(
        samples=samples,
        summary={"total_samples": 2, "diagnostic_value": float("nan")},
        split_summary={"counts": {"train": 1, "excluded_anomaly": 1}},
        config=load_config(),
        warnings=[],
        input_root=tmp_path / "dataset",
        output_dir=output,
        label_file=None,
        semantic_backend="fake",
    )

    artifacts = write_reports(result, output, result.config)

    for name in (
        "samples.csv",
        "summary.json",
        "report.html",
        "hard_samples.png",
        "anomaly_samples.png",
        "split_manifest.csv",
        "split_summary.json",
    ):
        assert (output / name).exists(), name
        assert name in artifacts
    summary_text = (output / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in summary_text
    assert "Infinity" not in summary_text
    json.loads(summary_text)
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "sample-a" in html
    assert "hard_samples.png" in html
    assert (output / "hard_samples.png").stat().st_size > 500


def test_preview_caption_wraps_and_truncates_long_identifiers() -> None:
    row = pd.Series(
        {
            "sample_id": "9e318e7d62033dbd72fb-very-long-identifier",
            "rel_path": "cat/a-very-long-file-name-that-would-overlap.png",
            "difficulty_band": "Anomaly",
            "anomaly_types": "exact_duplicate;quality_outlier;statistical_outlier",
        }
    )

    caption = _preview_caption(row)
    lines = caption.splitlines()

    assert len(lines) == 2
    assert "..." in lines[0]
    assert all(len(line) <= 30 for line in lines)


def test_preview_thumbnail_size_changes_output_dimensions(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    rng = np.random.default_rng(8)
    write_image(image, rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    samples = pd.DataFrame(
        {
            "sample_id": ["a"],
            "path": [image],
            "rel_path": ["cat/a.png"],
            "label": ["cat"],
            "semantic_available": [False],
            "semantic_complexity": [np.nan],
            "difficulty_score": [0.8],
            "difficulty_band": ["Hard"],
            "sample_status": ["Scored"],
            "is_anomaly": [False],
            "anomaly_types": [""],
            "split": ["train"],
        }
    )
    sizes = []
    for size in (80, 240):
        output = tmp_path / f"output_{size}"
        config = load_config(
            overrides={"report": {"image_thumbnail_size": size}}
        )
        result = AnalysisResult(
            samples=samples,
            summary={"total_samples": 1},
            split_summary={"counts": {"train": 1}},
            config=config,
            warnings=[],
            input_root=tmp_path / "dataset",
            output_dir=output,
            label_file=None,
            semantic_backend="disabled",
        )
        write_reports(result, output, config)
        with Image.open(output / "hard_samples.png") as preview:
            sizes.append(preview.size)

    assert sizes[1][0] > sizes[0][0]
