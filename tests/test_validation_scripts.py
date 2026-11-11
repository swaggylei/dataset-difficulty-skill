from __future__ import annotations

import json
from io import BytesIO
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_prepare_cifar_exports_balanced_images_and_labels(tmp_path: Path) -> None:
    from prepare_cifar10 import export_cifar_subset

    images = np.stack(
        [
            np.full((8, 8, 3), value, dtype=np.uint8)
            for value in (10, 20, 30, 40)
        ]
    )
    targets = [0, 0, 1, 1]
    output = tmp_path / "cifar"

    dataset = export_cifar_subset(
        images=images,
        targets=targets,
        class_names=["airplane", "automobile"],
        split="test",
        output=output,
        limit=4,
    )

    labels = pd.read_csv(output / "labels.csv")
    assert len(labels) == 4
    assert labels["label"].value_counts().to_dict() == {
        "airplane": 2,
        "automobile": 2,
    }
    assert labels["group_id"].is_unique
    assert dataset["exported_count"] == 4
    assert dataset["class_counts"] == {"airplane": 2, "automobile": 2}
    assert len(list((output / "images").rglob("*.png"))) == 4
    assert json.loads((output / "dataset_meta.json").read_text())["split"] == "test"


def test_compare_v1_v2_recomputes_scores_and_band_transitions(
    tmp_path: Path,
) -> None:
    from compare_v1_v2 import compare_analysis_frames

    sample_ids = [f"s{index}" for index in range(8)]
    frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": ["cat"] * 2 + ["dog"] * 6,
            "sample_status": [
                "Scored",
                "Scored",
                "Scored",
                "Statistically_Unusual",
                "Scored",
                "Scored",
                "Scored",
                "Scored",
            ],
            "difficulty_score": np.linspace(0.1, 0.8, 8),
            "difficulty_band": ["Easy", "Easy", "Medium", "Medium", "Medium", "Hard", "Hard", "Hard"],
            "blur_laplacian_var": [80, 70, 60, 50, 40, 30, 20, 10],
            "brightness": [0.5] * 8,
            "contrast": [0.3, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.16],
            "spectral_entropy": np.linspace(0.2, 0.9, 8),
            "high_freq_energy_ratio": np.linspace(0.1, 0.8, 8),
            "image_entropy": np.linspace(2, 8, 8),
            "semantic_complexity": np.linspace(0.1, 0.9, 8),
            "semantic_available": [True] * 8,
            "visual_contribution": np.linspace(0.02, 0.1, 8),
            "semantic_contribution": np.linspace(0.07, 0.6, 8),
            "label_rarity_contribution": [0.0] * 8,
        }
    )
    output = tmp_path / "comparison"

    report = compare_analysis_frames(frame, before=None, output=output)

    assert report["common_scored_samples"] == 8
    assert report["mean_v1_score"] > 0
    assert report["mean_v2_score"] > 0
    assert report["mean_label_contribution_v1"] > 0
    assert report["mean_label_contribution_v2"] == 0
    assert "band_transitions" in report
    assert (output / "comparison.json").exists()
    assert (output / "comparison.md").exists()


def test_prepare_cifar_exports_huggingface_parquet(tmp_path: Path) -> None:
    from prepare_cifar10 import export_cifar_parquet

    records = []
    for index, label in enumerate((0, 0, 1, 1)):
        image = Image.fromarray(
            np.full((8, 8, 3), 20 + index * 10, dtype=np.uint8)
        )
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        records.append({"img": {"bytes": buffer.getvalue()}, "label": label})
    parquet = tmp_path / "cifar.parquet"
    pd.DataFrame(records).to_parquet(parquet)
    output = tmp_path / "exported"

    metadata = export_cifar_parquet(
        parquet_path=parquet,
        output=output,
        class_names=["airplane", "automobile"],
        split="test",
        limit=4,
    )

    assert metadata["exported_count"] == 4
    labels = pd.read_csv(output / "labels.csv")
    assert labels["label"].value_counts().to_dict() == {
        "airplane": 2,
        "automobile": 2,
    }


def test_compare_v1_v2_uses_explicit_before_scores(tmp_path: Path) -> None:
    from compare_v1_v2 import compare_analysis_frames

    after = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "label": ["cat", "cat", "dog", "dog"],
            "sample_status": ["Scored"] * 4,
            "difficulty_score": [0.2, 0.4, 0.6, 0.8],
            "difficulty_band": ["Easy", "Medium", "Medium", "Hard"],
            "blur_laplacian_var": [40, 30, 20, 10],
            "brightness": [0.5] * 4,
            "contrast": [0.2] * 4,
            "spectral_entropy": [0.2, 0.4, 0.6, 0.8],
            "high_freq_energy_ratio": [0.1, 0.2, 0.3, 0.4],
            "image_entropy": [2, 3, 4, 5],
            "semantic_complexity": [0.2, 0.4, 0.6, 0.8],
            "semantic_available": [True] * 4,
            "visual_contribution": [0.06, 0.12, 0.18, 0.24],
            "semantic_contribution": [0.14, 0.28, 0.42, 0.56],
            "label_rarity_contribution": [0.0] * 4,
        }
    )
    before = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "difficulty_score": [0.9, 0.7, 0.5, 0.3],
            "difficulty_band": ["Hard", "Medium", "Medium", "Easy"],
        }
    )

    report = compare_analysis_frames(
        after,
        before=before,
        output=tmp_path / "comparison",
    )

    assert report["mean_v1_score"] == pytest.approx(0.6)
    assert report["band_transitions"]["Hard"]["Easy"] == 1
    assert report["band_transitions"]["Easy"]["Hard"] == 1
