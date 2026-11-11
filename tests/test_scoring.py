from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataset_difficulty.config import load_config
from dataset_difficulty.scoring import score_difficulty


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_ids = [f"s{index}" for index in range(6)]
    samples = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": ["cat", "cat", "dog", "dog", "dog", "dog"],
            "path": [f"/tmp/{sample_id}.png" for sample_id in sample_ids],
            "rel_path": [f"{sample_id}.png" for sample_id in sample_ids],
            "is_anomaly": [False, False, False, False, False, True],
            "anomaly_types": ["", "", "", "", "", "corrupt_image"],
        }
    )
    quality = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "blur_laplacian_var": [5.0, 10.0, 20.0, 40.0, 80.0, np.nan],
            "brightness": [0.05, 0.2, 0.4, 0.6, 0.8, np.nan],
            "contrast": [0.02, 0.08, 0.15, 0.25, 0.4, np.nan],
            "spectral_entropy": [0.1, 0.2, 0.4, 0.6, 0.8, np.nan],
            "high_freq_energy_ratio": [0.01, 0.05, 0.1, 0.2, 0.4, np.nan],
            "image_entropy": [1.0, 2.0, 3.0, 5.0, 7.0, np.nan],
            "quality_issue_flags": [
                "blurred;dark_image",
                "",
                "",
                "",
                "",
                "",
            ],
        }
    )
    semantic = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "semantic_available": [True, True, True, True, False, False],
            "semantic_complexity": [0.1, 0.2, 0.5, 0.8, np.nan, np.nan],
            "semantic_component_coverage": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        }
    )
    return samples, quality, semantic


def test_scoring_preserves_components_and_isolates_anomalies() -> None:
    samples, quality, semantic = _inputs()

    result = score_difficulty(samples, quality, semantic, load_config())

    assert result.loc[0, "blur_component"] > result.loc[4, "blur_component"]
    assert result.loc[0, "label_rarity_score"] > result.loc[2, "label_rarity_score"]
    assert {"Easy", "Medium", "Hard"}.issubset(set(result["difficulty_band"]))
    assert result.loc[5, "sample_status"] == "Anomaly"
    assert result.loc[5, "difficulty_band"] == "Anomaly"
    assert np.isnan(result.loc[5, "difficulty_score"])
    assert result.loc[0, "component_coverage"] == pytest.approx(1.0)
    assert result.loc[4, "component_coverage"] < 1.0
    assert result.loc[4, "difficulty_score"] >= 0.0
    assert np.allclose(
        quality["blur_laplacian_var"].to_numpy(dtype=float),
        result["blur_laplacian_var"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_scoring_semantic_component_drives_order_when_visual_is_equal() -> None:
    sample_ids = [f"s{index}" for index in range(4)]
    samples = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": ["cat"] * 4,
            "path": [f"/tmp/{sample_id}.png" for sample_id in sample_ids],
            "rel_path": [f"{sample_id}.png" for sample_id in sample_ids],
            "is_anomaly": [False] * 4,
            "anomaly_types": [""] * 4,
        }
    )
    quality = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "blur_laplacian_var": [20.0] * 4,
            "brightness": [0.5] * 4,
            "contrast": [0.2] * 4,
            "spectral_entropy": [0.5] * 4,
            "high_freq_energy_ratio": [0.2] * 4,
            "image_entropy": [5.0] * 4,
            "quality_issue_flags": [""] * 4,
        }
    )
    semantic = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "semantic_available": [True] * 4,
            "semantic_complexity": [0.05, 0.25, 0.55, 0.90],
            "semantic_component_coverage": [1.0] * 4,
        }
    )

    result = score_difficulty(samples, quality, semantic, load_config())

    assert result["difficulty_score"].is_monotonic_increasing
    band_order = {"Easy": 0, "Medium": 1, "Hard": 2}
    numeric_bands = result["difficulty_band"].map(band_order)
    assert numeric_bands.is_monotonic_increasing
    assert result.loc[0, "difficulty_band"] == "Easy"
    assert result.loc[3, "difficulty_band"] == "Hard"


def test_anomalies_do_not_change_clean_sample_components_or_bands() -> None:
    clean_ids = [f"clean{index}" for index in range(4)]
    clean_samples = pd.DataFrame(
        {
            "sample_id": clean_ids,
            "label": ["cat"] * 4,
            "path": [f"/tmp/{sample_id}.png" for sample_id in clean_ids],
            "rel_path": [f"{sample_id}.png" for sample_id in clean_ids],
            "is_anomaly": [False] * 4,
            "anomaly_types": [""] * 4,
        }
    )
    clean_quality = pd.DataFrame(
        {
            "sample_id": clean_ids,
            "blur_laplacian_var": [20.0, 30.0, 40.0, 50.0],
            "brightness": [0.4, 0.45, 0.5, 0.55],
            "contrast": [0.1, 0.15, 0.2, 0.25],
            "spectral_entropy": [0.2, 0.3, 0.4, 0.5],
            "high_freq_energy_ratio": [0.05, 0.1, 0.15, 0.2],
            "image_entropy": [2.0, 3.0, 4.0, 5.0],
            "quality_issue_flags": [""] * 4,
        }
    )
    clean_semantic = pd.DataFrame(
        {
            "sample_id": clean_ids,
            "semantic_available": [True] * 4,
            "semantic_complexity": [0.1, 0.2, 0.3, 0.4],
            "semantic_component_coverage": [1.0] * 4,
        }
    )
    clean_result = score_difficulty(
        clean_samples, clean_quality, clean_semantic, load_config()
    )

    anomaly_id = "outlier"
    all_samples = pd.concat(
        [
            clean_samples,
            pd.DataFrame(
                {
                    "sample_id": [anomaly_id],
                    "label": ["cat"],
                    "path": ["/tmp/outlier.png"],
                    "rel_path": ["outlier.png"],
                    "is_anomaly": [True],
                    "anomaly_types": ["corrupt_image"],
                }
            ),
        ],
        ignore_index=True,
    )
    all_quality = pd.concat(
        [
            clean_quality,
            pd.DataFrame(
                {
                    "sample_id": [anomaly_id],
                    "blur_laplacian_var": [0.001],
                    "brightness": [0.001],
                    "contrast": [0.001],
                    "spectral_entropy": [0.001],
                    "high_freq_energy_ratio": [0.001],
                    "image_entropy": [0.001],
                    "quality_issue_flags": ["blurred;dark_image"],
                }
            ),
        ],
        ignore_index=True,
    )
    all_semantic = pd.concat(
        [
            clean_semantic,
            pd.DataFrame(
                {
                    "sample_id": [anomaly_id],
                    "semantic_available": [True],
                    "semantic_complexity": [0.999],
                    "semantic_component_coverage": [1.0],
                }
            ),
        ],
        ignore_index=True,
    )

    all_result = score_difficulty(all_samples, all_quality, all_semantic, load_config())
    clean_from_all = all_result[all_result["sample_id"].isin(clean_ids)].reset_index(
        drop=True
    )

    assert np.allclose(
        clean_result["difficulty_score"].to_numpy(dtype=float),
        clean_from_all["difficulty_score"].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert clean_result["difficulty_band"].tolist() == clean_from_all[
        "difficulty_band"
    ].tolist()
