from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataset_difficulty.config import load_config
from dataset_difficulty.semantics import compute_semantic_metrics


def _embedding_fixture() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    samples = pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(6)],
            "label": ["cat", "cat", "cat", "dog", "dog", "dog"],
        }
    )
    base = np.asarray(
        [
            [1.0, 0.0],
            [0.98, 0.2],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.2, 0.98],
            [0.1, 0.995],
        ],
        dtype=np.float64,
    )
    augmented = np.repeat(base[:, None, :], 2, axis=1)
    augmented[2, 0] = np.asarray([1.0, 0.0])
    augmented[2, 1] = np.asarray([0.7, 0.7])
    return samples, base, augmented


def test_semantic_metrics_preserve_components_and_detect_instability() -> None:
    samples, base, augmented = _embedding_fixture()

    result = compute_semantic_metrics(samples, base, augmented, load_config())

    assert result["semantic_available"].all()
    assert result.loc[2, "class_centroid_distance"] > result.loc[0, "class_centroid_distance"]
    assert result.loc[2, "class_confusion_risk"] > result.loc[0, "class_confusion_risk"]
    assert result.loc[2, "augmentation_instability"] > result.loc[0, "augmentation_instability"]
    assert result["semantic_complexity"].notna().all()
    assert {
        "outlier_component",
        "confusion_component",
        "instability_component",
    }.issubset(result.columns)


def test_semantic_metrics_leave_unavailable_embedding_null() -> None:
    samples = pd.DataFrame({"sample_id": ["a", "b"], "label": ["cat", "dog"]})
    base = np.asarray([[1.0, 0.0], [np.nan, np.nan]])
    augmented = np.full((2, 1, 2), np.nan)

    result = compute_semantic_metrics(samples, base, augmented, load_config())

    assert result["semantic_embedding_available"].tolist() == [True, False]
    assert result["semantic_available"].tolist() == [False, False]
    assert np.isnan(result.loc[0, "semantic_complexity"])
    assert np.isnan(result.loc[1, "semantic_complexity"])
    assert np.isnan(result.loc[1, "class_centroid_distance"])


def test_singleton_class_does_not_fabricate_intra_class_distance() -> None:
    samples = pd.DataFrame({"sample_id": ["a"], "label": ["cat"]})
    base = np.asarray([[1.0, 0.0]])
    augmented = np.asarray([[[1.0, 0.0]]])

    result = compute_semantic_metrics(samples, base, augmented, load_config())

    assert np.isnan(result.loc[0, "class_centroid_distance"])
    assert np.isnan(result.loc[0, "outlier_component"])


def test_embedding_available_does_not_imply_usable_semantic_complexity() -> None:
    samples = pd.DataFrame({"sample_id": ["a"], "label": ["cat"]})
    base = np.asarray([[1.0, 0.0]])
    augmented = np.array([[[np.nan, np.nan]]])

    result = compute_semantic_metrics(samples, base, augmented, load_config())

    assert result.loc[0, "semantic_embedding_available"]
    assert not result.loc[0, "semantic_available"]
    assert np.isnan(result.loc[0, "semantic_complexity"])
