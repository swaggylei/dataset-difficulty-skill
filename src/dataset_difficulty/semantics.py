from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(
        array,
        norms,
        out=np.full_like(array, np.nan, dtype=np.float64),
        where=norms > np.finfo(float).eps,
    )


def _percentile_rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rank(pct=True, method="average", na_option="keep")


def _weighted_component(
    frame: pd.DataFrame,
    specifications: list[tuple[str, float]],
) -> tuple[pd.Series, pd.Series]:
    weighted_sum = pd.Series(0.0, index=frame.index)
    available_weight = pd.Series(0.0, index=frame.index)
    total_weight = sum(weight for _, weight in specifications)
    for column, weight in specifications:
        valid = frame[column].notna()
        weighted_sum.loc[valid] += frame.loc[valid, column] * weight
        available_weight.loc[valid] += weight
    complexity = weighted_sum / available_weight.replace(0.0, np.nan)
    coverage = available_weight / total_weight if total_weight > 0 else available_weight
    return complexity, coverage


def compute_semantic_metrics(
    samples: pd.DataFrame,
    base: np.ndarray,
    augmented: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compute CLIP embedding geometry without treating vector entropy as probability."""

    base_array = _normalize_rows(base)
    augmented_array = np.asarray(augmented, dtype=np.float64)
    if base_array.ndim != 2:
        raise ValueError("base embeddings must have shape (samples, dimensions)")
    if augmented_array.ndim != 3 or augmented_array.shape[0] != len(samples):
        raise ValueError(
            "augmented embeddings must have shape (samples, augmentations, dimensions)"
        )
    if augmented_array.shape[2] != base_array.shape[1]:
        raise ValueError("base and augmented embedding dimensions do not match")
    if len(samples) != base_array.shape[0]:
        raise ValueError("samples and embedding rows do not match")

    semantic_config = config.get("semantic", {})
    minimum_class_size = int(semantic_config.get("min_class_samples", 2))
    labels = samples["label"].fillna("").astype(str).to_numpy()
    available = np.isfinite(base_array).all(axis=1) & (labels != "")

    centroid_distances = np.full(len(samples), np.nan, dtype=np.float64)
    confusion_risk = np.full(len(samples), np.nan, dtype=np.float64)
    cross_class_margin = np.full(len(samples), np.nan, dtype=np.float64)
    centroid_by_label: dict[str, np.ndarray] = {}

    for label in sorted(set(labels[available])):
        member_indices = np.flatnonzero(available & (labels == label))
        if len(member_indices) == 0:
            continue
        centroid = base_array[member_indices].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > np.finfo(float).eps:
            centroid_by_label[label] = centroid / norm
        if len(member_indices) >= minimum_class_size and label in centroid_by_label:
            similarities = base_array[member_indices] @ centroid_by_label[label]
            centroid_distances[member_indices] = np.clip(1.0 - similarities, 0.0, 2.0)

    if len(centroid_by_label) >= 2:
        labels_with_centroid = list(centroid_by_label)
        centroid_matrix = np.vstack([centroid_by_label[label] for label in labels_with_centroid])
        for index in np.flatnonzero(available):
            label = labels[index]
            if label not in centroid_by_label:
                continue
            similarities = centroid_matrix @ base_array[index]
            own_position = labels_with_centroid.index(label)
            own_similarity = float(similarities[own_position])
            other_similarities = np.delete(similarities, own_position)
            nearest_other = float(np.max(other_similarities))
            margin = own_similarity - nearest_other
            cross_class_margin[index] = margin
            confusion_risk[index] = max(0.0, -margin)

    augmentation_stability = np.full(len(samples), np.nan, dtype=np.float64)
    augmentation_instability = np.full(len(samples), np.nan, dtype=np.float64)
    for index in range(len(samples)):
        if not available[index] or augmented_array.shape[1] == 0:
            continue
        variants = _normalize_rows(augmented_array[index])
        valid_variants = np.isfinite(variants).all(axis=1)
        if not valid_variants.any():
            continue
        similarities = variants[valid_variants] @ base_array[index]
        stability = float(np.clip(np.mean(similarities), -1.0, 1.0))
        augmentation_stability[index] = stability
        augmentation_instability[index] = float(np.clip(1.0 - stability, 0.0, 2.0))

    frame = pd.DataFrame(
        {
            "sample_id": samples["sample_id"].to_numpy(),
            "semantic_available": available,
            "class_centroid_distance": centroid_distances,
            "class_confusion_risk": confusion_risk,
            "cross_class_margin": cross_class_margin,
            "augmentation_stability": augmentation_stability,
            "augmentation_instability": augmentation_instability,
        }
    )
    frame["outlier_component"] = _percentile_rank(frame["class_centroid_distance"])
    frame["confusion_component"] = _percentile_rank(frame["class_confusion_risk"])
    frame["instability_component"] = _percentile_rank(frame["augmentation_instability"])

    semantic_weights = config.get("difficulty", {}).get("semantic_weights", {})
    specifications = [
        ("outlier_component", float(semantic_weights.get("outlier", 0.40))),
        ("confusion_component", float(semantic_weights.get("confusion", 0.40))),
        ("instability_component", float(semantic_weights.get("instability", 0.20))),
    ]
    complexity, coverage = _weighted_component(frame, specifications)
    complexity.loc[~frame["semantic_available"]] = np.nan
    coverage.loc[~frame["semantic_available"]] = 0.0
    frame["semantic_complexity"] = complexity
    frame["semantic_component_coverage"] = coverage
    return frame
