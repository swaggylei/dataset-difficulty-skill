from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd


METRIC_COLUMNS = (
    "blur_laplacian_var",
    "brightness",
    "contrast",
    "spectral_entropy",
    "image_entropy",
)


def _robust_z(values: pd.Series, threshold: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    deviations = (numeric - median).abs()
    mad = float(deviations.median())
    if not np.isfinite(mad) or mad <= np.finfo(float).eps:
        return pd.Series(False, index=values.index)
    z_values = 0.6745 * (numeric - median) / mad
    return z_values.abs() > threshold


def _statistical_outliers(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[set[Any], dict[Any, list[str]]]:
    anomaly_config = config.get("anomaly", {})
    minimum_group_size = int(anomaly_config.get("robust_min_group_size", 8))
    threshold = float(anomaly_config.get("robust_z_threshold", 3.5))
    outliers: set[Any] = set()
    metric_hits: dict[Any, list[str]] = defaultdict(list)

    valid = frame[frame["decode_ok"].fillna(False).astype(bool)]
    for _, group in valid.groupby("label", dropna=False):
        if len(group) < minimum_group_size:
            continue
        for metric in METRIC_COLUMNS:
            if metric not in group:
                continue
            mask = _robust_z(group[metric], threshold)
            for sample_id in group.loc[mask, "sample_id"]:
                outliers.add(sample_id)
                metric_hits[sample_id].append(metric)
    return outliers, metric_hits


def _hamming_hex(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def _near_duplicate_groups(
    frame: pd.DataFrame,
    threshold: int,
    max_samples: int,
) -> list[list[int]]:
    candidates = frame[
        frame["decode_ok"].fillna(False).astype(bool)
        & frame["average_hash"].fillna("").astype(str).ne("")
    ]
    if len(candidates) < 2 or len(candidates) > max_samples:
        return []

    representatives: dict[str, int] = {}
    for index, row in candidates.iterrows():
        content_hash = str(row.get("file_sha256", "") or "")
        representatives.setdefault(content_hash or f"row:{index}", int(index))
    indices = list(representatives.values())
    parent = {index: index for index in indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for position, first_index in enumerate(indices):
        first_hash = str(frame.at[first_index, "average_hash"])
        for second_index in indices[position + 1 :]:
            second_hash = str(frame.at[second_index, "average_hash"])
            if _hamming_hex(first_hash, second_hash) <= threshold:
                union(first_index, second_index)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        clusters[find(index)].append(index)
    return [members for members in clusters.values() if len(members) > 1]


def _ordered_reasons(reasons: Iterable[str]) -> str:
    order = {
        "corrupt_image": 0,
        "empty_label": 1,
        "conflicting_label": 2,
        "exact_duplicate": 3,
        "near_duplicate": 4,
        "insufficient_resolution": 5,
        "quality_outlier": 6,
        "statistical_outlier": 7,
    }
    return ";".join(sorted(set(reasons), key=lambda value: (order.get(value, 99), value)))


def detect_anomalies(
    samples: pd.DataFrame,
    quality: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Combine integrity, quality, duplicate, and statistical anomaly signals."""

    if samples.empty:
        frame = samples.copy()
        frame["anomaly_types"] = pd.Series(dtype="object")
        frame["is_anomaly"] = pd.Series(dtype=bool)
        frame["duplicate_group"] = pd.Series(dtype="object")
        frame["is_duplicate_canonical"] = pd.Series(dtype=bool)
        frame["statistical_outlier_metrics"] = pd.Series(dtype="object")
        frame["near_duplicate_status"] = pd.Series(dtype="object")
        return frame

    quality_columns = [
        column
        for column in quality.columns
        if column not in samples.columns or column == "sample_id"
    ]
    frame = samples.merge(quality[quality_columns], on="sample_id", how="left")
    reasons: dict[Any, set[str]] = {
        sample_id: set() for sample_id in frame["sample_id"].tolist()
    }

    for _, row in frame.iterrows():
        sample_id = row["sample_id"]
        if bool(row.get("label_conflict", False)):
            reasons[sample_id].add("conflicting_label")
        elif not str(row.get("label", "") or "").strip():
            reasons[sample_id].add("empty_label")
        if not bool(row.get("decode_ok", False)):
            reasons[sample_id].add("corrupt_image")
            continue

        minimum_size = int(config.get("anomaly", {}).get("min_image_size", 64))
        width = row.get("width")
        height = row.get("height")
        if (
            pd.isna(width)
            or pd.isna(height)
            or int(width) < minimum_size
            or int(height) < minimum_size
        ):
            reasons[sample_id].add("insufficient_resolution")
        if str(row.get("quality_issue_flags", "") or ""):
            reasons[sample_id].add("quality_outlier")

    statistical, metric_hits = _statistical_outliers(frame, config)
    for sample_id in statistical:
        reasons[sample_id].add("statistical_outlier")

    exact_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in frame.iterrows():
        if not bool(row.get("decode_ok", False)):
            continue
        content_hash = str(row.get("file_sha256", "") or "")
        if content_hash:
            exact_groups[content_hash].append(index)
    exact_groups_with_duplicates = {
        content_hash: indices
        for content_hash, indices in exact_groups.items()
        if len(indices) > 1
    }

    duplicate_group: dict[int, str] = {}
    canonical_indices: set[int] = set()
    for content_hash, indices in exact_groups_with_duplicates.items():
        group_name = f"exact:{content_hash[:12]}"
        ordered = sorted(indices, key=lambda index: frame.at[index, "rel_path"])
        canonical_indices.add(ordered[0])
        for index in indices:
            duplicate_group[index] = group_name
            reasons[frame.at[index, "sample_id"]].add("exact_duplicate")

    near_config = config.get("anomaly", {}).get("near_duplicate", {})
    near_status = "disabled"
    if bool(near_config.get("enabled", True)):
        near_threshold = int(near_config.get("hamming_threshold", 6))
        max_samples = int(near_config.get("max_samples", 5000))
        near_candidates = frame[
            frame["decode_ok"].fillna(False).astype(bool)
            & frame["average_hash"].fillna("").astype(str).ne("")
        ]
        if len(near_candidates) > max_samples:
            near_status = "skipped_large_dataset"
            clusters = []
        else:
            near_status = "enabled"
            clusters = _near_duplicate_groups(frame, near_threshold, max_samples)
        for cluster_number, cluster in enumerate(clusters, start=1):
            group_name = f"near:{cluster_number:04d}"
            for representative_index in cluster:
                content_hash = str(frame.at[representative_index, "file_sha256"] or "")
                member_indices = exact_groups.get(content_hash, [representative_index])
                for index in member_indices:
                    duplicate_group.setdefault(index, group_name)
                    reasons[frame.at[index, "sample_id"]].add("near_duplicate")

    anomaly_types = [
        _ordered_reasons(reasons[sample_id]) for sample_id in frame["sample_id"]
    ]
    statistical_metrics = [
        ";".join(metric_hits.get(sample_id, [])) for sample_id in frame["sample_id"]
    ]
    frame["anomaly_types"] = anomaly_types
    frame["is_anomaly"] = [bool(value) for value in anomaly_types]
    frame["duplicate_group"] = [
        duplicate_group.get(index, "") for index in frame.index
    ]
    frame["is_duplicate_canonical"] = [
        index in canonical_indices for index in frame.index
    ]
    frame["statistical_outlier_metrics"] = statistical_metrics
    frame["near_duplicate_status"] = near_status
    return frame
