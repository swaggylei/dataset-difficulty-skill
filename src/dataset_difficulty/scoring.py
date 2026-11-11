from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _merge_missing_columns(
    frame: pd.DataFrame,
    auxiliary: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    selected = [
        column for column in columns if column in auxiliary.columns and column not in frame.columns
    ]
    if "sample_id" not in auxiliary.columns:
        raise ValueError("Auxiliary table is missing sample_id")
    if not selected:
        return frame
    return frame.merge(auxiliary[["sample_id", *selected]], on="sample_id", how="left")


def _percentile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(pct=True, method="average", na_option="keep")


def _inverse_percentile(series: pd.Series) -> pd.Series:
    ranked = _percentile(series)
    return 1.0 - ranked


def _weighted_available(
    frame: pd.DataFrame,
    components: list[tuple[str, float]],
) -> tuple[pd.Series, pd.Series]:
    weighted_sum = pd.Series(0.0, index=frame.index)
    available_weight = pd.Series(0.0, index=frame.index)
    for column, weight in components:
        valid = frame[column].notna()
        weighted_sum.loc[valid] += frame.loc[valid, column] * weight
        available_weight.loc[valid] += weight
    score = weighted_sum / available_weight.replace(0.0, np.nan)
    total_weight = sum(weight for _, weight in components)
    coverage = (
        available_weight / total_weight if total_weight > 0 else available_weight
    )
    return score, coverage


def _component_contributions(
    frame: pd.DataFrame,
    components: list[tuple[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    effective = pd.DataFrame(index=frame.index, dtype=float)
    contributions = pd.DataFrame(index=frame.index, dtype=float)
    available_weight = pd.Series(0.0, index=frame.index)
    total_weight = sum(weight for _, weight in components)
    for column, weight in components:
        available_weight.loc[frame[column].notna()] += weight
        effective[column] = 0.0
        contributions[column] = 0.0
    for column, weight in components:
        valid = frame[column].notna() & (available_weight > 0)
        effective.loc[valid, column] = weight / available_weight.loc[valid]
        contributions.loc[valid, column] = (
            frame.loc[valid, column] * effective.loc[valid, column]
        )
    coverage = (
        available_weight / total_weight if total_weight > 0 else available_weight
    )
    return effective, contributions, coverage


def _assign_bands(
    scores: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    band_config = config.get("difficulty", {}).get("bands", {})
    quantiles = band_config.get("quantiles", [0.33, 0.67])
    labels = band_config.get("labels", ["Easy", "Medium", "Hard"])
    if len(quantiles) != 2 or len(labels) != 3:
        raise ValueError("Difficulty bands require two quantiles and three labels")
    valid = scores.dropna()
    bands = pd.Series("Unscored", index=scores.index, dtype="object")
    if valid.empty:
        return bands
    first_threshold = float(valid.quantile(float(quantiles[0])))
    second_threshold = float(valid.quantile(float(quantiles[1])))
    bands.loc[valid.index] = np.select(
        [
            valid <= first_threshold,
            valid <= second_threshold,
        ],
        [labels[0], labels[1]],
        default=labels[2],
    )
    return bands


def score_difficulty(
    samples: pd.DataFrame,
    quality: pd.DataFrame,
    semantic: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build transparent, dataset-relative difficulty components and bands."""

    quality_columns = [
        "blur_laplacian_var",
        "brightness",
        "contrast",
        "spectral_entropy",
        "high_freq_energy_ratio",
        "image_entropy",
        "quality_issue_flags",
    ]
    semantic_columns = [
        "semantic_available",
        "semantic_embedding_available",
        "semantic_complexity",
        "semantic_component_coverage",
        "class_centroid_distance",
        "class_confusion_risk",
        "cross_class_margin",
        "augmentation_stability",
        "augmentation_instability",
        "outlier_component",
        "confusion_component",
        "instability_component",
        "outlier_available",
        "confusion_available",
        "instability_available",
    ]
    frame = _merge_missing_columns(samples.copy(), quality, quality_columns)
    frame = _merge_missing_columns(frame, semantic, semantic_columns)
    anomaly_mask = frame.get(
        "is_anomaly", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool)
    anomaly_types = frame.get(
        "anomaly_types", pd.Series("", index=frame.index)
    ).fillna("").astype(str)
    statistical_only_mask = anomaly_mask & anomaly_types.map(
        lambda value: {item for item in value.split(";") if item}
        == {"statistical_outlier"}
    )
    include_statistical_only = bool(
        config.get("split", {}).get("include_statistical_only", True)
    )
    soft_anomaly_mask = statistical_only_mask & include_statistical_only
    excluded_mask = anomaly_mask & ~soft_anomaly_mask
    eligible = ~excluded_mask

    for metric in quality_columns[:-1]:
        if metric not in frame:
            frame[metric] = np.nan
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    if "quality_issue_flags" not in frame:
        frame["quality_issue_flags"] = ""
    frame["quality_issue_flags"] = frame["quality_issue_flags"].fillna("").astype(str)
    if "semantic_available" not in frame:
        frame["semantic_available"] = False
    frame["semantic_available"] = (
        frame["semantic_available"].fillna(False).astype(bool)
    )
    if "semantic_complexity" not in frame:
        frame["semantic_complexity"] = np.nan
    frame["semantic_complexity"] = pd.to_numeric(
        frame["semantic_complexity"], errors="coerce"
    )

    for component in (
        "blur_component",
        "brightness_component",
        "contrast_component",
        "frequency_complexity_component",
        "entropy_component",
        "visual_complexity_score",
        "visual_accessibility_risk",
        "visual_difficulty_component",
        "visual_component_coverage",
        "label_rarity_score",
        "label_rarity_weight",
        "label_rarity_contribution",
        "difficulty_visual_component",
        "difficulty_visual_weight",
        "difficulty_visual_effective_weight",
        "visual_contribution",
        "difficulty_semantic_component",
        "difficulty_semantic_weight",
        "difficulty_semantic_effective_weight",
        "semantic_contribution",
        "semantic_difficulty_component",
        "difficulty_score",
        "component_coverage",
        "excluded_from_split",
        "is_statistical_only_anomaly",
    ):
        frame[component] = np.nan

    frame.loc[eligible, "blur_component"] = _inverse_percentile(
        frame.loc[eligible, "blur_laplacian_var"]
    )
    frame.loc[eligible, "brightness_component"] = _percentile(
        (frame.loc[eligible, "brightness"] - 0.5).abs()
    )
    frame.loc[eligible, "contrast_component"] = _inverse_percentile(
        frame.loc[eligible, "contrast"]
    )
    spectral_rank = _percentile(frame.loc[eligible, "spectral_entropy"])
    frequency_rank = _percentile(frame.loc[eligible, "high_freq_energy_ratio"])
    frame.loc[eligible, "frequency_complexity_component"] = pd.concat(
        [spectral_rank, frequency_rank], axis=1
    ).mean(axis=1, skipna=True)
    frame.loc[eligible, "entropy_component"] = _percentile(
        frame.loc[eligible, "image_entropy"]
    )
    visual_complexity_weights = config.get("difficulty", {}).get(
        "visual_complexity_weights", {}
    )
    complexity_frame = frame.loc[eligible].copy()
    complexity_frame["spectral_entropy_component"] = _percentile(
        complexity_frame["spectral_entropy"]
    )
    complexity_frame["high_frequency_component"] = _percentile(
        complexity_frame["high_freq_energy_ratio"]
    )
    complexity_specifications = [
        ("entropy_component", float(visual_complexity_weights.get("image_entropy", 0.34))),
        (
            "spectral_entropy_component",
            float(visual_complexity_weights.get("spectral_entropy", 0.33)),
        ),
        (
            "high_frequency_component",
            float(visual_complexity_weights.get("high_frequency", 0.33)),
        ),
    ]
    complexity_score, _ = _weighted_available(
        complexity_frame, complexity_specifications
    )
    frame.loc[eligible, "visual_complexity_score"] = complexity_score.clip(0.0, 1.0)

    visual_weights = config.get("difficulty", {}).get("visual_weights", {})
    visual_specifications = [
        ("blur_component", float(visual_weights.get("blur", 0.50))),
        ("brightness_component", float(visual_weights.get("brightness", 0.20))),
        ("contrast_component", float(visual_weights.get("contrast", 0.30))),
    ]
    visual_score, visual_coverage = _weighted_available(
        frame.loc[eligible], visual_specifications
    )
    frame.loc[eligible, "visual_accessibility_risk"] = visual_score.clip(0.0, 1.0)
    frame.loc[eligible, "visual_difficulty_component"] = visual_score.clip(0.0, 1.0)
    frame.loc[eligible, "difficulty_visual_component"] = visual_score.clip(0.0, 1.0)
    frame.loc[eligible, "visual_component_coverage"] = visual_coverage

    issue_counts = frame["quality_issue_flags"].map(
        lambda value: len([item for item in str(value).split(";") if item])
    )
    frame["visual_quality_risk"] = (issue_counts / 6.0).clip(0.0, 1.0)

    labels = frame["label"].fillna("").astype(str)
    class_counts = labels[labels != ""].value_counts()
    maximum_count = float(class_counts.max()) if not class_counts.empty else np.nan
    frame["class_count"] = labels.map(class_counts).fillna(0).astype(int)
    for index in frame.index:
        label = labels.loc[index]
        count = frame.loc[index, "class_count"]
        if label and np.isfinite(maximum_count) and maximum_count > 0:
            frame.loc[index, "label_rarity_score"] = 1.0 - float(count) / maximum_count

    frame.loc[eligible, "difficulty_semantic_component"] = np.where(
        frame.loc[eligible, "semantic_available"],
        frame.loc[eligible, "semantic_complexity"].clip(0.0, 1.0),
        np.nan,
    )
    frame.loc[eligible, "semantic_difficulty_component"] = frame.loc[
        eligible, "difficulty_semantic_component"
    ]
    frame["label_rarity_weight"] = 0.0
    frame["label_rarity_contribution"] = 0.0
    frame["label_rarity_in_difficulty"] = False
    frame["is_statistical_only_anomaly"] = statistical_only_mask
    frame["excluded_from_split"] = excluded_mask

    weights = config.get("difficulty", {}).get("weights", {})
    visual_weight = float(weights.get("visual", 0.30))
    semantic_weight = float(weights.get("semantic", 0.70))
    main_specifications = [
        ("difficulty_visual_component", visual_weight),
        ("difficulty_semantic_component", semantic_weight),
    ]
    effective, contributions, coverage = _component_contributions(
        frame.loc[eligible], main_specifications
    )
    frame.loc[eligible, "difficulty_visual_weight"] = visual_weight
    frame.loc[eligible, "difficulty_semantic_weight"] = semantic_weight
    frame.loc[eligible, "difficulty_visual_effective_weight"] = effective[
        "difficulty_visual_component"
    ]
    frame.loc[eligible, "difficulty_semantic_effective_weight"] = effective[
        "difficulty_semantic_component"
    ]
    frame.loc[eligible, "visual_contribution"] = contributions[
        "difficulty_visual_component"
    ]
    frame.loc[eligible, "semantic_contribution"] = contributions[
        "difficulty_semantic_component"
    ]
    frame.loc[eligible, "difficulty_score"] = (
        frame.loc[eligible, "visual_contribution"]
        + frame.loc[eligible, "semantic_contribution"]
    ).clip(0.0, 1.0)
    frame.loc[eligible, "component_coverage"] = coverage
    frame.loc[excluded_mask, "component_coverage"] = 0.0

    frame.loc[eligible, "difficulty_scope"] = np.where(
        frame.loc[eligible, "difficulty_semantic_component"].notna(),
        "visual_semantic",
        np.where(
            frame.loc[eligible, "difficulty_visual_component"].notna(),
            "visual_only",
            "unscored",
        ),
    )
    frame.loc[excluded_mask, "difficulty_scope"] = "anomaly"
    frame["difficulty_band"] = _assign_bands(frame["difficulty_score"], config)
    frame.loc[excluded_mask, "difficulty_band"] = "Anomaly"
    frame["sample_status"] = np.select(
        [
            excluded_mask,
            soft_anomaly_mask,
            frame["difficulty_score"].notna(),
        ],
        ["Anomaly", "Statistically_Unusual", "Scored"],
        default="Unscored",
    )
    return frame
