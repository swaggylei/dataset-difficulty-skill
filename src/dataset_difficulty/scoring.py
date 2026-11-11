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
        "semantic_complexity",
        "semantic_component_coverage",
    ]
    frame = _merge_missing_columns(samples.copy(), quality, quality_columns)
    frame = _merge_missing_columns(frame, semantic, semantic_columns)
    anomaly_mask = frame.get(
        "is_anomaly", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool)
    eligible = ~anomaly_mask

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
        "visual_difficulty_component",
        "visual_component_coverage",
        "label_rarity_score",
        "semantic_difficulty_component",
        "difficulty_score",
        "component_coverage",
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

    visual_weights = config.get("difficulty", {}).get("visual_weights", {})
    visual_specifications = [
        ("blur_component", float(visual_weights.get("blur", 0.30))),
        ("brightness_component", float(visual_weights.get("brightness", 0.15))),
        ("contrast_component", float(visual_weights.get("contrast", 0.15))),
        (
            "frequency_complexity_component",
            float(visual_weights.get("frequency", 0.15)),
        ),
        ("entropy_component", float(visual_weights.get("entropy", 0.25))),
    ]
    visual_score, visual_coverage = _weighted_available(
        frame.loc[eligible], visual_specifications
    )
    frame.loc[eligible, "visual_difficulty_component"] = visual_score.clip(0.0, 1.0)
    frame.loc[eligible, "visual_component_coverage"] = visual_coverage

    issue_counts = frame["quality_issue_flags"].map(
        lambda value: len([item for item in str(value).split(";") if item])
    )
    frame["visual_quality_risk"] = (issue_counts / 6.0).clip(0.0, 1.0)

    labels = frame["label"].fillna("").astype(str)
    eligible_labels = labels[eligible & (labels != "")]
    class_counts = eligible_labels.value_counts()
    maximum_count = float(class_counts.max()) if not class_counts.empty else np.nan
    frame["class_count"] = labels.map(class_counts).fillna(0).astype(int)
    for index in frame.index[eligible]:
        label = labels.loc[index]
        count = frame.loc[index, "class_count"]
        if label and np.isfinite(maximum_count) and maximum_count > 0:
            frame.loc[index, "label_rarity_score"] = 1.0 - float(count) / maximum_count

    frame.loc[eligible, "semantic_difficulty_component"] = np.where(
        frame.loc[eligible, "semantic_available"],
        frame.loc[eligible, "semantic_complexity"].clip(0.0, 1.0),
        np.nan,
    )
    weights = config.get("difficulty", {}).get("weights", {})
    main_specifications = [
        ("visual_difficulty_component", float(weights.get("visual", 0.20))),
        ("semantic_difficulty_component", float(weights.get("semantic", 0.60))),
        ("label_rarity_score", float(weights.get("label", 0.20))),
    ]
    difficulty, coverage = _weighted_available(
        frame.loc[eligible], main_specifications
    )
    frame.loc[eligible, "difficulty_score"] = difficulty.clip(0.0, 1.0)
    frame.loc[eligible, "component_coverage"] = coverage
    frame.loc[anomaly_mask, "component_coverage"] = 0.0

    frame["difficulty_band"] = _assign_bands(frame["difficulty_score"], config)
    frame.loc[anomaly_mask, "difficulty_band"] = "Anomaly"
    frame["sample_status"] = np.select(
        [
            anomaly_mask,
            frame["difficulty_score"].notna(),
        ],
        ["Anomaly", "Scored"],
        default="Unscored",
    )
    return frame
