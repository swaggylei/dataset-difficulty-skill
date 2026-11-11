from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from dataset_difficulty.anomalies import detect_anomalies
from dataset_difficulty.features import (
    ClipFeatureExtractor,
    FeatureCache,
    FeatureExtractor,
    NullFeatureCache,
    extract_features,
)
from dataset_difficulty.loading import load_dataset
from dataset_difficulty.quality import analyze_image_quality
from dataset_difficulty.reports import write_reports
from dataset_difficulty.scoring import score_difficulty
from dataset_difficulty.semantics import compute_semantic_metrics
from dataset_difficulty.splitting import assign_splits


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AnalysisResult:
    samples: pd.DataFrame
    summary: dict[str, Any]
    split_summary: dict[str, Any]
    config: dict[str, Any]
    warnings: list[str]
    input_root: Path
    output_dir: Path
    label_file: Path | None
    semantic_backend: str


def _is_within(path: Path, parent: Path) -> bool:
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    return (
        resolved_path == resolved_parent
        or resolved_parent in resolved_path.parents
    )


def _quality_table(samples: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "decode_ok",
                "error",
                "width",
                "height",
                "blur_laplacian_var",
                "brightness",
                "contrast",
                "spectral_entropy",
                "high_freq_energy_ratio",
                "image_entropy",
                "file_sha256",
                "file_size",
                "average_hash",
                "quality_issue_flags",
            ]
        )
    rows: list[dict[str, Any]] = []
    total = len(samples)
    for position, (_, sample) in enumerate(samples.iterrows(), start=1):
        metrics = analyze_image_quality(Path(sample["path"]), config)
        metrics["sample_id"] = sample["sample_id"]
        rows.append(metrics)
        if total >= 20 and position % max(1, total // 10) == 0:
            LOGGER.info("Quality analysis %d/%d", position, total)
    return pd.DataFrame(rows)


def _empty_semantic(samples: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": samples["sample_id"].to_numpy(),
            "semantic_available": False,
            "class_centroid_distance": np.nan,
            "class_confusion_risk": np.nan,
            "cross_class_margin": np.nan,
            "augmentation_stability": np.nan,
            "augmentation_instability": np.nan,
            "outlier_component": np.nan,
            "confusion_component": np.nan,
            "instability_component": np.nan,
            "semantic_complexity": np.nan,
            "semantic_component_coverage": 0.0,
        }
    )


def _preflight_extractor(
    extractor: FeatureExtractor,
    samples: pd.DataFrame,
) -> None:
    last_image_error: Exception | None = None
    for _, sample in samples.iterrows():
        path = Path(sample["path"])
        if not path.exists():
            continue
        try:
            with Image.open(path) as image:
                image = image.convert("RGB").copy()
        except Exception as error:
            last_image_error = error
            continue
        extractor.encode_images([image])
        return
    raise RuntimeError(
        "No readable image is available for semantic preflight"
    ) from last_image_error


def _cache_covers_samples(
    cache: Any,
    samples: pd.DataFrame,
    augmentations: int,
) -> bool:
    if samples.empty:
        return True
    for _, sample in samples.iterrows():
        content_hash = str(sample.get("file_sha256", "") or "")
        if not content_hash:
            return False
        cached = cache.load(content_hash)
        if cached is None:
            return False
        _, augmented = cached
        if augmented.shape[0] != augmentations:
            return False
    return True


def _semantic_analysis(
    samples: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    extractor: FeatureExtractor | None,
    semantic_backend: str,
) -> tuple[pd.DataFrame, str, list[str]]:
    semantic_config = config.get("semantic", {})
    mode = semantic_backend.lower()
    if mode == "disabled" or not bool(semantic_config.get("enabled", True)):
        return (
            _empty_semantic(samples),
            "disabled",
            ["Semantic metrics are disabled; semantic fields remain null."],
        )

    warnings: list[str] = []
    clip_config = semantic_config.get("clip", {})
    anomaly_mask = samples.get(
        "is_anomaly", pd.Series(False, index=samples.index)
    ).fillna(False).astype(bool)
    eligible_samples = samples[~anomaly_mask].reset_index(drop=True)
    if eligible_samples.empty:
        return (
            _empty_semantic(samples),
            "unavailable",
            ["Semantic metrics are unavailable because no non-anomalous samples remain."],
        )

    active_extractor = extractor
    if active_extractor is None:
        active_extractor = ClipFeatureExtractor(
            model_name=str(
                clip_config.get("model_name", "openai/clip-vit-base-patch32")
            ),
            device=str(clip_config.get("device", "auto")),
            allow_download=bool(clip_config.get("allow_download", False)),
            batch_size=int(clip_config.get("batch_size", 16)),
        )

    cache_config = semantic_config.get("cache", {})
    cache_value = Path(
        cache_config.get("directory", ".dataset_difficulty_cache")
    )
    if not cache_value.is_absolute():
        cache_value = output_dir.parent / cache_value
    augmentations = int(clip_config.get("augmentations", 2))
    batch_size = int(clip_config.get("batch_size", 16))
    seed = int(config.get("project", {}).get("seed", 42))
    signature = (
        f"{active_extractor.signature}|augmentations={augmentations}|seed={seed}|aug-v2"
    )
    if bool(cache_config.get("enabled", True)):
        cache = FeatureCache(cache_value, signature)
    else:
        cache = NullFeatureCache(signature)
    if extractor is None and not _cache_covers_samples(
        cache, eligible_samples, augmentations
    ):
        try:
            _preflight_extractor(active_extractor, eligible_samples)
        except Exception as error:
            if mode == "clip":
                raise RuntimeError(
                    "CLIP is unavailable. Use --semantic-backend disabled for an "
                    "offline report, or explicitly allow model download."
                ) from error
            return (
                _empty_semantic(samples),
                "unavailable",
                [
                    "CLIP could not be loaded from local files; semantic fields remain "
                    f"null ({type(error).__name__}: {error})."
                ],
            )
    feature_result = extract_features(
        eligible_samples,
        extractor=active_extractor,
        cache=cache,
        augmentations=augmentations,
        seed=seed,
        batch_size=batch_size,
    )
    eligible_semantic = compute_semantic_metrics(
        eligible_samples,
        feature_result.base,
        feature_result.augmented,
        config,
    )
    semantic = _empty_semantic(samples)
    semantic_by_id = eligible_semantic.set_index("sample_id")
    for column in eligible_semantic.columns:
        if column == "sample_id":
            continue
        semantic[column] = semantic["sample_id"].map(semantic_by_id[column])
    semantic["semantic_available"] = semantic["semantic_available"].map(
        lambda value: bool(value) if pd.notna(value) else False
    ).astype(bool)
    semantic["semantic_component_coverage"] = (
        semantic["semantic_component_coverage"].fillna(0.0)
    )
    available_count = int(semantic["semantic_available"].sum())
    if available_count == 0:
        warnings.append("No sample received a usable semantic embedding.")
    elif available_count < len(eligible_samples):
        warnings.append(
            f"Semantic features are unavailable for "
            f"{len(eligible_samples) - available_count} "
            "samples."
        )
    backend_name = feature_result.backend
    return semantic, backend_name, warnings


def _count_anomaly_reasons(anomaly_types: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in anomaly_types.fillna("").astype(str):
        for reason in [item for item in value.split(";") if item]:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _build_summary(
    samples: pd.DataFrame,
    split_summary: dict[str, Any],
    warnings: list[str],
    semantic_backend: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    status_counts = samples["sample_status"].value_counts().to_dict()
    band_counts = samples["difficulty_band"].value_counts().to_dict()
    split_counts = samples["split"].value_counts().to_dict()
    class_counts = samples["label"].fillna("").replace("", "<empty>").value_counts()
    scored = samples[samples["sample_status"].eq("Scored")]
    scope_counts = scored["difficulty_scope"].value_counts().to_dict()
    scored_scopes = set(scope_counts)
    if not scored_scopes:
        difficulty_scope = "unscored"
    elif scored_scopes == {"visual_semantic"}:
        difficulty_scope = "visual_semantic"
    elif scored_scopes == {"visual_only"}:
        difficulty_scope = "visual_only"
    else:
        difficulty_scope = "mixed"
    interpretations = {
        "visual_semantic": (
            "Composite prior uses visual accessibility and CLIP semantic components."
        ),
        "visual_only": (
            "Visual-only prior: CLIP semantics were unavailable, so this score "
            "does not represent complete semantic difficulty."
        ),
        "mixed": (
            "Some scored samples include CLIP semantics and others are visual-only; "
            "consult difficulty_scope per row."
        ),
        "unscored": "No non-anomalous sample received a difficulty prior.",
    }
    return {
        "counts": {
            "total": int(len(samples)),
            "anomalies": int(samples["is_anomaly"].sum()),
            "scored": int(status_counts.get("Scored", 0)),
            "unscored": int(status_counts.get("Unscored", 0)),
            "semantic_available": int(
                samples["semantic_available"].fillna(False).sum()
            ),
        },
        "difficulty_bands": {
            str(key): int(value) for key, value in band_counts.items()
        },
        "splits": {str(key): int(value) for key, value in split_counts.items()},
        "classes": {str(key): int(value) for key, value in class_counts.items()},
        "anomaly_reasons": _count_anomaly_reasons(samples["anomaly_types"]),
        "mean_component_coverage": float(
            samples["component_coverage"].mean(skipna=True)
        ),
        "split_diagnostics": split_summary,
        "semantic_backend": semantic_backend,
        "difficulty_scope": difficulty_scope,
        "difficulty_scopes": {
            str(key): int(value) for key, value in scope_counts.items()
        },
        "score_interpretation": interpretations[difficulty_scope],
        "configuration": config,
        "warnings": warnings,
    }


def analyze_dataset(
    data_root: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    label_file: str | Path | None = None,
    extractor: FeatureExtractor | None = None,
    semantic_backend: str = "auto",
) -> AnalysisResult:
    """Run loading, quality, anomaly, semantic, scoring, split, and report stages."""

    root = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    data_config = config.get("data", {})
    cache_config = config.get("semantic", {}).get("cache", {})
    cache_value = Path(
        cache_config.get("directory", ".dataset_difficulty_cache")
    )
    if not cache_value.is_absolute():
        cache_value = output.parent / cache_value
    resolved_label_file = Path(label_file).resolve() if label_file else None
    if _is_within(output, root):
        raise ValueError("output directory must be outside the input dataset tree")
    if resolved_label_file and output == resolved_label_file.parent.resolve():
        raise ValueError("output directory cannot overwrite the label-file directory")
    if _is_within(cache_value, root):
        raise ValueError("cache directory must be outside the input dataset tree")
    output.mkdir(parents=True, exist_ok=True)
    exclusions = [output, cache_value]
    for ignored in data_config.get("ignore_dirs", []):
        exclusions.append(root / str(ignored))

    loaded = load_dataset(
        root,
        label_file=resolved_label_file,
        data_format=str(data_config.get("format", "auto")),
        extensions=list(data_config.get("extensions", [])),
        path_column=str(data_config.get("path_column", "path")),
        label_column=str(data_config.get("label_column", "label")),
        exclude_paths=exclusions,
    )
    LOGGER.info("Loaded %d samples", len(loaded.samples))
    quality = _quality_table(loaded.samples, config)
    anomalous = detect_anomalies(loaded.samples, quality, config)
    pipeline_warnings: list[str] = []
    if (
        "near_duplicate_status" in anomalous.columns
        and anomalous["near_duplicate_status"].eq("skipped_large_dataset").any()
    ):
        pipeline_warnings.append(
            "Near-duplicate comparison was skipped because the dataset exceeds "
            "anomaly.near_duplicate.max_samples."
        )
    semantic, backend_name, semantic_warnings = _semantic_analysis(
        anomalous,
        config,
        output,
        extractor=extractor,
        semantic_backend=semantic_backend,
    )
    scored = score_difficulty(anomalous, quality, semantic, config)
    scored_rows = scored["sample_status"].eq("Scored")
    if scored_rows.any() and not scored.loc[scored_rows, "semantic_available"].any():
        semantic_warnings.append(
            "Difficulty score is visual-only because CLIP semantics are unavailable; "
            "it is not a complete semantic difficulty assessment."
        )
    split_config = config.get("split", {})
    split_result = assign_splits(
        scored,
        ratios=split_config.get(
            "ratios", {"train": 0.70, "val": 0.15, "test": 0.15}
        ),
        seed=int(config.get("project", {}).get("seed", 42)),
        exclude_anomalies=bool(split_config.get("exclude_anomalies", True)),
        ensure_val_test_when_possible=bool(
            split_config.get("ensure_val_test_when_possible", True)
        ),
    )
    warnings = [
        *loaded.warnings,
        *pipeline_warnings,
        *semantic_warnings,
        *split_result.summary["warnings"],
    ]
    summary = _build_summary(
        split_result.assignment,
        split_result.summary,
        warnings,
        backend_name,
        config,
    )
    result = AnalysisResult(
        samples=split_result.assignment,
        summary=summary,
        split_summary=split_result.summary,
        config=config,
        warnings=warnings,
        input_root=root,
        output_dir=output,
        label_file=resolved_label_file,
        semantic_backend=backend_name,
    )
    artifacts = write_reports(result, output, config)
    result.summary["artifacts"] = artifacts
    return result
