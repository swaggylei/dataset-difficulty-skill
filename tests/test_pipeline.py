from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from dataset_difficulty.config import load_config
from dataset_difficulty.pipeline import _preflight_extractor, _semantic_analysis, analyze_dataset

from conftest import write_image


class TintedFakeExtractor:
    signature = "tinted-fake-v1"

    def __init__(self) -> None:
        self.calls = 0

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        self.calls += len(images)
        vectors = []
        for image in images:
            array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
            red, green, blue = array.mean(axis=(0, 1))
            vectors.append([red, green, blue, array.std(), 0.2])
        return np.asarray(vectors)


class FailingOneExtractor(TintedFakeExtractor):
    signature = "failing-one-v1"

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if any(
            np.asarray(image.convert("RGB")).mean(axis=(0, 1))[0]
            > 1.5 * np.asarray(image.convert("RGB")).mean(axis=(0, 1))[2]
            for image in images
        ):
            raise RuntimeError("synthetic encoder failure for bright image")
        return super().encode_images(images)


def _write_classification_dataset(root: Path) -> None:
    rng = np.random.default_rng(21)
    for class_name, tint in (("cat", (1.0, 0.4, 0.3)), ("dog", (0.2, 0.4, 1.0))):
        for index in range(4):
            noise = rng.integers(20, 220, (72, 72, 3), dtype=np.uint8)
            image = np.clip(noise * np.asarray(tint), 0, 255).astype(np.uint8)
            write_image(root / class_name / f"{index}.png", image)


def test_pipeline_with_injected_features_generates_reports(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_classification_dataset(root)
    config = load_config(
        overrides={
            "anomaly": {"near_duplicate": {"enabled": False}},
            "semantic": {"min_class_samples": 2},
        }
    )
    output = tmp_path / "output"

    result = analyze_dataset(
        root,
        output,
        config,
        extractor=TintedFakeExtractor(),
        semantic_backend="fake",
    )

    assert len(result.samples) == 8
    assert result.samples["semantic_available"].any()
    assert (output / "report.html").exists()
    assert (output / "summary.json").exists()
    assert (output / "split_manifest.csv").exists()
    assert result.summary["counts"]["total"] == 8


def test_pipeline_without_semantics_keeps_metrics_null(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_classification_dataset(root)
    output = tmp_path / "output"

    result = analyze_dataset(
        root,
        output,
        load_config(),
        semantic_backend="disabled",
    )

    assert not result.samples["semantic_available"].any()
    assert result.samples["semantic_complexity"].isna().all()
    assert any("semantic" in warning.lower() for warning in result.warnings)
    assert any("visual-only" in warning.lower() for warning in result.warnings)
    assert result.summary["difficulty_scope"] == "visual_only"
    assert pd.isna(result.samples["semantic_complexity"]).all()


def test_pipeline_rejects_output_inside_input_tree(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_classification_dataset(root)

    with pytest.raises(ValueError, match="output"):
        analyze_dataset(
            root,
            root / "reports",
            load_config(),
            semantic_backend="disabled",
        )


def test_pipeline_rejects_cache_inside_input_tree(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_classification_dataset(root)
    config = load_config(
        overrides={
            "semantic": {
                "cache": {"directory": str(root / ".cache"), "enabled": True}
            }
        }
    )

    with pytest.raises(ValueError, match="cache"):
        analyze_dataset(
            root,
            tmp_path / "output",
            config,
            extractor=TintedFakeExtractor(),
            semantic_backend="fake",
        )


def test_empty_dataset_produces_valid_empty_report(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    output = tmp_path / "output"

    result = analyze_dataset(
        root,
        output,
        load_config(),
        semantic_backend="disabled",
    )

    assert result.samples.empty
    assert result.summary["counts"]["total"] == 0
    assert (output / "report.html").exists()


def test_summary_contains_resolved_configuration(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_classification_dataset(root)

    result = analyze_dataset(
        root,
        tmp_path / "output",
        load_config(),
        semantic_backend="disabled",
    )

    assert result.summary["configuration"]["difficulty"]["weights"]["semantic"] == 0.70


def test_semantic_analysis_skips_anomaly_images(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_classification_dataset(root)
    samples = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "path": [
                root / "cat" / "0.png",
                root / "cat" / "1.png",
                root / "dog" / "0.png",
            ],
            "rel_path": ["cat/0.png", "cat/1.png", "dog/0.png"],
            "label": ["cat", "cat", "dog"],
            "is_anomaly": [False, True, False],
        }
    )
    extractor = TintedFakeExtractor()
    config = load_config(
        overrides={
            "semantic": {
                "cache": {"enabled": False},
                "clip": {"augmentations": 1},
            }
        }
    )

    semantic, _, _, runtime = _semantic_analysis(
        samples,
        config,
        tmp_path / "output",
        extractor=extractor,
        semantic_backend="fake",
    )

    assert extractor.calls == 4
    assert not semantic.loc[semantic["sample_id"] == "b", "semantic_available"].iloc[0]
    assert runtime["cache_hits"] == 0
    assert runtime["cache_misses"] == 2


def test_preflight_skips_unreadable_image_and_reaches_next(
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"broken")
    good = tmp_path / "good.png"
    write_image(good, np.full((32, 32, 3), 127, dtype=np.uint8))
    samples = pd.DataFrame({"path": [corrupt, good]})
    extractor = TintedFakeExtractor()

    _preflight_extractor(extractor, samples)


def test_preflight_uses_prepare_without_extra_inference(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    write_image(image, np.full((32, 32, 3), 127, dtype=np.uint8))
    samples = pd.DataFrame({"path": [image]})

    class PrepareOnlyExtractor(TintedFakeExtractor):
        def __init__(self) -> None:
            super().__init__()
            self.prepared = 0

        def prepare(self) -> None:
            self.prepared += 1

    extractor = PrepareOnlyExtractor()

    _preflight_extractor(extractor, samples)

    assert extractor.prepared == 1
    assert extractor.calls == 0


def test_pipeline_warns_when_near_duplicate_comparison_is_skipped(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _write_classification_dataset(root)
    config = load_config(
        overrides={
            "anomaly": {"near_duplicate": {"enabled": True, "max_samples": 1}}
        }
    )

    result = analyze_dataset(
        root,
        tmp_path / "output",
        config,
        semantic_backend="disabled",
    )

    assert any("near-duplicate" in warning.lower() for warning in result.warnings)


def test_pipeline_summary_records_runtime_cache_and_failure_ids(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _write_classification_dataset(root)
    config = load_config(
        overrides={
            "semantic": {
                "clip": {"augmentations": 0, "batch_size": 2},
                "cache": {"enabled": False},
                "min_class_samples": 2,
            },
            "anomaly": {"near_duplicate": {"enabled": False}},
        }
    )

    result = analyze_dataset(
        root,
        tmp_path / "output",
        config,
        extractor=FailingOneExtractor(),
        semantic_backend="fake",
    )

    performance = result.summary["performance"]
    runtime = result.summary["semantic_runtime"]
    assert {
        "quality_seconds",
        "semantic_seconds",
        "scoring_seconds",
        "split_seconds",
        "analysis_seconds",
        "report_seconds",
        "total_seconds",
    }.issubset(performance)
    assert runtime["batch_size"] == 2
    assert runtime["cache_hits"] == 0
    assert runtime["cache_misses"] == runtime["eligible_sample_count"]
    assert runtime["eligible_sample_count"] > 0
    assert runtime["failure_count"] > 0
    assert runtime["failure_sample_ids"]
    assert runtime["device"] in {"cpu", "cuda", "not_loaded"}
    assert "prepare_seconds" in runtime
