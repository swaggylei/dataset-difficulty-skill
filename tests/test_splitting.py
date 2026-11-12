from __future__ import annotations

import json

import pandas as pd
import pytest

from dataset_difficulty.splitting import _leakage_diagnostics, assign_splits


def _stratified_frame() -> pd.DataFrame:
    rows = []
    for label in ("cat", "dog"):
        for band in ("Easy", "Medium", "Hard"):
            for index in range(15):
                rows.append(
                    {
                        "sample_id": f"{label}-{band}-{index:02d}",
                        "rel_path": f"{label}/{band}/{index:02d}.png",
                        "label": label,
                        "difficulty_band": band,
                        "sample_status": "Scored",
                    }
                )
    return pd.DataFrame(rows)


def test_joint_split_is_reproducible_and_roughly_matches_ratios() -> None:
    frame = _stratified_frame()
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}

    first = assign_splits(frame, ratios, seed=42)
    second = assign_splits(frame, ratios, seed=42)

    assert first.assignment["split"].tolist() == second.assignment["split"].tolist()
    counts = first.assignment["split"].value_counts().to_dict()
    assert counts["train"] == 63
    assert counts["val"] == 13 or counts["val"] == 14
    assert counts["test"] == 13 or counts["test"] == 14
    for (_, _), group in first.assignment.groupby(["label", "difficulty_band"]):
        assert {"train", "val", "test"}.issubset(set(group["split"]))
    assert first.summary["counts"] == counts
    json.dumps(first.summary, allow_nan=False)


def test_tiny_strata_do_not_crash_and_anomalies_are_excluded() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(8)],
            "rel_path": [f"s{index}.png" for index in range(8)],
            "label": ["cat"] * 8,
            "difficulty_band": ["Easy", "Easy", "Medium", "Medium", "Hard", "Hard", "Hard", "Anomaly"],
            "sample_status": [
                "Scored",
                "Scored",
                "Scored",
                "Scored",
                "Scored",
                "Scored",
                "Scored",
                "Anomaly",
            ],
        }
    )

    result = assign_splits(
        frame,
        {"train": 0.70, "val": 0.15, "test": 0.15},
        seed=3,
    )

    assert result.assignment.loc[7, "split"] == "excluded_anomaly"
    assert set(result.assignment.loc[:6, "split"]).issubset(
        {"train", "val", "test"}
    )
    assert result.summary["excluded_anomalies"] == 1
    assert result.summary["warnings"]


def _grouped_frame() -> pd.DataFrame:
    rows = []
    for label in ("cat", "dog"):
        for band in ("Easy", "Medium", "Hard"):
            for group_index in range(10):
                group_id = f"{label}-{band}-group-{group_index}"
                for member in range(3):
                    rows.append(
                        {
                            "sample_id": f"{group_id}-{member}",
                            "rel_path": f"{label}/{band}/{group_index}/{member}.png",
                            "label": label,
                            "difficulty_band": band,
                            "sample_status": "Scored",
                            "group_id": group_id,
                        }
                    )
    return pd.DataFrame(rows)


def test_group_ids_never_span_splits_and_diagnostics_are_reported() -> None:
    frame = _grouped_frame()

    result = assign_splits(
        frame,
        {"train": 0.70, "val": 0.15, "test": 0.15},
        seed=42,
    )

    grouped = result.assignment[result.assignment["group_id"].ne("")].groupby(
        "group_id"
    )["split"].nunique()
    assert grouped.max() == 1
    assert result.summary["groups_spanning_splits"] == 0
    assert set(result.summary["class_distribution"]) == {"train", "val", "test"}
    assert set(result.summary["difficulty_distribution"]) == {
        "train",
        "val",
        "test",
    }
    assert result.summary["leakage_risk"]["level"] == "low"
    assert result.summary["group_count"] == 60
    counts = result.assignment["split"].value_counts(normalize=True)
    assert counts["train"] == pytest.approx(0.70, abs=0.06)


def test_mixed_group_stays_intact() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "rel_path": ["a", "b", "c", "d"],
            "label": ["cat", "dog", "cat", "dog"],
            "difficulty_band": ["Easy", "Hard", "Medium", "Medium"],
            "sample_status": ["Scored"] * 4,
            "group_id": ["mixed", "mixed", "g2", "g3"],
        }
    )

    result = assign_splits(
        frame,
        {"train": 0.70, "val": 0.15, "test": 0.15},
        seed=7,
    )

    mixed_splits = result.assignment.loc[
        result.assignment["group_id"] == "mixed", "split"
    ]
    assert mixed_splits.nunique() == 1
    assert any("mixed" in warning.lower() for warning in result.summary["warnings"])


def test_leakage_diagnostics_detects_intentional_group_crossing() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "group_id": ["shared", "shared", "safe"],
            "split": ["train", "test", "val"],
        }
    )

    diagnostics = _leakage_diagnostics(frame)

    assert diagnostics["groups_spanning_splits"] == 1
    assert diagnostics["level"] == "high"
    assert any("group_id" in message for message in diagnostics["messages"])
