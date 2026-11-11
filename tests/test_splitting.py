from __future__ import annotations

import json

import pandas as pd

from dataset_difficulty.splitting import assign_splits


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
