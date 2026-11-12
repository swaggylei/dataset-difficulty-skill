from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_difficulty.config import load_config
from dataset_difficulty.loading import load_dataset


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-decoded-by-loader")
    return path


def test_folder_loader_uses_parent_directory_as_label(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _touch(root / "cat" / "one.jpg")
    _touch(root / "dog" / "nested" / "two.png")

    result = load_dataset(root, data_format="folder")

    labels = result.samples.set_index("rel_path")["label"].to_dict()
    assert labels == {
        "cat/one.jpg": "cat",
        "dog/nested/two.png": "dog",
    }
    assert result.samples["sample_id"].is_unique


def test_csv_loader_resolves_paths_and_keeps_empty_labels(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image = _touch(root / "images" / "a.jpg")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "path,label\n"
        f"{image.as_posix()},cat\n"
        "images/missing.jpg,\n",
        encoding="utf-8",
    )

    result = load_dataset(root, label_file=labels, data_format="csv")

    assert len(result.samples) == 2
    assert set(result.samples["label"]) == {"cat", ""}
    assert result.samples.loc[result.samples["label"] == "cat", "path"].iloc[0] == image


def test_csv_loader_marks_conflicting_paths_without_choosing_a_label(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _touch(root / "a.jpg")
    labels = tmp_path / "labels.csv"
    labels.write_text("path,label\na.jpg,cat\na.jpg,dog\n", encoding="utf-8")

    result = load_dataset(root, label_file=labels, data_format="csv")

    assert len(result.samples) == 1
    row = result.samples.iloc[0]
    assert row["label"] == ""
    assert row["label_conflict"] is True or row["label_conflict"] == True
    assert row["conflicting_labels"] == "cat;dog"
    assert any("conflict" in warning.lower() for warning in result.warnings)


def test_csv_loader_reads_optional_group_id(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _touch(root / "a.jpg")
    _touch(root / "b.jpg")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "path,label,group_id\na.jpg,cat,subject-1\nb.jpg,cat,subject-1\n",
        encoding="utf-8",
    )

    result = load_dataset(
        root,
        label_file=labels,
        data_format="csv",
        group_column="group_id",
    )

    assert result.samples["group_id"].tolist() == ["subject-1", "subject-1"]


def test_explicit_missing_group_column_raises_instead_of_silent_leakage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _touch(root / "a.jpg")
    labels = tmp_path / "labels.csv"
    labels.write_text("path,label\na.jpg,cat\n", encoding="utf-8")

    with pytest.raises(ValueError, match="group column"):
        load_dataset(
            root,
            label_file=labels,
            data_format="csv",
            group_column="subject_id",
            require_group_column=True,
        )


@pytest.mark.parametrize(
    "payload",
    [
        [{"path": "cat/a.jpg", "label": "cat"}],
        {"cat/a.jpg": "cat"},
        {"items": [{"path": "cat/a.jpg", "label": "cat"}]},
    ],
)
def test_json_loader_supports_documented_shapes(tmp_path: Path, payload) -> None:
    root = tmp_path / "dataset"
    _touch(root / "cat" / "a.jpg")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(payload), encoding="utf-8")

    result = load_dataset(root, label_file=labels, data_format="json")

    assert result.samples[["rel_path", "label"]].to_dict("records") == [
        {"rel_path": "cat/a.jpg", "label": "cat"}
    ]


def test_json_loader_reads_group_id(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _touch(root / "cat" / "a.jpg")
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            [{"path": "cat/a.jpg", "label": "cat", "group_id": "subject-json"}]
        ),
        encoding="utf-8",
    )

    result = load_dataset(root, label_file=labels, data_format="json")

    assert result.samples.loc[0, "group_id"] == "subject-json"


def test_loader_excludes_generated_output_directory(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _touch(root / "cat" / "a.jpg")
    _touch(root / "reports" / "preview.png")

    result = load_dataset(
        root,
        data_format="folder",
        exclude_paths=[root / "reports"],
    )

    assert result.samples["rel_path"].tolist() == ["cat/a.jpg"]


def test_load_config_merges_user_values_recursively(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "split:\n  ratios:\n    train: 0.7\n    val: 0.15\n    test: 0.15\n"
        "report:\n  preview_count: 12\n",
        encoding="utf-8",
    )

    config = load_config(
        base,
        overrides={
            "split": {"ratios": {"test": 0.2}},
            "report": {"preview_count": 4},
        },
    )

    assert config["split"]["ratios"] == {
        "train": 0.7,
        "val": 0.15,
        "test": 0.2,
    }
    assert config["report"]["preview_count"] == 4
