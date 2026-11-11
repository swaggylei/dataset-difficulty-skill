from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from conftest import write_image


def test_cli_runs_end_to_end_without_semantics(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    rng = np.random.default_rng(91)
    for label in ("cat", "dog"):
        for index in range(3):
            image = rng.integers(10, 245, size=(72, 72, 3), dtype=np.uint8)
            write_image(root / label / f"{index}.png", image)
    output = tmp_path / "reports"
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_dataset.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(root),
            "--output",
            str(output),
            "--no-semantic",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output / "report.html").exists()
    assert (output / "samples.csv").exists()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["total"] == 6
    assert summary["semantic_backend"] == "disabled"


def test_cli_honors_semantic_backend_from_config(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    rng = np.random.default_rng(92)
    for label in ("cat", "dog"):
        for index in range(2):
            image = rng.integers(10, 245, size=(64, 64, 3), dtype=np.uint8)
            write_image(root / label / f"{index}.png", image)
    config_path = tmp_path / "offline.yaml"
    config_path.write_text(
        "semantic:\n  enabled: true\n  backend: disabled\n",
        encoding="utf-8",
    )
    output = tmp_path / "reports"
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_dataset.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(root),
            "--output",
            str(output),
            "--config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["semantic_backend"] == "disabled"


def test_cli_group_column_keeps_groups_in_one_split(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    rng = np.random.default_rng(93)
    rows = ["path,label,group_id"]
    for group_index in range(8):
        for member in range(3):
            relative = Path("cat") / f"g{group_index}_{member}.png"
            image = rng.integers(10, 245, size=(64, 64, 3), dtype=np.uint8)
            write_image(root / relative, image)
            rows.append(f"{relative.as_posix()},cat,group-{group_index}")
    labels = tmp_path / "labels.csv"
    labels.write_text("\n".join(rows) + "\n", encoding="utf-8")
    output = tmp_path / "reports"
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_dataset.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(root),
            "--labels",
            str(labels),
            "--format",
            "csv",
            "--group-column",
            "group_id",
            "--output",
            str(output),
            "--no-semantic",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = pd.read_csv(output / "split_manifest.csv")
    grouped = manifest[manifest["group_id"].ne("")].groupby("group_id")[
        "split"
    ].nunique()
    assert grouped.max() == 1
