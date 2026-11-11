from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

SKILL_ROOT = Path(__file__).resolve().parents[1]
SRC = SKILL_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_difficulty.reports import _preview_sheet  # noqa: E402


def _categorize_rows(frame: pd.DataFrame) -> list[str]:
    categories = []
    for _, row in frame.iterrows():
        reasons = {
            reason for reason in str(row.get("anomaly_types", "")).split(";") if reason
        }
        path = Path(str(row.get("path", "")))
        quality_flags = row.get("quality_issue_flags", "")
        quality_flags = "" if pd.isna(quality_flags) else str(quality_flags).strip()
        if not bool(row.get("decode_ok", False)) or not path.exists():
            categories.append("corrupt_or_unreadable")
        elif quality_flags or "quality_outlier" in reasons:
            categories.append("low_quality_valid")
        elif reasons == {"statistical_outlier"}:
            categories.append("statistical_only")
        else:
            categories.append("other_anomaly")
    return categories


def classify_anomalies(
    samples: pd.DataFrame,
    sample_count: int = 50,
    seed: int = 20261111,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anomalies = samples[samples["is_anomaly"].fillna(False).astype(bool)].copy()
    all_categories = _categorize_rows(anomalies)
    anomalies["audit_category"] = all_categories
    if anomalies.empty:
        selected = anomalies.copy()
    else:
        count = min(sample_count, len(anomalies))
        indices = np.random.default_rng(seed).choice(
            anomalies.index.to_numpy(), size=count, replace=False
        )
        selected = anomalies.loc[indices].copy()
        missing_categories = set(anomalies["audit_category"]) - set(
            selected["audit_category"]
        )
        for category in sorted(missing_categories):
            replacement_candidates = anomalies[
                anomalies["audit_category"].eq(category)
                & ~anomalies.index.isin(selected.index)
            ]
            if replacement_candidates.empty:
                continue
            replacements = replacement_candidates.sample(
                n=min(
                    len(replacement_candidates),
                    max(1, len(selected) // max(1, len(missing_categories))),
                ),
                random_state=seed,
            )
            selected = pd.concat(
                [selected.iloc[: max(0, len(selected) - len(replacements))], replacements]
            )
        selected = selected.sort_values("rel_path").reset_index(drop=True)
    summary = {
        "total_anomalies": int(len(anomalies)),
        "sample_count": int(len(selected)),
        "seed": int(seed),
        "category_counts_all_anomalies": anomalies[
            "audit_category"
        ].value_counts().to_dict(),
        "sample_category_counts": selected["audit_category"].value_counts().to_dict(),
        "reason_counts_all_anomalies": {
            reason: int(
                anomalies["anomaly_types"]
                .fillna("")
                .astype(str)
                .map(lambda value: reason in set(value.split(";")))
                .sum()
            )
            for reason in sorted(
                {
                    item
                    for value in anomalies["anomaly_types"].fillna("").astype(str)
                    for item in value.split(";")
                    if item
                }
            )
        },
    }
    return selected, summary


def write_audit(
    samples_path: str | Path,
    output_dir: str | Path,
    sample_count: int = 50,
    seed: int = 20261111,
) -> dict[str, Any]:
    samples = pd.read_csv(samples_path)
    selected, summary = classify_anomalies(samples, sample_count, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "anomaly_audit_50.csv", index=False)
    (output / "anomaly_audit_50.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _preview_sheet(
        selected,
        output / "anomaly_audit_50.png",
        "Random anomaly audit sample",
        sample_count,
        5,
        160,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a random anomaly sample.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261111)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = write_audit(
        args.samples,
        args.output,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
