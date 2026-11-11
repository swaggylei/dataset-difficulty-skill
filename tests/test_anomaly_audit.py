from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_anomaly_audit_separates_corrupt_quality_and_statistical_only(
    tmp_path: Path,
) -> None:
    from audit_anomalies import classify_anomalies

    good = tmp_path / "good.png"
    good.write_bytes(b"not-decoded-here")
    corrupt = tmp_path / "missing.png"
    samples = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "path": [good, corrupt, good, good],
            "rel_path": ["a.png", "b.png", "c.png", "d.png"],
            "is_anomaly": [True, True, True, True],
            "decode_ok": [True, False, True, True],
            "quality_issue_flags": ["", "", "blurred", ""],
            "anomaly_types": [
                "statistical_outlier",
                "corrupt_image",
                "quality_outlier",
                "statistical_outlier;quality_outlier",
            ],
        }
    )

    selected, summary = classify_anomalies(samples, sample_count=4, seed=1)

    assert selected["audit_category"].tolist() == [
        "statistical_only",
        "corrupt_or_unreadable",
        "low_quality_valid",
        "low_quality_valid",
    ]
    assert summary["category_counts_all_anomalies"]["statistical_only"] == 1
    assert summary["category_counts_all_anomalies"]["corrupt_or_unreadable"] == 1
