# CIFAR-10 Anomaly Audit

Deterministic random audit of 50 anomalies from the 794 flagged CIFAR-10
test images (`seed=20261111`).

## Classification of All 794 Anomalies

- `corrupt_or_unreadable`: 0
- `low_quality_valid`: 374
- `statistical_only`: 420
- `other_anomaly`: 0

Anomaly reasons overlap when a sample is both quality and statistically
unusual, so the reason counts are 374 `quality_outlier` and 532
`statistical_outlier`.

## Random 50-Image Sample

- `statistical_only`: 33
- `low_quality_valid`: 17
- `corrupt_or_unreadable`: 0

The 50 preview images were inspected: no unreadable files were found;
statistical-only examples include visible but unusual CIFAR objects such as
aircraft, birds, cats, and vehicles. Statistical-only samples are retained as
`Statistically_Unusual`, and low-quality readable samples are retained as
`Quality_Flagged`. Both are eligible for the split by default.

Artifacts:

- `anomaly_audit_50.csv`
- `anomaly_audit_50.json`
- `anomaly_audit_50.png`
