from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


SPLIT_ORDER = ("train", "val", "test")


@dataclass(slots=True)
class SplitResult:
    assignment: pd.DataFrame
    summary: dict[str, Any]


def _normalize_ratios(ratios: dict[str, float]) -> dict[str, float]:
    values = {
        name: float(ratios.get(name, 0.0)) for name in SPLIT_ORDER
    }
    if any(value < 0 for value in values.values()):
        raise ValueError("Split ratios must be non-negative")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("At least one split ratio must be positive")
    return {name: value / total for name, value in values.items()}


def _stable_group_seed(seed: int, label: str, band: str) -> int:
    digest = hashlib.sha256(f"{label}|{band}".encode("utf-8")).hexdigest()
    return (int(seed) + int(digest[:8], 16)) % (2**32)


def _balanced_counts(
    size: int,
    ratios: dict[str, float],
    allocated_before: dict[str, int],
    processed_after: int,
    group_seed: int,
    ensure_val_test: bool,
) -> tuple[dict[str, int], bool]:
    counts = {name: int(np.floor(size * ratios[name])) for name in SPLIT_ORDER}
    remaining = size - sum(counts.values())
    projected = {
        name: allocated_before[name] + counts[name] for name in SPLIT_ORDER
    }
    targets = {name: processed_after * ratios[name] for name in SPLIT_ORDER}
    rng = np.random.default_rng(group_seed)
    while remaining > 0:
        deficits = {name: targets[name] - projected[name] for name in SPLIT_ORDER}
        best = max(deficits.values())
        candidates = [name for name in SPLIT_ORDER if abs(deficits[name] - best) < 1e-12]
        if len(candidates) > 1:
            tie_order = list(rng.permutation(candidates))
            chosen = str(tie_order[0])
        else:
            chosen = candidates[0]
        counts[chosen] += 1
        projected[chosen] += 1
        remaining -= 1

    fallback = False
    if ensure_val_test and size >= 3:
        for required in ("val", "test"):
            if counts[required] == 0:
                donor_candidates = [
                    name for name in SPLIT_ORDER if name != required and counts[name] > 1
                ]
                if donor_candidates:
                    donor = max(donor_candidates, key=lambda name: counts[name])
                    counts[donor] -= 1
                    counts[required] += 1
                    fallback = True
    return counts, fallback or (size < 3 and ensure_val_test)


def assign_splits(
    scored: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
    exclude_anomalies: bool = True,
    ensure_val_test_when_possible: bool = True,
) -> SplitResult:
    """Assign seeded splits jointly across label and difficulty strata."""

    normalized = _normalize_ratios(ratios)
    frame = scored.copy()
    frame["split"] = "excluded_anomaly"
    anomaly_mask = frame.get(
        "sample_status", pd.Series("", index=frame.index)
    ).eq("Anomaly") | frame.get(
        "difficulty_band", pd.Series("", index=frame.index)
    ).eq("Anomaly")
    if not exclude_anomalies:
        anomaly_mask = pd.Series(False, index=frame.index)

    eligible = frame[~anomaly_mask]
    allocated_before = {name: 0 for name in SPLIT_ORDER}
    processed = 0
    warnings: list[str] = []
    strata_summary: list[dict[str, Any]] = []

    grouped = eligible.groupby(["label", "difficulty_band"], dropna=False, sort=True)
    for (label, band), group in grouped:
        size = len(group)
        group_seed = _stable_group_seed(seed, str(label), str(band))
        counts, fallback = _balanced_counts(
            size=size,
            ratios=normalized,
            allocated_before=allocated_before,
            processed_after=processed + size,
            group_seed=group_seed,
            ensure_val_test=ensure_val_test_when_possible,
        )
        if fallback:
            warnings.append(
                f"Strata label={label!r}, band={band!r}, size={size} "
                "cannot fully satisfy validation/test quotas."
            )
        ordered = group.sort_values(["sample_id", "rel_path"])
        indices = ordered.index.to_numpy()
        permutation = np.random.default_rng(group_seed).permutation(indices)
        start = 0
        for split_name in SPLIT_ORDER:
            end = start + counts[split_name]
            frame.loc[permutation[start:end], "split"] = split_name
            start = end
        allocated_before = {
            name: allocated_before[name] + counts[name] for name in SPLIT_ORDER
        }
        processed += size
        strata_summary.append(
            {
                "label": str(label),
                "difficulty_band": str(band),
                "size": int(size),
                "counts": {name: int(count) for name, count in counts.items()},
            }
        )

    counts = frame["split"].value_counts().sort_index().to_dict()
    scored_total = sum(counts.get(name, 0) for name in SPLIT_ORDER)
    summary: dict[str, Any] = {
        "ratios": normalized,
        "seed": int(seed),
        "exclude_anomalies": bool(exclude_anomalies),
        "counts": {str(name): int(value) for name, value in counts.items()},
        "actual_scored_ratios": (
            {
                name: counts.get(name, 0) / scored_total
                for name in SPLIT_ORDER
            }
            if scored_total
            else {name: 0.0 for name in SPLIT_ORDER}
        ),
        "excluded_anomalies": int((frame["split"] == "excluded_anomaly").sum()),
        "strata": strata_summary,
        "warnings": warnings,
    }
    return SplitResult(assignment=frame, summary=summary)
