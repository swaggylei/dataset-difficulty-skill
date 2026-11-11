from __future__ import annotations

import hashlib
from collections import defaultdict
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


def _stable_unit_order(seed: int, unit: str) -> str:
    return hashlib.sha256(f"{seed}|{unit}".encode("utf-8")).hexdigest()


def _assign_group_units(
    frame: pd.DataFrame,
    eligible_indices: pd.Index,
    ratios: dict[str, float],
    seed: int,
    ensure_val_test: bool,
) -> tuple[dict[str, Any], list[str], int, int]:
    eligible = frame.loc[eligible_indices].copy()
    group_ids = eligible["group_id"].fillna("").astype(str).str.strip()
    eligible["_split_unit"] = group_ids.where(
        group_ids.ne(""), "sample:" + eligible["sample_id"].astype(str)
    )
    units = []
    stratum_totals: dict[str, int] = defaultdict(int)
    mixed_group_count = 0
    for unit, group in eligible.groupby("_split_unit", sort=False):
        strata = (
            group.groupby(["label", "difficulty_band"], dropna=False)
            .size()
            .to_dict()
        )
        normalized_counts = {
            f"{label}\u0000{band}": int(count)
            for (label, band), count in strata.items()
        }
        for stratum, count in normalized_counts.items():
            stratum_totals[stratum] += count
        if len(normalized_counts) > 1 and not str(unit).startswith("sample:"):
            mixed_group_count += 1
        units.append(
            {
                "unit": str(unit),
                "indices": group.index.to_numpy(),
                "size": int(len(group)),
                "strata": normalized_counts,
            }
        )

    units.sort(
        key=lambda item: (
            -item["size"],
            _stable_unit_order(seed, item["unit"]),
        )
    )
    total_samples = len(eligible)
    current_by_stratum = {
        split: defaultdict(int) for split in SPLIT_ORDER
    }
    current_size = {split: 0 for split in SPLIT_ORDER}
    unit_split: dict[str, str] = {}
    for unit in units:
        candidate_costs: dict[str, float] = {}
        for split in SPLIT_ORDER:
            stratum_cost = 0.0
            for stratum, count in unit["strata"].items():
                target = stratum_totals[stratum] * ratios[split]
                current = current_by_stratum[split][stratum]
                stratum_cost += (current + count - target) ** 2 - (
                    current - target
                ) ** 2
            total_target = total_samples * ratios[split]
            total_cost = (
                current_size[split] + unit["size"] - total_target
            ) ** 2 - (current_size[split] - total_target) ** 2
            candidate_costs[split] = stratum_cost + 0.10 * total_cost
        chosen = min(SPLIT_ORDER, key=lambda split: (candidate_costs[split], SPLIT_ORDER.index(split)))
        unit_split[unit["unit"]] = chosen
        current_size[chosen] += unit["size"]
        for stratum, count in unit["strata"].items():
            current_by_stratum[chosen][stratum] += count

    if ensure_val_test and len(units) >= 3:
        for required in ("val", "test"):
            if current_size[required] == 0:
                donors = [
                    unit
                    for unit in units
                    if unit_split[unit["unit"]] == "train"
                ]
                if donors:
                    donor = min(donors, key=lambda item: (item["size"], item["unit"]))
                    unit_split[donor["unit"]] = required
                    current_size["train"] -= donor["size"]
                    current_size[required] += donor["size"]

    for unit in units:
        frame.loc[unit["indices"], "split"] = unit_split[unit["unit"]]

    warnings: list[str] = []
    if mixed_group_count:
        warnings.append(
            f"{mixed_group_count} group_id values contain mixed labels or difficulty bands."
        )
    for split in SPLIT_ORDER:
        if current_size[split] == 0 and total_samples:
            warnings.append(f"Split {split!r} received no eligible samples.")
    group_count = int(group_ids[group_ids.ne("")].nunique())
    return unit_split, warnings, group_count, len(units)


def _distribution(frame: pd.DataFrame, column: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in frame["split"].drop_duplicates().tolist():
        values = (
            frame.loc[frame["split"].eq(split), column]
            .fillna("")
            .replace("", "<empty>")
            .value_counts()
        )
        result[str(split)] = {
            str(key): int(value) for key, value in values.items()
        }
    return result


def _leakage_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    active_splits = set(SPLIT_ORDER)
    groups_spanning = 0
    duplicate_groups_spanning = 0
    if "group_id" in frame.columns:
        group_rows = frame[
            frame["group_id"].fillna("").astype(str).str.strip().ne("")
            & frame["split"].isin(active_splits)
        ]
        if not group_rows.empty:
            groups_spanning = int(
                (
                    group_rows.groupby("group_id")["split"].nunique() > 1
                ).sum()
            )
    if "duplicate_group" in frame.columns:
        duplicate_rows = frame[
            frame["duplicate_group"].fillna("").astype(str).ne("")
            & frame["split"].isin(active_splits)
        ]
        if not duplicate_rows.empty:
            duplicate_groups_spanning = int(
                (
                    duplicate_rows.groupby("duplicate_group")["split"].nunique()
                    > 1
                ).sum()
            )
    messages = []
    if groups_spanning:
        messages.append("Some group_id values appear in multiple splits.")
    if duplicate_groups_spanning:
        messages.append("Some duplicate groups appear in multiple splits.")
    if not messages:
        messages.append("No group_id or duplicate-group leakage detected.")
    return {
        "level": "high" if messages[0] != "No group_id or duplicate-group leakage detected." else "low",
        "groups_spanning_splits": groups_spanning,
        "duplicate_groups_spanning_splits": duplicate_groups_spanning,
        "messages": messages,
    }


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
    if "group_id" not in frame.columns:
        frame["group_id"] = ""
    else:
        frame["group_id"] = frame["group_id"].fillna("").astype(str).str.strip()
    frame["split"] = "excluded_anomaly"
    anomaly_mask = frame.get(
        "sample_status", pd.Series("", index=frame.index)
    ).eq("Anomaly") | frame.get(
        "difficulty_band", pd.Series("", index=frame.index)
    ).eq("Anomaly")
    if not exclude_anomalies:
        anomaly_mask = pd.Series(False, index=frame.index)

    eligible = frame[~anomaly_mask]
    warnings: list[str] = []
    strata_summary: list[dict[str, Any]] = []
    use_groups = bool(eligible["group_id"].ne("").any())
    group_count = int(eligible.loc[eligible["group_id"].ne(""), "group_id"].nunique())
    if use_groups:
        _, group_warnings, group_count, split_unit_count = _assign_group_units(
            frame,
            eligible.index,
            normalized,
            seed,
            ensure_val_test_when_possible,
        )
        warnings.extend(group_warnings)
        for split_name, class_counts in _distribution(
            frame[frame["split"].isin(SPLIT_ORDER)], "label"
        ).items():
            strata_summary.append(
                {
                    "split": split_name,
                    "class_counts": class_counts,
                }
            )
    else:
        group_count = 0
        split_unit_count = len(eligible)
        allocated_before = {name: 0 for name in SPLIT_ORDER}
        processed = 0
        grouped = eligible.groupby(
            ["label", "difficulty_band"], dropna=False, sort=True
        )
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
    actual_ratios = (
        {
            name: counts.get(name, 0) / scored_total
            for name in SPLIT_ORDER
        }
        if scored_total
        else {name: 0.0 for name in SPLIT_ORDER}
    )
    for name in SPLIT_ORDER:
        if abs(actual_ratios[name] - normalized[name]) > 0.05:
            warnings.append(
                f"Actual {name} ratio {actual_ratios[name]:.3f} differs from "
                f"requested {normalized[name]:.3f} by more than 0.05."
            )
    leakage = _leakage_diagnostics(frame)
    if leakage["level"] != "low":
        warnings.extend(leakage["messages"])
    summary: dict[str, Any] = {
        "ratios": normalized,
        "seed": int(seed),
        "exclude_anomalies": bool(exclude_anomalies),
        "split_mode": "group" if use_groups else "sample",
        "group_count": group_count,
        "split_unit_count": int(split_unit_count),
        "counts": {str(name): int(value) for name, value in counts.items()},
        "actual_scored_ratios": actual_ratios,
        "class_distribution": _distribution(frame, "label"),
        "difficulty_distribution": _distribution(frame, "difficulty_band"),
        "groups_spanning_splits": leakage["groups_spanning_splits"],
        "duplicate_groups_spanning_splits": leakage[
            "duplicate_groups_spanning_splits"
        ],
        "leakage_risk": leakage,
        "soft_statistical_anomalies": int(
            frame["sample_status"].eq("Statistically_Unusual").sum()
        ),
        "excluded_anomalies": int((frame["split"] == "excluded_anomaly").sum()),
        "strata": strata_summary,
        "warnings": warnings,
    }
    return SplitResult(assignment=frame, summary=summary)
