from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(slots=True)
class DatasetLoadResult:
    samples: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    root: Path | None = None
    label_file: Path | None = None
    data_format: str = "folder"


def _stable_sample_id(path: Path) -> str:
    normalized = path.resolve().as_posix().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _is_excluded(path: Path, exclusions: Iterable[Path]) -> bool:
    resolved = path.resolve()
    for excluded in exclusions:
        excluded_resolved = excluded.resolve()
        if resolved == excluded_resolved or excluded_resolved in resolved.parents:
            return True
    return False


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _normalize_extensions(extensions: Iterable[str] | None) -> set[str]:
    values = extensions or DEFAULT_EXTENSIONS
    return {value.lower() if value.startswith(".") else f".{value.lower()}" for value in values}


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_id",
            "path",
            "rel_path",
            "label",
            "group_id",
            "label_source",
            "label_conflict",
            "conflicting_labels",
        ]
    )


def _folder_rows(
    root: Path,
    extensions: set[str],
    exclusions: list[Path],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if _is_excluded(path, exclusions):
            continue
        relative_parts = path.relative_to(root).parts
        label = relative_parts[0] if len(relative_parts) > 1 else ""
        rows.append(
            {
                "path": path.resolve(),
                "rel_path": _relative_path(path, root),
                "label": label,
                "group_id": "",
                "label_source": "folder",
                "label_conflict": False,
                "conflicting_labels": "",
            }
        )
    if any(row["label"] == "" for row in rows):
        warnings.append("Some images are directly under the dataset root and have empty labels.")
    return rows, warnings


def _resolve_labeled_path(raw_path: str, root: Path, label_file: Path) -> Path:
    candidate = Path(raw_path.strip())
    if candidate.is_absolute():
        return candidate.resolve()
    for base in (label_file.parent, root):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (root / candidate).resolve()


def _labeled_rows_from_records(
    records: list[dict[str, Any]],
    root: Path,
    label_file: Path,
    path_column: str,
    label_column: str,
    group_column: str,
    require_group_column: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    if require_group_column and records:
        missing_group = [
            index
            for index, record in enumerate(records, start=2)
            if group_column not in record
        ]
        if missing_group:
            raise ValueError(
                f"Required group column {group_column!r} is missing from "
                f"label records {missing_group[:5]}"
            )
    for index, record in enumerate(records, start=2):
        raw_path = str(record.get(path_column, "") or "").strip()
        if not raw_path:
            warnings.append(f"Skipped label row {index} because the path is empty.")
            continue
        path = _resolve_labeled_path(raw_path, root, label_file)
        label = str(record.get(label_column, "") or "").strip()
        group_id = str(record.get(group_column, "") or "").strip()
        rows.append(
            {
                "path": path,
                "rel_path": _relative_path(path, root),
                "label": label,
                "group_id": group_id,
                "label_source": "label_file",
                "label_conflict": False,
                "conflicting_labels": "",
            }
        )
    return rows, warnings


def _read_csv_rows(
    root: Path,
    label_file: Path,
    path_column: str,
    label_column: str,
    group_column: str,
    require_group_column: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    frame = pd.read_csv(label_file, dtype=str, keep_default_na=False)
    missing = [column for column in (path_column, label_column) if column not in frame]
    if missing:
        raise ValueError(
            f"Label file is missing columns {missing}; available columns: {list(frame.columns)}"
        )
    return _labeled_rows_from_records(
        frame.to_dict("records"),
        root=root,
        label_file=label_file,
        path_column=path_column,
        label_column=label_column,
        group_column=group_column,
        require_group_column=require_group_column,
    )


def _json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("JSON list entries must be objects with path and label fields")
        return payload
    if isinstance(payload, dict):
        for key in ("items", "data", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                return _json_records(value)
        records = []
        for raw_path, value in payload.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("path", raw_path)
                records.append(record)
            else:
                records.append({"path": raw_path, "label": value})
        return records
    raise ValueError("JSON labels must be a list, mapping, or object containing items/data")


def _read_json_rows(
    root: Path,
    label_file: Path,
    path_column: str,
    label_column: str,
    group_column: str,
    require_group_column: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    payload = json.loads(label_file.read_text(encoding="utf-8"))
    records = _json_records(payload)
    return _labeled_rows_from_records(
        records,
        root=root,
        label_file=label_file,
        path_column=path_column,
        label_column=label_column,
        group_column=group_column,
        require_group_column=require_group_column,
    )


def _collapse_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    if not rows:
        return [], []
    by_path: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_path.setdefault(str(row["path"]), []).append(row)

    collapsed: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path_text, entries in by_path.items():
        labels = sorted({str(entry["label"]) for entry in entries if str(entry["label"])})
        groups = sorted(
            {
                str(entry.get("group_id", ""))
                for entry in entries
                if str(entry.get("group_id", ""))
            }
        )
        conflict = len(labels) > 1
        first = entries[0]
        collapsed.append(
            {
                "path": Path(path_text),
                "rel_path": first["rel_path"],
                "label": "" if conflict else (labels[0] if labels else ""),
                "group_id": groups[0] if groups else "",
                "label_source": first["label_source"],
                "label_conflict": conflict,
                "conflicting_labels": ";".join(labels),
            }
        )
        if conflict:
            warnings.append(
                f"Conflicting labels for {first['rel_path']}: {';'.join(labels)}"
            )
        if len(groups) > 1:
            warnings.append(
                f"Conflicting group IDs for {first['rel_path']}: {';'.join(groups)}"
            )
    return collapsed, warnings


def _finalize_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty_frame()
    frame = pd.DataFrame(rows)
    frame["sample_id"] = frame["path"].map(_stable_sample_id)
    frame["path"] = frame["path"].map(lambda value: Path(value).resolve())
    frame["label"] = frame["label"].fillna("").astype(str)
    frame["group_id"] = frame["group_id"].fillna("").astype(str)
    frame["label_conflict"] = frame["label_conflict"].fillna(False).astype(bool)
    columns = [
        "sample_id",
        "path",
        "rel_path",
        "label",
        "group_id",
        "label_source",
        "label_conflict",
        "conflicting_labels",
    ]
    return frame[columns].sort_values("rel_path").reset_index(drop=True)


def load_dataset(
    root: str | Path,
    label_file: str | Path | None = None,
    data_format: str = "auto",
    extensions: list[str] | None = None,
    path_column: str = "path",
    label_column: str = "label",
    group_column: str = "group_id",
    require_group_column: bool = False,
    exclude_paths: list[str | Path] | None = None,
) -> DatasetLoadResult:
    """Load folder, CSV, or JSON classification datasets without decoding images."""

    root_path = Path(root).resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root_path}")
    label_path = Path(label_file).resolve() if label_file is not None else None
    if label_path is not None and not label_path.exists():
        raise FileNotFoundError(f"Label file does not exist: {label_path}")

    exclusions = [Path(path) for path in (exclude_paths or [])]
    format_value = data_format.lower()
    if format_value == "auto":
        if label_path is None:
            format_value = "folder"
        elif label_path.suffix.lower() == ".csv":
            format_value = "csv"
        elif label_path.suffix.lower() == ".json":
            format_value = "json"
        else:
            raise ValueError("Cannot infer label format; use csv or json explicitly")

    if format_value == "folder":
        if require_group_column:
            raise ValueError(
                "A group column was requested but folder format has no group metadata"
            )
        rows, warnings = _folder_rows(
            root_path,
            _normalize_extensions(extensions),
            exclusions + ([label_path] if label_path else []),
        )
    elif format_value == "csv":
        if label_path is None:
            raise ValueError("CSV format requires a label file")
        rows, warnings = _read_csv_rows(
            root_path,
            label_path,
            path_column=path_column,
            label_column=label_column,
            group_column=group_column,
            require_group_column=require_group_column,
        )
        rows = [row for row in rows if not _is_excluded(row["path"], exclusions)]
    elif format_value == "json":
        if label_path is None:
            raise ValueError("JSON format requires a label file")
        rows, warnings = _read_json_rows(
            root_path,
            label_path,
            path_column=path_column,
            label_column=label_column,
            group_column=group_column,
            require_group_column=require_group_column,
        )
        rows = [row for row in rows if not _is_excluded(row["path"], exclusions)]
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    collapsed, conflict_warnings = _collapse_rows(rows)
    warnings.extend(conflict_warnings)
    if require_group_column and not any(
        str(row.get("group_id", "")).strip() for row in collapsed
    ):
        raise ValueError(
            f"Required group column {group_column!r} contains no non-empty values"
        )
    if not collapsed:
        warnings.append("No supported image files were found.")
    return DatasetLoadResult(
        samples=_finalize_frame(collapsed),
        warnings=warnings,
        root=root_path,
        label_file=label_path,
        data_format=format_value,
    )
