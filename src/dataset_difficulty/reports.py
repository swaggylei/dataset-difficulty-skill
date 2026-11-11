from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _sanitize(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    if value is None:
        return None
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    exported = frame.copy()
    for column in exported.columns:
        if exported[column].map(lambda value: isinstance(value, Path)).any():
            exported[column] = exported[column].map(
                lambda value: str(value) if isinstance(value, Path) else value
            )
    return exported


def _format_value(value: Any, digits: int = 4) -> str:
    if value is None or value is pd.NA:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    if isinstance(value, np.floating):
        numeric = float(value)
        return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "n/a"
    return html.escape(str(value))


def _html_table(
    frame: pd.DataFrame,
    columns: Iterable[str],
    limit: int = 50,
) -> str:
    selected = [column for column in columns if column in frame.columns]
    if frame.empty or not selected:
        return "<p class=\"empty\">No rows available.</p>"
    rows = []
    for _, row in frame.head(limit).iterrows():
        cells = "".join(
            f"<td>{_format_value(row[column])}</td>" for column in selected
        )
        rows.append(f"<tr>{cells}</tr>")
    headers = "".join(f"<th>{html.escape(column)}</th>" for column in selected)
    return (
        "<div class=\"table-wrap\"><table>"
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _preview_caption(row: pd.Series) -> str:
    name = Path(str(row.get("rel_path", "") or row.get("sample_id", ""))).name
    band = str(row.get("difficulty_band", "") or "")
    reasons = str(row.get("anomaly_types", "") or "").replace(";", ", ")
    first_line = f"{_truncate_text(name, 18)} | {_truncate_text(band, 8)}"
    second_line = _truncate_text(reasons or "no anomaly", 30)
    return f"{first_line}\n{second_line}"


def _preview_sheet(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
    count: int,
    columns: int,
    thumbnail_size: int = 180,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    selected = frame.head(max(1, count))
    if selected.empty:
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No samples available for this preview",
            ha="center",
            va="center",
            fontsize=13,
        )
        figure.suptitle(title)
        figure.savefig(output_path, dpi=max(72, thumbnail_size), bbox_inches="tight")
        plt.close(figure)
        return

    columns = max(1, min(columns, len(selected)))
    rows = int(math.ceil(len(selected) / columns))
    panel_inches = max(1.5, float(thumbnail_size) / 100.0)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            max(2.2, columns * panel_inches),
            max(2.2, rows * panel_inches),
        ),
        squeeze=False,
    )
    flattened = list(axes.ravel())
    for axis, (_, row) in zip(flattened, selected.iterrows(), strict=False):
        axis.axis("off")
        try:
            with Image.open(Path(row["path"])) as image:
                thumbnail = ImageOps.contain(
                    image.convert("RGB"),
                    (thumbnail_size, thumbnail_size),
                )
                axis.imshow(thumbnail)
        except Exception:
            axis.text(0.5, 0.5, "unreadable", ha="center", va="center")
        axis.set_title(_preview_caption(row), fontsize=7, pad=4)
    for axis in flattened[len(selected) :]:
        axis.axis("off")
    figure.suptitle(title, fontsize=14)
    figure.tight_layout(pad=1.5)
    figure.savefig(output_path, dpi=max(72, thumbnail_size), bbox_inches="tight")
    plt.close(figure)


def _build_html(
    result: Any,
    summary: dict[str, Any],
) -> str:
    title = str(result.config.get("report", {}).get("title", "Dataset Difficulty"))
    counts = summary.get("counts", {})
    bands = summary.get("difficulty_bands", {})
    splits = summary.get("splits", {})
    scored = result.samples[
        result.samples["sample_status"].eq("Scored")
    ].sort_values("difficulty_score", ascending=False)
    anomalies = result.samples[
        result.samples["is_anomaly"].fillna(False).astype(bool)
    ].sort_values("rel_path")
    warnings_html = "".join(
        f"<li>{html.escape(str(warning))}</li>" for warning in result.warnings
    )
    count_items = "".join(
        f"<div class=\"metric\"><span>{html.escape(str(key))}</span>"
        f"<strong>{html.escape(str(value))}</strong></div>"
        for key, value in counts.items()
    )
    table_columns = [
        "sample_id",
        "rel_path",
        "label",
        "blur_laplacian_var",
        "brightness",
        "contrast",
        "image_entropy",
        "spectral_entropy",
        "high_freq_energy_ratio",
        "visual_quality_risk",
        "visual_complexity_score",
        "visual_accessibility_risk",
        "visual_contribution",
        "semantic_complexity",
        "semantic_contribution",
        "difficulty_band",
        "difficulty_score",
        "difficulty_scope",
        "anomaly_types",
        "split",
    ]
    styles = """
    body { margin: 0; background: #f4f6f8; color: #17212b;
      font-family: "Segoe UI", Arial, sans-serif; line-height: 1.45; }
    main { max-width: 1180px; margin: 0 auto; padding: 28px; }
    header { border-bottom: 3px solid #176b87; padding-bottom: 14px; margin-bottom: 22px; }
    h1 { font-size: 29px; margin: 0 0 8px; letter-spacing: 0; }
    h2 { font-size: 20px; margin: 28px 0 12px; }
    .note { color: #4c5b67; max-width: 860px; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
      gap: 10px; margin: 18px 0; }
    .metric { background: white; border: 1px solid #d7e0e5; border-radius: 6px;
      padding: 11px 13px; }
    .metric span { display: block; color: #5b6a75; font-size: 12px; text-transform: uppercase; }
    .metric strong { font-size: 22px; }
    .table-wrap { overflow-x: auto; background: white; border: 1px solid #d7e0e5; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5ebee;
      white-space: nowrap; }
    th { background: #edf2f4; font-weight: 600; }
    img { max-width: 100%; border: 1px solid #d7e0e5; background: white; }
    a { color: #176b87; }
    .warnings { background: #fff8e6; border: 1px solid #ead697; border-radius: 6px;
      padding: 10px 18px; }
    .empty { color: #6a7780; }
    """
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{styles}</style>
</head>
<body>
<main>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="note">Training-free relative difficulty prior. Scores are not
    accuracy predictions. Label rarity is reported independently and does not
    affect the composite score. Anomalies are listed separately from Easy,
    Medium, and Hard bands.</p>
  </header>
  <section><h2>Dataset Counts</h2><div class="metrics">{count_items}</div></section>
  <section>
    <h2>Score Interpretation</h2>
    <p class="note"><strong>{html.escape(str(summary.get("difficulty_scope", "unscored")))}</strong>:
    {html.escape(str(summary.get("score_interpretation", "")))}</p>
    <p class="note">Image entropy and frequency complexity are descriptive
    visual-complexity signals represented by visual_complexity_score. They are
    not assumed to imply higher task difficulty and do not contribute to the
    composite score. Image quality, visual accessibility risk, semantic
    complexity, and the final prior are shown separately. When CLIP is
    unavailable, the report is visual-only.</p>
  </section>
  <section>
    <h2>Difficulty Bands</h2>
    <p class="note">{html.escape(json.dumps(_sanitize(bands), ensure_ascii=False))}</p>
    <h2>Split Counts</h2>
    <p class="note">{html.escape(json.dumps(_sanitize(splits), ensure_ascii=False))}</p>
  </section>
  <section>
    <h2>Hardest Scored Samples</h2>
    {_html_table(scored, table_columns)}
  </section>
  <section>
    <h2>Hard Sample Preview</h2>
    <img src="hard_samples.png" alt="Hard sample preview">
  </section>
  <section>
    <h2>Anomaly Samples</h2>
    {_html_table(anomalies, table_columns)}
    <img src="anomaly_samples.png" alt="Anomaly sample preview">
  </section>
  <section>
    <h2>Warnings</h2>
    <div class="warnings"><ul>{warnings_html or '<li>None</li>'}</ul></div>
  </section>
  <section>
    <h2>Artifacts</h2>
    <ul>
      <li><a href="samples.csv">samples.csv</a></li>
      <li><a href="summary.json">summary.json</a></li>
      <li><a href="split_manifest.csv">split_manifest.csv</a></li>
      <li><a href="split_summary.json">split_summary.json</a></li>
    </ul>
  </section>
</main>
</body>
</html>
"""


def write_reports(
    result: Any,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, str]:
    """Write all required CSV, JSON, HTML, PNG, and split artifacts."""

    output_path = Path(output_dir).resolve()
    input_root = getattr(result, "input_root", None)
    if input_root is not None:
        resolved_input = Path(input_root).resolve()
        if output_path == resolved_input or resolved_input in output_path.parents:
            raise ValueError("Report output must be outside the input dataset tree")
    output_path.mkdir(parents=True, exist_ok=True)
    report_config = config.get("report", {})
    preview_count = int(report_config.get("preview_count", 12))
    preview_columns = int(report_config.get("preview_columns", 4))
    thumbnail_size = int(report_config.get("image_thumbnail_size", 180))

    _csv_ready(result.samples).to_csv(
        output_path / "samples.csv", index=False, encoding="utf-8-sig"
    )
    manifest_columns = [
        "sample_id",
        "path",
        "rel_path",
        "label",
        "difficulty_band",
        "difficulty_score",
        "visual_contribution",
        "semantic_contribution",
        "difficulty_scope",
        "label_rarity_score",
        "label_rarity_contribution",
        "group_id",
        "component_coverage",
        "split",
        "is_anomaly",
        "anomaly_types",
    ]
    manifest_columns = [
        column for column in manifest_columns if column in result.samples.columns
    ]
    _csv_ready(result.samples[manifest_columns]).to_csv(
        output_path / "split_manifest.csv", index=False, encoding="utf-8-sig"
    )

    scored = result.samples[
        result.samples["sample_status"].eq("Scored")
    ].sort_values("difficulty_score", ascending=False)
    anomalies = result.samples[
        result.samples["is_anomaly"].fillna(False).astype(bool)
    ].sort_values("rel_path")
    _preview_sheet(
        scored,
        output_path / "hard_samples.png",
        "Hardest scored samples",
        preview_count,
        preview_columns,
        thumbnail_size,
    )
    _preview_sheet(
        anomalies,
        output_path / "anomaly_samples.png",
        "Anomaly samples",
        preview_count,
        preview_columns,
        thumbnail_size,
    )

    artifact_names = [
        "samples.csv",
        "summary.json",
        "report.html",
        "hard_samples.png",
        "anomaly_samples.png",
        "split_manifest.csv",
        "split_summary.json",
    ]
    summary = dict(result.summary)
    summary.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_root": str(result.input_root),
            "label_file": str(result.label_file) if result.label_file else None,
            "semantic_backend": result.semantic_backend,
            "warnings": list(result.warnings),
            "output_directory": str(output_path),
            "artifacts": artifact_names,
            "limitations": [
                "Difficulty scores are dataset-relative priors, not predicted accuracy.",
                "Quality issues and visual complexity are not equivalent to task difficulty.",
                "Label rarity is an independent dataset statistic and contributes zero.",
                "Visual-only scores do not represent complete semantic difficulty.",
                "Unavailable metrics remain null and are excluded from weighted scores.",
            ],
        }
    )
    split_summary = _sanitize(result.split_summary)
    _write_json(output_path / "summary.json", summary)
    _write_json(output_path / "split_summary.json", split_summary)
    (output_path / "report.html").write_text(
        _build_html(result, _sanitize(summary)),
        encoding="utf-8",
    )
    return {name: str(output_path / name) for name in artifact_names}
