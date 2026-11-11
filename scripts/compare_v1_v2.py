from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _percentile(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(
        pct=True, method="average", na_option="keep"
    )


def _inverse_percentile(series: pd.Series) -> pd.Series:
    return 1.0 - _percentile(series)


def _weighted_available(
    frame: pd.DataFrame,
    components: list[tuple[str, float]],
) -> tuple[pd.Series, pd.Series]:
    weighted_sum = pd.Series(0.0, index=frame.index)
    available_weight = pd.Series(0.0, index=frame.index)
    for column, weight in components:
        valid = frame[column].notna()
        weighted_sum.loc[valid] += frame.loc[valid, column] * weight
        available_weight.loc[valid] += weight
    score = weighted_sum / available_weight.replace(0.0, np.nan)
    return score, available_weight


def _bands(scores: pd.Series) -> pd.Series:
    valid = scores.dropna()
    result = pd.Series("Unscored", index=scores.index, dtype="object")
    if valid.empty:
        return result
    first = float(valid.quantile(0.33))
    second = float(valid.quantile(0.67))
    result.loc[valid.index] = np.select(
        [valid <= first, valid <= second],
        ["Easy", "Medium"],
        default="Hard",
    )
    return result


def recompute_v1(after: pd.DataFrame) -> pd.DataFrame:
    frame = after[after["sample_status"].eq("Scored")].copy()
    frame["v1_blur"] = _inverse_percentile(frame["blur_laplacian_var"])
    frame["v1_brightness"] = _percentile((frame["brightness"] - 0.5).abs())
    frame["v1_contrast"] = _inverse_percentile(frame["contrast"])
    spectral = _percentile(frame["spectral_entropy"])
    frequency = _percentile(frame["high_freq_energy_ratio"])
    frame["v1_frequency"] = pd.concat([spectral, frequency], axis=1).mean(
        axis=1, skipna=True
    )
    frame["v1_entropy"] = _percentile(frame["image_entropy"])
    visual, _ = _weighted_available(
        frame,
        [
            ("v1_blur", 0.30),
            ("v1_brightness", 0.15),
            ("v1_contrast", 0.15),
            ("v1_frequency", 0.15),
            ("v1_entropy", 0.25),
        ],
    )
    frame["v1_visual"] = visual.clip(0.0, 1.0)
    labels = frame["label"].fillna("").astype(str)
    counts = labels[labels.ne("")].value_counts()
    maximum = float(counts.max()) if not counts.empty else np.nan
    frame["v1_label_rarity"] = labels.map(counts).map(
        lambda count: (
            1.0 - float(count) / maximum
            if pd.notna(count) and np.isfinite(maximum) and maximum > 0
            else np.nan
        )
    )
    frame["v1_semantic"] = np.where(
        frame["semantic_available"].fillna(False).astype(bool),
        pd.to_numeric(frame["semantic_complexity"], errors="coerce"),
        np.nan,
    )
    score, available_weight = _weighted_available(
        frame,
        [
            ("v1_visual", 0.20),
            ("v1_semantic", 0.60),
            ("v1_label_rarity", 0.20),
        ],
    )
    frame["v1_difficulty_score"] = score.clip(0.0, 1.0)
    frame["v1_difficulty_band"] = _bands(frame["v1_difficulty_score"])
    frame["v1_visual_contribution"] = (
        frame["v1_visual"] * 0.20 / available_weight
    )
    frame["v1_semantic_contribution"] = (
        frame["v1_semantic"] * 0.60 / available_weight
    )
    frame["v1_label_contribution"] = (
        frame["v1_label_rarity"] * 0.20 / available_weight
    )
    return frame


def compare_analysis_frames(
    after: pd.DataFrame,
    before: pd.DataFrame | None,
    output: str | Path,
) -> dict[str, Any]:
    recomputed = recompute_v1(after)
    if before is not None:
        v1 = before.set_index("sample_id")[
            ["difficulty_score", "difficulty_band"]
        ].rename(
            columns={
                "difficulty_score": "v1_difficulty_score",
                "difficulty_band": "v1_difficulty_band",
            }
        )
        merged = recomputed.set_index("sample_id").join(v1, how="inner")
        if "v1_visual_contribution" not in merged:
            merged["v1_visual_contribution"] = np.nan
        if "v1_semantic_contribution" not in merged:
            merged["v1_semantic_contribution"] = np.nan
        if "v1_label_contribution" not in merged:
            merged["v1_label_contribution"] = np.nan
    else:
        merged = recomputed.set_index("sample_id")

    transitions = pd.crosstab(
        merged["v1_difficulty_band"], merged["difficulty_band"]
    )
    report = {
        "common_scored_samples": int(len(merged)),
        "mean_v1_score": float(merged["v1_difficulty_score"].mean()),
        "mean_v2_score": float(pd.to_numeric(merged["difficulty_score"]).mean()),
        "median_v1_score": float(merged["v1_difficulty_score"].median()),
        "median_v2_score": float(pd.to_numeric(merged["difficulty_score"]).median()),
        "mean_label_contribution_v1": float(
            merged["v1_label_contribution"].fillna(0).mean()
        ),
        "mean_label_contribution_v2": float(
            pd.to_numeric(
                merged.get("label_rarity_contribution", 0.0), errors="coerce"
            ).fillna(0).mean()
        ),
        "mean_visual_contribution_v1": float(
            merged["v1_visual_contribution"].fillna(0).mean()
        ),
        "mean_visual_contribution_v2": float(
            pd.to_numeric(merged["visual_contribution"], errors="coerce")
            .fillna(0)
            .mean()
        ),
        "mean_semantic_contribution_v1": float(
            merged["v1_semantic_contribution"].fillna(0).mean()
        ),
        "mean_semantic_contribution_v2": float(
            pd.to_numeric(merged["semantic_contribution"], errors="coerce")
            .fillna(0)
            .mean()
        ),
        "spearman_rank_correlation": float(
            merged["v1_difficulty_score"].corr(
                pd.to_numeric(merged["difficulty_score"]), method="spearman"
            )
        ),
        "changed_band_count": int(
            (
                merged["v1_difficulty_band"]
                != merged["difficulty_band"]
            ).sum()
        ),
        "band_transitions": {
            str(v1_band): {
                str(v2_band): int(value)
                for v2_band, value in row.items()
            }
            for v1_band, row in transitions.iterrows()
        },
    }
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    markdown = [
        "# V1 vs V2 Difficulty Comparison",
        "",
        f"- Common scored samples: {report['common_scored_samples']}",
        f"- Mean V1 score: {report['mean_v1_score']:.4f}",
        f"- Mean V2 score: {report['mean_v2_score']:.4f}",
        f"- Spearman rank correlation: {report['spearman_rank_correlation']:.4f}",
        f"- Changed difficulty bands: {report['changed_band_count']}",
        f"- Mean Label Rarity contribution V1: {report['mean_label_contribution_v1']:.4f}",
        f"- Mean Label Rarity contribution V2: {report['mean_label_contribution_v2']:.4f}",
        "",
        "## Band Transitions",
        "",
        "```json",
        json.dumps(report["band_transitions"], indent=2),
        "```",
    ]
    (output_path / "comparison.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare V1 scoring with V2 analysis output."
    )
    parser.add_argument("--after-csv", required=True)
    parser.add_argument("--before-csv", default=None)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    after = pd.read_csv(args.after_csv)
    before = pd.read_csv(args.before_csv) if args.before_csv else None
    report = compare_analysis_frames(after, before, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
