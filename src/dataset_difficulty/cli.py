from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from dataset_difficulty import __version__
from dataset_difficulty.config import load_config
from dataset_difficulty.pipeline import analyze_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset-difficulty",
        description="Training-free difficulty prior analysis for image classification.",
    )
    parser.add_argument("data_root", help="Dataset directory")
    parser.add_argument("--labels", help="CSV or JSON label file")
    parser.add_argument(
        "--format",
        choices=["auto", "folder", "csv", "json"],
        default="auto",
        help="Dataset format",
    )
    parser.add_argument("--output", required=True, help="Report output directory")
    parser.add_argument("--config", help="YAML configuration override")
    parser.add_argument("--path-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--group-column", default=None)
    parser.add_argument(
        "--semantic-backend",
        choices=["auto", "clip", "disabled"],
        default=None,
    )
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    overrides: dict = {"data": {"format": args.format}}
    if args.path_column:
        overrides["data"]["path_column"] = args.path_column
    if args.label_column:
        overrides["data"]["label_column"] = args.label_column
    if args.group_column:
        overrides["data"]["group_column"] = args.group_column
    if args.seed is not None:
        overrides["project"] = {"seed": args.seed}
    clip_overrides: dict = {}
    if args.clip_model:
        clip_overrides["model_name"] = args.clip_model
    if args.allow_model_download:
        clip_overrides["allow_download"] = True
    if args.device:
        clip_overrides["device"] = args.device
    if clip_overrides:
        overrides["semantic"] = {"clip": clip_overrides}
    config = load_config(args.config, overrides=overrides)
    backend = (
        "disabled"
        if args.no_semantic
        else (
            args.semantic_backend
            or str(config.get("semantic", {}).get("backend", "auto"))
        )
    )
    try:
        result = analyze_dataset(
            Path(args.data_root),
            Path(args.output),
            config,
            label_file=Path(args.labels) if args.labels else None,
            semantic_backend=backend,
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Report: {result.output_dir / 'report.html'}")
    print(
        f"Samples: {result.summary['counts']['total']} | "
        f"Anomalies: {result.summary['counts']['anomalies']} | "
        f"Semantic backend: {result.semantic_backend}"
    )
    return 0
