from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def select_balanced_indices(
    targets: Sequence[int],
    limit: int | None,
    num_classes: int,
) -> list[int]:
    if limit is None or limit <= 0 or limit >= len(targets):
        return list(range(len(targets)))
    per_class = max(1, limit // num_classes)
    selected: list[int] = []
    counts = [0] * num_classes
    for index, target in enumerate(targets):
        if counts[int(target)] < per_class:
            selected.append(index)
            counts[int(target)] += 1
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected_set = set(selected)
        for index in range(len(targets)):
            if index not in selected_set:
                selected.append(index)
                if len(selected) >= limit:
                    break
    return selected


def export_cifar_subset(
    images: np.ndarray,
    targets: Sequence[int],
    class_names: Sequence[str],
    split: str,
    output: str | Path,
    limit: int | None = 10000,
) -> dict[str, Any]:
    output_path = Path(output).resolve()
    image_root = output_path / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    indices = select_balanced_indices(targets, limit, len(class_names))
    rows = ["path,label,group_id"]
    for index in indices:
        target = int(targets[index])
        label = str(class_names[target])
        relative = (
            Path("images") / label / f"{split}_{index:06d}.png"
        )
        path = output_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        image_bgr = cv2.cvtColor(np.asarray(images[index]), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(path.suffix, image_bgr)
        if not ok:
            raise RuntimeError(f"Unable to encode CIFAR image {index}")
        encoded.tofile(path)
        group_id = f"cifar10-{split}-{index:06d}"
        rows.append(f"{relative.as_posix()},{label},{group_id}")
    (output_path / "labels.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    metadata = {
        "source": "CIFAR-10",
        "split": split,
        "requested_limit": limit,
        "exported_count": len(indices),
        "class_names": list(class_names),
        "class_counts": {
            label: sum(1 for row in rows[1:] if f",{label}," in row)
            for label in class_names
        },
    }
    (output_path / "dataset_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def _image_bytes(record: Any) -> bytes:
    if isinstance(record, dict):
        value = record.get("bytes")
        if value is not None:
            return bytes(value)
    if isinstance(record, (bytes, bytearray)):
        return bytes(record)
    raise ValueError("Parquet img field does not contain encoded image bytes")


def export_cifar_parquet(
    parquet_path: str | Path,
    output: str | Path,
    class_names: Sequence[str],
    split: str,
    limit: int = 10000,
) -> dict[str, Any]:
    import pandas as pd

    frame = pd.read_parquet(parquet_path)
    if "img" not in frame or "label" not in frame:
        raise ValueError("CIFAR parquet must contain img and label columns")
    images = []
    for record in frame["img"]:
        encoded = np.frombuffer(_image_bytes(record), dtype=np.uint8)
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Unable to decode an image from CIFAR parquet")
        images.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return export_cifar_subset(
        images=np.stack(images),
        targets=frame["label"].astype(int).tolist(),
        class_names=class_names,
        split=split,
        output=output,
        limit=limit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a reproducible CIFAR-10 image/label directory."
    )
    parser.add_argument("--root", help="Torchvision CIFAR-10 data root")
    parser.add_argument(
        "--parquet",
        help="Hugging Face-style CIFAR-10 parquet file",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.parquet:
        metadata = export_cifar_parquet(
            parquet_path=args.parquet,
            output=args.output,
            class_names=CIFAR10_CLASSES,
            split=args.split,
            limit=args.limit,
        )
        print(json.dumps(metadata, indent=2))
        return 0
    if not args.root:
        raise ValueError("--root is required unless --parquet is supplied")
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(
        root=args.root,
        train=args.split == "train",
        download=args.download,
    )
    metadata = export_cifar_subset(
        images=dataset.data,
        targets=dataset.targets,
        class_names=dataset.classes,
        split=args.split,
        output=args.output,
        limit=args.limit,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
