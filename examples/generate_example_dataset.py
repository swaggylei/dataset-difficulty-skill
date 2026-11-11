from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "toy_dataset"


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Unable to encode {path}")
    encoded.tofile(path)


def _tinted_texture(
    rng: np.random.Generator,
    tint: tuple[float, float, float],
    size: int = 80,
) -> np.ndarray:
    noise = rng.integers(15, 235, size=(size, size, 3), dtype=np.uint8)
    return np.clip(noise * np.asarray(tint), 0, 255).astype(np.uint8)


def main() -> None:
    rng = np.random.default_rng(20261111)
    for index in range(3):
        _write_image(
            DATASET / "cat" / f"{index:02d}_texture.png",
            _tinted_texture(rng, (1.0, 0.45, 0.30)),
        )
        _write_image(
            DATASET / "dog" / f"{index:02d}_texture.png",
            _tinted_texture(rng, (0.25, 0.45, 1.0)),
        )

    cat_source = DATASET / "cat" / "00_texture.png"
    duplicate = DATASET / "cat" / "03_exact_duplicate.png"
    duplicate.write_bytes(cat_source.read_bytes())
    source_image = cv2.imdecode(np.fromfile(cat_source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source_image is None:
        raise RuntimeError("Generated source image could not be decoded")
    _write_image(
        DATASET / "cat" / "04_blurred.png",
        cv2.GaussianBlur(source_image, (21, 21), 8),
    )
    (DATASET / "dog" / "04_corrupt.png").write_bytes(b"not-an-image")

    rows = ["path,label"]
    for path in sorted(DATASET.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            relative = path.relative_to(DATASET).as_posix()
            label = relative.split("/", 1)[0]
            rows.append(f"{relative},{label}")
    (DATASET / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Example dataset: {DATASET}")


if __name__ == "__main__":
    main()
