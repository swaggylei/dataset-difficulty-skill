from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_path(request) -> Path:
    """Provide a sandbox-friendly per-test directory under the project."""

    node_hash = hashlib.sha256(request.node.nodeid.encode("utf-8")).hexdigest()[:12]
    path = Path(__file__).resolve().parents[1] / ".test_artifacts" / node_hash
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    yield path


@pytest.fixture
def sample_array() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)


def require_cv2():
    pytest.importorskip("cv2", reason="OpenCV is required for image tests")


def write_image(path: Path, image: np.ndarray) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    assert ok
    encoded.tofile(path)
