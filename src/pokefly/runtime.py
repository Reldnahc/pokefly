"""Keep model data and compiler caches local to the project by default."""

from __future__ import annotations

import os
from pathlib import Path


def configure_runtime() -> Path:
    root = Path.cwd()
    os.environ.setdefault("FLY_DATA", str(root / "fly-data"))
    for variable, directory in (
        ("CUPY_CACHE_DIR", "cupy"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("NUMBA_CACHE_DIR", "numba"),
    ):
        os.environ.setdefault(variable, str(root / ".cache" / directory))
    return Path(os.environ["FLY_DATA"]).resolve()
