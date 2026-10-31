"""Atomic file writes: readers never see a half-written splat or JSON file."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_write_bytes(path, buffer.getvalue())
