"""Where scene data lives. Override with the HEATSPLAT_DATA environment variable."""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def data_root() -> Path:
    return Path(os.environ.get("HEATSPLAT_DATA", REPO_ROOT / "data" / "scenes"))


def scene_dir(scene_id: str) -> Path:
    if not SCENE_ID_PATTERN.fullmatch(scene_id or ""):
        raise ValueError(f"invalid scene id: {scene_id[:80]!r}")
    return data_root() / scene_id
