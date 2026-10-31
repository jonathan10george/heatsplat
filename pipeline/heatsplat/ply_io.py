"""Read and write 3D Gaussian Splatting `.ply` files (the format trainers and phone apps export).

The standard layout stores log-scales, logit-opacity and spherical-harmonic DC colour
coefficients; this module converts to and from the linear values `GaussianScene` uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from heatsplat.fsutil import atomic_write_bytes
from heatsplat.scene import SCALE_FLOOR, GaussianScene

SH_C0 = 0.28209479177387814  # zeroth-order spherical harmonic basis constant
PLY_TYPES = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
}
REQUIRED = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")


def _parse_header(raw: bytes) -> tuple[int, list[tuple[str, str]], int]:
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError("not a PLY file: no end_header")
    newline = raw.find(b"\n", end)  # header ends after end_header plus \n or \r\n
    if newline < 0:
        raise ValueError("not a PLY file: truncated header")
    lines = raw[:end].decode("ascii", errors="replace").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError("not a PLY file: missing 'ply' magic")
    if not any(line.startswith("format binary_little_endian") for line in lines):
        raise ValueError("only binary_little_endian PLY is supported")
    count = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in lines:
        parts = line.split()
        if parts[:1] == ["element"]:
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                count = int(parts[2])
        elif parts[:1] == ["property"] and in_vertex:
            if parts[1] == "list":
                raise ValueError("list properties are not supported in the vertex element")
            if parts[1] not in PLY_TYPES:
                raise ValueError(f"unsupported PLY property type {parts[1]!r} for {parts[2]!r}")
            props.append((parts[2], PLY_TYPES[parts[1]]))
    return count, props, newline + 1


def read_ply(path: Path) -> GaussianScene:
    """Load a 3DGS .ply into a GaussianScene (linear scales, sigmoid opacity, RGB from SH DC)."""
    raw = Path(path).read_bytes()
    count, props, offset = _parse_header(raw)
    names = {name for name, _ in props}
    missing = [name for name in REQUIRED if name not in names]
    if missing:
        raise ValueError(f"PLY is missing Gaussian fields: {missing}")
    dtype = np.dtype(props)
    rows = np.frombuffer(raw, dtype=dtype, count=count, offset=offset)
    positions = np.stack([rows["x"], rows["y"], rows["z"]], axis=1).astype(np.float32)
    colors = np.clip(0.5 + SH_C0 * np.stack([rows["f_dc_0"], rows["f_dc_1"], rows["f_dc_2"]], axis=1), 0.0, 1.0).astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-rows["opacity"].astype(np.float64)))).astype(np.float32)
    scales = np.maximum(np.exp(np.stack([rows["scale_0"], rows["scale_1"], rows["scale_2"]], axis=1).astype(np.float64)), SCALE_FLOOR).astype(np.float32)
    rotations = np.stack([rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"]], axis=1).astype(np.float64)
    rotations = (rotations / np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    finite = np.isfinite(positions).all(axis=1) & np.isfinite(scales).all(axis=1) & np.isfinite(rotations).all(axis=1) & np.isfinite(opacities)
    if not finite.all():
        positions, scales, rotations, colors, opacities = (a[finite] for a in (positions, scales, rotations, colors, opacities))
    return GaussianScene(positions, scales, rotations, colors, opacities)


def write_ply(path: Path, scene: GaussianScene) -> int:
    """Write a GaussianScene as a standard 3DGS .ply (DC colour only, no higher SH). Returns N."""
    n = len(scene)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
              ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"), ("opacity", "<f4"),
              ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
              ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4")]
    rows = np.zeros(n, dtype=np.dtype(fields))
    rows["x"], rows["y"], rows["z"] = scene.positions.T
    dc = (scene.colors.astype(np.float64) - 0.5) / SH_C0
    rows["f_dc_0"], rows["f_dc_1"], rows["f_dc_2"] = dc.T
    o = np.clip(scene.opacities.astype(np.float64), 1e-6, 1 - 1e-6)
    rows["opacity"] = np.log(o / (1 - o))
    rows["scale_0"], rows["scale_1"], rows["scale_2"] = np.log(scene.scales.astype(np.float64)).T
    rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"] = scene.rotations.T
    header = "\n".join(["ply", "format binary_little_endian 1.0", f"element vertex {n}", *(f"property float {name}" for name, _ in fields), "end_header", ""])
    atomic_write_bytes(Path(path), header.encode("ascii") + rows.tobytes())
    return n
