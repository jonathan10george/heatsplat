# Heatsplat

3D thermal building scans with Gaussian splatting. Finds heat leaks, ranks them, prices them,
and explains them in an orbitable browser view.

See [PLAN.md](PLAN.md) for the product brief, architecture, and build plan.

## Run the demo

```bash
# 1. Python deps (uv manages the venv at the repo root)
uv sync --all-extras
```

```bash
# 2. Build the synthetic demo scene (≈6 s) and detect leaks
uv run python -m heatsplat.cli synth --scene demo-house
```

```bash
# 3. API on :8000 (OpenAPI docs at /docs)
uv run uvicorn app.main:app --app-dir backend --reload --port 8000
```

```bash
# 4. Viewer on :5173 (proxies /api to the backend)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173.

## Tests

```bash
uv run pytest -q
```

## Re-run detection with different thresholds

```bash
uv run python -m heatsplat.cli analyze --scene demo-house --min-delta 2.5
```

## Ingest a real capture

```bash
uv run python -m heatsplat.cli ingest --scene my-building --ply splat.ply --colmap sparse_txt --thermal-dir thermal --thermal-range 0,20 --thermal-hfov 56 --up auto --scale 0.83 --camera-height 1.5
```

`--scale` is metres per reconstruction unit (COLMAP units are arbitrary; measure one known
length). `--camera-height` is how high the phone was held; it puts the ground at y=0 and drops
reconstruction junk far below it. Poses come from a COLMAP text model (`scripts/colmap_sfm.sh
photos ws` makes one on CPU) or a nerfstudio `transforms.json`.

## No GPU for a splat? Use the sparse points

```bash
uv run python -m heatsplat.cli points-to-splat --points ws/sparse/0_txt/points3D.txt --out sparse.ply
```

COLMAP's sparse points become surface-aligned disk Gaussians, so the thermal bake and the
detector run on a real capture within minutes. Detection adapts its clustering radius to the
point spacing, but a sparse cloud only supports coarse anomalies: train a real splat with
OpenSplat (see `docs/duke-scan-plan.md`) for the full analysis.

## Real building: ThermoScenes

[ThermoScenes](https://zenodo.org/records/10835108) (Hassan et al., CC-BY-4.0) ships paired
RGB + thermal photos of real façades taken with a FLIR One on a phone, with COLMAP poses.
`scripts/thermoscenes_scene.sh` turns a scene folder into a heatsplat scene in one go:

```bash
scripts/thermoscenes_scene.sh data/captures/thermoscenes-raw/ThermoScenes/buildingA_winter thermoscenes-buildingA-winter 0.9 "Building A, winter"
```

The third argument is metres per unit (0.9 for `buildingA_winter`, estimated from the storey
pitch and the photographer's stride). With `tools/OpenSplat/build/opensplat` present the script
trains a dense splat first; otherwise it falls back to the sparse points.

`--transforms transforms.json --up z` works for nerfstudio / Polycam / Luma poses instead of `--colmap`.
The capture, reconstruction and scan-day checklist live in [docs/duke-scan-plan.md](docs/duke-scan-plan.md).
