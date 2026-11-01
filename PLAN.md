# Heatsplat — product brief and build plan

**One line:** Scan a building with a phone and a clip-on thermal camera, get a 3D Gaussian splat
where every heat leak glows, is ranked, priced, and explained.

## Who it is for and why it wins

- **User:** homeowners, home-energy auditors, and property managers who today get a stack of flat
  thermal photos and have to guess where each one was taken.
- **Problem:** thermal photos lose spatial context. A 3D thermal model answers "where exactly is
  the leak, how big is it, and what is it costing me" in one orbitable view.
- **Differentiation vs. prior art:** ThermalGaussian, Thermal-NeRF and ThermSplit-GS are research
  training code. Nothing packages capture → splat → thermal bake → leak detection → cost estimate →
  audit report into a product a judge can click through in 90 seconds. That pipeline is the project.

## Demo script (what judges see)

1. Pick a scanned building. The splat loads and auto-orbits.
2. Toggle **RGB → Thermal → Leaks**. Leaks glow, everything else dims.
3. Click a leak marker. Side panel: temperature delta, area, estimated annual loss in dollars,
   likely cause, recommended fix.
4. Drag the sensitivity slider. Detection reruns on the backend live and markers update.
5. Hit **Generate audit report**. A prioritised remediation report with a fix per leak opens and
   downloads as markdown. Today it is template-generated (the modal says so); phase 2b swaps in
   a Claude agent with retrieval over weatherization guidance behind the same endpoint. Do not
   claim agent or RAG output on stage until that lands.

Every step is a visible feature. Reliability work stays invisible, so it comes second.

## Architecture

```
phone RGB video + thermal frames
        │
        ▼
 pipeline/ (Python)
   capture ingest ──► poses (COLMAP / Polycam export) ──► Gaussian splat (.ply)
   thermal bake   ──► per-Gaussian temperature (back-project thermal frames through poses)
   detect         ──► local-baseline residual + DBSCAN clusters = leaks (unsupervised ML)
   cost           ──► ΔT × area × degree-days ──► kWh/yr ──► $/yr
   export         ──► rgb.splat, thermal.splat, leaks.splat, leaks.json
        │
        ▼
 backend/ (FastAPI)
   GET  /api/scenes                      list scans
   GET  /api/scenes/{id}                 metadata + stats
   GET  /api/scenes/{id}/splat?mode=     rgb | thermal | leaks
   GET  /api/scenes/{id}/leaks           ranked leak list
   POST /api/scenes/{id}/analyze         rerun detection with new thresholds
   POST /api/scenes/{id}/report          agent-written audit report (Claude + RAG)
        │
        ▼
 frontend/ (React + Vite + three.js + gaussian-splats-3d)
   Viewer · mode toggle · leak markers · leak panel · sensitivity slider · report button
```

### Why these choices

- **Gaussian splats, not NeRF:** real-time in a browser, so the demo is interactive, not a video.
- **Pre-baked per-Gaussian temperature:** the viewer never does thermal math. Swapping between
  rgb/thermal/leaks is a file swap, so it is instant on stage.
- **Unsupervised detection (local median baseline + DBSCAN):** no labeled thermal data exists for
  this, and anomaly-vs-neighborhood is exactly what a human auditor does with a thermal camera.
- **FastAPI:** the detection and report code is Python; FastAPI exposes it with zero glue and
  gives free OpenAPI docs at /docs, which is itself a demo-able artifact.
- **React + three.js:** the splat renderer is a three.js scene, so leak markers are ordinary
  three.js meshes. Any 3D overlay is a few lines.
- **Mac has no CUDA:** training a splat happens off this machine (Polycam/Luma phone export,
  OpenSplat on Metal, or a cloud GPU). The pipeline consumes a finished .ply plus poses, and a
  synthetic scene generator makes the whole stack demoable before any real capture exists.

## Build phases

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Synthetic house scene with injected leaks, detection, three .splat exports, FastAPI, React viewer with mode toggle and markers | done (Oct 31) |
| 1 | Leak detail panel with cost estimate, live sensitivity slider, stats bar, click-to-fly camera | done (Oct 31) |
| 2a | Audit report: template-based markdown with per-leak fixes, modal + download | done (Oct 31) |
| 2b | Audit-report agent (Claude tool use + retrieval over DOE weatherization guides) behind the same endpoint | next |
| 3 | Real capture ingest: `ingest` CLI takes a splat `.ply` + poses (nerfstudio `transforms.json` or a COLMAP text model) + thermal frames, bakes with occlusion, auto up-axis; proven end to end on a simulated capture (both pose routes) | done (Oct 31); waiting on a camera for a real scan, see docs/duke-scan-plan.md |
| 3b | Real building: ThermoScenes `buildingA_winter` (96 FLIR One frames, COLMAP poses) ingested via `scripts/thermoscenes_scene.sh`; sparse-point splat path (`points-to-splat`) for machines without a GPU; up axis from the walking plane, ground from camera height, density-adaptive clustering | done (Nov 1): dense OpenSplat splat (7000 iters, ~3 min on Metal via `scripts/build_opensplat.sh` + MPS patch); thermal bake uses the dataset's full calibration (fy + distortion); detector lights the glazed lobby and window frames (12 patches at 1.5 °C); sparse-point fallback kept |
| 4 | Polish: screenshot export, loading skeletons, mobile layout check | final day |

### Engineering notes worth repeating to judges

- **Surface-aware baseline:** each Gaussian is compared only with neighbours whose normal (its
  thinnest axis) faces the same way, so a wall is never baselined against the roof that meets it.
  This removed every false positive along the eaves.
- **Density-independent support:** the baseline neighbourhood is a voxel-downsampled support set,
  so the 2 m baseline radius means the same thing on a dense wall and a sparse lawn.
- **Significance vs. scene noise:** anomaly significance uses the global residual noise, not the
  local spread, so a leak next to a cold window is not masked by the window.
- **Hysteresis clustering:** clusters grow from a lower edge threshold but must contain a seed
  above the chosen one, so raising the threshold never fragments a leak into more leaks.
- **Residual caching:** the expensive baseline is cached per scene; the slider re-runs only
  thresholding and clustering, under a second for 110k Gaussians.

## Real capture path (hardware day)

Full field plan, camera picks and the Mac-only reconstruction commands: [docs/duke-scan-plan.md](docs/duke-scan-plan.md).

- Thermal camera: FLIR One Edge Pro or a TOPDON TC001 clipped rigidly to the phone.
- Record RGB video and a thermal burst on the same walk. Keep a fixed rigid offset.
- Get poses + splat: Polycam or Luma export (fastest), or COLMAP + OpenSplat (Metal) locally.
- Bake: project each thermal frame through its pose, accumulate per-Gaussian temperature
  weighted by splat opacity along the ray, median across frames.
- Best capture conditions: dusk or early morning, ≥10 °C indoor/outdoor difference, no direct sun.

## Repo layout

```
heatsplat/
  PLAN.md              this file
  pipeline/            Python package `heatsplat` + tests
  backend/             FastAPI app
  frontend/            React + Vite viewer
  data/scenes/<id>/    gaussians.npz, meta.json, exported .splat files, leaks.json
  docs/                notes, capture checklist, report corpus
```
