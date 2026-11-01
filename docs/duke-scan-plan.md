# Duke building scan: field plan

Goal: one real Duke façade as a thermal Gaussian splat with detected leaks, loaded in the
viewer next to the synthetic house. A real anomaly on a real building is the demo's headline.

## 1. Camera

Buy (or borrow, see below) one phone-attachable thermal camera. Two tiers:

| Pick | Sensor | Connects | Street price (Oct 2026, verify) | Notes |
|---|---|---|---|---|
| **TOPDON TC001** (recommended) | 256×192, InfiRay core | USB-C (Android; iPhone 15+ via USB-C, confirm the iOS app on the listing) | ~$200–320 | Same core as the P2 Pro, usually cheaper. 56° lens. |
| **InfiRay P2 Pro** | 256×192 | USB-C | ~$250–400 | 9 g, same class; macro lens we don't need. |
| FLIR ONE Edge Pro | 160×120 native | Wi-Fi, any phone | ~$600–700 | Wireless is convenient but it is 2–3× the price for a lower native resolution. |
| TOPDON TC002 / FLIR ONE Pro (Lightning) | 256×192 / 160×120 | Lightning | similar | Only if the phone is an iPhone 14 or older. |

What the pipeline actually needs from the app: **grayscale ("white hot") palette and a locked
temperature span** (for example 0–20 °C) for every frame. Then each image maps linearly to °C
and `ingest --thermal-range 0,20` reads it; no radiometric export is required. If the app can
export radiometric CSV or 16-bit TIFF, better, but it is optional.

Borrowing options to try before buying (both unconfirmed; the Link lends no thermal camera):
- Email **Duke Facilities, Energy Management** (facilities.duke.edu → Energy & Resource
  Management). Energy auditors own thermal cameras, and "two students mapping heat loss on an
  East Campus dorm in 3D" is the kind of project they tend to say yes to. Offer them the result.
- Ask the **Innovation Co-Lab** and **Pratt IT**; neither lists one publicly.

## 2. Target buildings (East Campus first)

1. **Gilbert-Addoms (GA)**, East Campus dorm, ~60 years old. A Bass Connections team already
   documented its energy inefficiency against the newer LEED-certified Keohane; our scan turns
   that paper into a 3D map of where the heat leaves. Flat brick walls and regular windows are
   ideal for splatting. Scan the exterior only, from the path.
2. **Smith Warehouse** (backup): long, flat brick tobacco-warehouse façades, many windows,
   quiet in the evening. Easy geometry, strong visuals.
3. **A West Campus Gothic façade** (showpiece only): stone and leaded glass make a beautiful
   splat, but the geometry is complex and the stone's thermal mass flattens the signal.

Etiquette: public paths only, no close-ups of occupied windows at night (thermal cannot see
through glass, but be visibly considerate), and a courtesy email to Housing/the RA for a dorm.

## 3. When

- **After dusk, 1–2 h after sunset**, so the sun's heat has left the bricks. Sunset in Durham
  is ~5:15 pm once DST ends on Nov 1; scan 7–9 pm.
- A dry evening with the indoor/outdoor difference ≥ 10 °C. Early-November Durham nights
  average ~7 °C, so a heated dorm gives ~14 °C: plenty. Check weather.gov for Durham, NC.
- No rain in the previous 12 h (wet brick evaporatively cools and confuses the signal).

## 4. Capture (two people, 15 minutes per façade)

1. Tape the thermal camera rigidly to the phone, lens next to the phone's main camera.
2. Record an **RGB video** (4K/30, exposure locked) walking a slow arc at 3–6 m from the
   façade, phone landscape, keeping the whole wall plus eaves and ground line in frame; one
   pass at eye level, a second pass ~1 m lower if the wall is tall. 60–90 s per pass.
3. At the same time record a **thermal video** in the app (white-hot, locked span).
   **Clap once in view of both cameras at the start** to sync them.
4. Note the time, outside temperature, and whether the building is heated.

## 5. Reconstruct and bake (Mac only, no GPU needed)

```bash
# frames from both videos at the same rate, trimmed to the clap (offsets in seconds)
ffmpeg -ss 1.4 -i rgb.mp4 -vf fps=2 frames/frame_%04d.jpg
ffmpeg -ss 0.9 -i thermal.mp4 -vf fps=2 -pix_fmt gray thermal/frame_%04d.png
```

```bash
# poses + sparse points from the RGB frames, CPU only (brew install colmap); ~10 min for 150 photos
scripts/colmap_sfm.sh frames ws
```

OpenSplat (github.com/pierotofy/OpenSplat) trains the splat on Apple Metal. Building it is a
one-off (Homebrew cmake, opencv and pytorch plus the Xcode Metal toolchain must be installed):

```bash
scripts/build_opensplat.sh
```

The script clones the pinned commit under `tools/OpenSplat`, applies
`scripts/opensplat-mps-backward.patch` (upstream fetches a Metal command buffer off the MPS
dispatch queue in the backward pass, which libtorch 2.9+ aborts on with a SIGTRAP) and builds
incrementally, so rerunning it is cheap.

```bash
# splat in COLMAP's coordinate frame (the default; do not pass --center); 7000 iterations take ~3 min on an M4 Air for ~100 frames
tools/OpenSplat/build/opensplat ws -n 7000 -o ga.ply
```

No GPU and no time to build? `uv run python -m heatsplat.cli points-to-splat --points
ws/sparse/0_txt/points3D.txt --out ga.ply` turns COLMAP's sparse points into disk Gaussians in
seconds. The thermal view works; detection on a few thousand points only catches coarse
anomalies, so treat it as the preview and the trained splat as the result.

```bash
# bake thermal onto the splat, detect, export; poses and splat share COLMAP's frame
uv run python -m heatsplat.cli ingest --scene gilbert-addoms --ply ga.ply --colmap ws/sparse/0_txt \
  --thermal-dir thermal --thermal-range 0,20 --thermal-hfov 56 --up auto --scale 0.83 --camera-height 1.5 \
  --name "Gilbert-Addoms, East Campus"
```

Then open the viewer: the scene appears in the picker. Check thermal mode first. If the thermal
layer is shifted relative to the bricks, the lens FOV is wrong (`--thermal-hfov`) or the clap
offset is off by a frame; both are quick to redo. Fallback if time is short: Luma, Polycam or
KIRI export a splat `.ply` from the same RGB video on the phone, and `--transforms` takes
nerfstudio-style poses. Their world up axis varies (ARKit exports are often Y up): try
`--up auto` first, then `--up y` or `--up z` if the house lies on its side.

Three flags matter for the numbers: `--scale` (metres per reconstruction unit; measure one known
length such as a door height in the splat and divide, or use the storey pitch) so areas and
costs are metric, `--camera-height` (how high the phone was held, usually 1.5 m) so the ground
sits at y=0 and junk far below it is dropped, and `--thermal-hfov` for the real thermal lens.
The CLI warns when scale or FOV is missing. `--up auto` takes gravity from the plane the camera
centres lie in (a walk at constant phone height), so keep the phone at one height and cover a
patch of ground, not a single straight line.

This path is exercised end to end on a synthetic capture (`pipeline/tests/test_ingest.py`) and
on a real one: ThermoScenes `buildingA_winter`, 96 FLIR One frames of a four-storey office
block in winter with COLMAP poses, goes through `scripts/thermoscenes_scene.sh` and appears in
the viewer as "Building A, winter, trained". OpenSplat trains it in about 3 minutes on an M4 Air
(7000 iterations, 11,109 Gaussians); ingest keeps 10,159 of them after dropping 634 oversized or
needle-shaped Gaussians, 316 below the ground and any far floaters. The script reuses an existing
`splat.ply`; set `RETRAIN=1` to train again.

What the detector finds there is the glazing: the glass entrance lobby and the window frames run
2–3 °C warmer than the insulated cladding, reported as 12 small patches at the default 1.5 °C
threshold (3 at 2.5 °C, none at 3 °C). That is real heat loss but not a defect, and it marks the
limit of a detector tuned for localised leaks on a façade that is mostly glass; a brick dorm with
punched windows is the intended target. Two estimates feed those numbers: the 0.9 m/unit scale
(from the storey pitch) and the thermal calibration, which for this dataset is the RGB camera's
(the thermal frames share its image grid), distortion included. The trained splat is soft:
low-opacity floaters between the cameras and the façade blur the thermal view, and more frames
or a longer run sharpen it. On the sparse-point preview the detector reports no leak.

## 6. Demo-day story

"Facilities knows this dorm is inefficient on paper. We walked past it for 15 minutes with a
$250 camera and here, in 3D, are the four places the heat leaves, ranked, with a dollar figure
on each." Keep the synthetic house in the picker as the controlled reference.
