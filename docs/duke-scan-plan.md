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
# poses from the RGB frames (brew install colmap)
colmap automatic_reconstructor --workspace_path ws --image_path frames --camera_model PINHOLE --single_camera 1
colmap model_converter --input_path ws/sparse/0 --output_path ws/sparse_txt --output_type TXT
```

```bash
# splat on Apple Metal (github.com/pierotofy/OpenSplat); 10–20 min for a façade
opensplat ws -n 7000 -o ga.ply
```

```bash
# bake thermal onto the splat, detect, export; poses and splat share COLMAP's frame
uv run python -m heatsplat.cli ingest --scene gilbert-addoms --ply ga.ply --colmap ws/sparse_txt \
  --thermal-dir thermal --thermal-range 0,20 --thermal-hfov 56 --up auto --scale 0.83 --name "Gilbert-Addoms, East Campus"
```

Then open the viewer: the scene appears in the picker. Check thermal mode first. If the thermal
layer is shifted relative to the bricks, the lens FOV is wrong (`--thermal-hfov`) or the clap
offset is off by a frame; both are quick to redo. Fallback if time is short: Luma, Polycam or
KIRI export a splat `.ply` from the same RGB video on the phone, and `--transforms` takes
nerfstudio-style poses. Their world up axis varies (ARKit exports are often Y up): try
`--up auto` first, then `--up y` or `--up z` if the house lies on its side.

Two flags matter for the numbers: `--scale` (metres per reconstruction unit; measure one known
length such as a door height in the splat and divide) so areas and costs are metric, and
`--thermal-hfov` for the real thermal lens. The CLI warns when either is missing.

This whole path is already exercised end to end on a synthetic capture
(`pipeline/tests/test_ingest.py`), including the COLMAP route and automatic up-axis recovery.

## 6. Demo-day story

"Facilities knows this dorm is inefficient on paper. We walked past it for 15 minutes with a
$250 camera and here, in 3D, are the four places the heat leaves, ranked, with a dollar figure
on each." Keep the synthetic house in the picker as the controlled reference.
