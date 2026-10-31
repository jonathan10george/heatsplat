# Real capture checklist (hardware day)

Goal: one building façade (or one room's exterior wall) as a Gaussian splat with a baked
thermal layer, loaded into the viewer in place of the synthetic house.

## Kit

- Phone with a good camera (RGB video) and a clip-on thermal camera (FLIR One Edge Pro,
  TOPDON TC001, or similar) fixed rigidly to the phone. Tape it; it must not shift between shots.
- A tape measure or a known-size object in frame (door height works) to fix scene scale.
- Capture at dusk or pre-dawn: no direct sun, ≥10 °C indoor/outdoor difference, dry surfaces.

## Capture

1. Walk a slow arc around the façade at 2–4 m, phone in landscape, overlapping views (~70 %).
   Keep the whole wall in frame as much as possible; include the eaves and the ground line.
2. Every ~1 m along the arc, stop and grab a thermal frame (radiometric export if the app
   allows it, otherwise the palette image plus its min/max temperatures).
3. Note ambient temperature, time, and whether the heating was on.

## Reconstruct (pick one)

- **Fastest:** Polycam or Luma on the phone → export Gaussian splat (.ply / .splat) and camera
  poses. Convert poses to nerfstudio `transforms.json` if they are not already in that form.
- **Local, Mac:** `ns-process-data video` (COLMAP) → OpenSplat with the Metal backend →
  `.ply` + `transforms.json`.
- **Cloud GPU:** nerfstudio `splatfacto` on a rented GPU; export `.ply`.

## Bake and detect

1. Convert the splat `.ply` into `GaussianScene` arrays (positions, linear scales, quaternions
   as w,x,y,z, colours, opacities). The .splat writer and `scene.py` document the layout.
2. Convert thermal images to °C arrays (radiometric) or linearly map the palette image between
   its recorded min/max.
3. Align thermal frames to RGB poses: apply the fixed rigid offset between the two lenses and
   the thermal intrinsics (calibrate once against a checkerboard warmed by a lamp, or use the
   vendor's published FOV).
4. `bake.bake_temperature(positions, poses, intrinsics, frames)` → `fill_unobserved` →
   `scene.with_temperature(...)` → `export.export_scene(...)`.
5. Open the viewer, check thermal mode first. If the thermal layer looks mirrored or offset,
   the pose convention is wrong: try `bake.opengl_to_opencv` on the poses.

## Demo-day fallbacks

- Keep the synthetic scene in the scene picker as "Synthetic reference".
- If the thermal registration is rough, show thermal mode on the real scan anyway: judges
  react to the real splat, and the leak overlay can run on the synthetic scene.
