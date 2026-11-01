#!/usr/bin/env bash
# Turn a ThermoScenes scene (Hassan et al., https://zenodo.org/records/10835108) into a heatsplat scene.
#
#   scripts/thermoscenes_scene.sh <raw_scene_dir> <scene_id> <metres_per_unit> [display name]
#
# The raw scene folder holds images/, thermal/ (8-bit PNG normalised to temperature_bounds.json),
# colmap/sparse/0 and transforms_thermal.json. Uses tools/OpenSplat/build/opensplat for a trained
# splat when it exists, otherwise the COLMAP sparse points become disk Gaussians (no GPU needed).
set -euo pipefail

RAW="${1:?raw scene dir}"
SCENE_ID="${2:?scene id}"
SCALE="${3:?metres per COLMAP unit}"
NAME="${4:-$SCENE_ID (ThermoScenes)}"
ITERS="${OPENSPLAT_ITERS:-3000}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/data/captures/thermoscenes/$(basename "$RAW")"
OPENSPLAT="$ROOT/tools/OpenSplat/build/opensplat"
UV="${UV:-uv}"

mkdir -p "$WORK/colmap_txt"
ln -sfn "$RAW/images" "$WORK/images"
ln -sfn "$RAW/colmap/sparse" "$WORK/sparse"
colmap model_converter --input_path "$RAW/colmap/sparse/0" --output_path "$WORK/colmap_txt" --output_type TXT >/dev/null

RANGE="$(python3 -I -c 'import json,sys; b=json.load(open(sys.argv[1])); print(f"{b[\"absolute_min_temperature\"]},{b[\"absolute_max_temperature\"]}")' "$RAW/temperature_bounds.json")"

if [ -x "$OPENSPLAT" ]; then
  echo "training splat with OpenSplat ($ITERS iterations)..."
  "$OPENSPLAT" "$WORK" -n "$ITERS" -o "$WORK/splat.ply"
  PLY="$WORK/splat.ply"
  KIND="trained"
else
  echo "no OpenSplat binary at $OPENSPLAT; using COLMAP sparse points as disk Gaussians"
  "$UV" run --project "$ROOT" python -m heatsplat.cli points-to-splat --points "$WORK/colmap_txt/points3D.txt" --out "$WORK/sparse_splat.ply"
  PLY="$WORK/sparse_splat.ply"
  KIND="sparse"
fi

"$UV" run --project "$ROOT" python -m heatsplat.cli ingest \
  --scene "$SCENE_ID" --ply "$PLY" --colmap "$WORK/colmap_txt" \
  --thermal-dir "$RAW/thermal" --thermal-range="$RANGE" --up auto --scale "$SCALE" --camera-height "${CAMERA_HEIGHT:-1.5}" \
  --name "$NAME, $KIND"
