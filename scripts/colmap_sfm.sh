#!/usr/bin/env bash
# CPU-only COLMAP structure-from-motion for a folder of photos (no CUDA needed on a Mac).
#
#   scripts/colmap_sfm.sh <images_dir> <workspace_dir> [matcher]
#
# Produces <workspace>/sparse/0 (binary model, what OpenSplat reads) and
# <workspace>/sparse/0_txt (text model, what `heatsplat ingest --colmap` reads).
# matcher: exhaustive (default, best for <~300 photos) or sequential (video frames).
set -euo pipefail

IMAGES="${1:?images dir}"
WORKSPACE="${2:?workspace dir}"
MATCHER="${3:-exhaustive}"
COLMAP="${COLMAP:-colmap}"
THREADS="$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)"
DB="$WORKSPACE/database.db"

mkdir -p "$WORKSPACE/sparse"

"$COLMAP" feature_extractor \
  --database_path "$DB" --image_path "$IMAGES" \
  --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL \
  --SiftExtraction.use_gpu 0 --SiftExtraction.num_threads "$THREADS"

"$COLMAP" "${MATCHER}_matcher" --database_path "$DB" --SiftMatching.use_gpu 0 --SiftMatching.num_threads "$THREADS"

"$COLMAP" mapper --database_path "$DB" --image_path "$IMAGES" --output_path "$WORKSPACE/sparse" --Mapper.num_threads "$THREADS"

if [ ! -f "$WORKSPACE/sparse/0/cameras.bin" ]; then
  echo "mapper produced no model in $WORKSPACE/sparse/0" >&2
  exit 1
fi

mkdir -p "$WORKSPACE/sparse/0_txt"
"$COLMAP" model_converter --input_path "$WORKSPACE/sparse/0" --output_path "$WORKSPACE/sparse/0_txt" --output_type TXT

REGISTERED="$(grep -c '\.jpg\|\.JPG\|\.png\|\.PNG\|\.jpeg' "$WORKSPACE/sparse/0_txt/images.txt" || true)"
POINTS="$(grep -vc '^#' "$WORKSPACE/sparse/0_txt/points3D.txt" || true)"
echo "COLMAP done: $REGISTERED images registered, $POINTS sparse points -> $WORKSPACE/sparse/0"
