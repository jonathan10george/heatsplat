#!/usr/bin/env bash
# Double-click this file in Finder (it opens a Terminal window) or run it from a terminal.
# Builds OpenSplat with Metal support (incremental after the first time), then trains the
# ThermoScenes buildingA_winter scan (7000 iterations) and ingests it as "Building A, winter".
set -euo pipefail
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"

cd "$(dirname "$0")/.."
echo "repo: $(pwd)"

echo "== building OpenSplat =="
scripts/build_opensplat.sh

echo "== training + ingesting buildingA_winter =="
OPENSPLAT_ITERS="${OPENSPLAT_ITERS:-7000}" scripts/thermoscenes_scene.sh \
  data/captures/thermoscenes-raw/ThermoScenes/buildingA_winter thermoscenes-buildingA-winter 0.9 "Building A, winter"

echo "== done: open the viewer and pick 'Building A, winter, trained' =="
