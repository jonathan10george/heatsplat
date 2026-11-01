#!/usr/bin/env bash
# Build OpenSplat (github.com/pierotofy/OpenSplat) with Metal support into tools/OpenSplat/build.
# Idempotent: clones the pinned commit when missing, applies scripts/opensplat-mps-backward.patch
# once (the Metal backward pass fetched a command buffer off the MPS dispatch queue, which
# libtorch >= 2.9 aborts on), then lets cmake do an incremental build.
# Needs Homebrew cmake, opencv and pytorch (libtorch) plus the Xcode Metal toolchain.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/tools/OpenSplat"
PATCH="$ROOT/scripts/opensplat-mps-backward.patch"
PINNED_COMMIT="${OPENSPLAT_COMMIT:-e33f9dd}"
TORCH_CMAKE="${TORCH_CMAKE:-/opt/homebrew/opt/pytorch/share/cmake}"

if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/pierotofy/OpenSplat "$SRC"
  git -C "$SRC" checkout -q "$PINNED_COMMIT"
fi

if git -C "$SRC" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "MPS backward patch already applied"
elif git -C "$SRC" apply --check "$PATCH" 2>/dev/null; then
  git -C "$SRC" apply "$PATCH"
  echo "applied $(basename "$PATCH")"
else
  echo "warning: $(basename "$PATCH") does not apply to this OpenSplat checkout; building it unpatched" >&2
fi

cmake -S "$SRC" -B "$SRC/build" -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SRC/build" -j"$(sysctl -n hw.logicalcpu)"
echo "built $SRC/build/opensplat"
