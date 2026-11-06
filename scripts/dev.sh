#!/usr/bin/env bash
# Demo-day launcher: backend on :8000 and the Vite frontend on :5173 in one terminal; Ctrl-C stops both.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
uv run uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 &
BACK=$!
npm run dev --prefix frontend &
FRONT=$!
trap 'kill "$BACK" "$FRONT" 2>/dev/null' EXIT INT TERM
echo "heatsplat: http://localhost:5173  (API on :8000; Ctrl-C stops both)"
wait
