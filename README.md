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
