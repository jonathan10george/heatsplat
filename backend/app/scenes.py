"""Scene endpoints: listing, metadata, splat files, leaks, re-analysis, report."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.report import GENERATOR_TEMPLATE, render_report
from app.schemas import AnalyzeRequest, ReportOut, SceneSummaryOut, SplatMode
from heatsplat.cost import CostModel
from heatsplat.detect import DetectionParams
from heatsplat.export import LEAKS_FILE, META_FILE, SPLAT_MODES, analyze_scene, is_complete_scene, load_summary
from heatsplat.paths import data_root, scene_dir

log = logging.getLogger("heatsplat.api")
router = APIRouter(prefix="/api/scenes", tags=["scenes"])
_analyze_lock = threading.Lock()  # analyses rewrite scene files; never let two overlap


def _analyze_locked(path: Path, params: DetectionParams, cost_model: CostModel) -> dict:
    with _analyze_lock:
        return analyze_scene(path, params, cost_model)


def _scene_path(scene_id: str) -> Path:
    try:
        path = scene_dir(scene_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not (path / META_FILE).exists():
        raise HTTPException(status_code=404, detail=f"scene {scene_id!r} not found")
    return path


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.exception("could not read %s", path)
        raise HTTPException(status_code=500, detail=f"scene file {path.name} is unreadable") from exc


@router.get("", response_model=list[SceneSummaryOut])
def list_scenes() -> list[SceneSummaryOut]:
    root = data_root()
    if not root.exists():
        return []
    summaries = []
    for path in sorted(root.iterdir()):
        if is_complete_scene(path):
            try:
                summaries.append(SceneSummaryOut(**load_summary(path).__dict__))
            except (OSError, KeyError, ValueError):
                log.exception("skipping unreadable scene %s", path.name)
    return summaries


@router.get("/{scene_id}")
def get_scene(scene_id: str) -> dict:
    path = _scene_path(scene_id)
    return _read_json(path / META_FILE)


@router.get("/{scene_id}/splat")
def get_splat(scene_id: str, mode: SplatMode = Query("rgb")) -> FileResponse:
    path = _scene_path(scene_id) / f"{mode}.splat"
    if mode not in SPLAT_MODES or not path.exists():
        raise HTTPException(status_code=404, detail=f"no {mode} splat for scene {scene_id!r}")
    # Re-analysis rewrites leaks.splat in place, so the browser must revalidate rather than reuse a cached body.
    return FileResponse(path, media_type="application/octet-stream", filename=f"{scene_id}-{mode}.splat", headers={"Cache-Control": "no-cache"})


@router.get("/{scene_id}/leaks")
def get_leaks(scene_id: str) -> dict:
    return _read_json(_scene_path(scene_id) / LEAKS_FILE)


@router.post("/{scene_id}/analyze")
async def analyze(scene_id: str, request: AnalyzeRequest) -> dict:
    path = _scene_path(scene_id)
    params = DetectionParams(
        min_delta_c=request.min_delta_c,
        min_zscore=request.min_zscore,
        cluster_eps_m=request.cluster_eps_m,
        cluster_min_points=request.cluster_min_points,
        sign=request.sign,
    )
    cost_model = CostModel(price_per_kwh=request.price_per_kwh, heating_hours_per_year=request.heating_hours_per_year)
    try:
        return await run_in_threadpool(_analyze_locked, path, params, cost_model)
    except (OSError, ValueError) as exc:
        log.exception("analysis failed for %s", scene_id)
        raise HTTPException(status_code=500, detail="analysis failed; see the server log") from exc


@router.post("/{scene_id}/report", response_model=ReportOut)
def report(scene_id: str) -> ReportOut:
    path = _scene_path(scene_id)
    markdown = render_report(_read_json(path / META_FILE), _read_json(path / LEAKS_FILE))
    return ReportOut(scene_id=scene_id, generator=GENERATOR_TEMPLATE, markdown=markdown)
