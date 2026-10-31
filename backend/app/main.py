"""Heatsplat API. Run with: uv run uvicorn app.main:app --app-dir backend --reload --port 8000"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.scenes import router as scenes_router

DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def create_app() -> FastAPI:
    logging.basicConfig(level=os.environ.get("HEATSPLAT_LOG_LEVEL", "INFO"))
    app = FastAPI(title="Heatsplat API", version="0.1.0", description="3D thermal scans: find, rank and price heat leaks.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in os.environ.get("HEATSPLAT_CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(scenes_router)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # The default handler echoes the offending input, which fails for NaN/Infinity bodies
        # (not JSON-serialisable) and turns a 422 into a 500. Report type, location and message only.
        errors = [{key: str(err[key]) if key == "msg" else err[key] for key in ("type", "loc", "msg") if key in err} for err in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
