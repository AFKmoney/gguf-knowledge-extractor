"""
FastAPI server exposing the GGUF Knowledge Extractor as an interactive web UI.

Endpoints:
  GET  /                     → UI (index.html)
  GET  /api/packs            → list available probe packs
  POST /api/extract          → run extraction (multipart: gguf file + options)
  GET  /api/jobs             → list past extraction jobs (in-memory)
  GET  /api/jobs/{id}        → get job status / results
  GET  /api/jobs/{id}/download/{format}  → download result file
                                       formats: json, markdown, graphml, turtle, sqlite
  GET  /api/backends         → check which inference backends are available
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..core.extractor import KnowledgeExtractor
from ..core.inference.base import list_available_backends
from ..core.probes.base import list_default_packs, load_probe_packs
from ..core.exporters.json_exporter import export_json
from ..core.exporters.markdown_exporter import export_markdown
from ..core.exporters.graph_exporter import export_graphml, export_turtle
from ..core.exporters.sqlite_exporter import export_sqlite


STATIC_DIR = Path(__file__).parent / "static"
JOBS_DIR = Path("/home/z/my-project/download/extraction_jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)


# In-memory job registry (sufficient for a local single-user tool)
JOBS: Dict[str, Dict[str, Any]] = {}


def create_app() -> FastAPI:
    app = FastAPI(title="GGUF Knowledge Extractor", version="1.0.0")

    # Static files (CSS/JS)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ------------------------------------------------------------------ #
    # Pages
    # ------------------------------------------------------------------ #
    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    @app.get("/api/packs")
    async def api_packs():
        packs = list_default_packs()
        return [
            {
                "name": p.name,
                "description": p.description,
                "category": p.category,
                "domain": p.domain,
                "n_probes": len(p),
                "path": p.path,
            }
            for p in packs
        ]

    @app.get("/api/backends")
    async def api_backends(server_url: str = "http://127.0.0.1:8080"):
        return list_available_backends(server_url=server_url)

    @app.post("/api/extract")
    async def api_extract(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        packs: str = Form("all"),  # comma-separated pack names, or "all"
        do_metadata: bool = Form(True),
        do_weights: bool = Form(True),
        do_probes: bool = Form(True),
        server_url: str = Form("http://127.0.0.1:8080"),
        prefer_backend: str = Form("auto"),
        n_ctx: int = Form(4096),
        n_gpu_layers: int = Form(0),
    ):
        if not file.filename or not file.filename.lower().endswith(".gguf"):
            raise HTTPException(400, "File must be a .gguf file")

        job_id = str(uuid.uuid4())[:8]
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        gguf_path = job_dir / file.filename
        with open(gguf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "gguf_path": str(gguf_path),
            "gguf_filename": file.filename,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "progress": {"message": "queued", "current": 0, "total": 0},
            "result_paths": {},
            "error": None,
            "options": {
                "packs": packs,
                "do_metadata": do_metadata,
                "do_weights": do_weights,
                "do_probes": do_probes,
                "server_url": server_url,
                "prefer_backend": prefer_backend,
                "n_ctx": n_ctx,
                "n_gpu_layers": n_gpu_layers,
            },
        }

        background_tasks.add_task(
            _run_extraction,
            job_id,
            str(gguf_path),
            packs,
            do_metadata,
            do_weights,
            do_probes,
            server_url,
            prefer_backend,
            n_ctx,
            n_gpu_layers,
        )
        return {"job_id": job_id, "status": "queued"}

    @app.get("/api/jobs")
    async def api_jobs():
        return list(JOBS.values())

    @app.get("/api/jobs/{job_id}")
    async def api_job(job_id: str):
        if job_id not in JOBS:
            raise HTTPException(404, "Job not found")
        return JOBS[job_id]

    @app.get("/api/jobs/{job_id}/download/{fmt}")
    async def api_download(job_id: str, fmt: str):
        if job_id not in JOBS:
            raise HTTPException(404, "Job not found")
        job = JOBS[job_id]
        if job["status"] != "completed":
            raise HTTPException(400, f"Job is {job['status']}, not completed")
        paths = job["result_paths"]
        fmt_map = {
            "json": ("json", "application/json"),
            "markdown": ("markdown", "text/markdown"),
            "graphml": ("graphml", "application/xml"),
            "turtle": ("turtle", "text/turtle"),
            "sqlite": ("sqlite", "application/x-sqlite3"),
        }
        if fmt not in fmt_map:
            raise HTTPException(400, f"Unknown format: {fmt}")
        key, mime = fmt_map[fmt]
        if key not in paths:
            raise HTTPException(404, f"Format {fmt} not available for this job")
        path = Path(paths[key])
        if not path.exists():
            raise HTTPException(404, "File missing on disk")
        return FileResponse(str(path), media_type=mime, filename=path.name)

    return app


# ---------------------------------------------------------------------- #
# Background extraction worker
# ---------------------------------------------------------------------- #
def _run_extraction(
    job_id: str,
    gguf_path: str,
    packs_str: str,
    do_metadata: bool,
    do_weights: bool,
    do_probes: bool,
    server_url: str,
    prefer_backend: str,
    n_ctx: int,
    n_gpu_layers: int,
):
    job = JOBS[job_id]
    job["status"] = "running"
    try:
        # Choose packs
        all_packs = list_default_packs()
        if packs_str == "all":
            selected = all_packs
        else:
            names = {n.strip() for n in packs_str.split(",") if n.strip()}
            selected = [p for p in all_packs if p.name in names]
            if not selected:
                selected = all_packs

        # Progress callback updates the job dict
        def progress(msg, cur, total):
            job["progress"] = {"message": msg, "current": cur, "total": total}

        extractor = KnowledgeExtractor(
            gguf_path=gguf_path,
            server_url=server_url,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            prefer_backend=prefer_backend,
            progress_cb=progress,
        )
        report = extractor.extract(
            packs=selected,
            do_metadata=do_metadata,
            do_weights=do_weights,
            do_probes=do_probes,
        )

        job_dir = Path(gguf_path).parent
        base_name = Path(gguf_path).stem
        paths: Dict[str, str] = {}

        # Always produce JSON
        paths["json"] = export_json(report, job_dir / f"{base_name}_report.json")
        paths["markdown"] = export_markdown(report, job_dir / f"{base_name}_report.md")
        paths["graphml"] = export_graphml(report, job_dir / f"{base_name}_knowledge.graphml")
        paths["turtle"] = export_turtle(report, job_dir / f"{base_name}_knowledge.ttl")
        paths["sqlite"] = export_sqlite(report, job_dir / f"{base_name}_knowledge.db")

        # Also store a copy of the report dict in the job for in-UI browsing
        import dataclasses
        report_dict = dataclasses.asdict(report)
        # Truncate large fields for the in-memory preview
        report_dict["probe_results"] = report_dict["probe_results"][:500]
        job["report_preview"] = _to_jsonable(report_dict)
        job["result_paths"] = paths
        job["status"] = "completed"
        job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception as e:
        import traceback
        job["status"] = "failed"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()


def _to_jsonable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return str(obj)
    if isinstance(obj, set):
        return list(obj)
    return obj


# Module-level app instance for `uvicorn gguf_knowledge_extractor.web.server:app`
app = create_app()
