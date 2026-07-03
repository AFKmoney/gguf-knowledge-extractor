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
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..core.extractor import KnowledgeExtractor
from ..core.inference.base import list_available_backends
from ..core.probes.base import list_default_packs
from ..core.exporters.json_exporter import export_json
from ..core.exporters.markdown_exporter import export_markdown
from ..core.exporters.graph_exporter import export_graphml, export_turtle
from ..core.exporters.sqlite_exporter import export_sqlite
from ..core.causal_tracer import CausalTracer
from ..core.rome_editor import RomeEditor, EditRequest
from ..core.fingerprint_compare import FingerprintComparator
from ..core.model_manager import ModelManager, format_bytes


STATIC_DIR = Path(__file__).parent / "static"
JOBS_DIR = Path("/home/z/my-project/download/extraction_jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory registries (sufficient for a local single-user tool)
JOBS: Dict[str, Dict[str, Any]] = {}
DOWNLOADS: Dict[str, Dict[str, Any]] = {}  # v4: track active downloads from HF Hub


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
        do_attribution: bool = Form(False),
        attribution_top_k: int = Form(20),
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
                "do_attribution": do_attribution,
                "attribution_top_k": attribution_top_k,
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
            do_attribution,
            attribution_top_k,
            server_url,
            prefer_backend,
            n_ctx,
            n_gpu_layers,
        )
        return {"job_id": job_id, "status": "queued"}

    @app.post("/api/extract-local")
    async def api_extract_local(
        background_tasks: BackgroundTasks,
        local_path: str = Form(...),
        packs: str = Form("all"),
        do_metadata: bool = Form(True),
        do_weights: bool = Form(True),
        do_probes: bool = Form(True),
        do_attribution: bool = Form(False),
        attribution_top_k: int = Form(20),
        server_url: str = Form("http://127.0.0.1:8080"),
        prefer_backend: str = Form("auto"),
        n_ctx: int = Form(4096),
        n_gpu_layers: int = Form(0),
    ):
        """Extract from a local file path (e.g. a downloaded HF model)."""
        gguf_path = Path(local_path)
        if not gguf_path.exists() or not gguf_path.name.lower().endswith(".gguf"):
            raise HTTPException(400, f"Invalid local path: {local_path}")

        job_id = str(uuid.uuid4())[:8]
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "gguf_path": str(gguf_path),
            "gguf_filename": gguf_path.name,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "progress": {"message": "queued", "current": 0, "total": 0},
            "result_paths": {},
            "error": None,
            "options": {
                "packs": packs,
                "do_metadata": do_metadata,
                "do_weights": do_weights,
                "do_probes": do_probes,
                "do_attribution": do_attribution,
                "attribution_top_k": attribution_top_k,
                "server_url": server_url,
                "prefer_backend": prefer_backend,
                "n_ctx": n_ctx,
                "n_gpu_layers": n_gpu_layers,
                "local_path": str(gguf_path),
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
            do_attribution,
            attribution_top_k,
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

    # ------------------------------------------------------------------ #
    # v3 API: causal trace, ROME edit, fingerprint compare
    # ------------------------------------------------------------------ #

    @app.post("/api/trace")
    async def api_trace(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        packs: str = Form(""),
        top_k: int = Form(10),
    ):
        """v3: Run logit-lens causal tracing on fact probes."""
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
            "options": {"packs": packs, "top_k": top_k},
            "kind": "trace",
        }

        background_tasks.add_task(_run_trace, job_id, str(gguf_path), packs, top_k)
        return {"job_id": job_id, "status": "queued"}

    @app.post("/api/edit")
    async def api_edit(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        edits_json: str = Form(...),  # JSON string of [{subject, prompt, target_object}]
    ):
        """v3: Run ROME rank-1 fact editing."""
        if not file.filename or not file.filename.lower().endswith(".gguf"):
            raise HTTPException(400, "File must be a .gguf file")

        import json as _json
        try:
            edits_data = _json.loads(edits_json)
        except Exception as e:
            raise HTTPException(400, f"Invalid edits JSON: {e}")

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
            "options": {"edits": edits_data},
            "kind": "edit",
        }

        background_tasks.add_task(_run_edit, job_id, str(gguf_path), edits_data)
        return {"job_id": job_id, "status": "queued"}

    @app.post("/api/compare")
    async def api_compare(reports: List[UploadFile] = File(...)):
        """v3: Cross-model fingerprint comparison. Upload 2+ attribution JSON reports."""
        if len(reports) < 2:
            raise HTTPException(400, "Need at least 2 reports to compare")

        job_id = str(uuid.uuid4())[:8]
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        report_paths = []
        for f in reports:
            p = job_dir / f.filename
            with open(p, "wb") as out:
                shutil.copyfileobj(f.file, out)
            report_paths.append(str(p))

        comparator = FingerprintComparator()
        report = comparator.compare_all(report_paths)

        import dataclasses, json as _json
        report_dict = _to_jsonable(dataclasses.asdict(report))

        # Save
        out_path = job_dir / "comparison.json"
        with open(out_path, "w") as f:
            _json.dump(report_dict, f, indent=2, default=str)

        JOBS[job_id] = {
            "id": job_id,
            "status": "completed",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "result_paths": {"json": str(out_path)},
            "error": None,
            "kind": "compare",
            "report_preview": report_dict,
        }
        return {"job_id": job_id, "status": "completed", "report": report_dict}

    # ------------------------------------------------------------------ #
    # v4 API: Hugging Face Hub model browser & downloader
    # ------------------------------------------------------------------ #
    MODELS_DIR = Path(os.environ.get("GGUF_MODELS_DIR", "/home/z/my-project/models"))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_model_manager() -> ModelManager:
        return ModelManager(models_dir=MODELS_DIR)

    @app.get("/api/models/search")
    async def api_models_search(q: str = "", limit: int = 20, sort: str = "downloads"):
        """Search HF Hub for GGUF models."""
        if not q:
            raise HTTPException(400, "Query parameter 'q' is required")
        try:
            mgr = _get_model_manager()
            results = mgr.search(q, limit=min(limit, 50), sort=sort, gguf_only=True)
            out = [_to_jsonable({
                "repo_id": r.repo_id, "author": r.author, "model_name": r.model_name,
                "downloads": r.downloads, "likes": r.likes,
                "pipeline_tag": r.pipeline_tag, "tags": r.tags[:8],
                "last_modified": r.last_modified, "gated": r.gated,
                "gguf_file_count": len(r.gguf_files),
            }) for r in results]
            mgr.close()
            return {"results": out, "count": len(out)}
        except Exception as e:
            raise HTTPException(500, f"Search failed: {e}")

    @app.get("/api/models/info/{repo_id:path}")
    async def api_models_info(repo_id: str):
        """Get detailed info for a HF model repo."""
        try:
            mgr = _get_model_manager()
            info = mgr.get_model_info(repo_id)
            out = _to_jsonable({
                "repo_id": info.repo_id, "author": info.author,
                "downloads": info.downloads, "likes": info.likes,
                "pipeline_tag": info.pipeline_tag, "tags": info.tags[:15],
                "last_modified": info.last_modified, "gated": info.gated,
                "created_at": info.created_at,
                "files": [
                    {"filename": f.filename, "size_bytes": f.size_bytes,
                     "size_human": format_bytes(f.size_bytes) if f.size_bytes else "?",
                     "is_gguf": f.is_gguf, "download_url": f.download_url}
                    for f in info.files
                ],
                "card_data": info.card_data,
            })
            mgr.close()
            return out
        except Exception as e:
            raise HTTPException(500, f"Info failed: {e}")

    @app.post("/api/models/download")
    async def api_models_download(
        background_tasks: BackgroundTasks,
        repo_id: str = Form(...),
        filename: str = Form(...),
    ):
        """Start downloading a GGUF file from HF Hub. Returns a download_id for progress polling."""
        download_id = str(uuid.uuid4())[:8]
        DOWNLOADS[download_id] = {
            "id": download_id,
            "repo_id": repo_id,
            "filename": filename,
            "status": "queued",
            "bytes_downloaded": 0,
            "total_bytes": 0,
            "speed_mbps": 0,
            "eta_seconds": 0,
            "percent": 0,
            "error": None,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "completed_at": None,
            "local_path": None,
        }
        background_tasks.add_task(_run_download, download_id, repo_id, filename)
        return {"download_id": download_id, "status": "queued"}

    @app.get("/api/models/downloads")
    async def api_models_downloads():
        """List all downloads (active and completed)."""
        return list(DOWNLOADS.values())

    @app.get("/api/models/downloads/{download_id}")
    async def api_models_download_status(download_id: str):
        if download_id not in DOWNLOADS:
            raise HTTPException(404, "Download not found")
        return DOWNLOADS[download_id]

    @app.get("/api/models/local")
    async def api_models_local():
        """List locally-downloaded models."""
        try:
            mgr = _get_model_manager()
            local = mgr.list_local_models()
            out = [_to_jsonable({
                "filename": m.filename, "path": m.path,
                "size_bytes": m.size_bytes,
                "size_human": format_bytes(m.size_bytes),
                "repo_id": m.repo_id, "downloaded_at": m.downloaded_at,
            }) for m in local]
            mgr.close()
            return {"models": out, "count": len(out), "models_dir": str(MODELS_DIR)}
        except Exception as e:
            raise HTTPException(500, f"List failed: {e}")

    @app.delete("/api/models/local/{filename}")
    async def api_models_delete(filename: str):
        """Delete a local model file."""
        try:
            mgr = _get_model_manager()
            ok = mgr.delete_local_model(filename)
            mgr.close()
            if not ok:
                raise HTTPException(404, "File not found")
            return {"status": "deleted", "filename": filename}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Delete failed: {e}")

    return app


# ---------------------------------------------------------------------- #
# v4 download worker
# ---------------------------------------------------------------------- #
def _run_download(download_id: str, repo_id: str, filename: str):
    """Background worker for HF downloads."""
    download = DOWNLOADS[download_id]
    download["status"] = "downloading"

    def progress_cb(p):
        download["bytes_downloaded"] = p.bytes_downloaded
        download["total_bytes"] = p.total_bytes
        download["speed_mbps"] = p.speed_mbps
        download["eta_seconds"] = p.eta_seconds
        download["percent"] = p.percent

    try:
        mgr = ModelManager(models_dir=Path(os.environ.get("GGUF_MODELS_DIR", "/home/z/my-project/models")))
        result = mgr.download(repo_id, filename, progress_cb=progress_cb)
        if result.success:
            download["status"] = "completed"
            download["local_path"] = result.local_path
            download["percent"] = 100.0
        else:
            download["status"] = "failed"
            download["error"] = result.error
        download["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        mgr.close()
    except Exception as e:
        download["status"] = "failed"
        download["error"] = str(e)
        download["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------- #
# v3 background workers
# ---------------------------------------------------------------------- #
def _run_trace(job_id: str, gguf_path: str, packs_str: str, top_k: int):
    """Background worker for causal tracing."""
    import gguf
    job = JOBS[job_id]
    job["status"] = "running"
    try:
        reader = gguf.GGUFReader(gguf_path)
        fields = _load_fields(reader)

        tracer = CausalTracer(reader, fields, top_k=top_k)
        if not tracer.is_available():
            job["status"] = "failed"
            job["error"] = "Forward pass not available for this GGUF (need Llama-arch with full tensors)"
            return

        # Load fact probes
        all_packs = list_default_packs()
        fact_packs = [p for p in all_packs if p.category == "facts"]
        if packs_str:
            names = {n.strip() for n in packs_str.split(",") if n.strip()}
            fact_packs = [p for p in fact_packs if p.name in names]
        if not fact_packs:
            fact_packs = [p for p in all_packs if p.category == "facts"]

        facts = []
        for pack in fact_packs:
            for probe in pack.probes:
                facts.append({
                    "probe_id": probe.id,
                    "prompt": probe.prompt,
                    "expected": probe.expected,
                    "domain": pack.domain,
                    "source_pack": pack.name,
                })

        job["progress"] = {"message": f"Tracing {len(facts)} facts", "current": 0, "total": len(facts)}
        report = tracer.trace_facts(facts)

        import dataclasses
        report_dict = _to_jsonable(dataclasses.asdict(report))

        job_dir = Path(gguf_path).parent
        out_path = job_dir / f"{Path(gguf_path).stem}_causal_trace.json"
        with open(out_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        job["result_paths"] = {"json": str(out_path)}
        job["report_preview"] = report_dict
        job["status"] = "completed"
        job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception as e:
        import traceback
        job["status"] = "failed"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()


def _run_edit(job_id: str, gguf_path: str, edits_data):
    """Background worker for ROME editing."""
    import gguf
    job = JOBS[job_id]
    job["status"] = "running"
    try:
        reader = gguf.GGUFReader(gguf_path)
        fields = _load_fields(reader)

        editor = RomeEditor(reader, fields)
        if not editor.is_available():
            job["status"] = "failed"
            job["error"] = "Forward pass not available for this GGUF"
            return

        requests = [EditRequest(**e) for e in edits_data]
        job_dir = Path(gguf_path).parent
        output_gguf = job_dir / f"{Path(gguf_path).stem}_edited.gguf"

        job["progress"] = {"message": f"Editing {len(requests)} facts", "current": 0, "total": len(requests)}
        report = editor.edit_facts(requests, gguf_path, str(output_gguf))

        import dataclasses
        report_dict = _to_jsonable(dataclasses.asdict(report))

        report_path = job_dir / f"{Path(gguf_path).stem}_edit_report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        job["result_paths"] = {
            "json": str(report_path),
            "edited_gguf": report.output_gguf_path,
        }
        job["report_preview"] = report_dict
        job["status"] = "completed"
        job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception as e:
        import traceback
        job["status"] = "failed"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()


def _load_fields(reader):
    """Load GGUF metadata fields into a plain dict."""
    import gguf
    fields = {}
    for fname in reader.fields.keys():
        try:
            f = reader.get_field(fname)
            if len(f.types) == 1 and f.types[0] == gguf.GGUFValueType.ARRAY:
                arr = f.parts[f.data[0]] if f.data else None
                fields[fname] = arr.tolist() if arr is not None and hasattr(arr, "tolist") else None
            else:
                try:
                    v = f.contents()
                    if hasattr(v, "tolist"):
                        fields[fname] = v.tolist()
                    elif hasattr(v, "item"):
                        fields[fname] = v.item()
                    else:
                        fields[fname] = v
                except Exception:
                    fields[fname] = None
        except Exception:
            pass
    return fields


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
    do_attribution: bool,
    attribution_top_k: int,
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
            do_attribution=do_attribution,
            attribution_top_k=attribution_top_k,
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
