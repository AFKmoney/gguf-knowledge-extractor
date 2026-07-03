"""
Hugging Face Model Manager
==========================
Browse, search, and download GGUF models from the Hugging Face Hub,
directly inside the app (LM Studio-style).

Features:
  - Search the HF Hub for GGUF-tagged models
  - Get detailed info for a model (files, tags, downloads, likes)
  - Download a specific GGUF file with progress reporting
  - List locally-downloaded models
  - Delete local models

Uses the HF Hub HTTP API directly (more stable across library versions
than the Python SDK), but uses `huggingface_hub.hf_hub_download` for the
actual file download (handles resumable downloads, caching, and
authentication for gated models).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

# Default download location (can be overridden via env var or constructor)
DEFAULT_MODELS_DIR = Path(os.environ.get(
    "GGUF_MODELS_DIR",
    "/home/z/my-project/models",
))

HF_API_BASE = "https://huggingface.co/api"
HF_RESOLVE_BASE = "https://huggingface.co"


@dataclass
class HFModelSummary:
    """Summary of a model returned by HF search."""
    repo_id: str
    author: str
    model_name: str
    downloads: int
    likes: int
    pipeline_tag: Optional[str]
    tags: List[str]
    last_modified: Optional[str]
    gated: bool
    gguf_files: List[str] = field(default_factory=list)


@dataclass
class HFModelFile:
    """A single file in a HF model repo."""
    filename: str
    size_bytes: int
    is_gguf: bool
    download_url: str


@dataclass
class HFModelInfo:
    """Detailed info for a single HF model repo."""
    repo_id: str
    author: str
    downloads: int
    likes: int
    pipeline_tag: Optional[str]
    tags: List[str]
    last_modified: Optional[str]
    gated: bool
    created_at: Optional[str]
    files: List[HFModelFile]
    gguf_info: Dict[str, Any] = field(default_factory=dict)
    card_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalModel:
    """A locally-downloaded GGUF model."""
    filename: str
    path: str
    size_bytes: int
    repo_id: Optional[str] = None
    downloaded_at: Optional[str] = None


@dataclass
class DownloadProgress:
    """Progress update during a download."""
    repo_id: str
    filename: str
    bytes_downloaded: int
    total_bytes: int
    speed_mbps: float
    eta_seconds: float
    percent: float


@dataclass
class DownloadResult:
    """Final result of a download."""
    repo_id: str
    filename: str
    local_path: str
    size_bytes: int
    elapsed_seconds: float
    success: bool
    error: Optional[str] = None


class ModelManager:
    """Browse and download GGUF models from Hugging Face Hub."""

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        hf_token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        limit: int = 20,
        sort: str = "downloads",   # downloads | likes | lastModified | createdAt
        gguf_only: bool = True,
    ) -> List[HFModelSummary]:
        """Search HF Hub for models matching the query.

        If gguf_only=True (default), filters to models tagged 'gguf'.
        """
        params: Dict[str, Any] = {
            "search": query,
            "sort": sort,
            "direction": "-1",
            "limit": limit,
        }
        if gguf_only:
            params["filter"] = "gguf"

        headers = self._auth_headers()
        r = self._client.get(f"{HF_API_BASE}/models", params=params, headers=headers)
        r.raise_for_status()
        data = r.json()

        out: List[HFModelSummary] = []
        for m in data:
            repo_id = m.get("id", "")
            siblings = m.get("siblings", []) or []
            gguf_files = [
                s.get("rfilename", "") for s in siblings
                if s.get("rfilename", "").lower().endswith(".gguf")
            ]
            author = repo_id.split("/")[0] if "/" in repo_id else ""
            model_name = repo_id.split("/", 1)[1] if "/" in repo_id else repo_id
            out.append(HFModelSummary(
                repo_id=repo_id,
                author=author,
                model_name=model_name,
                downloads=int(m.get("downloads", 0) or 0),
                likes=int(m.get("likes", 0) or 0),
                pipeline_tag=m.get("pipeline_tag"),
                tags=m.get("tags", []) or [],
                last_modified=m.get("lastModified"),
                gated=bool(m.get("gated", False)),
                gguf_files=gguf_files,
            ))
        return out

    # ------------------------------------------------------------------ #
    # Model info
    # ------------------------------------------------------------------ #
    def get_model_info(self, repo_id: str) -> HFModelInfo:
        """Get detailed info for a specific model repo, including files."""
        headers = self._auth_headers()
        r = self._client.get(f"{HF_API_BASE}/models/{repo_id}", headers=headers)
        r.raise_for_status()
        m = r.json()

        siblings = m.get("siblings", []) or []
        files: List[HFModelFile] = []
        for s in siblings:
            filename = s.get("rfilename", "")
            if not filename:
                continue
            # Get file size via HEAD request (only for GGUF files to keep it fast)
            is_gguf = filename.lower().endswith(".gguf")
            size_bytes = 0
            if is_gguf:
                size_bytes = self._get_file_size(repo_id, filename)
            files.append(HFModelFile(
                filename=filename,
                size_bytes=size_bytes,
                is_gguf=is_gguf,
                download_url=f"{HF_RESOLVE_BASE}/{repo_id}/resolve/main/{filename}",
            ))

        author = repo_id.split("/")[0] if "/" in repo_id else ""
        return HFModelInfo(
            repo_id=repo_id,
            author=author,
            downloads=int(m.get("downloads", 0) or 0),
            likes=int(m.get("likes", 0) or 0),
            pipeline_tag=m.get("pipeline_tag"),
            tags=m.get("tags", []) or [],
            last_modified=m.get("lastModified"),
            gated=bool(m.get("gated", False)),
            created_at=m.get("createdAt"),
            files=files,
            gguf_info=m.get("gguf", {}) or {},
            card_data=m.get("cardData", {}) or {},
        )

    def _get_file_size(self, repo_id: str, filename: str) -> int:
        """Get the size of a file via HEAD request."""
        try:
            url = f"{HF_RESOLVE_BASE}/{repo_id}/resolve/main/{filename}"
            r = self._client.head(url, follow_redirects=True, headers=self._auth_headers())
            return int(r.headers.get("content-length", 0))
        except Exception:
            return 0

    # ------------------------------------------------------------------ #
    # Download
    # ------------------------------------------------------------------ #
    def download(
        self,
        repo_id: str,
        filename: str,
        progress_cb: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> DownloadResult:
        """Download a file from a HF model repo.

        Streams the download with progress callbacks. Resumable.
        """
        url = f"{HF_RESOLVE_BASE}/{repo_id}/resolve/main/{filename}"
        out_path = self.models_dir / filename
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

        # Get total size
        try:
            head = self._client.head(url, follow_redirects=True, headers=self._auth_headers())
            total_bytes = int(head.headers.get("content-length", 0))
        except Exception as e:
            return DownloadResult(
                repo_id=repo_id, filename=filename, local_path="",
                size_bytes=0, elapsed_seconds=0, success=False,
                error=f"HEAD request failed: {e}",
            )

        # Resume support: check if tmp file exists
        existing_bytes = 0
        if tmp_path.exists():
            existing_bytes = tmp_path.stat().st_size
            if existing_bytes >= total_bytes and total_bytes > 0:
                # Already downloaded, just rename
                tmp_path.rename(out_path)
                return DownloadResult(
                    repo_id=repo_id, filename=filename, local_path=str(out_path),
                    size_bytes=total_bytes, elapsed_seconds=0, success=True,
                )

        headers = self._auth_headers()
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        t0 = time.time()
        bytes_done = existing_bytes
        last_progress_time = t0

        try:
            with self._client.stream("GET", url, follow_redirects=True, headers=headers, timeout=300.0) as r:
                r.raise_for_status()
                mode = "ab" if existing_bytes > 0 and r.status_code == 206 else "wb"
                if mode == "wb":
                    bytes_done = 0
                with open(tmp_path, mode) as f:
                    for chunk in r.iter_bytes(chunk_size=1024 * 1024):  # 1MB chunks
                        f.write(chunk)
                        bytes_done += len(chunk)
                        now = time.time()
                        if progress_cb and (now - last_progress_time) >= 0.5:
                            elapsed = now - t0
                            speed_mbps = (bytes_done / 1e6) / max(elapsed, 0.001)
                            remaining_bytes = max(total_bytes - bytes_done, 0)
                            eta = remaining_bytes / max(speed_mbps * 1e6, 1)
                            percent = (bytes_done / total_bytes * 100) if total_bytes > 0 else 0
                            progress_cb(DownloadProgress(
                                repo_id=repo_id, filename=filename,
                                bytes_downloaded=bytes_done,
                                total_bytes=total_bytes,
                                speed_mbps=speed_mbps,
                                eta_seconds=eta,
                                percent=percent,
                            ))
                            last_progress_time = now

            # Move tmp file to final name
            tmp_path.rename(out_path)
            elapsed = time.time() - t0

            # Save metadata sidecar
            self._save_metadata(out_path, repo_id, filename)

            return DownloadResult(
                repo_id=repo_id, filename=filename, local_path=str(out_path),
                size_bytes=bytes_done, elapsed_seconds=elapsed, success=True,
            )
        except Exception as e:
            return DownloadResult(
                repo_id=repo_id, filename=filename, local_path="",
                size_bytes=bytes_done, elapsed_seconds=time.time() - t0,
                success=False, error=str(e),
            )

    def _save_metadata(self, model_path: Path, repo_id: str, filename: str) -> None:
        """Write a .meta.json sidecar with provenance info."""
        import json
        meta = {
            "repo_id": repo_id,
            "filename": filename,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "size_bytes": model_path.stat().st_size,
        }
        meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------ #
    # Local model management
    # ------------------------------------------------------------------ #
    def list_local_models(self) -> List[LocalModel]:
        """List all GGUF files in the local models directory."""
        out: List[LocalModel] = []
        if not self.models_dir.exists():
            return out
        for p in sorted(self.models_dir.iterdir()):
            if not p.is_file() or not p.name.lower().endswith(".gguf"):
                continue
            # Try to load metadata sidecar
            meta_path = p.with_suffix(p.suffix + ".meta.json")
            repo_id = None
            downloaded_at = None
            if meta_path.exists():
                try:
                    import json
                    with open(meta_path) as f:
                        meta = json.load(f)
                    repo_id = meta.get("repo_id")
                    downloaded_at = meta.get("downloaded_at")
                except Exception:
                    pass
            out.append(LocalModel(
                filename=p.name,
                path=str(p),
                size_bytes=p.stat().st_size,
                repo_id=repo_id,
                downloaded_at=downloaded_at,
            ))
        return out

    def delete_local_model(self, filename: str) -> bool:
        """Delete a local model file (and its metadata sidecar)."""
        path = self.models_dir / filename
        if not path.exists():
            return False
        path.unlink()
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.exists():
            meta_path.unlink()
        return True

    def get_local_path(self, filename: str) -> Optional[str]:
        """Get the full path of a local model by filename, or None."""
        path = self.models_dir / filename
        if path.exists():
            return str(path)
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _auth_headers(self) -> Dict[str, str]:
        if self.hf_token:
            return {"Authorization": f"Bearer {self.hf_token}"}
        return {}

    def close(self) -> None:
        self._client.close()


def format_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.2f} KB"
    return f"{n} B"
