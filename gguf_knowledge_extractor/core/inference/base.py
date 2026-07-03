"""
Inference Backend Abstraction
=============================
Provides a uniform interface for running prompts through a GGUF model,
regardless of whether the user has:
  - llama.cpp server running (HTTP API, OpenAI-compatible)
  - llama-cpp-python installed (direct in-process bindings)

If neither is available, all probes gracefully degrade to "skipped" status.
"""
from __future__ import annotations

import abc
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenerationResult:
    """Result of a single generation call."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_seconds: float = 0.0
    backend: str = ""
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


class InferenceBackend(abc.ABC):
    """Abstract base for inference backends."""

    name: str = "abstract"

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can be used right now."""
        ...

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        seed: int = 42,
    ) -> GenerationResult:
        ...

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------- #
# llama.cpp server backend (HTTP, OpenAI-compatible)
# ---------------------------------------------------------------------- #
class ServerBackend(InferenceBackend):
    """Calls a running llama-server (or any OpenAI-compatible endpoint)."""

    name = "llama.cpp_server"

    def __init__(self, base_url: str = "http://127.0.0.1:8080", api_key: str = "none"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = None
        try:
            import httpx
            self._client = httpx.Client(timeout=120.0)
        except ImportError:
            pass

    def is_available(self) -> bool:
        if self._client is None:
            return False
        try:
            r = self._client.get(f"{self.base_url}/health", timeout=2.0)
            return r.status_code in (200, 503)  # 503 = loading
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        seed: int = 42,
    ) -> GenerationResult:
        if self._client is None:
            return GenerationResult(text="", backend=self.name, error="httpx not installed")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        t0 = time.time()
        try:
            r = self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                    "stop": stop or [],
                    "seed": seed,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=180.0,
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            elapsed = time.time() - t0
            return GenerationResult(
                text=text,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                elapsed_seconds=elapsed,
                backend=self.name,
                raw=data,
            )
        except Exception as e:
            return GenerationResult(
                text="", backend=self.name, error=str(e), elapsed_seconds=time.time() - t0
            )

    def close(self) -> None:
        if self._client:
            self._client.close()


# ---------------------------------------------------------------------- #
# llama-cpp-python backend (in-process)
# ---------------------------------------------------------------------- #
class PythonBackend(InferenceBackend):
    """Uses llama-cpp-python to load the GGUF directly in-process."""

    name = "llama_cpp_python"

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads or os.cpu_count() or 4
        self.verbose = verbose
        self._llm = None
        self._import_error: Optional[str] = None
        try:
            from llama_cpp import Llama  # type: ignore
            self._Llama = Llama
        except ImportError as e:
            self._import_error = str(e)
            self._Llama = None

    def is_available(self) -> bool:
        if self._Llama is None:
            return False
        if not os.path.isfile(self.model_path):
            return False
        return True

    def _ensure_loaded(self):
        if self._llm is None and self._Llama is not None:
            self._llm = self._Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=self.verbose,
            )
        return self._llm

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        seed: int = 42,
    ) -> GenerationResult:
        if self._Llama is None:
            return GenerationResult(text="", backend=self.name, error=self._import_error or "llama_cpp not installed")

        try:
            llm = self._ensure_loaded()
        except Exception as e:
            return GenerationResult(text="", backend=self.name, error=f"load failed: {e}")

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        t0 = time.time()
        try:
            out = llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or [],
                echo=False,
                seed=seed,
            )
            text = out["choices"][0]["text"]
            usage = out.get("usage", {})
            return GenerationResult(
                text=text,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                elapsed_seconds=time.time() - t0,
                backend=self.name,
                raw=out,
            )
        except Exception as e:
            return GenerationResult(
                text="", backend=self.name, error=str(e), elapsed_seconds=time.time() - t0
            )

    def close(self) -> None:
        self._llm = None  # release reference


# ---------------------------------------------------------------------- #
# Auto-selecting composite backend
# ---------------------------------------------------------------------- #
class AutoBackend(InferenceBackend):
    """Tries server first, falls back to python, then to a stub."""

    name = "auto"

    def __init__(
        self,
        model_path: Optional[str] = None,
        server_url: str = "http://127.0.0.1:8080",
        server_api_key: str = "none",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        prefer: str = "auto",  # "auto" | "server" | "python"
    ):
        self.server = ServerBackend(server_url, server_api_key)
        self.python = PythonBackend(model_path or "", n_ctx=n_ctx, n_gpu_layers=n_gpu_layers) if model_path else None
        self.prefer = prefer
        self._chosen: Optional[InferenceBackend] = None

    def is_available(self) -> bool:
        return self.server.is_available() or (self.python is not None and self.python.is_available())

    def _select(self) -> InferenceBackend:
        if self._chosen is not None and self._chosen.is_available():
            return self._chosen

        order: List[InferenceBackend] = []
        if self.prefer in ("auto", "server"):
            order.append(self.server)
            if self.python:
                order.append(self.python)
        elif self.prefer == "python":
            if self.python:
                order.append(self.python)
            order.append(self.server)

        for b in order:
            if b.is_available():
                self._chosen = b
                return b

        # If nothing is available, return the server (will return errors)
        self._chosen = self.server
        return self._chosen

    def generate(self, *args, **kwargs) -> GenerationResult:
        return self._select().generate(*args, **kwargs)

    def close(self) -> None:
        if self.server:
            self.server.close()
        if self.python:
            self.python.close()


def list_available_backends(
    model_path: Optional[str] = None,
    server_url: str = "http://127.0.0.1:8080",
) -> Dict[str, bool]:
    """Return a dict of {backend_name: is_available}."""
    out = {}
    s = ServerBackend(server_url)
    out["server"] = s.is_available()
    s.close()
    if model_path:
        p = PythonBackend(model_path)
        out["python"] = p.is_available()
        p.close()
    else:
        out["python"] = False
    return out
