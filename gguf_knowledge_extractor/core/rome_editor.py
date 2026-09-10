"""ROME-style rank-1 knowledge editing for GGUF models.

Stages:
1. extract MLP key k* (paper: at subject last token when subject is known);
2. optimize target value v* with paper_approx_adam (C-weighted ΔW in-loop);
3. solve and apply the full-matrix rank-1 ROME update with ridge > 0 by default.

``paper_approx_adam`` is a NumPy Adam + Rademacher-grad approximation of the
ROME objective, not the original PyTorch autograd solver — reports must keep
``paper_equivalent: False`` until a full contract pass.
"""
from __future__ import annotations

import shutil
import struct
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import gguf

from .forward_pass import NumpyLlamaForward
from .mlp_analyzer import MLPAnalyzer
from .causal_tracer import CausalTracer
from .rome_reference import collect_key_statistics, solve_rome_update, covariance_condition_number
from .rome_target import (
    extract_mlp_key,
    optimize_target_value,
    find_subject_token_index,
)


@dataclass
class EditRequest:
    subject: str
    prompt: str
    target_object: str
    preserve_other_facts: bool = True
    calibration_prompts: Optional[List[str]] = None
    target_iterations: int = 32
    target_step_size: float = 0.5
    target_probe_scale: float = 0.02
    target_seed: int = 0
    target_kl_weight: float = 0.5
    target_l2_weight: float = 1e-4
    target_method: str = "paper_approx_adam"
    ridge: float = 0.01
    token_ids: Optional[List[int]] = None
    subject_token_ids: Optional[List[int]] = None


@dataclass
class EditResult:
    subject: str
    prompt: str
    target_object: str
    target_token_id: Optional[int]
    edited_layer: Optional[int]
    edited_neuron: Optional[int]
    key_vector_norm: float
    old_value_norm: float
    new_value_norm: float
    delta_norm: float
    pre_edit_prediction: str
    post_edit_prediction: str
    edit_successful: bool
    method: str
    error: Optional[str] = None
    target_initial_probability: Optional[float] = None
    target_final_probability: Optional[float] = None
    target_initial_loss: Optional[float] = None
    target_final_loss: Optional[float] = None
    calibration_key_count: int = 0
    target_seed: Optional[int] = None
    subject_token_index: Optional[int] = None
    covariance_condition: Optional[float] = None
    ridge: Optional[float] = None
    paper_equivalent: bool = False


@dataclass
class EditReport:
    n_edits_requested: int
    n_edits_successful: int
    edits: List[Dict[str, Any]]
    output_gguf_path: Optional[str]
    stats: Dict[str, Any] = field(default_factory=dict)


class RomeEditor:
    """ROME-style rank-1 fact editor with explicit target optimization."""

    def __init__(self, reader: gguf.GGUFReader, fields: Dict[str, Any]):
        self.reader = reader
        self.fields = fields
        self.forward_pass = NumpyLlamaForward(reader, fields)
        tokens = [str(t) for t in fields.get("tokenizer.ggml.tokens", [])] if isinstance(fields.get("tokenizer.ggml.tokens"), list) else []
        self.tokens = tokens
        self.mlp_analyzer = MLPAnalyzer(reader, tokens=tokens, top_k_per_layer=20, top_k_tokens=10)
        self.causal_tracer = CausalTracer(reader, fields, top_k=10)
        self._llm = None

    def is_available(self) -> bool:
        return self.forward_pass.is_available()

    def tokenize(self, text: str, add_bos: bool = True) -> List[int]:
        """Prefer llama.cpp SPM (+BOS); fall back to CausalTracer greedy."""
        try:
            from llama_cpp import Llama
            if self._llm is None:
                # Lazy: reuse a tiny context for tokenization only if a path is known.
                # Callers doing heavy edits should pass token_ids explicitly.
                raise RuntimeError("no lazy llm without path")
        except Exception:
            pass
        if self._llm is not None:
            ids = self._llm.tokenize(text.encode("utf-8"), add_bos=add_bos)
            return list(ids)
        ids = self.causal_tracer.tokenize(text)
        return ids

    def attach_llama_tokenizer(self, llm) -> None:
        """Attach a llama-cpp Llama instance for SPM tokenization."""
        self._llm = llm

    def compute_edit(self, request: EditRequest, target_layer: Optional[int] = None):
        if not self.is_available():
            return self._failed(request, "forward_pass_unavailable", "Forward pass not available"), None

        target_id = self.fo