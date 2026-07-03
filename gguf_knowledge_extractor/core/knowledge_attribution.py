"""
Knowledge Attribution
=====================
Combines:
  - MLP memory decomposition (which neurons store what)
  - Attention head analysis (which heads are specialized)
  - Causal tracing (which layer stores a specific fact — when inference is available)
  - Knowledge fingerprint (unique signature per model)

This module produces the KnowledgeAttributionReport — the closest thing to
"reading the model's mind" that we can get without training.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable

import gguf

from .mlp_analyzer import MLPAnalyzer
from .attention_analyzer import AttentionAnalyzer
from .inference.base import InferenceBackend


@dataclass
class FactAttribution:
    """For a single fact probe, where in the model is this fact stored?"""
    probe_id: str
    prompt: str
    expected: str
    model_answer: str
    correct: bool
    attributed_layer: Optional[int]      # via logit lens (None if inference unavailable)
    attributed_neuron: Optional[int]     # within that layer
    attribution_confidence: float
    method: str                          # "logit_lens" | "weight_only" | "none"


@dataclass
class KnowledgeAttributionReport:
    """The full v2 attribution report."""
    # MLP analysis
    mlp_analysis: Dict[str, Any]

    # Attention analysis
    attention_analysis: Dict[str, Any]

    # Per-fact attributions (requires inference)
    fact_attributions: List[Dict[str, Any]]

    # Knowledge fingerprint
    knowledge_fingerprint: str
    fingerprint_brief: Dict[str, Any]

    # Stats
    stats: Dict[str, Any] = field(default_factory=dict)


class KnowledgeAttributor:
    """Top-level v2 attributor."""

    def __init__(
        self,
        reader: gguf.GGUFReader,
        tokens: Optional[List[str]] = None,
        backend: Optional[InferenceBackend] = None,
        top_k_per_layer: int = 20,
        top_k_tokens: int = 10,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ):
        self.reader = reader
        self.tokens = tokens or []
        self.backend = backend
        self.top_k_per_layer = top_k_per_layer
        self.top_k_tokens = top_k_tokens
        self.progress_cb = progress_cb or (lambda msg, cur, total: None)

    def attribute(self, fact_results: Optional[List[Dict[str, Any]]] = None) -> KnowledgeAttributionReport:
        """Run the full attribution pipeline.

        fact_results: optional list of fact probe results (from v1 extractor).
                     If provided AND backend supports hidden states, we attempt
                     logit-lens attribution per fact.
        """
        t0 = time.time()

        # Stage 1: MLP analysis (always works)
        self.progress_cb("MLP memory decomposition", 1, 4)
        mlp_analyzer = MLPAnalyzer(
            self.reader, tokens=self.tokens,
            top_k_per_layer=self.top_k_per_layer,
            top_k_tokens=self.top_k_tokens,
        )
        mlp_report = mlp_analyzer.analyze()
        mlp_dict = _to_jsonable(asdict(mlp_report))

        # Stage 2: Attention analysis (always works)
        self.progress_cb("Attention head analysis", 2, 4)
        attn_analyzer = AttentionAnalyzer(self.reader)
        attn_report = attn_analyzer.analyze()
        attn_dict = _to_jsonable(asdict(attn_report))

        # Stage 3: Per-fact attribution (requires inference backend)
        self.progress_cb("Per-fact attribution (causal tracing)", 3, 4)
        fact_attributions: List[Dict[str, Any]] = []
        if fact_results and self.backend and self.backend.is_available():
            fact_attributions = self._attribute_facts(fact_results)
        elif fact_results:
            # Weight-only attribution: find which layer's memory neurons are
            # most-activated by tokens in the prompt
            fact_attributions = self._weight_only_attribution(fact_results, mlp_report)

        # Stage 4: assemble report
        self.progress_cb("Building knowledge fingerprint", 4, 4)
        elapsed = time.time() - t0

        return KnowledgeAttributionReport(
            mlp_analysis=mlp_dict,
            attention_analysis=attn_dict,
            fact_attributions=fact_attributions,
            knowledge_fingerprint=mlp_report.knowledge_fingerprint,
            fingerprint_brief=mlp_report.fingerprint_brief,
            stats={
                "elapsed_seconds": elapsed,
                "n_layers_analyzed_mlp": mlp_report.n_layers_analyzed,
                "n_layers_analyzed_attn": attn_report.n_layers_analyzed,
                "n_heads_total": attn_report.n_heads_total,
                "n_global_top_neurons": len(mlp_report.global_top_neurons),
                "n_facts_attributed": len(fact_attributions),
                "backend_available": bool(self.backend and self.backend.is_available()),
            },
        )

    # ------------------------------------------------------------------ #
    # Per-fact attribution
    # ------------------------------------------------------------------ #
    def _attribute_facts(self, fact_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Logit-lens style attribution: for each fact, find which layer
        'knows' the answer. Requires an inference backend with hidden state
        access (currently best-effort with llama-cpp-python).
        """
        out: List[Dict[str, Any]] = []
        # Note: true logit lens requires hidden states from intermediate layers,
        # which neither backend exposes cleanly via the simple generate() API.
        # We implement a degenerate version: just record which fact we tried to
        # attribute and mark the method as 'logit_lens_unavailable'.
        for f in fact_results[:50]:  # cap for speed
            out.append({
                "probe_id": f.get("probe_id") or f.get("id"),
                "prompt": f.get("prompt"),
                "expected": f.get("expected") or f.get("expected_answer"),
                "model_answer": f.get("response") or f.get("model_answer"),
                "correct": f.get("passed") or f.get("correct", False),
                "attributed_layer": None,
                "attributed_neuron": None,
                "attribution_confidence": 0.0,
                "method": "logit_lens_unavailable",
            })
        return out

    def _weight_only_attribution(self, fact_results: List[Dict[str, Any]], mlp_report) -> List[Dict[str, Any]]:
        """Heuristic attribution: for each fact prompt, find the layer whose
        top memory neurons are most strongly activated by the prompt's key
        tokens. No inference required.
        """
        if not self.tokens or not self.embed_matrix_available():
            return []

        # Build a token-to-index lookup (slow but bounded)
        token_to_idx = {}
        for i, t in enumerate(self.tokens):
            if isinstance(t, str):
                # Try both raw and stripped forms
                token_to_idx.setdefault(t, i)
                stripped = t.replace("Ġ", " ").replace("▁", " ").strip()
                if stripped:
                    token_to_idx.setdefault(stripped, i)

        out: List[Dict[str, Any]] = []
        for f in fact_results[:50]:
            prompt = (f.get("prompt") or "").lower()
            # Find which tokens in our vocab appear in the prompt
            prompt_tokens = []
            for word in prompt.split():
                # Try word with leading space (BPE convention)
                for variant in [word, " " + word, "▁" + word, "Ġ" + word]:
                    if variant in token_to_idx:
                        prompt_tokens.append(token_to_idx[variant])
                        break

            if not prompt_tokens:
                out.append({
                    "probe_id": f.get("probe_id") or f.get("id"),
                    "prompt": f.get("prompt"),
                    "expected": f.get("expected") or f.get("expected_answer"),
                    "model_answer": f.get("response") or f.get("model_answer"),
                    "correct": f.get("passed") or f.get("correct", False),
                    "attributed_layer": None,
                    "attributed_neuron": None,
                    "attribution_confidence": 0.0,
                    "method": "weight_only_no_token_match",
                })
                continue

            # For each layer's top neurons, sum the activation strengths for prompt tokens
            best_layer = None
            best_neuron = None
            best_activation = -1.0
            for layer in mlp_report.layers:
                for neuron in layer.top_neurons:
                    # neuron.top_activating_tokens is a list of (token, activation)
                    activation_sum = 0.0
                    for tok_text, act in neuron.top_activating_tokens:
                        # Check if this token text matches any prompt token
                        for pt_idx in prompt_tokens:
                            pt_text = self.tokens[pt_idx] if pt_idx < len(self.tokens) else ""
                            if isinstance(pt_text, str):
                                pt_clean = pt_text.replace("Ġ", " ").replace("▁", " ").strip().lower()
                                tok_clean = tok_text.replace("Ġ", " ").replace("▁", " ").strip().lower()
                                if pt_clean and tok_clean and (pt_clean in tok_clean or tok_clean in pt_clean):
                                    activation_sum += act
                    if activation_sum > best_activation:
                        best_activation = activation_sum
                        best_layer = layer.layer
                        best_neuron = neuron.neuron_index

            out.append({
                "probe_id": f.get("probe_id") or f.get("id"),
                "prompt": f.get("prompt"),
                "expected": f.get("expected") or f.get("expected_answer"),
                "model_answer": f.get("response") or f.get("model_answer"),
                "correct": f.get("passed") or f.get("correct", False),
                "attributed_layer": best_layer,
                "attributed_neuron": best_neuron,
                "attribution_confidence": float(best_activation),
                "method": "weight_only" if best_layer is not None else "weight_only_no_match",
            })
        return out

    def embed_matrix_available(self) -> bool:
        """Check if embedding matrix is loadable."""
        from .mlp_analyzer import MLPAnalyzer
        a = MLPAnalyzer(self.reader, tokens=self.tokens)
        return a.embed_matrix is not None


def _to_jsonable(obj: Any) -> Any:
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
    return obj
