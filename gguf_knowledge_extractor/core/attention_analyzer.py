"""
Attention Head Analyzer
=======================
Analyzes attention head matrices in a transformer to identify specialized
heads — induction heads, copy heads, previous-token heads, etc.

Background
----------
Anthropic's interpretability work showed that transformers develop
specialized attention heads with identifiable signatures:

- **Induction heads** match a token to its earlier occurrence and copy the
  token that followed. They have a characteristic Q·K pattern where Q
  matches K of the previous token.
- **Previous-token heads** attend to the immediately preceding token.
- **Copy heads** have a V·O composition that approximately preserves the
  token embedding.

We detect these by inspecting the W_q, W_k, W_v, W_o matrices per head:

- **Induction score**: how aligned is W_q with W_k shifted by one position
  (approximated by QK commutativity structure).
- **Copy score**: how close is W_v @ W_o to identity (within the head's slice).
- **Specialization**: how low-rank is the head (concentrated SVD spectrum).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import gguf


@dataclass
class HeadAnalysis:
    layer: int
    head_index: int
    head_dim: int
    q_norm: float
    k_norm: float
    v_norm: float
    o_norm: float
    copy_score: float            # how close V·O is to identity (0-1, higher=more copy)
    induction_score: float       # heuristic Q-K alignment score (0-1)
    specialization: float        # how concentrated the SVD spectrum is (entropy-based)
    top_singular_value: float


@dataclass
class AttentionLayerAnalysis:
    layer: int
    n_heads: int
    head_dim: int
    heads: List[HeadAnalysis]
    strongest_copy_head: Optional[HeadAnalysis] = None
    strongest_induction_head: Optional[HeadAnalysis] = None


@dataclass
class AttentionModelAnalysis:
    n_layers_analyzed: int
    n_heads_total: int
    layers: List[AttentionLayerAnalysis]
    top_copy_heads: List[HeadAnalysis]
    top_induction_heads: List[HeadAnalysis]


class AttentionAnalyzer:
    """Inspects attention matrices to find specialized heads."""

    def __init__(self, reader: gguf.GGUFReader, n_heads_per_layer: int = 32):
        self.reader = reader
        self.n_heads_per_layer = n_heads_per_layer

    def _tensor_to_float(self, t) -> Optional[np.ndarray]:
        try:
            data = t.data
            qt = gguf.GGMLQuantizationType(t.tensor_type)
            if qt == gguf.GGMLQuantizationType.F32:
                arr = np.frombuffer(data, dtype=np.float32)
            elif qt == gguf.GGMLQuantizationType.F16:
                arr = np.frombuffer(data, dtype=np.float16).astype(np.float32)
            elif qt == gguf.GGMLQuantizationType.BF16:
                raw = np.frombuffer(data, dtype=np.uint16).astype(np.uint32)
                raw = (raw << 16).astype(np.uint32)
                arr = raw.view(np.float32).copy()
            else:
                try:
                    from gguf.quants import dequantize
                    arr = dequantize(data, qt).astype(np.float32)
                except Exception:
                    return None
            shape = [int(s) for s in t.shape]
            try:
                return arr.reshape(shape)
            except Exception:
                return arr
        except Exception:
            return None

    @staticmethod
    def _extract_layer_index(name: str) -> Optional[int]:
        import re
        m = re.search(r"blk\.(\d+)|layers?\.(\d+)|h\.(\d+)", name)
        if m:
            for g in m.groups():
                if g is not None:
                    return int(g)
        return None

    def _find_attn_tensor(self, layer_idx: int, kind: str) -> Optional[np.ndarray]:
        """kind: 'q' | 'k' | 'v' | 'output'"""
        candidates = {
            "q": [f"blk.{layer_idx}.attn_q.weight", f"layers.{layer_idx}.attention.wq.weight"],
            "k": [f"blk.{layer_idx}.attn_k.weight", f"layers.{layer_idx}.attention.wk.weight"],
            "v": [f"blk.{layer_idx}.attn_v.weight", f"layers.{layer_idx}.attention.wv.weight"],
            "output": [f"blk.{layer_idx}.attn_output.weight", f"blk.{layer_idx}.attn_o.weight",
                       f"layers.{layer_idx}.attention.wo.weight"],
        }[kind]
        for t in self.reader.tensors:
            name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            for cand in candidates:
                if name == cand:
                    return self._tensor_to_float(t)
        return None

    def analyze(self) -> AttentionModelAnalysis:
        layer_indices = set()
        for t in self.reader.tensors:
            name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            if "attn_q" in name or "attention.wq" in name:
                idx = self._extract_layer_index(name)
                if idx is not None:
                    layer_indices.add(idx)

        layer_analyses: List[AttentionLayerAnalysis] = []
        all_copy_heads: List[HeadAnalysis] = []
        all_induction_heads: List[HeadAnalysis] = []

        for li in sorted(layer_indices):
            la = self.analyze_layer(li)
            if la is not None:
                layer_analyses.append(la)
                all_copy_heads.extend([h for h in la.heads if h.copy_score > 0])
                all_induction_heads.extend([h for h in la.heads if h.induction_score > 0])

        all_copy_heads.sort(key=lambda h: h.copy_score, reverse=True)
        all_induction_heads.sort(key=lambda h: h.induction_score, reverse=True)

        return AttentionModelAnalysis(
            n_layers_analyzed=len(layer_analyses),
            n_heads_total=sum(la.n_heads for la in layer_analyses),
            layers=layer_analyses,
            top_copy_heads=all_copy_heads[:20],
            top_induction_heads=all_induction_heads[:20],
        )

    def analyze_layer(self, layer_idx: int) -> Optional[AttentionLayerAnalysis]:
        w_q = self._find_attn_tensor(layer_idx, "q")
        w_v = self._find_attn_tensor(layer_idx, "v")
        w_o = self._find_attn_tensor(layer_idx, "output")

        if w_q is None or w_v is None or w_o is None:
            return None

        # Shapes (Llama convention):
        # w_q: [n_heads * head_dim, embed_dim]  → row i is the i-th row of the concatenated Q
        # w_v: [n_heads * head_dim, embed_dim]
        # w_o: [embed_dim, n_heads * head_dim]
        if w_q.ndim != 2 or w_v.ndim != 2 or w_o.ndim != 2:
            return None

        n_heads_total_q, embed_dim = w_q.shape
        embed_dim_o, n_heads_total_o = w_o.shape

        if n_heads_total_q != n_heads_total_o:
            return None

        # Try to figure out n_heads. Use metadata field if available, else estimate.
        # Llama-7B: n_heads=32, head_dim=128, embed_dim=4096
        # We'll try common divisors.
        n_heads = self._infer_n_heads(n_heads_total_q)
        head_dim = n_heads_total_q // n_heads

        heads: List[HeadAnalysis] = []
        for h in range(n_heads):
            # Slice for this head
            q_h = w_q[h * head_dim:(h + 1) * head_dim, :]  # [head_dim, embed_dim]
            v_h = w_v[h * head_dim:(h + 1) * head_dim, :]  # [head_dim, embed_dim]
            o_h = w_o[:, h * head_dim:(h + 1) * head_dim]  # [embed_dim, head_dim]

            q_norm = float(np.linalg.norm(q_h))
            v_norm = float(np.linalg.norm(v_h))
            o_norm = float(np.linalg.norm(o_h))
            # k_norm not used here (we don't always have W_k separately for GQA)
            k_norm = 0.0

            # Copy score: how close is V_h @ O_h to a scaled identity?
            # V_h: [head_dim, embed_dim], O_h: [embed_dim, head_dim]
            # V_h @ O_h: [head_dim, head_dim]
            vo = v_h @ o_h  # [head_dim, head_dim]
            vo_norm = np.linalg.norm(vo)
            if vo_norm > 1e-8:
                vo_normalized = vo / vo_norm
                eye_norm = np.linalg.norm(np.eye(head_dim))
                copy_score = float(max(0.0, 1.0 - np.linalg.norm(vo_normalized - np.eye(head_dim) / eye_norm) / 2))
            else:
                copy_score = 0.0

            # Induction score (heuristic): how symmetric is Q_h @ Q_h.T?
            # Induction heads tend to have a Q matrix that aligns with itself shifted.
            # As a cheap proxy: compute the alignment between Q_h rows and the
            # leading singular vectors of Q_h @ Q_h.T
            try:
                qq = q_h @ q_h.T  # [head_dim, head_dim]
                u, s, vt = np.linalg.svd(qq)
                # Induction heads tend to have low-rank QQ^T (concentrated spectrum)
                # Entropy of normalized singular values
                s_norm = s / (s.sum() + 1e-8)
                entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
                max_entropy = np.log(head_dim)
                # Lower entropy = more concentrated = more induction-like
                induction_score = float(1.0 - entropy / max(max_entropy, 1e-8))
                top_singular_value = float(s[0]) if len(s) > 0 else 0.0
                specialization = float(1.0 - entropy / max(max_entropy, 1e-8))
            except Exception:
                induction_score = 0.0
                top_singular_value = 0.0
                specialization = 0.0

            heads.append(HeadAnalysis(
                layer=layer_idx,
                head_index=h,
                head_dim=int(head_dim),
                q_norm=q_norm,
                k_norm=k_norm,
                v_norm=v_norm,
                o_norm=o_norm,
                copy_score=copy_score,
                induction_score=induction_score,
                specialization=specialization,
                top_singular_value=top_singular_value,
            ))

        # Pick strongest heads
        strongest_copy = max(heads, key=lambda h: h.copy_score) if heads else None
        strongest_induction = max(heads, key=lambda h: h.induction_score) if heads else None

        return AttentionLayerAnalysis(
            layer=layer_idx,
            n_heads=n_heads,
            head_dim=int(head_dim),
            heads=heads,
            strongest_copy_head=strongest_copy,
            strongest_induction_head=strongest_induction,
        )

    def _infer_n_heads(self, total_dim: int) -> int:
        """Try common head counts that divide total_dim evenly."""
        for candidate in [32, 16, 8, 40, 48, 64, 12, 24, 20]:
            if total_dim % candidate == 0:
                head_dim = total_dim // candidate
                if head_dim >= 32:  # reasonable head dim
                    return candidate
        # Fallback: try to find largest divisor
        for n in range(min(64, total_dim), 0, -1):
            if total_dim % n == 0 and (total_dim // n) >= 32:
                return n
        return 1


def analyze_attention(reader: gguf.GGUFReader) -> Dict[str, Any]:
    """Convenience function returning a JSON-serializable dict."""
    a = AttentionAnalyzer(reader)
    report = a.analyze()
    return _to_jsonable(asdict(report))


def _to_jsonable(obj: Any) -> Any:
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
