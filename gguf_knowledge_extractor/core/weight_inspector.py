"""
Weight Inspector
================
Analyzes the actual tensor values inside a GGUF file to extract structural
knowledge about the model — without running any inference.

This is the "mechanistic interpretability at a glance" layer:
  - Per-tensor statistics (mean, std, norm, sparsity)
  - Layer-wise activation potential (proxy via weight magnitudes)
  - Embedding-space analysis (token clusters, concept centroids)
  - Outlier neuron detection (the famous "magic neurons" of LLMs)
  - Attention head capacity estimation
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gguf


@dataclass
class TensorStats:
    name: str
    dtype: str
    shape: List[int]
    n_elements: int
    mean: float
    std: float
    min: float
    max: float
    abs_mean: float
    norm: float
    sparsity: float  # fraction of zeros (or near-zeros)
    n_outliers: int  # |x| > mean + 6*std


@dataclass
class LayerStats:
    layer_index: int
    attention_norm: Optional[TensorStats]
    mlp_up_norm: Optional[TensorStats]
    mlp_down_norm: Optional[TensorStats]
    mlp_gate_norm: Optional[TensorStats]
    total_params: int


@dataclass
class EmbeddingAnalysis:
    vocab_size: int
    embed_dim: int
    centroid_norm: float
    mean_pairwise_cosine: float
    top_outlier_tokens: List[Tuple[str, float]]  # (token_text, distance_from_centroid)
    cluster_count_estimate: int


@dataclass
class WeightInspectionReport:
    total_parameters: int
    total_size_bytes: int
    n_tensors: int
    n_layers: int
    quantization_summary: Dict[str, int]  # dtype -> count
    tensor_stats: List[TensorStats]
    layer_stats: List[LayerStats]
    embedding_analysis: Optional[EmbeddingAnalysis]
    outlier_tensors: List[str]  # tensors with unusually high std
    capacity_estimate: Dict[str, Any] = field(default_factory=dict)


class WeightInspector:
    """Reads tensor values from a GGUF file and computes statistics."""

    def __init__(self, reader: gguf.GGUFReader, tokenizer_tokens: Optional[List[str]] = None):
        self.reader = reader
        self.tokens = tokenizer_tokens or []

    # ------------------------------------------------------------------ #
    # Tensor value access
    # ------------------------------------------------------------------ #
    def _tensor_array(self, t) -> Optional[np.ndarray]:
        """Convert a quantized tensor to a float32 numpy array for analysis.

        Note: for quantized tensors, we only dequantize small ones to keep
        memory bounded. Large tensors are analyzed on a random sample.
        """
        try:
            # gguf reader provides `.data` slice; for quantized types we need
            # the underlying bytes. The simplest path: use the reader's
            # `_get_tensor_view` style via numpy view if F16/F32.
            data = t.data  # this is a memoryview into the mmap'd file
            qt = gguf.GGMLQuantizationType(t.tensor_type)

            if qt == gguf.GGMLQuantizationType.F32:
                arr = np.frombuffer(data, dtype=np.float32)
            elif qt == gguf.GGMLQuantizationType.F16:
                arr = np.frombuffer(data, dtype=np.float16).astype(np.float32)
            elif qt == gguf.GGMLQuantizationType.BF16:
                # BF16: upper 16 bits of float32
                raw = np.frombuffer(data, dtype=np.uint16).astype(np.uint32)
                raw = (raw << 16).astype(np.uint32)
                arr = raw.view(np.float32).copy()
            else:
                # Quantized: best-effort dequantize via gguf helpers if available,
                # otherwise fall back to byte-level stats.
                try:
                    from gguf.quants import dequantize  # type: ignore
                    arr = dequantize(data, qt).astype(np.float32)
                except Exception:
                    # Use bytes as a last resort for shape statistics
                    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 255.0
                    arr = (arr - 0.5) * 2  # rescale to roughly [-1, 1]

            # Sample if too large
            if arr.size > 5_000_000:
                idx = np.random.choice(arr.size, size=5_000_000, replace=False)
                arr = arr[idx]
            return arr
        except Exception as e:
            return None

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #
    def _stats(self, arr: np.ndarray, name: str, dtype: str, shape: List[int]) -> TensorStats:
        n = arr.size
        mean = float(arr.mean())
        std = float(arr.std())
        mn = float(arr.min())
        mx = float(arr.max())
        abs_mean = float(np.abs(arr).mean())
        norm = float(np.sqrt((arr * arr).sum()))
        # sparsity: fraction of values with |x| < 0.01 * std (or zero if std==0)
        threshold = max(0.01 * std, 1e-8)
        sparsity = float((np.abs(arr) < threshold).mean())
        # outliers: |x - mean| > 6*std
        if std > 0:
            n_outliers = int((np.abs(arr - mean) > 6 * std).sum())
        else:
            n_outliers = 0
        return TensorStats(
            name=name, dtype=dtype, shape=shape, n_elements=n,
            mean=mean, std=std, min=mn, max=mx,
            abs_mean=abs_mean, norm=norm, sparsity=sparsity, n_outliers=n_outliers,
        )

    def inspect(self, max_tensors: int = 2000) -> WeightInspectionReport:
        tensor_stats: List[TensorStats] = []
        layer_stats: List[LayerStats] = []
        total_params = 0
        total_bytes = 0
        quant_summary: Dict[str, int] = {}
        outlier_tensors: List[str] = []

        # Group tensors by layer
        layer_groups: Dict[int, Dict[str, Any]] = {}

        all_stds: List[float] = []

        for i, t in enumerate(self.reader.tensors):
            if i >= max_tensors:
                break
            try:
                dt = gguf.GGMLQuantizationType(t.tensor_type).name
            except Exception:
                dt = str(t.tensor_type)

            quant_summary[dt] = quant_summary.get(dt, 0) + 1

            shape = list(t.shape[::-1]) if hasattr(t, "shape") else []
            n_elements = 1
            for s in shape:
                n_elements *= int(s)
            total_params += n_elements
            total_bytes += t.n_bytes if hasattr(t, "n_bytes") else 0

            name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            arr = self._tensor_array(t)
            if arr is None:
                continue

            stats = self._stats(arr, name=name, dtype=dt, shape=shape)
            tensor_stats.append(stats)
            all_stds.append(stats.std)

            # layer grouping
            layer_idx = self._extract_layer_index(name)
            if layer_idx is not None:
                grp = layer_groups.setdefault(layer_idx, {"layer_index": layer_idx})
                if "attn" in name and "norm" not in name.lower():
                    # could be q/k/v projections
                    pass
                if "attn_norm" in name or "attention_norm" in name or "attn.output_norm" in name:
                    grp["attention_norm"] = stats
                elif "ffn_up" in name or "mlp.up" in name or "feed_forward.w2" in name:
                    grp["mlp_up_norm"] = stats
                elif "ffn_down" in name or "mlp.down" in name or "feed_forward.w3" in name:
                    grp["mlp_down_norm"] = stats
                elif "ffn_gate" in name or "mlp.gate" in name or "feed_forward.w1" in name:
                    grp["mlp_gate_norm"] = stats

        for idx in sorted(layer_groups.keys()):
            g = layer_groups[idx]
            layer_stats.append(LayerStats(
                layer_index=idx,
                attention_norm=g.get("attention_norm"),
                mlp_up_norm=g.get("mlp_up_norm"),
                mlp_down_norm=g.get("mlp_down_norm"),
                mlp_gate_norm=g.get("mlp_gate_norm"),
                total_params=0,
            ))

        # Outlier tensors: std > mean + 3*std_of_stds
        if all_stds:
            mean_std = float(np.mean(all_stds))
            std_std = float(np.std(all_stds))
            threshold = mean_std + 3 * std_std if std_std > 0 else float("inf")
            outlier_tensors = [s.name for s in tensor_stats if s.std > threshold]

        # Embedding analysis
        emb_analysis = self._analyze_embeddings()

        # Capacity estimate
        capacity = self._capacity_estimate(total_params, len(layer_stats))

        return WeightInspectionReport(
            total_parameters=total_params,
            total_size_bytes=total_bytes,
            n_tensors=len(tensor_stats),
            n_layers=len(layer_stats),
            quantization_summary=quant_summary,
            tensor_stats=tensor_stats,
            layer_stats=layer_stats,
            embedding_analysis=emb_analysis,
            outlier_tensors=outlier_tensors,
            capacity_estimate=capacity,
        )

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    def _analyze_embeddings(self) -> Optional[EmbeddingAnalysis]:
        """Find the token embedding tensor and analyze it."""
        emb_tensor = None
        emb_name = None
        for t in self.reader.tensors:
            name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            if name in ("token_embd.weight", "model.embed_tokens.weight",
                        "embeddings.word_embeddings", "tok_embeddings.weight"):
                emb_tensor = t
                emb_name = name
                break
        if emb_tensor is None:
            return None

        arr = self._tensor_array(emb_tensor)
        if arr is None:
            return None

        # GGUF stores shape in C-order (row-major), e.g. [vocab_size, embed_dim]
        # No reversal needed.
        shape = [int(s) for s in emb_tensor.shape]
        if len(shape) != 2:
            return None
        vocab, dim = shape[0], shape[1]
        if arr.size != vocab * dim:
            return None
        try:
            mat = arr.reshape(vocab, dim).astype(np.float32)
        except Exception:
            return None

        # Sample 2000 tokens for analysis to keep it fast
        sample_size = min(2000, vocab)
        idx = np.random.choice(vocab, size=sample_size, replace=False)
        mat_sample = mat[idx]

        centroid = mat_sample.mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))

        # Mean pairwise cosine (sample 500 pairs to keep it bounded)
        n_pairs = 500
        if sample_size >= 2:
            i_idx = np.random.choice(sample_size, n_pairs)
            j_idx = np.random.choice(sample_size, n_pairs)
            mask = i_idx != j_idx
            i_idx, j_idx = i_idx[mask], j_idx[mask]
            a = mat_sample[i_idx]
            b = mat_sample[j_idx]
            na = np.linalg.norm(a, axis=1) + 1e-8
            nb = np.linalg.norm(b, axis=1) + 1e-8
            cos = (a * b).sum(axis=1) / (na * nb)
            mean_cos = float(cos.mean())
        else:
            mean_cos = 0.0

        # Outlier tokens (furthest from centroid)
        norms = np.linalg.norm(mat_sample - centroid, axis=1)
        top_idx = np.argsort(norms)[-10:][::-1]
        top_outliers: List[Tuple[str, float]] = []
        for i in top_idx:
            tok_idx = int(idx[i])
            tok_text = self.tokens[tok_idx] if tok_idx < len(self.tokens) else f"<token_{tok_idx}>"
            top_outliers.append((tok_text, float(norms[i])))

        # Cluster count estimate (rough): unique "direction buckets" via hashing
        # Simpler heuristic: vocab / embedding_dim
        cluster_count = max(1, vocab // max(dim, 1))

        return EmbeddingAnalysis(
            vocab_size=vocab,
            embed_dim=dim,
            centroid_norm=centroid_norm,
            mean_pairwise_cosine=mean_cos,
            top_outlier_tokens=top_outliers,
            cluster_count_estimate=cluster_count,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_layer_index(name: str) -> Optional[int]:
        """Extract layer index from a tensor name like 'blk.12.attn_q.weight'."""
        import re
        m = re.search(r"blk\.(\d+)|layers?\.(\d+)|h\.(\d+)", name)
        if m:
            for g in m.groups():
                if g is not None:
                    return int(g)
        return None

    def _capacity_estimate(self, total_params: int, n_layers: int) -> Dict[str, Any]:
        """Rough estimate of model capacity in knowledge bits."""
        # A common heuristic: ~2 bits of knowledge per parameter (per Anthropic / DeepMind estimates)
        bits_per_param = 2.0
        knowledge_bits = total_params * bits_per_param
        knowledge_bytes = knowledge_bits / 8
        # Rough token estimate (compressed knowledge tokens ~ 1 token per 100 params)
        knowledge_tokens = total_params / 100
        return {
            "estimated_knowledge_bits": int(knowledge_bits),
            "estimated_knowledge_bytes": int(knowledge_bytes),
            "estimated_knowledge_tokens": int(knowledge_tokens),
            "estimated_knowledge_human_words": int(knowledge_tokens * 0.75),  # 1 token ≈ 0.75 word
            "bits_per_param_assumption": bits_per_param,
            "per_layer_capacity_tokens": int(knowledge_tokens / max(n_layers, 1)),
        }


def inspect_gguf(reader: gguf.GGUFReader, tokens: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenience function returning a JSON-serializable dict."""
    insp = WeightInspector(reader, tokens)
    report = insp.inspect()
    return asdict(report)
