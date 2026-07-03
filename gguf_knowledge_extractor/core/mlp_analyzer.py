"""
MLP Memory Analyzer (ROME / MEMIT-inspired)
============================================
Decomposes the MLP modules of a transformer into their constituent
"memories" — without ever running inference.

Background
----------
In a transformer MLP block (Llama-style), the architecture is:

    hidden = W_gate @ x          # up-projection: [hidden_dim, embed_dim] @ [embed_dim]
    gate   = SiLU(W_gate @ x)    # gated activation
    up     = W_up   @ x          # up projection
    h      = gate * up           # element-wise
    y      = W_down @ h          # down-projection: [embed_dim, hidden_dim] @ [hidden_dim]

In the simpler (non-gated) FFN form used by GPT-2 / early Llama variants:

    h = activation(W_up @ x)
    y = W_down @ h

Either way, **each column of W_down corresponds to a single "memory neuron"**
whose key is the matching column/row of W_up (or W_gate) and whose value is
the column of W_down. ROME (Rank-One Model Editing) showed that these
neurons are where factual associations are physically stored.

This module extracts, for every layer:
  - Per-neuron key vector (input direction)
  - Per-neuron value vector (output direction)
  - Per-neuron "memory strength" = ||key|| * ||value||
  - Top-K tokens from the vocabulary that most strongly activate each top neuron
    (by computing token_emb @ key for every vocab token)
  - Top-K tokens whose embedding is most aligned with each neuron's value
    direction (i.e., the tokens this memory would "produce")

This is the closest you can get to "reading the model's memory" without
running it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gguf


@dataclass
class NeuronMemory:
    """A single 'memory neuron' extracted from the MLP."""
    layer: int
    neuron_index: int
    key_norm: float
    value_norm: float
    memory_strength: float          # ||key|| * ||value||
    top_activating_tokens: List[Tuple[str, float]]   # (token_text, activation)
    top_output_tokens: List[Tuple[str, float]]       # (token_text, cosine_sim)


@dataclass
class MLPLayerAnalysis:
    layer: int
    hidden_dim: int
    embed_dim: int
    is_gated: bool
    n_neurons: int
    top_neurons: List[NeuronMemory]
    layer_strength: float           # total memory strength = sum of ||key||*||value||
    concentration: float            # how concentrated memory is in top neurons (0-1)


@dataclass
class MLPModelAnalysis:
    n_layers_analyzed: int
    layers: List[MLPLayerAnalysis]
    global_top_neurons: List[NeuronMemory]   # top neurons across all layers
    knowledge_fingerprint: str                # sha256 of the global top neurons
    fingerprint_brief: Dict[str, Any]         # quick-summary dict


class MLPAnalyzer:
    """Extracts MLP memory neurons from a GGUF file."""

    def __init__(
        self,
        reader: gguf.GGUFReader,
        tokens: Optional[List[str]] = None,
        top_k_per_layer: int = 20,
        top_k_tokens: int = 10,
        max_neurons_per_layer: int = 100000,  # safety bound
    ):
        self.reader = reader
        self.tokens = tokens or []
        self.top_k_per_layer = top_k_per_layer
        self.top_k_tokens = top_k_tokens
        self.max_neurons_per_layer = max_neurons_per_layer

        # Find embedding matrix (for token attribution)
        self.embed_matrix = self._find_embedding_matrix()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _find_embedding_matrix(self) -> Optional[np.ndarray]:
        """Find and load the token embedding matrix as a 2D float32 array."""
        for t in self.reader.tensors:
            name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            if name in ("token_embd.weight", "model.embed_tokens.weight",
                        "embeddings.word_embeddings", "tok_embeddings.weight"):
                arr = self._tensor_to_float(t)
                if arr is not None and arr.ndim == 1:
                    shape = [int(s) for s in t.shape]
                    if len(shape) == 2:
                        try:
                            arr = arr.reshape(shape)
                        except Exception:
                            return None
                if arr is not None and arr.ndim == 2:
                    return arr.astype(np.float32)
        return None

    def _tensor_to_float(self, t) -> Optional[np.ndarray]:
        """Convert a tensor (any dtype) to a 1D float32 numpy array."""
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
            return arr
        except Exception:
            return None

    def _find_layer_tensor(self, layer_idx: int, candidates: List[str]) -> Optional[np.ndarray]:
        """Find a tensor by name pattern (e.g., 'blk.5.ffn_up.weight')."""
        for t in self.reader.tensors:
            name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            for cand in candidates:
                # Build a regex-like check
                if self._name_matches(name, layer_idx, cand):
                    arr = self._tensor_to_float(t)
                    if arr is not None:
                        shape = [int(s) for s in t.shape]
                        try:
                            return arr.reshape(shape)
                        except Exception:
                            return None
        return None

    @staticmethod
    def _name_matches(name: str, layer_idx: int, pattern: str) -> bool:
        """Pattern like 'blk.{i}.ffn_up.weight' or 'layers.{i}.feed_forward.w1.weight'."""
        import re
        pat = pattern.replace("{i}", str(layer_idx))
        # Allow either dots or underscores in the actual name
        pat_re = re.escape(pat).replace(r"\{i\}", r"\d+")
        return bool(re.search(pat_re, name))

    @staticmethod
    def _extract_layer_index(name: str) -> Optional[int]:
        import re
        m = re.search(r"blk\.(\d+)|layers?\.(\d+)|h\.(\d+)", name)
        if m:
            for g in m.groups():
                if g is not None:
                    return int(g)
        return None

    def _token_text(self, idx: int) -> str:
        if 0 <= idx < len(self.tokens):
            tok = self.tokens[idx]
            # Strip BPE prefix
            if isinstance(tok, str):
                return tok.replace("Ġ", " ").replace("▁", " ")[:30]
            return str(tok)[:30]
        return f"<tok_{idx}>"

    # ------------------------------------------------------------------ #
    # Main analysis
    # ------------------------------------------------------------------ #
    def analyze(self) -> MLPModelAnalysis:
        # Find all unique layer indices that have an ffn_up / mlp_gate tensor
        layer_indices = set()
        for t in self.reader.tensors:
            name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            if any(kw in name for kw in ("ffn_up", "mlp.up", "feed_forward.w2",
                                          "mlp.gate", "ffn_gate", "feed_forward.w1",
                                          "mlp.down", "ffn_down", "feed_forward.w3")):
                idx = self._extract_layer_index(name)
                if idx is not None:
                    layer_indices.add(idx)

        layer_analyses: List[MLPLayerAnalysis] = []
        all_neurons: List[NeuronMemory] = []

        for li in sorted(layer_indices):
            analysis = self.analyze_layer(li)
            if analysis is not None:
                layer_analyses.append(analysis)
                all_neurons.extend(analysis.top_neurons)

        # Global top neurons
        all_neurons.sort(key=lambda n: n.memory_strength, reverse=True)
        global_top = all_neurons[:50]

        # Knowledge fingerprint: hash of (layer, neuron_idx, top_activating_tokens)
        fp_input = ""
        for n in global_top:
            fp_input += f"L{n.layer}N{n.neuron_index}|"
            fp_input += ",".join(t for t, _ in n.top_activating_tokens[:5])
            fp_input += "\n"
        fingerprint = hashlib.sha256(fp_input.encode("utf-8")).hexdigest()

        # Brief fingerprint summary
        brief = {
            "n_layers_analyzed": len(layer_analyses),
            "n_global_top_neurons": len(global_top),
            "strongest_layer": max(layer_analyses, key=lambda x: x.layer_strength).layer if layer_analyses else None,
            "most_concentrated_layer": max(layer_analyses, key=lambda x: x.concentration).layer if layer_analyses else None,
            "top_5_neurons": [
                {"layer": n.layer, "neuron": n.neuron_index,
                 "strength": round(n.memory_strength, 4),
                 "top_token": n.top_activating_tokens[0][0] if n.top_activating_tokens else ""}
                for n in global_top[:5]
            ],
        }

        return MLPModelAnalysis(
            n_layers_analyzed=len(layer_analyses),
            layers=layer_analyses,
            global_top_neurons=global_top,
            knowledge_fingerprint=fingerprint,
            fingerprint_brief=brief,
        )

    def analyze_layer(self, layer_idx: int) -> Optional[MLPLayerAnalysis]:
        """Analyze the MLP at a single transformer block."""
        # Try to find gate/up/down tensors
        w_gate = self._find_layer_tensor(layer_idx, [
            f"blk.{layer_idx}.ffn_gate.weight",
            f"blk.{layer_idx}.mlp.gate.weight",
            f"layers.{layer_idx}.feed_forward.w1.weight",
        ])
        w_up = self._find_layer_tensor(layer_idx, [
            f"blk.{layer_idx}.ffn_up.weight",
            f"blk.{layer_idx}.mlp.up.weight",
            f"layers.{layer_idx}.feed_forward.w2.weight",
        ])
        w_down = self._find_layer_tensor(layer_idx, [
            f"blk.{layer_idx}.ffn_down.weight",
            f"blk.{layer_idx}.mlp.down.weight",
            f"layers.{layer_idx}.feed_forward.w3.weight",
        ])

        if w_down is None:
            return None  # can't analyze without down projection

        is_gated = w_gate is not None

        # For non-gated MLPs: w_up is the up projection
        # For gated MLPs (Llama-style): w_gate is the "gate" branch, w_up is the "up" branch
        # Either way, the "key" of neuron i is column i of the up/gate projection,
        # and the "value" is column i of w_down (i.e., row i of w_down.T).
        #
        # Conventions vary: in GGUF, ffn_up.weight is stored as [hidden_dim, embed_dim]
        # (output, input), so column i of ffn_up = neuron i's input direction (in embed_dim).
        # ffn_down.weight is stored as [embed_dim, hidden_dim], so column i of ffn_down
        # = neuron i's output direction (in embed_dim).
        #
        # We use the gate if available (gated activation), else the up.

        key_source = w_gate if is_gated else w_up
        if key_source is None:
            return None

        # Shapes
        # key_source: [hidden_dim, embed_dim]  → column i is neuron i's input direction
        # w_down:     [embed_dim, hidden_dim]  → column i is neuron i's output direction
        if key_source.ndim != 2 or w_down.ndim != 2:
            return None

        hidden_dim, embed_dim_key = key_source.shape
        embed_dim_val, hidden_dim_down = w_down.shape

        # Sanity check
        if embed_dim_key != embed_dim_val:
            # Maybe transposed; try to fix
            if hidden_dim == embed_dim_val and embed_dim_key == hidden_dim_down:
                # key_source is actually [embed_dim, hidden_dim]
                key_source = key_source.T
                hidden_dim, embed_dim_key = key_source.shape
            else:
                return None
        if hidden_dim != hidden_dim_down:
            return None

        # We now have:
        # key_source: [hidden_dim, embed_dim]  (rows are neurons, columns are embed dims)
        # w_down:     [embed_dim, hidden_dim]  (rows are embed dims, columns are neurons)

        # Limit neurons for tractability
        n_neurons = min(hidden_dim, self.max_neurons_per_layer)

        # Compute per-neuron norms
        key_norms = np.linalg.norm(key_source[:n_neurons], axis=1)  # [n_neurons]
        value_norms = np.linalg.norm(w_down[:, :n_neurons], axis=0)  # [n_neurons]
        memory_strengths = key_norms * value_norms

        # Pick top-K neurons
        top_k = min(self.top_k_per_layer, n_neurons)
        top_indices = np.argsort(memory_strengths)[::-1][:top_k]

        # Embedding matrix (if available)
        emb = self.embed_matrix
        # If embedding matrix is huge, sample to keep memory bounded
        if emb is not None and emb.shape[0] > 10000:
            # Use full vocab for the search — it's just a matrix multiply
            pass

        top_neurons: List[NeuronMemory] = []
        for ni in top_indices:
            ni = int(ni)
            key_vec = key_source[ni]  # [embed_dim]
            val_vec = w_down[:, ni]   # [embed_dim]

            top_activating: List[Tuple[str, float]] = []
            top_output: List[Tuple[str, float]] = []

            if emb is not None and emb.shape[1] == len(key_vec):
                # Top activating tokens: argmax of (emb @ key_vec)
                activations = emb @ key_vec
                # Get top-K token indices
                top_act_idx = np.argsort(activations)[::-1][:self.top_k_tokens]
                top_activating = [
                    (self._token_text(int(i)), float(activations[i]))
                    for i in top_act_idx
                ]

                # Top output tokens: highest cosine similarity between emb rows and val_vec
                emb_norms = np.linalg.norm(emb, axis=1) + 1e-8
                val_norm = np.linalg.norm(val_vec) + 1e-8
                cosines = (emb @ val_vec) / (emb_norms * val_norm)
                top_out_idx = np.argsort(cosines)[::-1][:self.top_k_tokens]
                top_output = [
                    (self._token_text(int(i)), float(cosines[i]))
                    for i in top_out_idx
                ]

            top_neurons.append(NeuronMemory(
                layer=layer_idx,
                neuron_index=ni,
                key_norm=float(key_norms[ni]),
                value_norm=float(value_norms[ni]),
                memory_strength=float(memory_strengths[ni]),
                top_activating_tokens=top_activating,
                top_output_tokens=top_output,
            ))

        layer_strength = float(memory_strengths.sum())
        # Concentration: fraction of total strength held by top-K neurons
        top_k_strength = float(memory_strengths[top_indices].sum())
        concentration = top_k_strength / max(layer_strength, 1e-8)

        return MLPLayerAnalysis(
            layer=layer_idx,
            hidden_dim=int(hidden_dim),
            embed_dim=int(embed_dim_key),
            is_gated=is_gated,
            n_neurons=int(n_neurons),
            top_neurons=top_neurons,
            layer_strength=layer_strength,
            concentration=concentration,
        )


def analyze_mlp_memories(
    reader: gguf.GGUFReader,
    tokens: Optional[List[str]] = None,
    top_k_per_layer: int = 20,
    top_k_tokens: int = 10,
) -> Dict[str, Any]:
    """Convenience function returning a JSON-serializable dict."""
    analyzer = MLPAnalyzer(
        reader, tokens=tokens,
        top_k_per_layer=top_k_per_layer,
        top_k_tokens=top_k_tokens,
    )
    report = analyzer.analyze()
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
