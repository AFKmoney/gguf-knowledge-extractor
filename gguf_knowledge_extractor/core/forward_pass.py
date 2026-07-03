"""
Numpy Llama Forward Pass
========================
A minimal numpy implementation of the Llama-arch transformer forward pass.
This is the foundation for:
  - Logit lens (project hidden states through lm_head at each layer)
  - ROME rank-1 editing (compute key vectors k* for subjects)
  - True causal tracing (find which layer "knows" each fact)

We do NOT use llama-cpp-python here — we read the GGUF weights directly and
run the forward pass in pure numpy. This is slower than llama.cpp but gives
us access to every intermediate hidden state, which is exactly what we need.

Architecture supported:
  - Llama / Llama-2 / Llama-3
  - Mistral (same arch)
  - Qwen-2 (similar)
  - Any model with the standard `blk.{i}.attn_q/k/v/output.weight` +
    `blk.{i}.ffn_gate/up/down.weight` + `blk.{i}.attn_norm/ffn_norm.weight` +
    `output_norm.weight` + `output.weight` + `token_embd.weight` layout.

If the architecture doesn't match (e.g. GPT-2, Phi), we gracefully fall back
to "no forward pass available" and the caller can degrade to weight-only
attribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterator

import numpy as np
import gguf


@dataclass
class LayerTensors:
    layer_index: int
    attn_norm: Optional[np.ndarray]      # [dim]
    attn_q: Optional[np.ndarray]         # [dim, dim]
    attn_k: Optional[np.ndarray]         # [dim, dim] or [kv_dim, dim] for GQA
    attn_v: Optional[np.ndarray]         # [dim, dim] or [kv_dim, dim]
    attn_output: Optional[np.ndarray]    # [dim, dim]
    ffn_norm: Optional[np.ndarray]       # [dim]
    ffn_gate: Optional[np.ndarray]       # [hidden, dim]  (None for non-gated)
    ffn_up: Optional[np.ndarray]         # [hidden, dim]
    ffn_down: Optional[np.ndarray]       # [dim, hidden]


@dataclass
class ForwardPassConfig:
    n_heads: int
    n_kv_heads: int
    head_dim: int
    dim: int
    hidden_dim: int
    rope_freq_base: float = 10000.0
    rope_dim: Optional[int] = None       # default = head_dim
    norm_eps: float = 1e-5
    max_seq_len: int = 2048


@dataclass
class LogitLensResult:
    """The result of a single logit-lens forward pass."""
    token_ids: List[int]
    tokens_text: List[str]
    per_layer_logits: List[Tuple[int, np.ndarray]]   # (layer_idx, logits[last_token])
    final_logits: np.ndarray
    predicted_token_id: int
    predicted_token_text: str
    top_k_tokens: List[Tuple[int, str, float]] = field(default_factory=list)


class NumpyLlamaForward:
    """Pure-numpy forward pass for Llama-architecture models."""

    def __init__(self, reader: gguf.GGUFReader, fields: Dict[str, Any]):
        self.reader = reader
        self.fields = fields
        self.config: Optional[ForwardPassConfig] = None
        self.layers: List[LayerTensors] = []
        self.token_embd: Optional[np.ndarray] = None
        self.output_norm: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None
        self.tokens_vocab: List[str] = []
        self._load()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _tensor_to_float32(self, t) -> Optional[np.ndarray]:
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

    def _load(self):
        # Load all tensors into a dict {name: array}
        tensors: Dict[str, np.ndarray] = {}
        for t in self.reader.tensors:
            name = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
            arr = self._tensor_to_float32(t)
            if arr is not None:
                # Handle scalars/1D reshape
                shape = [int(s) for s in t.shape]
                try:
                    tensors[name] = arr.reshape(shape)
                except Exception:
                    tensors[name] = arr
        self.tensors = tensors

        # Find embedding + output
        self.token_embd = tensors.get("token_embd.weight")
        _output_norm = tensors.get("output_norm.weight")
        if _output_norm is None:
            _output_norm = tensors.get("norm.weight")
        self.output_norm = _output_norm
        self.output = tensors.get("output.weight")
        if self.output is None:
            # Tied embeddings
            self.output = self.token_embd

        if self.token_embd is None:
            return  # cannot do forward pass

        # Load vocab
        tok_field = self.fields.get("tokenizer.ggml.tokens")
        if isinstance(tok_field, list):
            self.tokens_vocab = [str(t) for t in tok_field]
        elif tok_field is not None and hasattr(tok_field, "__iter__"):
            try:
                self.tokens_vocab = [str(t) for t in tok_field]
            except Exception:
                self.tokens_vocab = []

        # Config from metadata
        arch = self.fields.get("general.architecture", "llama")
        prefix = f"{arch}."
        n_heads = self._get_int(f"{arch}.attention.head_count", 32)
        n_kv_heads = self._get_int(f"{arch}.attention.head_count_kv", n_heads)
        dim = self.token_embd.shape[1]
        head_dim = dim // n_heads
        hidden_dim = self._get_int(f"{arch}.feed_forward_length", dim * 4)
        rope_base = self._get_float(f"{arch}.rope.freq_base", 10000.0)
        rope_dim = self._get_int(f"{arch}.rope.dimension_count", head_dim)
        eps = self._get_float(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5)

        self.config = ForwardPassConfig(
            n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
            dim=dim, hidden_dim=hidden_dim,
            rope_freq_base=rope_base, rope_dim=rope_dim,
            norm_eps=float(eps) if eps else 1e-5,
        )

        # Load each layer's tensors
        layer_idx = 0
        while True:
            lt = self._extract_layer(tensors, layer_idx)
            if lt is None:
                break
            self.layers.append(lt)
            layer_idx += 1

    def _extract_layer(self, tensors: Dict[str, np.ndarray], idx: int) -> Optional[LayerTensors]:
        prefixes = [
            f"blk.{idx}.",
            f"layers.{idx}.",
            f"h.{idx}.",
        ]
        def find(name_suffixes: List[str]) -> Optional[np.ndarray]:
            for p in prefixes:
                for sfx in name_suffixes:
                    full = p + sfx
                    if full in tensors:
                        return tensors[full]
            return None

        attn_norm = find(["attn_norm.weight", "attention_norm.weight", "input_layernorm.weight"])
        attn_q = find(["attn_q.weight", "attention.wq.weight", "self_attn.q_proj.weight"])
        attn_k = find(["attn_k.weight", "attention.wk.weight", "self_attn.k_proj.weight"])
        attn_v = find(["attn_v.weight", "attention.wv.weight", "self_attn.v_proj.weight"])
        attn_output = find(["attn_output.weight", "attention.wo.weight", "self_attn.o_proj.weight"])
        ffn_norm = find(["ffn_norm.weight", "ffn_norm.weight", "post_attention_layernorm.weight"])
        ffn_gate = find(["ffn_gate.weight", "mlp.gate.weight", "mlp.gate_proj.weight"])
        ffn_up = find(["ffn_up.weight", "mlp.up.weight", "mlp.up_proj.weight"])
        ffn_down = find(["ffn_down.weight", "mlp.down.weight", "mlp.down_proj.weight"])

        # Need at minimum attn_q, attn_v, attn_output, ffn_up, ffn_down to do a forward pass
        if attn_q is None or attn_v is None or attn_output is None:
            return None
        if ffn_up is None or ffn_down is None:
            return None

        return LayerTensors(
            layer_index=idx,
            attn_norm=attn_norm, attn_q=attn_q, attn_k=attn_k, attn_v=attn_v,
            attn_output=attn_output, ffn_norm=ffn_norm,
            ffn_gate=ffn_gate, ffn_up=ffn_up, ffn_down=ffn_down,
        )

    def _get_int(self, key: str, default: int) -> int:
        v = self.fields.get(key)
        if v is None:
            return default
        try:
            return int(v)
        except Exception:
            return default

    def _get_float(self, key: str, default: float) -> float:
        v = self.fields.get(key)
        if v is None:
            return default
        try:
            return float(v)
        except Exception:
            return default

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        return (
            self.token_embd is not None
            and self.output is not None
            and self.output_norm is not None
            and len(self.layers) > 0
            and self.config is not None
        )

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def forward_with_lens(self, token_ids: List[int]) -> LogitLensResult:
        """Run forward pass, yielding logit lens at each layer.

        Returns the full result including per-layer logits at the LAST position
        (which is what we care about for next-token prediction tasks).
        """
        if not self.is_available():
            raise RuntimeError("Forward pass not available — missing tensors")

        cfg = self.config
        seq_len = len(token_ids)
        if seq_len == 0:
            raise ValueError("Empty token sequence")
        if seq_len > cfg.max_seq_len:
            token_ids = token_ids[:cfg.max_seq_len]
            seq_len = cfg.max_seq_len

        # Embedding lookup
        x = self.token_embd[np.array(token_ids)]  # [seq, dim]
        positions = np.arange(seq_len, dtype=np.float32)

        per_layer_logits: List[Tuple[int, np.ndarray]] = []

        for layer_idx, layer in enumerate(self.layers):
            x = self._apply_layer(x, layer, positions)
            # Logit lens at the last position
            last_hidden = x[-1]  # [dim]
            lens_logits = self._logit_lens(last_hidden)
            per_layer_logits.append((layer_idx, lens_logits))

        # Final
        final_hidden = x[-1]
        final_normed = self._rms_norm(final_hidden, self.output_norm)
        final_logits = final_normed @ self.output.T

        predicted_id = int(np.argmax(final_logits))
        predicted_text = self._token_text(predicted_id)
        top_k = self._top_k(final_logits, k=10)

        return LogitLensResult(
            token_ids=token_ids,
            tokens_text=[self._token_text(i) for i in token_ids],
            per_layer_logits=per_layer_logits,
            final_logits=final_logits,
            predicted_token_id=predicted_id,
            predicted_token_text=predicted_text,
            top_k_tokens=top_k,
        )

    def get_layer_hidden_state(self, token_ids: List[int], layer_idx: int) -> Optional[np.ndarray]:
        """Run forward pass up to layer_idx and return the hidden state at the last position."""
        if not self.is_available():
            return None
        seq_len = len(token_ids)
        if seq_len == 0 or seq_len > self.config.max_seq_len:
            token_ids = token_ids[:self.config.max_seq_len]
            seq_len = len(token_ids)

        x = self.token_embd[np.array(token_ids)]
        positions = np.arange(seq_len, dtype=np.float32)

        for i, layer in enumerate(self.layers):
            if i > layer_idx:
                break
            x = self._apply_layer(x, layer, positions)

        return x[-1]  # hidden state at last position

    # ------------------------------------------------------------------ #
    # Layer application
    # ------------------------------------------------------------------ #
    def _apply_layer(self, x: np.ndarray, layer: LayerTensors, positions: np.ndarray) -> np.ndarray:
        """Apply one transformer layer to x [seq, dim]."""
        cfg = self.config
        seq_len, dim = x.shape

        # Attention
        if layer.attn_norm is not None:
            attn_input = self._rms_norm(x, layer.attn_norm)
        else:
            attn_input = x

        # Projections
        # GGUF stores W_q as [dim, dim] (out, in), so x @ W_q.T gives [seq, dim]
        q = attn_input @ layer.attn_q.T  # [seq, dim]
        if layer.attn_k is not None and layer.attn_v is not None:
            k = attn_input @ layer.attn_k.T  # [seq, kv_dim]
            v = attn_input @ layer.attn_v.T  # [seq, kv_dim]
        else:
            return x  # can't do attention

        # Reshape to heads
        # q: [seq, n_heads * head_dim] -> [n_heads, seq, head_dim]
        q = q.reshape(seq_len, cfg.n_heads, cfg.head_dim).transpose(1, 0, 2)
        # k, v: [seq, n_kv_heads * head_dim] -> [n_kv_heads, seq, head_dim]
        k = k.reshape(seq_len, cfg.n_kv_heads, cfg.head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, cfg.n_kv_heads, cfg.head_dim).transpose(1, 0, 2)

        # Apply RoPE
        q = self._apply_rope(q, positions)
        k = self._apply_rope(k, positions)

        # GQA: if n_kv_heads < n_heads, repeat k and v
        if cfg.n_kv_heads < cfg.n_heads:
            n_rep = cfg.n_heads // cfg.n_kv_heads
            k = np.repeat(k, n_rep, axis=0)
            v = np.repeat(v, n_rep, axis=0)

        # Attention scores
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(cfg.head_dim)  # [n_heads, seq, seq]
        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores[:, mask] = -1e30
        # Softmax
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        # Attention output
        attn_out = attn @ v  # [n_heads, seq, head_dim]
        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, dim)
        attn_out = attn_out @ layer.attn_output.T
        x = x + attn_out

        # MLP
        if layer.ffn_norm is not None:
            ffn_input = self._rms_norm(x, layer.ffn_norm)
        else:
            ffn_input = x

        if layer.ffn_gate is not None:
            # Gated MLP (Llama): SiLU(W_gate @ x) * (W_up @ x) -> W_down
            gate = self._silu(ffn_input @ layer.ffn_gate.T)
            up = ffn_input @ layer.ffn_up.T
            h = gate * up
        else:
            # Non-gated MLP (GPT-2 style): activation(W_up @ x) -> W_down
            h = self._gelu(ffn_input @ layer.ffn_up.T)

        mlp_out = h @ layer.ffn_down.T
        x = x + mlp_out

        return x

    def _logit_lens(self, hidden: np.ndarray) -> np.ndarray:
        """Project a hidden state through final norm + lm_head to get logits."""
        normed = self._rms_norm(hidden, self.output_norm)
        return normed @ self.output.T

    # ------------------------------------------------------------------ #
    # Math helpers
    # ------------------------------------------------------------------ #
    def _rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
        eps = eps if eps is not None else self.config.norm_eps
        if x.ndim == 1:
            rms = np.sqrt((x ** 2).mean() + eps)
            return x * weight / rms
        else:
            rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
            return x * weight / rms

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return x * (1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        # GELU approximation
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    def _apply_rope(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Apply rotary positional embeddings.
        x: [n_heads, seq, head_dim]
        positions: [seq]
        """
        cfg = self.config
        rope_dim = cfg.rope_dim or cfg.head_dim
        # Only apply to first rope_dim dimensions; leave the rest unchanged
        n_heads, seq_len, head_dim = x.shape

        # Compute frequencies
        half = rope_dim // 2
        freqs = 1.0 / (cfg.rope_freq_base ** (np.arange(0, half, dtype=np.float32) / half))
        # angles: [seq, half]
        angles = np.outer(positions, freqs)
        cos = np.cos(angles)  # [seq, half]
        sin = np.sin(angles)

        # Split x into two halves along the head_dim
        x1 = x[..., :half]   # [n_heads, seq, half]
        x2 = x[..., half:rope_dim]  # [n_heads, seq, half]
        x_rest = x[..., rope_dim:]  # [n_heads, seq, head_dim - rope_dim]

        # Apply rotation
        # cos, sin: [seq, half] -> broadcast to [1, seq, half]
        cos_b = cos[None, :, :]
        sin_b = sin[None, :, :]
        new_x1 = x1 * cos_b - x2 * sin_b
        new_x2 = x1 * sin_b + x2 * cos_b

        # Recombine
        return np.concatenate([new_x1, new_x2, x_rest], axis=-1)

    # ------------------------------------------------------------------ #
    # Token helpers
    # ------------------------------------------------------------------ #
    def _token_text(self, idx: int) -> str:
        if 0 <= idx < len(self.tokens_vocab):
            tok = self.tokens_vocab[idx]
            if isinstance(tok, str):
                return tok.replace("Ġ", " ").replace("▁", " ")
            return str(tok)
        return f"<tok_{idx}>"

    def _top_k(self, logits: np.ndarray, k: int = 10) -> List[Tuple[int, str, float]]:
        # Numerically stable softmax
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()
        top_idx = np.argsort(probs)[::-1][:k]
        return [(int(i), self._token_text(int(i)), float(probs[i])) for i in top_idx]

    def find_token_for_text(self, text: str) -> Optional[int]:
        """Find the vocab index for a piece of text (tries exact match, then
        match with BPE space prefix, then substring)."""
        text = text.strip()
        if not text:
            return None
        # Exact match
        for i, tok in enumerate(self.tokens_vocab):
            if tok == text:
                return i
        # With BPE space prefix
        for prefix in [" ", "Ġ", "▁"]:
            target = prefix + text
            for i, tok in enumerate(self.tokens_vocab):
                if tok == target:
                    return i
        # First token that starts with the text
        for i, tok in enumerate(self.tokens_vocab):
            if isinstance(tok, str):
                clean = tok.replace("Ġ", " ").replace("▁", " ").strip()
                if clean.lower() == text.lower():
                    return i
        return None


def is_forward_pass_available(reader: gguf.GGUFReader, fields: Dict[str, Any]) -> bool:
    """Quick check: can we build a forward pass from this GGUF?"""
    try:
        fp = NumpyLlamaForward(reader, fields)
        return fp.is_available()
    except Exception:
        return False
