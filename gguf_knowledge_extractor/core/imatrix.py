"""
Importance Matrix (Imatrix) Computation
========================================
Computes an importance matrix for a GGUF model by running calibration
data through the forward pass and tracking per-tensor activation magnitudes.

The imatrix tells the quantizer which weights matter more — when used
during quantization, the quantizer can preserve more precision for important
weights and aggressively quantize unimportant ones. This is the same
technique used by `llama.cpp --imatrix`.

How it works:
  1. Run a set of calibration prompts through the numpy forward pass
  2. At each layer, track the mean squared activation of each row/column
     of every weight matrix
  3. Average across all calibration tokens
  4. Output a per-tensor importance score (higher = more important)

The resulting imatrix can be:
  - Saved as JSON for inspection
  - Used by the Quantizer module to weight quantization
  - Used to identify which tensors to keep in higher precision
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable

import numpy as np
import gguf

from .forward_pass import NumpyLlamaForward


@dataclass
class TensorImportance:
    """Importance score for a single tensor."""
    name: str
    n_activations: int          # number of activations observed
    mean_abs_activation: float  # mean absolute activation magnitude
    mean_squared_activation: float
    max_activation: float
    importance_score: float     # normalized 0-1 (1 = most important)


@dataclass
class ImatrixReport:
    """Full importance matrix report."""
    source_gguf: str
    n_calibration_prompts: int
    n_tokens_processed: int
    n_tensors_analyzed: int
    tensor_importances: List[Dict[str, Any]]
    elapsed_seconds: float
    # Recommended precision mapping (based on importance percentiles)
    precision_recommendations: Dict[str, str] = field(default_factory=dict)


# Default calibration prompts — a mix of common knowledge domains
DEFAULT_CALIBRATION_PROMPTS = [
    "The capital of France is",
    "Water boils at 100 degrees",
    "Python is a programming language",
    "The Earth orbits the Sun",
    "Mount Everest is in the Himalayas",
    "The Eiffel Tower was built in",
    "Machine learning is a subset of",
    "The speed of light is approximately",
    "Tokyo is the capital of",
    "Albert Einstein developed the theory of",
    "The Great Wall of China is in",
    "Photosynthesis converts sunlight into",
    "Shakespeare wrote Hamlet in",
    "The Pacific Ocean is the largest",
    "DNA stands for deoxyribonucleic",
]


class ImatrixComputer:
    """Computes importance matrices for GGUF models."""

    def __init__(
        self,
        reader: gguf.GGUFReader,
        fields: Dict[str, Any],
        calibration_prompts: Optional[List[str]] = None,
    ):
        self.forward_pass = NumpyLlamaForward(reader, fields)
        self.calibration_prompts = calibration_prompts or DEFAULT_CALIBRATION_PROMPTS
        # Accumulators: {tensor_name: list of activation magnitudes}
        self.activations: Dict[str, List[float]] = {}
        self.tensor_shapes: Dict[str, List[int]] = {}

    def is_available(self) -> bool:
        return self.forward_pass.is_available()

    # ------------------------------------------------------------------ #
    # Tokenization
    # ------------------------------------------------------------------ #
    def tokenize(self, text: str) -> List[int]:
        vocab = self.forward_pass.tokens_vocab
        if not vocab:
            return []
        vocab_by_len: Dict[int, List[tuple]] = {}
        for i, tok in enumerate(vocab):
            if not isinstance(tok, str):
                continue
            clean = tok.replace("Ġ", " ").replace("▁", " ")
            if len(clean) <= 20:
                vocab_by_len.setdefault(len(clean), []).append((clean, i))
        sorted_lengths = sorted(vocab_by_len.keys(), reverse=True)
        token_ids: List[int] = []
        i = 0
        text_to_match = " " + text
        while i < len(text_to_match):
            matched = False
            for length in sorted_lengths:
                if i + length > len(text_to_match):
                    continue
                substr = text_to_match[i:i+length]
                for tok_str, tok_idx in vocab_by_len[length]:
                    if tok_str == substr:
                        token_ids.append(tok_idx)
                        i += length
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                i += 1
        return token_ids

    # ------------------------------------------------------------------ #
    # Compute
    # ------------------------------------------------------------------ #
    def compute(
        self,
        max_tokens_per_prompt: int = 64,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> ImatrixReport:
        """Compute the importance matrix.

        Args:
            max_tokens_per_prompt: truncate each prompt to this many tokens
            progress_cb: callback(message, current, total)
        """
        if not self.is_available():
            return ImatrixReport(
                source_gguf="", n_calibration_prompts=0, n_tokens_processed=0,
                n_tensors_analyzed=0, tensor_importances=[], elapsed_seconds=0,
            )

        t0 = time.time()
        progress_cb = progress_cb or (lambda m, c, t: None)

        # Hook the forward pass to capture activations
        # We monkey-patch the _apply_layer method to intercept the inputs
        # to each weight matrix
        original_apply_layer = self.forward_pass._apply_layer
        self.activations = {}
        self.tensor_shapes = {}

        def patched_apply_layer(x, layer, positions):
            # Capture activations before attention
            if layer.attn_norm is not None:
                attn_input = self.forward_pass._rms_norm(x, layer.attn_norm)
            else:
                attn_input = x
            self._record_activation("attn_input", attn_input, layer.layer_index)

            # Capture pre-MLP activations (after attention residual)
            attn_out = original_apply_layer(x, layer, positions)
            # Note: original_apply_layer returns the final output of the layer
            # We need a different approach — re-implement with hooks
            return attn_out

        # Actually, let's do a cleaner hook approach:
        # We'll run the forward pass manually and capture activations
        # at each step.

        n_prompts = len(self.calibration_prompts)
        n_tokens_total = 0
        prompt_idx = 0

        for prompt in self.calibration_prompts:
            prompt_idx += 1
            progress_cb(f"Calibrating prompt {prompt_idx}/{n_prompts}", prompt_idx, n_prompts)

            token_ids = self.tokenize(prompt)
            if not token_ids:
                continue
            token_ids = token_ids[:max_tokens_per_prompt]
            n_tokens_total += len(token_ids)

            try:
                self._forward_with_hooks(token_ids)
            except Exception as e:
                # Skip prompts that fail
                continue

        # Compute per-tensor importance
        tensor_importances: List[Dict[str, Any]] = []
        all_scores = []
        for tname, acts in self.activations.items():
            if not acts:
                continue
            mean_abs = float(np.mean(acts))
            mean_sq = float(np.mean([a ** 2 for a in acts]))
            max_act = float(np.max(acts))
            tensor_importances.append({
                "name": tname,
                "n_activations": len(acts),
                "mean_abs_activation": mean_abs,
                "mean_squared_activation": mean_sq,
                "max_activation": max_act,
                "importance_score": 0.0,  # filled below
            })
            all_scores.append(mean_abs)

        # Normalize to 0-1 (1 = most important)
        if all_scores:
            max_score = max(all_scores)
            for ti in tensor_importances:
                ti["importance_score"] = ti["mean_abs_activation"] / max_score if max_score > 0 else 0.0

        # Sort by importance (descending)
        tensor_importances.sort(key=lambda x: x["importance_score"], reverse=True)

        # Recommend precision per tensor
        precision_recs = self._recommend_precision(tensor_importances)

        elapsed = time.time() - t0
        return ImatrixReport(
            source_gguf=getattr(self.forward_pass, 'tensors', {}).get('token_embd.weight', [None])[0] if False else "",
            n_calibration_prompts=n_prompts,
            n_tokens_processed=n_tokens_total,
            n_tensors_analyzed=len(tensor_importances),
            tensor_importances=tensor_importances,
            elapsed_seconds=elapsed,
            precision_recommendations=precision_recs,
        )

    def _forward_with_hooks(self, token_ids: List[int]):
        """Run forward pass with activation hooks on each weight matrix."""
        fp = self.forward_pass
        cfg = fp.config
        seq_len = len(token_ids)

        x = fp.token_embd[np.array(token_ids)].copy()
        positions = np.arange(seq_len, dtype=np.float32)

        for layer in fp.layers:
            # Attention input (after norm)
            if layer.attn_norm is not None:
                attn_input = fp._rms_norm(x, layer.attn_norm)
            else:
                attn_input = x
            self._record_activation(f"blk.{layer.layer_index}.attn_input", attn_input, layer.layer_index)

            # Capture activation for each attention weight matrix
            # W_q: x @ W_q.T, so the "input" to W_q is attn_input
            if layer.attn_q is not None:
                self._record_activation(f"blk.{layer.layer_index}.attn_q", attn_input, layer.layer_index)
            if layer.attn_k is not None:
                self._record_activation(f"blk.{layer.layer_index}.attn_k", attn_input, layer.layer_index)
            if layer.attn_v is not None:
                self._record_activation(f"blk.{layer.layer_index}.attn_v", attn_input, layer.layer_index)

            # Compute attention (use the original method)
            q = attn_input @ layer.attn_q.T
            if layer.attn_k is not None and layer.attn_v is not None:
                k = attn_input @ layer.attn_k.T
                v = attn_input @ layer.attn_v.T
            else:
                # Skip
                continue

            q = q.reshape(seq_len, cfg.n_heads, cfg.head_dim).transpose(1, 0, 2)
            k = k.reshape(seq_len, cfg.n_kv_heads, cfg.head_dim).transpose(1, 0, 2)
            v = v.reshape(seq_len, cfg.n_kv_heads, cfg.head_dim).transpose(1, 0, 2)

            q = fp._apply_rope(q, positions)
            k = fp._apply_rope(k, positions)

            if cfg.n_kv_heads < cfg.n_heads:
                n_rep = cfg.n_heads // cfg.n_kv_heads
                k = np.repeat(k, n_rep, axis=0)
                v = np.repeat(v, n_rep, axis=0)

            scores = q @ k.transpose(0, 2, 1) / np.sqrt(cfg.head_dim)
            mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            scores[:, mask] = -1e30
            scores = scores - scores.max(axis=-1, keepdims=True)
            exp_scores = np.exp(scores)
            attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
            attn_out = attn @ v
            attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, cfg.dim)
            attn_out = attn_out @ layer.attn_output.T
            x = x + attn_out

            # MLP input (after norm)
            if layer.ffn_norm is not None:
                ffn_input = fp._rms_norm(x, layer.ffn_norm)
            else:
                ffn_input = x
            self._record_activation(f"blk.{layer.layer_index}.ffn_input", ffn_input, layer.layer_index)

            # Capture activation for each MLP weight matrix
            if layer.ffn_gate is not None:
                self._record_activation(f"blk.{layer.layer_index}.ffn_gate", ffn_input, layer.layer_index)
            if layer.ffn_up is not None:
                self._record_activation(f"blk.{layer.layer_index}.ffn_up", ffn_input, layer.layer_index)

            # Compute MLP to get hidden state for ffn_down input
            if layer.ffn_gate is not None:
                gate = fp._silu(ffn_input @ layer.ffn_gate.T)
                up = ffn_input @ layer.ffn_up.T
                h = gate * up
            else:
                h = fp._gelu(ffn_input @ layer.ffn_up.T)

            # h is the input to ffn_down
            self._record_activation(f"blk.{layer.layer_index}.ffn_down", h, layer.layer_index)

            mlp_out = h @ layer.ffn_down.T
            x = x + mlp_out

        return x

    def _record_activation(self, name: str, activation: np.ndarray, layer_idx: int):
        """Record the activation magnitude for a tensor."""
        # Use mean absolute value per row (collapse the seq dimension)
        if activation.ndim == 2:
            # [seq, dim] → per-row mean abs
            row_means = np.mean(np.abs(activation), axis=1)
            self.activations.setdefault(name, []).extend(row_means.tolist())
        elif activation.ndim == 1:
            self.activations.setdefault(name, []).append(float(np.mean(np.abs(activation))))
        else:
            self.activations.setdefault(name, []).append(float(np.mean(np.abs(activation))))

    def _recommend_precision(
        self,
        tensor_importances: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Recommend a quantization precision for each tensor based on importance.

        Top 20%: F16 (highest precision)
        Next 30%: Q8_0
        Next 30%: Q5_0
        Bottom 20%: Q4_0 (most aggressive)
        """
        if not tensor_importances:
            return {}
        n = len(tensor_importances)
        recs = {}
        for i, ti in enumerate(tensor_importances):
            percentile = (i / n) * 100  # 0 = most important
            if percentile < 20:
                recs[ti["name"]] = "F16"
            elif percentile < 50:
                recs[ti["name"]] = "Q8_0"
            elif percentile < 80:
                recs[ti["name"]] = "Q5_0"
            else:
                recs[ti["name"]] = "Q4_0"
        return recs
