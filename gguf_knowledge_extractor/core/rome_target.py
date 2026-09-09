"""Explicit target-vector optimization for the ROME research path.

The optimizer treats the ROME value vector as an actual model intervention:
for a candidate v it applies the rank-one constraint W'k=v, runs the real
NumPy model, and scores the target-token objective. This removes the old
embedding-copy heuristic from the target construction step.

The optimizer is deliberately marked research/experimental: SPSA is a
black-box optimizer, not the exact autograd optimization procedure used in
the original ROME implementation. Therefore this module does not claim
paper-equivalence by itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .forward_pass import NumpyLlamaForward


@dataclass
class RomeTargetOptimizationResult:
    target_value: np.ndarray
    initial_loss: float
    final_loss: float
    initial_target_probability: float
    final_target_probability: float
    iterations: int
    seed: int
    method: str = "rome_target_spsa"


def extract_mlp_key(forward: NumpyLlamaForward, token_ids: List[int], layer_idx: int) -> np.ndarray:
    """Return the actual gated MLP activation h at the final token position.

    For Llama-style layers h = SiLU(W_gate x_ffn) * (W_up x_ffn). The helper
    obtains x_ffn by executing the real target layer's attention path while
    zeroing W_down, so it does not confuse a gate row/column with the ROME key.
    """
    if layer_idx < 0 or layer_idx >= len(forward.layers):
        raise ValueError(f"Layer {layer_idx} out of range")
    layer = forward.layers[layer_idx]
    if layer.ffn_gate is None or layer.ffn_up is None or layer.ffn_down is None:
        raise ValueError("ROME key extraction requires gated MLP tensors")
    original = layer.ffn_down
    try:
        layer.ffn_down = np.zeros_like(original, dtype=np.float32)
        attn_residual = forward.get_layer_hidden_state(token_ids, layer_idx)
    finally:
        layer.ffn_down = original
    if attn_residual is None:
        raise RuntimeError("Could not compute target-layer attention residual")
    ffn_input = forward._rms_norm(attn_residual, layer.ffn_norm) if layer.ffn_norm is not None else attn_residual
    gate = forward._silu(ffn_input @ layer.ffn_gate.T)
    up = ffn_input @ layer.ffn_up.T
    key = np.asarray(gate * up, dtype=np.float32)
    if key.ndim != 1 or key.shape[0] != layer.ffn_down.shape[1]:
        raise ValueError(f"Unexpected MLP key shape: {key.shape}")
    if not np.isfinite(key).all() or float(np.linalg.norm(key)) < 1e-10:
        raise ValueError("ROME MLP key is non-finite or near-zero")
    return key


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)
    return z - np.log(np.exp(z).sum())


def _objective(
    forward: NumpyLlamaForward,
    token_ids: List[int],
    layer_idx: int,
    key: np.ndarray,
    candidate: np.ndarray,
    target_id: int,
    base_logits: np.ndarray,
    kl_weight: float,
    l2_weight: float,
    baseline_value: np.ndarray,
) -> Tuple[float, float]:
    """Evaluate v by imposing the exact constraint W'k=v and running the model."""
    layer = forward.layers[layer_idx]
    w = np.asarray(layer.ffn_down, dtype=np.float32)
    key = np.asarray(key, dtype=np.float32).reshape(-1)
    residual = np.asarray(candidate, dtype=np.float32) - (w @ key)
    denom = float(key @ key)
    if denom <= 1e-12:
        raise ValueError("ROME key has near-zero norm")
    delta = np.outer(residual, key) / denom
    layer.ffn_down = w + delta
    try:
        result = forward.forward_with_lens(token_ids)
        logits = result.final_logits
    finally:
        layer.ffn_down = w
    logp = _log_softmax(logits)
    base_logp = _log_softmax(base_logits)
    p_base = np.exp(base_logp)
    kl = float(np.sum(p_base * (base_logp - logp)))
    l2 = float(np.mean((candidate - baseline_value) ** 2))
    loss = float(-logp[target_id] + kl_weight * kl + l2_weight * l2)
    return loss, float(np.exp(logp[target_id]))


def optimize_target_value(
    forward: NumpyLlamaForward,
    token_ids: List[int],
    layer_idx: int,
    key: np.ndarray,
    target_id: int,
    *,
    iterations: int = 24,
    step_size: float = 0.05,
    probe_scale: float = 0.01,
    kl_weight: float = 0.01,
    l2_weight: float = 1e-4,
    seed: int = 0,
) -> RomeTargetOptimizationResult:
    """Optimize v using SPSA against the actual model output.

    SPSA needs only two model evaluations per iteration regardless of hidden
    dimension, making this usable with large GGUF models. It is an explicit
    optimization procedure, but not yet the original ROME autograd solver.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if probe_scale <= 0 or step_size <= 0:
        raise ValueError("step_size and probe_scale must be positive")
    key = np.asarray(key, dtype=np.float32)
    layer = forward.layers[layer_idx]
    baseline_value = layer.ffn_down.astype(np.float32) @ key
    base = forward.forward_with_lens(token_ids)
    base_logits = base.final_logits.copy()
    initial_loss, initial_prob = _objective(
        forward, token_ids, layer_idx, key, baseline_value, target_id,
        base_logits, kl_weight, l2_weight, baseline_value,
    )
    rng = np.random.default_rng(seed)
    value = baseline_value.copy()
    for _ in range(iterations):
        direction = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=value.shape)
        plus = value + probe_scale * direction
        minus = value - probe_scale * direction
        loss_plus, _ = _objective(forward, token_ids, layer_idx, key, plus, target_id, base_logits, kl_weight, l2_weight, baseline_value)
        loss_minus, _ = _objective(forward, token_ids, layer_idx, key, minus, target_id, base_logits, kl_weight, l2_weight, baseline_value)
        grad = ((loss_plus - loss_minus) / (2.0 * probe_scale)) * direction
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1e-12:
            grad = grad / max(grad_norm, 1.0)
        value = value - step_size * grad.astype(np.float32)
    final_loss, final_prob = _objective(
        forward, token_ids, layer_idx, key, value, target_id,
        base_logits, kl_weight, l2_weight, baseline_value,
    )
    return RomeTargetOptimizationResult(
        target_value=value.astype(np.float32), initial_loss=initial_loss,
        final_loss=final_loss, initial_target_probability=initial_prob,
        final_target_probability=final_prob, iterations=iterations, seed=seed,
    )
