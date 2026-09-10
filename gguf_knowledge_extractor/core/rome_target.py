"""Explicit target-vector optimization for the ROME research path.

Two optimizers are exposed:

* ``paper_approx_adam`` — Adam on v* with the **same** C-weighted rank-one
  ΔW used at apply time (fixes the H2 SPSA/single-key vs C-weighted mismatch),
  NLL(target) + KL(base) + L2(v, Wk), optional subject-last-token key. This is
  still not bit-identical to the original ROME autograd stack (no PyTorch),
  but it is the intended paper-faithful *approximation* path.
* ``rome_target_spsa`` — legacy black-box SPSA with single-key ΔW in the
  objective (experimental; kept for unit tests / ablation).

Method strings must not claim full paper-equivalence unless a real-model
benchmark contract passes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .forward_pass import NumpyLlamaForward
from .research_math import rome_rank_one_update


@dataclass
class RomeTargetOptimizationResult:
    target_value: np.ndarray
    initial_loss: float
    final_loss: float
    initial_target_probability: float
    final_target_probability: float
    iterations: int
    seed: int
    method: str = "paper_approx_adam"
    paper_equivalent: bool = False


def find_subject_token_index(
    prompt_token_ids: Sequence[int],
    subject_token_ids: Sequence[int],
) -> int:
    """Return the last index of the subject span inside the prompt (paper ROME).

    Falls back to the final prompt token when the subject span is not found.
    """
    prompt = list(prompt_token_ids)
    subject = list(subject_token_ids)
    if not prompt:
        raise ValueError("empty prompt token ids")
    if not subject:
        return len(prompt) - 1
    # Prefer non-BOS subject pieces when subject tokenization included BOS.
    if len(subject) > 1 and subject[0] == prompt[0]:
        subject = subject[1:]
    n, m = len(prompt), len(subject)
    if m == 0:
        return n - 1
    for start in range(n - m, -1, -1):
        if prompt[start : start + m] == subject:
            return start + m - 1
    # Fallback: match last subject token alone.
    last = subject[-1]
    for i in range(n - 1, -1, -1):
        if prompt[i] == last:
            return i
    return n - 1


def extract_mlp_key(
    forward: NumpyLlamaForward,
    token_ids: List[int],
    layer_idx: int,
    token_pos: Optional[int] = None,
) -> np.ndarray:
    """Return the gated MLP activation h at ``token_pos`` (default: last token).

    Paper ROME extracts k* at the **last subject token**, not necessarily the
    final prompt token. Passing ``token_pos`` selects that index.
    """
    if layer_idx < 0 or layer_idx >= len(forward.layers):
        raise ValueError(f"Layer {layer_idx} out of range")
    layer = forward.layers[layer_idx]
    if layer.ffn_gate is None or layer.ffn_up is None or layer.ffn_down is None:
        raise ValueError("ROME key extraction requires gated MLP tensors")
    residual = forward.get_residual_after_attn(token_ids, layer_idx)
    keys = forward.mlp_keys_from_residual(residual, layer_idx)
    if token_pos is None:
        token_pos = keys.shape[0] - 1
    if token_pos < 0:
        token_pos = keys.shape[0] + token_pos
    if token_pos < 0 or token_pos >= keys.shape[0]:
        raise ValueError(f"token_pos {token_pos} out of range for seq={keys.shape[0]}")
    key = np.asarray(keys[token_pos], dtype=np.float32)
    if key.ndim != 1 or key.shape[0] != layer.ffn_down.shape[1]:
        raise ValueError(f"Unexpected MLP key shape: {key.shape}")
    if not np.isfinite(key).all() or float(np.linalg.norm(key)) < 1e-10:
        raise ValueError("ROME MLP key is non-finite or near-zero")
    return key


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)
    return z - np.log(np.exp(z).sum())


def _apply_delta_and_forward(
    forward: NumpyLlamaForward,
    token_ids: List[int],
    layer_idx: int,
    delta: np.ndarray,
) -> np.ndarray:
    layer = forward.layers[layer_idx]
    original = layer.ffn_down
    layer.ffn_down = (np.asarray(original, dtype=np.float32) + delta).astype(np.float32)
    try:
        return forward.forward_with_lens(token_ids).final_logits
    finally:
        layer.ffn_down = original


def _delta_for_candidate(
    w: np.ndarray,
    key: np.ndarray,
    candidate: np.ndarray,
    key_covariance: Optional[np.ndarray],
    ridge: float,
) -> np.ndarray:
    """Build ΔW for a candidate v* — C-weighted when covariance is supplied."""
    if key_covariance is None:
        # Legacy single-key matched update (experimental SPSA path).
        residual = np.asarray(candidate, dtype=np.float32) - (w @ key)
        denom = float(key @ key)
        if denom <= 1e-12:
            raise ValueError("ROME key has near-zero norm")
        return np.outer(residual, key) / denom
    return rome_rank_one_update(
        W=w,
        k=key,
        v_target=candidate,
        key_covariance=key_covariance,
        ridge=ridge,
    )


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
    key_covariance: Optional[np.ndarray] = None,
    ridge: float = 0.01,
) -> Tuple[float, float]:
    """Evaluate v by imposing W'k=v with the **apply-time** ΔW rule."""
    layer = forward.layers[layer_idx]
    w = np.asarray(layer.ffn_down, dtype=np.float32)
    key = np.asarray(key, dtype=np.float32).reshape(-1)
    delta = _delta_for_candidate(w, key, candidate, key_covariance, ridge)
    logits = _apply_delta_and_forward(forward, token_ids, layer_idx, delta)
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
    kl_weight: float = 0.1,
    l2_weight: float = 1e-4,
    seed: int = 0,
    method: str = "paper_approx_adam",
    key_covariance: Optional[np.ndarray] = None,
    ridge: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
) -> RomeTargetOptimizationResult:
    """Optimize v* against the real NumPy forward.

    ``paper_approx_adam`` (default): Adam with 2-point Rademacher gradient
    estimates; each evaluation applies the C-weighted ROME update when
    ``key_covariance`` is set (must match the final apply path).

    ``rome_target_spsa``: legacy SPSA + single-key ΔW (ignores covariance).
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if probe_scale <= 0 or step_size <= 0:
        raise ValueError("step_size and probe_scale must be positive")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    key = np.asarray(key, dtype=np.float32)
    layer = forward.layers[layer_idx]
    baseline_value = layer.ffn_down.astype(np.float32) @ key
    base = forward.forward_with_lens(token_ids)
    base_logits = base.final_logits.copy()

    if method == "rome_target_spsa":
        use_cov = None
        method_name = "rome_target_spsa"
    elif method in ("paper_approx_adam", "paper_approx"):
        use_cov = key_covariance
        method_name = "paper_approx_adam"
    else:
        raise ValueError(f"Unknown target optimization method: {method}")
    kl_w = float(kl_weight)

    initial_loss, initial_prob = _objective(
        forward, token_ids, layer_idx, key, baseline_value, target_id,
        base_logits, kl_w, l2_weight, baseline_value, use_cov, ridge,
    )

    rng = np.random.default_rng(seed)
    value = baseline_value.copy().astype(np.float32)
    m = np.zeros_like(value)
    v = np.zeros_like(value)

    for t in range(1, iterations + 1):
        direction = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=value.shape)
        plus = value + probe_scale * direction
        minus = value - probe_scale * direction
        loss_plus, _ = _objective(
            forward, token_ids, layer_idx, key, plus, target_id,
            base_logits, kl_w, l2_weight, baseline_value, use_cov, ridge,
        )
        loss_minus, _ = _objective(
            forward, token_ids, layer_idx, key, minus, target_id,
            base_logits, kl_w, l2_weight, baseline_value, use_cov, ridge,
        )
        grad = ((loss_plus - loss_minus) / (2.0 * probe_scale)) * direction

        if method == "rome_target_spsa":
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > 1e-12:
                grad = grad / max(grad_norm, 1.0)
            value = value - step_size * grad.astype(np.float32)
        else:
            # Adam
            g = grad.astype(np.float32)
            m = adam_beta1 * m + (1.0 - adam_beta1) * g
            v = adam_beta2 * v + (1.0 - adam_beta2) * (g * g)
            m_hat = m / (1.0 - adam_beta1 ** t)
            v_hat = v / (1.0 - adam_beta2 ** t)
            # Normalize Adam step to avoid explosion in high dim
            step = m_hat / (np.sqrt(v_hat) + adam_eps)
            step_norm = float(np.linalg.norm(step))
            if step_norm > 1.0:
                step = step / step_norm
            value = value - step_size * step

    final_loss, final_prob = _objective(
        forward, token_ids, layer_idx, key, value, target_id,
        base_logits, kl_w, l2_weight, baseline_value, use_cov, ridge,
    )
    return RomeTargetOptimizationResult(
        target_value=value.astype(np.float32),
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_target_probability=initial_prob,
        final_target_probability=final_prob,
        iterations=iterations,
        seed=seed,
        method=method_name,
        paper_equivalent=False,
    )
