"""Reference-faithful mathematical primitives for research methods.

These functions deliberately implement the equations of the named methods,
rather than merely using a similar heuristic. They operate on dense NumPy
arrays and are independent of GGUF I/O so they can be unit-tested against
paper equations before being wired into model surgery.

References:
- ROME: Meng et al., 2022, arXiv:2202.05262
- MEMIT: Meng et al., 2022, arXiv:2210.07229
- TIES-Merging: Yadav et al., 2023, arXiv:2306.01708
- Task Arithmetic: Ilharco et al., 2022/ICLR 2023, arXiv:2212.04089
- Wanda: Sun et al., 2023, arXiv:2306.11695
- SmoothQuant: Xiao et al., 2022, arXiv:2211.10438
- DARE: Yu et al., 2023, arXiv:2311.03099

The module intentionally does not claim paper-equivalence for methods whose
runtime/model architecture requirements are not represented by these pure
math primitives (e.g. full RepE, LEACE, causal scrubbing, or constitutional
AI). Those require additional activation collection and/or intervention
semantics at the model level.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def _as_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _check_matrix(name: str, x: np.ndarray, ndim: int = 2) -> np.ndarray:
    x = _as_f32(x)
    if x.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got {x.ndim}-D")
    return x


def rome_rank_one_update(
    W: np.ndarray,
    k: np.ndarray,
    v_target: np.ndarray,
    key_covariance: np.ndarray,
    ridge: float = 0.0,
) -> np.ndarray:
    """Return the ROME rank-one update using the key second moment.

    ROME solves a constrained least-squares problem. With C = E[k k^T],
    u = C^{-1} k and Lambda = (v* - W k)/(u^T k), the update is

        Delta W = Lambda u^T.

    A small ridge may be supplied for numerical stability when C is estimated
    from finite calibration samples. Setting ridge=0 follows the paper's
    algebra exactly when C is nonsingular.
    """
    W = _check_matrix("W", W)
    C = _check_matrix("key_covariance", key_covariance)
    k = _as_f32(k).reshape(-1)
    v = _as_f32(v_target).reshape(-1)
    if W.shape[1] != k.size or C.shape != (k.size, k.size):
        raise ValueError("W, k, and key_covariance dimensions are inconsistent")
    if W.shape[0] != v.size:
        raise ValueError("v_target dimension must equal W output dimension")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    A = C + np.eye(C.shape[0], dtype=np.float32) * float(ridge)
    try:
        u = np.linalg.solve(A, k)
    except np.linalg.LinAlgError as exc:
        raise ValueError("key covariance is singular; provide calibration data or ridge") from exc
    denom = float(u @ k)
    if abs(denom) < 1e-12:
        raise ValueError("ROME constraint is numerically singular")
    lam = (v - W @ k) / denom
    return np.outer(lam, u).astype(np.float32)


def apply_rank_one_update(W: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply and validate a ROME/MEMIT rank-one update."""
    W = _check_matrix("W", W)
    delta = _check_matrix("delta", delta)
    if W.shape != delta.shape:
        raise ValueError("W and delta must have identical shapes")
    return (W + delta).astype(np.float32)


def memit_layer_update(
    W: np.ndarray,
    K: np.ndarray,
    V_target: np.ndarray,
    key_covariance: Optional[np.ndarray] = None,
    ridge: float = 0.0,
) -> np.ndarray:
    """Compute the closed-form single-layer MEMIT least-squares update.

    K has shape [d, n] (one key per edit) and V_target has shape [m, n].
    Let R = V_target - W K. MEMIT's layer update is

        Delta = R K^T (C + K K^T)^{-1},

    where C is the key second-moment matrix from unedited calibration text.
    This is the core per-layer MEMIT solve; full paper MEMIT additionally
    constructs target residuals and distributes them over several layers.
    """
    W = _check_matrix("W", W)
    K = _check_matrix("K", K)
    V = _check_matrix("V_target", V_target)
    if K.shape[1] != V.shape[1] or W.shape[1] != K.shape[0] or W.shape[0] != V.shape[0]:
        raise ValueError("W/K/V dimensions are inconsistent")
    d = K.shape[0]
    if key_covariance is None:
        C = np.zeros((d, d), dtype=np.float32)
    else:
        C = _check_matrix("key_covariance", key_covariance)
        if C.shape != (d, d):
            raise ValueError("key_covariance has wrong shape")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    A = C + K @ K.T + ridge * np.eye(d, dtype=np.float32)
    R = V - W @ K
    try:
        # Solve A X = K R^T, then transpose to obtain R K^T A^-1.
        X = np.linalg.solve(A, K @ R.T)
    except np.linalg.LinAlgError as exc:
        raise ValueError("MEMIT system is singular; increase calibration/ridge") from exc
    return X.T.astype(np.float32)


def task_vector(base: np.ndarray, finetuned: np.ndarray) -> np.ndarray:
    """Task Arithmetic vector tau = theta_finetuned - theta_base."""
    base = _as_f32(base)
    finetuned = _as_f32(finetuned)
    if base.shape != finetuned.shape:
        raise ValueError("base and finetuned tensors must have identical shapes")
    return finetuned - base


def apply_task_arithmetic(base: np.ndarray, vectors: Sequence[np.ndarray], coefficients: Sequence[float]) -> np.ndarray:
    """Apply weighted task-vector arithmetic to a base tensor."""
    if len(vectors) != len(coefficients):
        raise ValueError("vectors and coefficients must have the same length")
    out = _as_f32(base).copy()
    for vec, coeff in zip(vectors, coefficients):
        vec = _as_f32(vec)
        if vec.shape != out.shape:
            raise ValueError("all task vectors must match the base shape")
        out += float(coeff) * vec
    return out


def ties_merge(
    task_vectors: np.ndarray,
    trim_fraction: float = 0.2,
    merge_scale: float = 1.0,
) -> np.ndarray:
    """TIES merge of task vectors.

    Input shape is [n_tasks, ...]. The implementation follows the three
    substantive TIES stages: trim low-magnitude entries, elect the dominant
    sign, then merge only entries agreeing with that sign.
    """
    tv = _as_f32(task_vectors)
    if tv.ndim < 2:
        raise ValueError("task_vectors must have shape [n_tasks, ...]")
    if not 0.0 <= trim_fraction < 1.0:
        raise ValueError("trim_fraction must be in [0, 1)")
    flat = tv.reshape(tv.shape[0], -1)
    keep = max(1, int(np.ceil((1.0 - trim_fraction) * flat.shape[1])))
    trimmed = np.zeros_like(flat)
    for i in range(flat.shape[0]):
        idx = np.argpartition(np.abs(flat[i]), -keep)[-keep:]
        trimmed[i, idx] = flat[i, idx]

    # Elect sign by total signed magnitude across tasks.
    sign_score = np.sum(trimmed, axis=0)
    elected = np.sign(sign_score)
    merged = np.zeros(flat.shape[1], dtype=np.float32)
    for j in range(flat.shape[1]):
        if elected[j] == 0:
            continue
        vals = trimmed[:, j]
        vals = vals[np.sign(vals) == elected[j]]
        if vals.size:
            merged[j] = float(np.mean(vals)) * float(merge_scale)
    return merged.reshape(tv.shape[1:]).astype(np.float32)


def dare(
    task_vector_: np.ndarray,
    drop_rate: float,
    seed: int = 0,
) -> np.ndarray:
    """DARE drop-and-rescale a task vector with a reproducible RNG."""
    tau = _as_f32(task_vector_)
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError("drop_rate must be in [0, 1)")
    rng = np.random.default_rng(seed)
    keep = rng.random(tau.shape) >= drop_rate
    return (tau * keep.astype(np.float32) / (1.0 - drop_rate)).astype(np.float32)


def wanda_scores(weight: np.ndarray, mean_abs_input: np.ndarray) -> np.ndarray:
    """Wanda importance scores |W_ij| * E|x_j|."""
    W = _check_matrix("weight", weight)
    x = _as_f32(mean_abs_input).reshape(-1)
    if W.shape[1] != x.size:
        raise ValueError("mean_abs_input must match weight input dimension")
    if np.any(x < 0):
        raise ValueError("mean_abs_input must be non-negative")
    return np.abs(W) * x[None, :]


def wanda_row_mask(scores: np.ndarray, sparsity: float) -> np.ndarray:
    """Return the Wanda per-output-row pruning mask."""
    S = _check_matrix("scores", scores)
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    n_keep = max(1, int(np.ceil((1.0 - sparsity) * S.shape[1])))
    mask = np.zeros_like(S, dtype=bool)
    for row in range(S.shape[0]):
        idx = np.argpartition(S[row], -n_keep)[-n_keep:]
        mask[row, idx] = True
    return mask


def smoothquant_scales(
    weight: np.ndarray,
    activation_max: np.ndarray,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute the SmoothQuant channel scales s_j.

    The caller must also apply x' = x / s at runtime. Weight-only rewriting is
    not mathematically equivalent to SmoothQuant and is therefore not done by
    this primitive.
    """
    W = _check_matrix("weight", weight)
    a = _as_f32(activation_max).reshape(-1)
    if W.shape[1] != a.size:
        raise ValueError("activation_max must match weight input dimension")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if np.any(a < 0):
        raise ValueError("activation_max must be non-negative")
    wmax = np.maximum(np.max(np.abs(W), axis=0), eps)
    amax = np.maximum(a, eps)
    return (amax ** alpha / (wmax ** (1.0 - alpha))).astype(np.float32)


def smoothquant_transform(weight: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Apply W' = W diag(s), the weight side of the SmoothQuant transform."""
    W = _check_matrix("weight", weight)
    s = _as_f32(scales).reshape(-1)
    if W.shape[1] != s.size or np.any(s <= 0):
        raise ValueError("scales must be positive and match weight input dimension")
    return (W * s[None, :]).astype(np.float32)


def smoothquant_inverse_activation(x: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Apply x' = diag(s)^-1 x, the runtime side of SmoothQuant."""
    x = _as_f32(x)
    s = _as_f32(scales).reshape(-1)
    if x.shape[-1] != s.size or np.any(s <= 0):
        raise ValueError("scales must be positive and match activation dimension")
    return x / s
