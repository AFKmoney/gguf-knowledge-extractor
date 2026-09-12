"""Reference-faithful ROME calibration helpers.

This module contains the pieces that are architecture-independent but required
before a ROME edit can honestly be called paper-faithful: collecting the MLP
key second moment from calibration contexts and solving the constrained
rank-one update with that covariance.

It intentionally does NOT invent a target value from a token embedding. The
paper's target construction is a separate optimization problem and remains an
explicit integration gate in the runtime editor.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from .research_math import rome_rank_one_update


def collect_key_statistics(keys: Iterable[np.ndarray], ridge: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean key and empirical second moment E[k k^T].

    Each key is a one-dimensional MLP activation vector for the same layer.
    The covariance is deliberately a second moment, matching the quantity
    used by ROME (often called the key covariance in implementations).
    """
    rows = [np.asarray(k, dtype=np.float32).reshape(-1) for k in keys]
    if not rows:
        raise ValueError("at least one calibration key is required")
    d = rows[0].size
    if d == 0 or any(k.size != d for k in rows):
        raise ValueError("all calibration keys must have the same non-zero dimension")
    K = np.stack(rows, axis=0)
    C = (K.T @ K) / float(K.shape[0])
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if ridge:
        C = C + float(ridge) * np.eye(d, dtype=np.float32)
    return K.mean(axis=0), C.astype(np.float32)


def solve_rome_update(
    W: np.ndarray,
    subject_key: np.ndarray,
    target_value: np.ndarray,
    calibration_keys: Iterable[np.ndarray],
    ridge: float = 0.01,
) -> np.ndarray:
    """Solve the ROME rank-one update from calibration activations.

    ``target_value`` must already be the ROME optimized target representation;
    this function never substitutes a token embedding for it.
    """
    _, C = collect_key_statistics(calibration_keys, ridge=0.0)
    return rome_rank_one_update(
        W=np.asarray(W, dtype=np.float32),
        k=np.asarray(subject_key, dtype=np.float32),
        v_target=np.asarray(target_value, dtype=np.float32),
        key_covariance=C,
        ridge=ridge,
    )


def rank_one_constraint_error(
    W_before: np.ndarray,
    delta: np.ndarray,
    key: np.ndarray,
    target: np.ndarray,
) -> float:
    """Return ||(W+Δ)k-target||₂ for the ROME constraint."""
    W = np.asarray(W_before, dtype=np.float32)
    D = np.asarray(delta, dtype=np.float32)
    k = np.asarray(key, dtype=np.float32).reshape(-1)
    v = np.asarray(target, dtype=np.float32).reshape(-1)
    return float(np.linalg.norm((W + D) @ k - v))


def covariance_condition_number(C: np.ndarray, ridge: float = 0.0) -> float:
    """Return cond(C + ridge I) for diagnostics (finite with ridge > 0)."""
    C = np.asarray(C, dtype=np.float64)
    d = C.shape[0]
    A = C + float(ridge) * np.eye(d)
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-30))
