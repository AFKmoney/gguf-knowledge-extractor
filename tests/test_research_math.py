"""Equation-level tests for reference-faithful research primitives."""
import numpy as np

from gguf_knowledge_extractor.core.research_math import (
    apply_rank_one_update,
    apply_task_arithmetic,
    dare,
    memit_layer_update,
    rome_rank_one_update,
    smoothquant_inverse_activation,
    smoothquant_scales,
    smoothquant_transform,
    task_vector,
    ties_merge,
    wanda_row_mask,
    wanda_scores,
)


def test_rome_hits_constraint_and_is_rank_one():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(5, 4)).astype(np.float32)
    C = rng.normal(size=(4, 4)).astype(np.float32)
    C = C @ C.T + np.eye(4, dtype=np.float32)
    k = rng.normal(size=4).astype(np.float32)
    v = rng.normal(size=5).astype(np.float32)
    delta = rome_rank_one_update(W, k, v, C)
    W2 = apply_rank_one_update(W, delta)
    assert np.allclose(W2 @ k, v, atol=1e-4)
    assert np.linalg.matrix_rank(delta, tol=1e-5) <= 1


def test_memit_satisfies_regularized_normal_equation():
    rng = np.random.default_rng(2)
    W = rng.normal(size=(6, 4)).astype(np.float32)
    K = rng.normal(size=(4, 3)).astype(np.float32)
    V = rng.normal(size=(6, 3)).astype(np.float32)
    C = np.eye(4, dtype=np.float32) * 0.1
    delta = memit_layer_update(W, K, V, C)
    A = C + K @ K.T
    residual = (V - (W + delta) @ K) @ K.T
    # The regularized optimum obeys Delta A = (V-WK)K^T.
    assert np.allclose(delta @ A, residual + delta @ A, atol=1e-4)
    expected = (V - W @ K) @ K.T
    assert np.allclose(delta @ A, expected, atol=1e-4)


def test_task_arithmetic_is_exact_vector_addition():
    base = np.array([1, 2, 3], dtype=np.float32)
    ft = np.array([2, 4, 8], dtype=np.float32)
    tau = task_vector(base, ft)
    assert np.array_equal(tau, np.array([1, 2, 5], dtype=np.float32))
    assert np.array_equal(apply_task_arithmetic(base, [tau], [1.0]), ft)


def test_ties_elects_consistent_sign_after_trim():
    tv = np.array([[1.0, -0.1, 2.0], [2.0, -0.2, -3.0]], dtype=np.float32)
    merged = ties_merge(tv, trim_fraction=0.0)
    assert merged[0] > 0
    assert merged[1] < 0
    assert merged[2] < 0


def test_dare_is_reproducible():
    tau = np.arange(100, dtype=np.float32)
    a = dare(tau, 0.8, seed=123)
    b = dare(tau, 0.8, seed=123)
    assert np.array_equal(a, b)


def test_wanda_is_rowwise_not_global():
    W = np.array([[10, 1, 1, 1], [1, 1, 1, 10]], dtype=np.float32)
    x = np.ones(4, dtype=np.float32)
    scores = wanda_scores(W, x)
    mask = wanda_row_mask(scores, 0.5)
    assert mask[0, 0] and mask[1, 3]
    assert int(mask.sum()) == 4


def test_smoothquant_transform_is_exactly_compensated():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(4, 3)).astype(np.float32)
    x = rng.normal(size=(7, 3)).astype(np.float32)
    scales = smoothquant_scales(W, np.max(np.abs(x), axis=0), alpha=0.5)
    W2 = smoothquant_transform(W, scales)
    x2 = smoothquant_inverse_activation(x, scales)
    assert np.allclose(x @ W.T, x2 @ W2.T, atol=2e-5)
