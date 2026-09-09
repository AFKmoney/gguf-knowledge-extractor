import numpy as np
import pytest

from gguf_knowledge_extractor.core.research_math import rome_rank_one_update
from gguf_knowledge_extractor.core.rome_reference import (
    collect_key_statistics,
    rank_one_constraint_error,
    solve_rome_update,
)


def test_collect_key_statistics_is_empirical_second_moment():
    keys = [np.array([1.0, 0.0]), np.array([0.0, 2.0]), np.array([1.0, 2.0])]
    mean, C = collect_key_statistics(keys)
    K = np.stack(keys)
    np.testing.assert_allclose(mean, K.mean(axis=0))
    np.testing.assert_allclose(C, K.T @ K / 3.0)


def test_rome_update_satisfies_target_constraint():
    rng = np.random.default_rng(7)
    d, out = 8, 5
    W = rng.normal(size=(out, d)).astype(np.float32)
    keys = [rng.normal(size=d).astype(np.float32) for _ in range(32)]
    subject = rng.normal(size=d).astype(np.float32)
    target = rng.normal(size=out).astype(np.float32)
    delta = solve_rome_update(W, subject, target, keys, ridge=1e-5)
    error = rank_one_constraint_error(W, delta, subject, target)
    assert error < 1e-4
    # A ROME update is rank-one (up to numerical precision).
    s = np.linalg.svd(delta, compute_uv=False)
    assert s[1] < max(s[0], 1e-8) * 1e-5


def test_reference_solver_matches_direct_math():
    rng = np.random.default_rng(11)
    W = rng.normal(size=(4, 6)).astype(np.float32)
    keys = [rng.normal(size=6).astype(np.float32) for _ in range(20)]
    k = rng.normal(size=6).astype(np.float32)
    v = rng.normal(size=4).astype(np.float32)
    _, C = collect_key_statistics(keys)
    expected = rome_rank_one_update(W, k, v, C, ridge=1e-4)
    actual = solve_rome_update(W, k, v, keys, ridge=1e-4)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_empty_calibration_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        collect_key_statistics([])
