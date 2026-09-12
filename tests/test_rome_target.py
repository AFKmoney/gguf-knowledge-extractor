import numpy as np

from gguf_knowledge_extractor.core.rome_target import (
    optimize_target_value,
    find_subject_token_index,
)


class _Layer:
    def __init__(self):
        self.ffn_down = np.eye(3, dtype=np.float32)


class _Result:
    def __init__(self, logits):
        self.final_logits = np.asarray(logits, dtype=np.float32)
        self.predicted_token_id = int(np.argmax(self.final_logits))
        self.predicted_token_text = str(self.predicted_token_id)


class _ToyForward:
    def __init__(self):
        self.layers = [_Layer()]

    def forward_with_lens(self, token_ids):
        value = self.layers[0].ffn_down @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        score = float(value[0])
        return _Result(np.array([0.0, score, -score], dtype=np.float32))


def test_target_optimizer_is_seed_deterministic():
    key = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    a = _ToyForward()
    b = _ToyForward()
    ra = optimize_target_value(a, [1, 2], 0, key, 1, iterations=6, step_size=0.1, probe_scale=0.02, seed=123, method="rome_target_spsa", kl_weight=0.01)
    rb = optimize_target_value(b, [1, 2], 0, key, 1, iterations=6, step_size=0.1, probe_scale=0.02, seed=123, method="rome_target_spsa", kl_weight=0.01)
    np.testing.assert_allclose(ra.target_value, rb.target_value)
    assert ra.final_target_probability == rb.final_target_probability
    assert ra.method == "rome_target_spsa"


def test_target_optimizer_reduces_target_loss_on_toy_model():
    key = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    result = optimize_target_value(
        _ToyForward(), [1], 0, key, 1,
        iterations=12, step_size=0.2, probe_scale=0.02, seed=7,
        method="rome_target_spsa", kl_weight=0.01,
    )
    assert result.final_loss < result.initial_loss
    assert result.final_target_probability > result.initial_target_probability


def test_paper_approx_adam_reduces_loss_on_toy_model():
    key = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    result = optimize_target_value(
        _ToyForward(), [1], 0, key, 1,
        iterations=16, step_size=0.3, probe_scale=0.02, seed=3,
        method="paper_approx_adam", kl_weight=0.01,
    )
    assert result.method == "paper_approx_adam"
    assert result.paper_equivalent is False
    assert result.final_loss <= result.initial_loss + 1e-6
    assert result.final_target_probability >= result.initial_target_probability - 1e-6


def test_find_subject_token_index_prefers_span_end():
    prompt = [1, 10, 20, 30, 40]
    subject = [20, 30]
    assert find_subject_token_index(prompt, subject) == 3  # last token of span
    assert find_subject_token_index(prompt, []) == 4
