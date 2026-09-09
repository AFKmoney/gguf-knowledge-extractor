"""
Activation Patcher
===================
True causal mediation analysis — find which hidden states at which layers
are causally responsible for a model's output.

The implementation keeps one fixed corruption realization for every layer
comparison and restores the clean post-layer activation at the requested
layer. This makes the layer effects comparable and matches the intended
causal-mediation intervention.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gguf

from .forward_pass import NumpyLlamaForward


@dataclass
class PatchResult:
    """Result of patching a single layer."""
    layer: int
    clean_top1_token: str
    clean_top1_prob: float
    corrupt_top1_token: str
    corrupt_top1_prob: float
    restored_top1_token: str
    restored_top1_prob: float
    clean_answer_prob_clean: float
    clean_answer_prob_corrupt: float
    clean_answer_prob_restored: float
    causal_effect: float
    indirect_effect: float


@dataclass
class CausalMediationReport:
    """Full causal mediation analysis report for one fact."""
    probe_id: str
    prompt: str
    expected_answer: Optional[str]
    expected_token_id: Optional[int]
    subject_token_indices: List[int]
    corruption_noise_std: float
    layer_results: List[Dict[str, Any]]
    best_layer: Optional[int]
    best_causal_effect: float
    clean_prediction: str
    corrupt_prediction: str
    elapsed_seconds: float
    method: str = "causal_mediation"


class ActivationPatcher:
    """Causal mediation analysis via activation patching."""

    def __init__(self, reader: gguf.GGUFReader, fields: Dict[str, Any]):
        self.forward_pass = NumpyLlamaForward(reader, fields)
        self.fields = fields

    def is_available(self) -> bool:
        return self.forward_pass.is_available()

    def tokenize(self, text: str) -> List[int]:
        """Greedy longest-match fallback tokenizer using the GGUF vocabulary."""
        vocab = self.forward_pass.tokens_vocab
        if not vocab:
            return []

        vocab_by_len: Dict[int, List[Tuple[str, int]]] = {}
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
                substr = text_to_match[i:i + length]
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

    @staticmethod
    def _softmax_prob(logits: np.ndarray, idx: int) -> float:
        m = float(np.max(logits))
        exp = np.exp(logits - m)
        return float(exp[idx] / np.sum(exp))

    def analyze(
        self,
        probe_id: str,
        prompt: str,
        expected_answer: Optional[str] = None,
        subject_indices: Optional[List[int]] = None,
        corruption_noise_std: float = 1.0,
        seed: int = 42,
    ) -> CausalMediationReport:
        """Run causal mediation analysis on a single fact.

        A single Gaussian corruption tensor is generated once and reused for
        every condition. For layer L, the corrupted run is restored with the
        *post-L clean activation*, then computation resumes at L+1. This avoids
        the previous off-by-one intervention and makes causal effects directly
        comparable across layers.
        """
        t0 = time.time()

        if not self.is_available():
            return CausalMediationReport(
                probe_id=probe_id, prompt=prompt,
                expected_answer=expected_answer, expected_token_id=None,
                subject_token_indices=[], corruption_noise_std=corruption_noise_std,
                layer_results=[], best_layer=None, best_causal_effect=0.0,
                clean_prediction="", corrupt_prediction="",
                elapsed_seconds=time.time() - t0,
                method="forward_pass_unavailable",
            )

        token_ids = self.tokenize(prompt)
        if not token_ids:
            return CausalMediationReport(
                probe_id=probe_id, prompt=prompt,
                expected_answer=expected_answer, expected_token_id=None,
                subject_token_indices=[], corruption_noise_std=corruption_noise_std,
                layer_results=[], best_layer=None, best_causal_effect=0.0,
                clean_prediction="", corrupt_prediction="",
                elapsed_seconds=time.time() - t0,
                method="tokenization_failed",
            )

        expected_id = self.forward_pass.find_token_for_text(expected_answer) if expected_answer else None

        if subject_indices is None:
            n_subj = min(3, max(1, len(token_ids) - 1))
            subject_indices = list(range(len(token_ids) - n_subj, len(token_ids)))
        subject_indices = [i for i in subject_indices if 0 <= i < len(token_ids)]

        # Generate corruption exactly once. Reusing this tensor is essential:
        # otherwise each comparison changes both the intervention and layer.
        rng = np.random.RandomState(seed)
        corruption = np.zeros_like(self.forward_pass.token_embd[np.array(token_ids)], dtype=np.float32)
        if corruption_noise_std > 0:
            for idx in subject_indices:
                corruption[idx] = rng.randn(self.forward_pass.config.dim).astype(np.float32) * corruption_noise_std

        clean_hidden_states = self._forward_with_hidden_states(token_ids)
        corrupt_hidden_states = self._forward_with_hidden_states(token_ids, corruption=corruption)

        clean_logits = self._logits_from_hidden(clean_hidden_states[-1][-1])
        corrupt_logits = self._logits_from_hidden(corrupt_hidden_states[-1][-1])
        clean_prediction = self.forward_pass._token_text(int(np.argmax(clean_logits)))
        corrupt_prediction = self.forward_pass._token_text(int(np.argmax(corrupt_logits)))

        clean_answer_prob = self._softmax_prob(clean_logits, expected_id) if expected_id is not None else 0.0
        corrupt_answer_prob = self._softmax_prob(corrupt_logits, expected_id) if expected_id is not None else 0.0

        layer_results: List[Dict[str, Any]] = []
        best_layer: Optional[int] = None
        best_effect = -float("inf")

        for layer_idx in range(len(self.forward_pass.layers)):
            restored_logits = self._forward_with_restoration(
                token_ids=token_ids,
                restore_layer=layer_idx,
                restore_hidden=clean_hidden_states[layer_idx + 1],
                corruption=corruption,
            )

            restored_top1 = int(np.argmax(restored_logits))
            restored_answer_prob = (
                self._softmax_prob(restored_logits, expected_id)
                if expected_id is not None else 0.0
            )
            causal_effect = restored_answer_prob - corrupt_answer_prob
            indirect_effect = clean_answer_prob - restored_answer_prob

            clean_lens = self._logit_lens_from_hidden(clean_hidden_states[layer_idx + 1][-1])
            corrupt_lens = self._logit_lens_from_hidden(corrupt_hidden_states[layer_idx + 1][-1])
            clean_top1 = int(np.argmax(clean_lens))
            corrupt_top1 = int(np.argmax(corrupt_lens))

            layer_results.append({
                "layer": layer_idx,
                "clean_top1_token": self.forward_pass._token_text(clean_top1),
                "clean_top1_prob": self._softmax_prob(clean_lens, clean_top1),
                "corrupt_top1_token": self.forward_pass._token_text(corrupt_top1),
                "corrupt_top1_prob": self._softmax_prob(corrupt_lens, corrupt_top1),
                "restored_top1_token": self.forward_pass._token_text(restored_top1),
                "restored_top1_prob": self._softmax_prob(restored_logits, restored_top1),
                "clean_answer_prob_clean": clean_answer_prob,
                "clean_answer_prob_corrupt": corrupt_answer_prob,
                "clean_answer_prob_restored": restored_answer_prob,
                "causal_effect": causal_effect,
                "indirect_effect": indirect_effect,
            })

            if causal_effect > best_effect:
                best_effect = causal_effect
                best_layer = layer_idx

        if best_effect == -float("inf"):
            best_effect = 0.0

        return CausalMediationReport(
            probe_id=probe_id,
            prompt=prompt,
            expected_answer=expected_answer,
            expected_token_id=expected_id,
            subject_token_indices=subject_indices,
            corruption_noise_std=corruption_noise_std,
            layer_results=layer_results,
            best_layer=best_layer,
            best_causal_effect=float(best_effect),
            clean_prediction=clean_prediction,
            corrupt_prediction=corrupt_prediction,
            elapsed_seconds=time.time() - t0,
        )

    def _forward_with_hidden_states(
        self,
        token_ids: List[int],
        corruption: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Return hidden states before layer 0 and after every layer."""
        fp = self.forward_pass
        x = fp.token_embd[np.array(token_ids)].copy()
        if corruption is not None:
            if corruption.shape != x.shape:
                raise ValueError(f"corruption shape {corruption.shape} != hidden shape {x.shape}")
            x += corruption

        positions = np.arange(len(token_ids), dtype=np.float32)
        hidden_states = [x.copy()]
        for layer in fp.layers:
            x = fp._apply_layer(x, layer, positions)
            hidden_states.append(x.copy())
        return hidden_states

    def _forward_with_restoration(
        self,
        token_ids: List[int],
        restore_layer: int,
        restore_hidden: np.ndarray,
        corruption: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run corrupted inference and restore the clean post-layer state."""
        fp = self.forward_pass
        x = fp.token_embd[np.array(token_ids)].copy()
        if corruption is not None:
            if corruption.shape != x.shape:
                raise ValueError(f"corruption shape {corruption.shape} != hidden shape {x.shape}")
            x += corruption

        positions = np.arange(len(token_ids), dtype=np.float32)
        for i, layer in enumerate(fp.layers):
            x = fp._apply_layer(x, layer, positions)
            if i == restore_layer:
                if restore_hidden.shape != x.shape:
                    raise ValueError(
                        f"restore hidden shape {restore_hidden.shape} != current shape {x.shape}"
                    )
                x = restore_hidden.copy()

        return self._logits_from_hidden(x[-1])

    def _logits_from_hidden(self, hidden: np.ndarray) -> np.ndarray:
        fp = self.forward_pass
        return fp._rms_norm(hidden, fp.output_norm) @ fp.output.T

    def _logit_lens_from_hidden(self, hidden: np.ndarray) -> np.ndarray:
        return self._logits_from_hidden(hidden)


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
    return obj
