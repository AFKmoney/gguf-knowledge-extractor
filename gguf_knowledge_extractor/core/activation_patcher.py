"""
Activation Patcher
===================
True causal mediation analysis — find which hidden states at which layers
are causally responsible for a model's output.

Method (from Meng et al., "Locating and Editing Factual Associations in GPT"):
  1. Run a clean forward pass on prompt P → get output O_clean
  2. Run a corrupted forward pass on a noisy version of P → get output O_corrupt
     (corruption = add noise to the subject token's embedding)
  3. For each layer L, restore the clean hidden state at layer L during the
     corrupted run → get output O_restore(L)
  4. The "causal effect" of layer L = how much restoring L recovers O_clean
     from O_corrupt. Layers with high effect are where the knowledge lives.

This is the gold-standard method for finding which layer stores a fact,
used by ROME and follow-up papers.

Requires the numpy forward pass (Llama-arch only).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gguf

from .forward_pass import NumpyLlamaForward


@dataclass
class PatchResult:
    """Result of patching a single layer."""
    layer: int
    # Logit lens at this layer for the clean run
    clean_top1_token: str
    clean_top1_prob: float
    # Logit lens at this layer for the corrupted run
    corrupt_top1_token: str
    corrupt_top1_prob: float
    # Logit lens at this layer when we restore the clean hidden state
    restored_top1_token: str
    restored_top1_prob: float
    # Causal effect: how much did restoring recover the clean prediction?
    # Measured as the probability of the clean answer token
    clean_answer_prob_clean: float      # P(answer) in clean run
    clean_answer_prob_corrupt: float   # P(answer) in corrupt run
    clean_answer_prob_restored: float  # P(answer) after restoring this layer
    causal_effect: float               # = restored - corrupt (how much we recovered)
    indirect_effect: float             # = clean - restored (how much is still missing)


@dataclass
class CausalMediationReport:
    """Full causal mediation analysis report for one fact."""
    probe_id: str
    prompt: str
    expected_answer: Optional[str]
    expected_token_id: Optional[int]
    # The subject token(s) that were corrupted
    subject_token_indices: List[int]
    corruption_noise_std: float
    # Per-layer results
    layer_results: List[Dict[str, Any]]
    # The layer with the highest causal effect (the "mediating layer")
    best_layer: Optional[int]
    best_causal_effect: float
    # Final prediction
    clean_prediction: str
    corrupt_prediction: str
    # Stats
    elapsed_seconds: float
    method: str = "causal_mediation"


class ActivationPatcher:
    """Causal mediation analysis via activation patching."""

    def __init__(self, reader: gguf.GGUFReader, fields: Dict[str, Any]):
        self.forward_pass = NumpyLlamaForward(reader, fields)
        self.fields = fields

    def is_available(self) -> bool:
        return self.forward_pass.is_available()

    # ------------------------------------------------------------------ #
    # Tokenization (same as causal_tracer)
    # ------------------------------------------------------------------ #
    def tokenize(self, text: str) -> List[int]:
        """Greedy longest-match tokenizer."""
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
    # Causal mediation analysis
    # ------------------------------------------------------------------ #
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

        Args:
            probe_id: identifier for this probe
            prompt: the fact prompt
            expected_answer: the expected answer text
            subject_indices: which token indices in the prompt are the "subject"
                             (the entity whose knowledge we're testing).
                             If None, we use the last 2-3 tokens.
            corruption_noise_std: std of Gaussian noise added to subject embeddings
            seed: random seed for reproducibility
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

        # Tokenize
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

        # Find expected token
        expected_id = None
        if expected_answer:
            expected_id = self.forward_pass.find_token_for_text(expected_answer)

        # Determine subject indices (default: last 2-3 tokens)
        if subject_indices is None:
            n_subj = min(3, len(token_ids) - 1)
            subject_indices = list(range(len(token_ids) - n_subj, len(token_ids)))

        # 1. Clean forward pass — collect hidden states at each layer
        clean_hidden_states = self._forward_with_hidden_states(token_ids)
        clean_logits = self.forward_pass.forward_with_lens(token_ids)
        clean_prediction = clean_logits.predicted_token_text

        # Clean answer probability (at final logits)
        clean_answer_prob = 0.0
        if expected_id is not None:
            logits = clean_logits.final_logits
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            clean_answer_prob = float(probs[expected_id])

        # 2. Corrupted forward pass — add noise to subject embeddings
        rng = np.random.RandomState(seed)
        corrupted_token_ids = list(token_ids)
        # We need to modify the embedding, not the token ID
        # So we do a custom forward pass with noise added to the embedding

        corrupt_hidden_states = self._forward_with_hidden_states(
            token_ids, corrupt_indices=subject_indices, noise_std=corruption_noise_std, rng=rng
        )
        corrupt_logits_result = self._forward_with_lens_custom(
            token_ids, corrupt_indices=subject_indices, noise_std=corruption_noise_std, rng=rng
        )
        corrupt_prediction = corrupt_logits_result.predicted_token_text

        corrupt_answer_prob = 0.0
        if expected_id is not None:
            logits = corrupt_logits_result.final_logits
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            corrupt_answer_prob = float(probs[expected_id])

        # 3. For each layer: restore the clean hidden state and re-run from that layer
        layer_results: List[Dict[str, Any]] = []
        best_layer = None
        best_effect = -1.0

        for layer_idx in range(len(self.forward_pass.layers)):
            # Run corrupted pass but restore clean hidden state at layer_idx
            restored_logits = self._forward_with_restoration(
                token_ids, layer_idx, clean_hidden_states[layer_idx + 1],
                corrupt_indices=subject_indices, noise_std=corruption_noise_std, rng=rng,
            )

            restored_prediction = self.forward_pass._token_text(int(np.argmax(restored_logits)))

            restored_answer_prob = 0.0
            if expected_id is not None:
                logits = restored_logits
                probs = np.exp(logits - logits.max())
                probs = probs / probs.sum()
                restored_answer_prob = float(probs[expected_id])

            # Causal effect = how much did restoring recover the clean answer prob
            causal_effect = restored_answer_prob - corrupt_answer_prob
            indirect_effect = clean_answer_prob - restored_answer_prob

            # Logit lens at this layer for each condition
            clean_lens = self._logit_lens_from_hidden(clean_hidden_states[layer_idx + 1][-1])
            corrupt_lens = self._logit_lens_from_hidden(corrupt_hidden_states[layer_idx + 1][-1])
            restored_lens = restored_logits  # already computed via lm_head

            clean_top1 = int(np.argmax(clean_lens))
            corrupt_top1 = int(np.argmax(corrupt_lens))
            restored_top1 = int(np.argmax(restored_lens))

            def _softmax_prob(logits, idx):
                m = logits.max()
                exp = np.exp(logits - m)
                return float(exp[idx] / exp.sum())

            layer_results.append({
                "layer": layer_idx,
                "clean_top1_token": self.forward_pass._token_text(clean_top1),
                "clean_top1_prob": _softmax_prob(clean_lens, clean_top1),
                "corrupt_top1_token": self.forward_pass._token_text(corrupt_top1),
                "corrupt_top1_prob": _softmax_prob(corrupt_lens, corrupt_top1),
                "restored_top1_token": self.forward_pass._token_text(restored_top1),
                "restored_top1_prob": _softmax_prob(restored_lens, restored_top1),
                "clean_answer_prob_clean": clean_answer_prob,
                "clean_answer_prob_corrupt": corrupt_answer_prob,
                "clean_answer_prob_restored": restored_answer_prob,
                "causal_effect": causal_effect,
                "indirect_effect": indirect_effect,
            })

            if causal_effect > best_effect:
                best_effect = causal_effect
                best_layer = layer_idx

        return CausalMediationReport(
            probe_id=probe_id,
            prompt=prompt,
            expected_answer=expected_answer,
            expected_token_id=expected_id,
            subject_token_indices=subject_indices,
            corruption_noise_std=corruption_noise_std,
            layer_results=layer_results,
            best_layer=best_layer,
            best_causal_effect=best_effect,
            clean_prediction=clean_prediction,
            corrupt_prediction=corrupt_prediction,
            elapsed_seconds=time.time() - t0,
        )

    # ------------------------------------------------------------------ #
    # Forward pass variants
    # ------------------------------------------------------------------ #
    def _forward_with_hidden_states(
        self,
        token_ids: List[int],
        corrupt_indices: Optional[List[int]] = None,
        noise_std: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[np.ndarray]:
        """Run forward pass and return hidden states at each layer.

        Returns: [hidden_before_layer_0, hidden_after_layer_0, hidden_after_layer_1, ...]
        Length = n_layers + 1
        """
        fp = self.forward_pass
        cfg = fp.config
        seq_len = len(token_ids)

        # Embedding
        x = fp.token_embd[np.array(token_ids)].copy()  # [seq, dim]

        # Corrupt subject tokens
        if corrupt_indices and noise_std > 0:
            if rng is None:
                rng = np.random.RandomState(42)
            for idx in corrupt_indices:
                if 0 <= idx < seq_len:
                    noise = rng.randn(cfg.dim).astype(np.float32) * noise_std
                    x[idx] += noise

        positions = np.arange(seq_len, dtype=np.float32)

        hidden_states = [x.copy()]  # before layer 0
        for layer in fp.layers:
            x = fp._apply_layer(x, layer, positions)
            hidden_states.append(x.copy())

        return hidden_states

    def _forward_with_lens_custom(
        self,
        token_ids: List[int],
        corrupt_indices: Optional[List[int]] = None,
        noise_std: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
    ):
        """Forward pass with corruption, returning a LogitLensResult-like object."""
        from .forward_pass import LogitLensResult

        fp = self.forward_pass
        hidden_states = self._forward_with_hidden_states(token_ids, corrupt_indices, noise_std, rng)

        # Final logits
        final_hidden = hidden_states[-1][-1]  # last token, last layer
        final_normed = fp._rms_norm(final_hidden, fp.output_norm)
        final_logits = final_normed @ fp.output.T

        predicted_id = int(np.argmax(final_logits))
        predicted_text = fp._token_text(predicted_id)
        top_k = fp._top_k(final_logits, k=10)

        return LogitLensResult(
            token_ids=token_ids,
            tokens_text=[fp._token_text(i) for i in token_ids],
            per_layer_logits=[],  # not needed here
            final_logits=final_logits,
            predicted_token_id=predicted_id,
            predicted_token_text=predicted_text,
            top_k_tokens=top_k,
        )

    def _forward_with_restoration(
        self,
        token_ids: List[int],
        restore_layer: int,
        restore_hidden: np.ndarray,
        corrupt_indices: Optional[List[int]] = None,
        noise_std: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Run corrupted forward pass but restore the clean hidden state at restore_layer.

        Returns: final logits (at the last position)
        """
        fp = self.forward_pass
        cfg = fp.config
        seq_len = len(token_ids)

        # Embedding
        x = fp.token_embd[np.array(token_ids)].copy()

        # Corrupt subject tokens
        if corrupt_indices and noise_std > 0:
            if rng is None:
                rng = np.random.RandomState(42)
            for idx in corrupt_indices:
                if 0 <= idx < seq_len:
                    noise = rng.randn(cfg.dim).astype(np.float32) * noise_std
                    x[idx] += noise

        positions = np.arange(seq_len, dtype=np.float32)

        # Run layers 0..restore_layer-1 with corrupted input
        for i, layer in enumerate(fp.layers):
            if i == restore_layer:
                # Restore the clean hidden state
                x = restore_hidden.copy()
            x = fp._apply_layer(x, layer, positions)

        # Final logits
        final_hidden = x[-1]
        final_normed = fp._rms_norm(final_hidden, fp.output_norm)
        return final_normed @ fp.output.T

    def _logit_lens_from_hidden(self, hidden: np.ndarray) -> np.ndarray:
        """Project a hidden state through final norm + lm_head."""
        fp = self.forward_pass
        normed = fp._rms_norm(hidden, fp.output_norm)
        return normed @ fp.output.T


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
