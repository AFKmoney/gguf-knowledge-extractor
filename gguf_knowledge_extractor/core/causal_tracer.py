"""
Causal Tracer (Logit Lens)
==========================
For each fact probe, runs a forward pass and identifies the EARLIEST layer
where the expected answer token appears in the top-K predictions.

This is "true" causal tracing — we don't just heuristically match prompt
tokens to neuron keys (as in v2's weight_only method); we actually run the
model's forward pass in numpy and project hidden states through the lm_head
at every layer.

Two modes:
  1. Logit lens: at each layer, project the hidden state through the final
     norm + lm_head to get "early exit" predictions. The earliest layer
     where the answer appears in top-K = "the layer that knows this fact".
  2. (Future) Causal mediation analysis: corrupt the subject embedding,
     restore hidden states at each layer, and find the layer whose
     restoration most recovers the answer. [Not implemented — would require
     modifying the forward pass to support intervention.]

Requires the numpy forward pass to be available (i.e. the GGUF must be
Llama-architecture and have all standard tensors).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gguf

from .forward_pass import NumpyLlamaForward, LogitLensResult
from .mlp_analyzer import MLPAnalyzer


@dataclass
class FactTrace:
    """Causal trace result for one fact probe."""
    probe_id: str
    prompt: str
    expected: Optional[str]
    expected_token_id: Optional[int]
    predicted_token_id: int
    predicted_token_text: str
    correct: bool
    # Per-layer: did the expected token appear in top-K?
    layer_first_correct: Optional[int]      # earliest layer where expected is in top-K
    layer_first_predicted: Optional[int]    # earliest layer where model's final prediction is in top-K
    per_layer_top1: List[Tuple[int, str, float]]   # (layer_idx, top1_token_text, prob)
    per_layer_expected_rank: List[Tuple[int, int, float]]  # (layer_idx, rank_of_expected, prob)
    # Attribution: which neurons in layer_first_correct are most activated?
    attributed_neuron: Optional[int]
    attributed_neuron_strength: float
    method: str


@dataclass
class CausalTraceReport:
    n_facts_traced: int
    n_correct: int
    n_with_logit_lens: int       # how many had a forward pass available
    avg_first_correct_layer: Optional[float]
    traces: List[Dict[str, Any]]
    stats: Dict[str, Any] = field(default_factory=dict)


class CausalTracer:
    """Runs logit-lens causal tracing on fact probes."""

    def __init__(
        self,
        reader: gguf.GGUFReader,
        fields: Dict[str, Any],
        top_k: int = 10,
    ):
        self.reader = reader
        self.fields = fields
        self.top_k = top_k
        self.forward_pass = NumpyLlamaForward(reader, fields)
        # Also set up MLP analyzer for per-fact neuron attribution
        tokens = []
        tok_field = fields.get("tokenizer.ggml.tokens")
        if isinstance(tok_field, list):
            tokens = [str(t) for t in tok_field]
        self.mlp_analyzer = MLPAnalyzer(reader, tokens=tokens, top_k_per_layer=20, top_k_tokens=10)

    def is_available(self) -> bool:
        return self.forward_pass.is_available()

    # ------------------------------------------------------------------ #
    # Tokenization (simple, GGUF-vocab-based)
    # ------------------------------------------------------------------ #
    def tokenize(self, text: str) -> List[int]:
        """Greedy longest-match tokenizer using the GGUF vocab.

        This is a fallback when no real tokenizer is available. For real
        BPE/SPM tokenization, you'd want to use llama-cpp-python or
        sentencepiece. For our purposes (logit lens on short prompts), this
        works well enough.
        """
        vocab = self.forward_pass.tokens_vocab
        if not vocab:
            return []

        # Build a lookup of vocab tokens by length (descending)
        # Limit to tokens of reasonable length to keep this fast
        vocab_by_len: Dict[int, List[Tuple[str, int]]] = {}
        for i, tok in enumerate(vocab):
            if not isinstance(tok, str):
                continue
            # Strip BPE prefix for matching
            clean = tok.replace("Ġ", " ").replace("▁", " ")
            if len(clean) <= 20:
                vocab_by_len.setdefault(len(clean), []).append((clean, i))

        sorted_lengths = sorted(vocab_by_len.keys(), reverse=True)

        # Greedy match
        token_ids: List[int] = []
        i = 0
        text_to_match = " " + text  # add leading space (BPE convention)
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
                # Skip one character (unknown)
                i += 1
        return token_ids

    # ------------------------------------------------------------------ #
    # Trace a single fact
    # ------------------------------------------------------------------ #
    def trace_fact(
        self,
        probe_id: str,
        prompt: str,
        expected: Optional[str] = None,
    ) -> FactTrace:
        if not self.is_available():
            return FactTrace(
                probe_id=probe_id, prompt=prompt, expected=expected,
                expected_token_id=None, predicted_token_id=-1,
                predicted_token_text="", correct=False,
                layer_first_correct=None, layer_first_predicted=None,
                per_layer_top1=[], per_layer_expected_rank=[],
                attributed_neuron=None, attributed_neuron_strength=0.0,
                method="forward_pass_unavailable",
            )

        # Tokenize prompt
        token_ids = self.tokenize(prompt)
        if len(token_ids) == 0:
            return FactTrace(
                probe_id=probe_id, prompt=prompt, expected=expected,
                expected_token_id=None, predicted_token_id=-1,
                predicted_token_text="", correct=False,
                layer_first_correct=None, layer_first_predicted=None,
                per_layer_top1=[], per_layer_expected_rank=[],
                attributed_neuron=None, attributed_neuron_strength=0.0,
                method="tokenization_failed",
            )

        # Run forward pass with logit lens
        result = self.forward_pass.forward_with_lens(token_ids)

        # Find expected token ID
        expected_id = None
        if expected:
            expected_id = self.forward_pass.find_token_for_text(expected)

        # Per-layer analysis
        per_layer_top1: List[Tuple[int, str, float]] = []
        per_layer_expected_rank: List[Tuple[int, int, float]] = []
        layer_first_correct: Optional[int] = None
        layer_first_predicted: Optional[int] = None

        predicted_id = result.predicted_token_id
        predicted_text = result.predicted_token_text

        for layer_idx, logits in result.per_layer_logits:
            # Top-1
            top1_id = int(np.argmax(logits))
            top1_text = self.forward_pass._token_text(top1_id)
            # Top-K
            top_k = self.forward_pass._top_k(logits, k=self.top_k)
            top_k_ids = [t[0] for t in top_k]
            per_layer_top1.append((layer_idx, top1_text, top_k[0][2]))

            # Expected rank
            if expected_id is not None:
                if expected_id in top_k_ids:
                    rank = top_k_ids.index(expected_id) + 1
                    prob = top_k[top_k_ids.index(expected_id)][2]
                    per_layer_expected_rank.append((layer_idx, rank, prob))
                    if layer_first_correct is None:
                        layer_first_correct = layer_idx
                else:
                    # Compute actual rank (could be very large)
                    sorted_idx = np.argsort(logits)[::-1]
                    rank_arr = np.where(sorted_idx == expected_id)[0]
                    rank = int(rank_arr[0]) + 1 if len(rank_arr) > 0 else -1
                    prob = float(np.exp(logits[expected_id] - logits.max()) /
                                 np.sum(np.exp(logits - logits.max())))
                    per_layer_expected_rank.append((layer_idx, rank, prob))

            # Predicted (final answer) rank
            if predicted_id in top_k_ids:
                if layer_first_predicted is None:
                    layer_first_predicted = layer_idx

        # Correct?
        correct = False
        if expected_id is not None:
            correct = (predicted_id == expected_id)
        elif expected:
            # Fall back to substring match on predicted text
            correct = expected.lower() in predicted_text.lower()

        # Attribute to neuron in layer_first_correct
        attributed_neuron = None
        attributed_neuron_strength = 0.0
        if layer_first_correct is not None:
            attributed_neuron, attributed_neuron_strength = self._attribute_to_neuron(
                token_ids, layer_first_correct, expected_id
            )

        return FactTrace(
            probe_id=probe_id, prompt=prompt, expected=expected,
            expected_token_id=expected_id,
            predicted_token_id=predicted_id,
            predicted_token_text=predicted_text,
            correct=correct,
            layer_first_correct=layer_first_correct,
            layer_first_predicted=layer_first_predicted,
            per_layer_top1=per_layer_top1,
            per_layer_expected_rank=per_layer_expected_rank,
            attributed_neuron=attributed_neuron,
            attributed_neuron_strength=attributed_neuron_strength,
            method="logit_lens",
        )

    def _attribute_to_neuron(
        self,
        token_ids: List[int],
        layer_idx: int,
        expected_token_id: Optional[int],
    ) -> Tuple[Optional[int], float]:
        """Find which neuron in layer_idx most strongly fires for the prompt.

        We compute the hidden state at layer_idx, then for each of the top
        neurons in that layer (from v2 MLP analysis), compute the activation
        dot(hidden, key_vector). Return the neuron with the strongest activation.
        """
        try:
            hidden = self.forward_pass.get_layer_hidden_state(token_ids, layer_idx)
            if hidden is None:
                return None, 0.0

            # Get top neurons for this layer
            mlp_report = self.mlp_analyzer.analyze()
            layer_analysis = None
            for L in mlp_report.layers:
                if L.layer == layer_idx:
                    layer_analysis = L
                    break
            if layer_analysis is None:
                return None, 0.0

            # For each top neuron, compute activation = hidden @ key_vector
            # (where key_vector = column i of W_gate or W_up)
            # Find the layer's tensors
            from .forward_pass import NumpyLlamaForward
            # Reuse the forward_pass's tensor loading
            fp = self.forward_pass
            key_source = None
            for prefix in [f"blk.{layer_idx}.", f"layers.{layer_idx}."]:
                gate_name = prefix + "ffn_gate.weight"
                up_name = prefix + "ffn_up.weight"
                if gate_name in fp.tensors:
                    key_source = fp.tensors[gate_name]
                    break
                elif up_name in fp.tensors:
                    key_source = fp.tensors[up_name]
                    break

            if key_source is None:
                return None, 0.0

            # Compute activation for each top neuron
            best_neuron = None
            best_activation = -1e30
            for neuron in layer_analysis.top_neurons:
                ni = neuron.neuron_index
                key_vec = key_source[ni]  # [embed_dim]
                activation = float(hidden @ key_vec)
                if activation > best_activation:
                    best_activation = activation
                    best_neuron = ni

            return best_neuron, best_activation
        except Exception:
            return None, 0.0

    # ------------------------------------------------------------------ #
    # Trace many facts
    # ------------------------------------------------------------------ #
    def trace_facts(self, facts: List[Dict[str, Any]]) -> CausalTraceReport:
        """Trace a list of fact dicts. Each should have: probe_id, prompt, expected."""
        t0 = time.time()
        traces: List[Dict[str, Any]] = []
        n_correct = 0
        n_with_lens = 0
        first_correct_layers: List[int] = []

        for i, f in enumerate(facts):
            probe_id = f.get("probe_id") or f.get("id") or f"fact_{i}"
            prompt = f.get("prompt", "")
            expected = f.get("expected") or f.get("expected_answer")

            trace = self.trace_fact(probe_id, prompt, expected)
            traces.append(_to_jsonable(asdict(trace)))

            if trace.method == "logit_lens":
                n_with_lens += 1
            if trace.correct:
                n_correct += 1
            if trace.layer_first_correct is not None:
                first_correct_layers.append(trace.layer_first_correct)

        avg_first = float(np.mean(first_correct_layers)) if first_correct_layers else None

        return CausalTraceReport(
            n_facts_traced=len(traces),
            n_correct=n_correct,
            n_with_logit_lens=n_with_lens,
            avg_first_correct_layer=avg_first,
            traces=traces,
            stats={
                "elapsed_seconds": time.time() - t0,
                "forward_pass_available": self.is_available(),
                "top_k": self.top_k,
                "n_layers": len(self.forward_pass.layers) if self.is_available() else 0,
            },
        )


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
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return str(obj)
    return obj
