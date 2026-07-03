"""
Cross-Model Fingerprint Comparator
==================================
Loads two (or more) attribution reports and computes similarity metrics
to detect shared training data lineage.

Metrics:
  1. Fingerprint hash match (binary: same/different)
  2. Top-neuron overlap (Jaccard on (layer, neuron_index) pairs in global top-K)
  3. Top-neuron key-vector cosine similarity (only meaningful for same-arch models)
  4. Top activating token set overlap (Jaccard on token strings)
  5. Concept mastery correlation (Pearson on per-domain pass rates)
  6. Behavioral profile similarity (refusal rate, persona)
  7. Architecture fingerprint distance (a "lineage score" from 0-1)

Outputs:
  - Similarity matrix (for N models)
  - Pairwise comparison detail
  - Lineage hypothesis (e.g. "model A appears to be a fine-tune of model B")
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

import numpy as np


@dataclass
class ModelSummary:
    """Summary of one model for comparison."""
    name: str
    arch: str
    n_params: int
    n_layers: int
    fingerprint: str
    fingerprint_brief: Dict[str, Any]
    concept_mastery: Dict[str, float]    # domain -> pass_rate
    behavioral: Dict[str, Any]
    top_neurons: List[Dict[str, Any]]    # global top neurons with layer/neuron_idx/tokens
    top_activating_tokens: List[str]     # unique tokens from top neurons


@dataclass
class PairwiseComparison:
    model_a: str
    model_b: str
    fingerprint_match: bool
    same_arch: bool
    top_neuron_jaccard: float              # Jaccard on (layer, neuron) pairs
    top_token_jaccard: float               # Jaccard on top activating token strings
    concept_mastery_correlation: float     # Pearson
    behavioral_similarity: float           # 0-1
    lineage_score: float                   # 0-1, higher = more likely related
    lineage_hypothesis: str


@dataclass
class ComparisonReport:
    n_models: int
    models: List[Dict[str, Any]]
    pairwise: List[Dict[str, Any]]
    similarity_matrix: List[List[float]]
    stats: Dict[str, Any] = field(default_factory=dict)


class FingerprintComparator:
    """Compare multiple attribution reports."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load_report(self, path: str) -> ModelSummary:
        """Load an attribution JSON report and extract a ModelSummary."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        md = data.get("metadata", {}) or {}
        attr = data.get("attribution", {}) or {}
        mlp = attr.get("mlp_analysis", {}) or {}
        brief = attr.get("fingerprint_brief", {}) or {}

        # Name
        name = md.get("name") or data.get("gguf_filename") or os.path.basename(path)

        # Concept mastery
        concepts = data.get("concepts", []) or []
        concept_mastery = {c.get("domain", "?"): float(c.get("pass_rate", 0)) for c in concepts}

        # Behavioral
        bp = data.get("behavioral_profile", {}) or {}
        behavioral = {
            "refusal_rate": float(bp.get("refusal_rate", 0)),
            "n_refusal_probes": int(bp.get("n_refusal_probes", 0)),
            "n_refused": int(bp.get("n_refused", 0)),
        }

        # Top neurons
        global_top = mlp.get("global_top_neurons", []) or []
        top_neurons = []
        top_tokens_set = set()
        for n in global_top:
            neuron_dict = {
                "layer": n.get("layer"),
                "neuron_index": n.get("neuron_index"),
                "memory_strength": n.get("memory_strength", 0),
                "top_activating_tokens": [],
            }
            for t in n.get("top_activating_tokens", []) or []:
                if isinstance(t, list) and len(t) >= 1:
                    tok = str(t[0])
                    neuron_dict["top_activating_tokens"].append(tok)
                    top_tokens_set.add(tok)
            top_neurons.append(neuron_dict)

        # N params (from weight inspection)
        wi = data.get("weight_inspection", {}) or {}
        n_params = int(wi.get("total_parameters", 0))

        return ModelSummary(
            name=name,
            arch=md.get("arch", "unknown"),
            n_params=n_params,
            n_layers=int(md.get("block_count", 0)),
            fingerprint=attr.get("knowledge_fingerprint", ""),
            fingerprint_brief=brief,
            concept_mastery=concept_mastery,
            behavioral=behavioral,
            top_neurons=top_neurons,
            top_activating_tokens=list(top_tokens_set),
        )

    # ------------------------------------------------------------------ #
    # Pairwise comparison
    # ------------------------------------------------------------------ #
    def compare_pair(self, a: ModelSummary, b: ModelSummary) -> PairwiseComparison:
        # 1. Fingerprint hash match
        fp_match = (a.fingerprint == b.fingerprint) and (a.fingerprint != "")

        # 2. Same arch?
        same_arch = (a.arch == b.arch) and (a.n_layers == b.n_layers)

        # 3. Top neuron Jaccard (on (layer, neuron_index) pairs)
        set_a = {(n["layer"], n["neuron_index"]) for n in a.top_neurons if n["layer"] is not None}
        set_b = {(n["layer"], n["neuron_index"]) for n in b.top_neurons if n["layer"] is not None}
        neuron_jaccard = self._jaccard(set_a, set_b)

        # 4. Top token Jaccard
        tok_a = set(a.top_activating_tokens)
        tok_b = set(b.top_activating_tokens)
        token_jaccard = self._jaccard(tok_a, tok_b)

        # 5. Concept mastery correlation
        common_domains = set(a.concept_mastery.keys()) & set(b.concept_mastery.keys())
        if len(common_domains) >= 2:
            rates_a = [a.concept_mastery[d] for d in common_domains]
            rates_b = [b.concept_mastery[d] for d in common_domains]
            concept_corr = self._pearson(rates_a, rates_b)
        else:
            concept_corr = 0.0

        # 6. Behavioral similarity
        behav_sim = 1.0 - abs(a.behavioral.get("refusal_rate", 0) - b.behavioral.get("refusal_rate", 0))

        # 7. Lineage score (weighted combination)
        # Higher = more likely related
        # Same arch is a strong prerequisite
        if not same_arch:
            lineage_score = 0.0
            hypothesis = f"Different architectures ({a.arch} vs {b.arch}) — likely unrelated"
        else:
            # Weighted combination
            lineage_score = (
                0.30 * neuron_jaccard +
                0.25 * token_jaccard +
                0.25 * max(0, concept_corr) +
                0.10 * behav_sim +
                0.10 * (1.0 if fp_match else 0.0)
            )
            if fp_match:
                hypothesis = f"**IDENTICAL FINGERPRINT** — {a.name} and {b.name} are very likely the SAME model (or exact fine-tunes of each other)"
            elif lineage_score > 0.7:
                hypothesis = f"**STRONG lineage** — {a.name} appears to be a fine-tune or quantization of {b.name} (score: {lineage_score:.2f})"
            elif lineage_score > 0.4:
                hypothesis = f"**MODERATE similarity** — {a.name} and {b.name} may share training data lineage (score: {lineage_score:.2f})"
            elif lineage_score > 0.2:
                hypothesis = f"**WEAK similarity** — same architecture but different training (score: {lineage_score:.2f})"
            else:
                hypothesis = f"**No clear lineage** — different models despite same architecture (score: {lineage_score:.2f})"

        return PairwiseComparison(
            model_a=a.name, model_b=b.name,
            fingerprint_match=fp_match, same_arch=same_arch,
            top_neuron_jaccard=neuron_jaccard,
            top_token_jaccard=token_jaccard,
            concept_mastery_correlation=concept_corr,
            behavioral_similarity=behav_sim,
            lineage_score=lineage_score,
            lineage_hypothesis=hypothesis,
        )

    # ------------------------------------------------------------------ #
    # N-way comparison
    # ------------------------------------------------------------------ #
    def compare_all(self, reports: List[str]) -> ComparisonReport:
        """Compare N attribution reports. Returns pairwise + similarity matrix."""
        models = [self.load_report(p) for p in reports]
        n = len(models)

        pairwise: List[Dict[str, Any]] = []
        sim_matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                cmp = self.compare_pair(models[i], models[j])
                pairwise.append(_to_jsonable(asdict(cmp)))
                sim_matrix[i][j] = cmp.lineage_score
                sim_matrix[j][i] = cmp.lineage_score
            sim_matrix[i][i] = 1.0

        return ComparisonReport(
            n_models=n,
            models=[_to_jsonable(asdict(m)) for m in models],
            pairwise=pairwise,
            similarity_matrix=sim_matrix,
            stats={
                "n_pairwise_comparisons": len(pairwise),
                "n_identical_fingerprints": sum(1 for p in pairwise if p.get("fingerprint_match")),
                "n_strong_lineage": sum(1 for p in pairwise if p.get("lineage_score", 0) > 0.7),
                "n_moderate_lineage": sum(1 for p in pairwise if 0.4 < p.get("lineage_score", 0) <= 0.7),
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    @staticmethod
    def _pearson(x: List[float], y: List[float]) -> float:
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        if x_arr.std() == 0 or y_arr.std() == 0:
            return 0.0
        return float(np.corrcoef(x_arr, y_arr)[0, 1])


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
