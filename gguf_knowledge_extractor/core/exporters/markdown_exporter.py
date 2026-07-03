"""Markdown report exporter — human-readable summary of the extracted knowledge."""
from __future__ import annotations

from pathlib import Path
from typing import Union

from ..extractor import KnowledgeReport


def export_markdown(report: KnowledgeReport, out_path: Union[str, Path]) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    md = report.metadata if isinstance(report.metadata, dict) else {}

    # ------------------------- Title ------------------------- #
    lines.append(f"# Knowledge Extraction Report")
    lines.append("")
    lines.append(f"**Model:** `{md.get('name') or report.gguf_filename}`  ")
    lines.append(f"**File:** `{report.gguf_path}`  ")
    lines.append(f"**Architecture:** `{md.get('arch', 'unknown')}`  ")
    lines.append(f"**Quantization:** `{md.get('quantization', 'unknown')}`  ")
    lines.append(f"**Extracted:** {report.extraction_timestamp}  ")
    lines.append(f"**Extractor version:** v{report.extractor_version}  ")
    lines.append(f"**Backend used:** `{report.stats.get('backend_used', 'n/a')}`  ")
    lines.append("")

    # ------------------------- TOC ------------------------- #
    lines.append("## Table of Contents")
    lines.append("")
    lines.append("1. [Model Metadata](#1-model-metadata)")
    lines.append("2. [Weight Inspection](#2-weight-inspection)")
    lines.append("3. [Pack Summaries](#3-pack-summaries)")
    lines.append("4. [Extracted Facts](#4-extracted-facts)")
    lines.append("5. [Concept Mastery](#5-concept-mastery)")
    lines.append("6. [Behavioral Profile](#6-behavioral-profile)")
    lines.append("7. [Calibration & Hallucination](#7-calibration--hallucination)")
    lines.append("8. [Knowledge Attribution (v2)](#8-knowledge-attribution-v2)")
    lines.append("9. [Raw Probe Results](#9-raw-probe-results)")
    lines.append("")

    # ------------------------- 1. Metadata ------------------------- #
    lines.append("## 1. Model Metadata")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    fields_to_show = [
        ("name", "Name"), ("arch", "Architecture"), ("quantization", "Quantization"),
        ("vocab_size", "Vocab size"), ("context_length", "Context length"),
        ("embedding_length", "Embedding dim"), ("block_count", "Layers"),
        ("feed_forward_length", "FFN dim"), ("attention_head_count", "Heads"),
        ("attention_head_count_kv", "KV heads"), ("rope_freq_base", "RoPE base"),
        ("tokenizer_model", "Tokenizer"), ("license", "License"),
        ("author", "Author"), ("organization", "Organization"),
        ("file_size_bytes", "File size (bytes)"),
    ]
    for k, label in fields_to_show:
        v = md.get(k)
        if v is None or v == "":
            continue
        lines.append(f"| {label} | `{v}` |")
    lines.append("")

    # ------------------------- 2. Weights ------------------------- #
    lines.append("## 2. Weight Inspection")
    lines.append("")
    wi = report.weight_inspection if isinstance(report.weight_inspection, dict) else {}
    if not wi:
        lines.append("_Weight inspection was not run._")
        lines.append("")
    else:
        cap = wi.get("capacity_estimate", {})
        lines.append("### Capacity Estimate")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Total parameters | {wi.get('total_parameters', 0):,} |")
        lines.append(f"| Total tensors | {wi.get('n_tensors', 0)} |")
        lines.append(f"| Layers detected | {wi.get('n_layers', 0)} |")
        lines.append(f"| Total size (bytes) | {wi.get('total_size_bytes', 0):,} |")
        lines.append(f"| Est. knowledge bits | {cap.get('estimated_knowledge_bits', 0):,} |")
        lines.append(f"| Est. knowledge tokens | {cap.get('estimated_knowledge_tokens', 0):,} |")
        lines.append(f"| Est. knowledge words (human) | {cap.get('estimated_knowledge_human_words', 0):,} |")
        lines.append(f"| Per-layer capacity (tokens) | {cap.get('per_layer_capacity_tokens', 0):,} |")
        lines.append("")

        lines.append("### Quantization Summary")
        lines.append("")
        lines.append("| Type | Count |")
        lines.append("|---|---|")
        for dt, n in wi.get("quantization_summary", {}).items():
            lines.append(f"| {dt} | {n} |")
        lines.append("")

        emb = wi.get("embedding_analysis")
        if emb:
            lines.append("### Embedding Analysis")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            lines.append(f"| Vocab size | {emb.get('vocab_size', 0):,} |")
            lines.append(f"| Embedding dim | {emb.get('embed_dim', 0)} |")
            lines.append(f"| Centroid norm | {emb.get('centroid_norm', 0):.4f} |")
            lines.append(f"| Mean pairwise cosine | {emb.get('mean_pairwise_cosine', 0):.4f} |")
            lines.append(f"| Cluster count estimate | {emb.get('cluster_count_estimate', 0):,} |")
            lines.append("")
            lines.append("#### Most Outlier Tokens (furthest from embedding centroid)")
            lines.append("")
            lines.append("| Token | Distance |")
            lines.append("|---|---|")
            for tok, dist in emb.get("top_outlier_tokens", [])[:10]:
                tok_safe = str(tok).replace("|", "\\|").replace("\n", " ")[:60]
                lines.append(f"| `{tok_safe}` | {dist:.4f} |")
            lines.append("")

        outliers = wi.get("outlier_tensors", [])
        if outliers:
            lines.append(f"### Outlier Tensors ({len(outliers)})")
            lines.append("")
            lines.append("These tensors have unusually high weight standard deviation, "
                         "often corresponding to specialized processing layers.")
            lines.append("")
            for t in outliers[:20]:
                lines.append(f"- `{t}`")
            if len(outliers) > 20:
                lines.append(f"- ... and {len(outliers)-20} more")
            lines.append("")

    # ------------------------- 3. Pack summaries ------------------------- #
    lines.append("## 3. Pack Summaries")
    lines.append("")
    if not report.pack_summaries:
        lines.append("_No probe packs were run._")
        lines.append("")
    else:
        lines.append("| Pack | Category | Domain | Probes | Passed | Failed | Skipped | Errors | Pass Rate | Avg latency (s) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in report.pack_summaries:
            lines.append(
                f"| {s.get('pack_name','')} | {s.get('category','')} | {s.get('domain') or ''} "
                f"| {s.get('n_probes',0)} | {s.get('n_passed',0)} | {s.get('n_failed',0)} "
                f"| {s.get('n_skipped',0)} | {s.get('n_errors',0)} "
                f"| {s.get('pass_rate',0)*100:.1f}% | {s.get('avg_latency_seconds',0):.2f} |"
            )
        lines.append("")

    # ------------------------- 4. Facts ------------------------- #
    lines.append("## 4. Extracted Facts")
    lines.append("")
    if not report.facts:
        lines.append("_No fact probes were run._")
        lines.append("")
    else:
        # Group by domain
        by_domain: dict[str, list] = {}
        for f in report.facts:
            by_domain.setdefault(f.get("domain", "general"), []).append(f)
        for domain, fs in by_domain.items():
            lines.append(f"### {domain.title()} ({len(fs)} facts)")
            lines.append("")
            lines.append("| ID | Prompt | Expected | Model Answer | Correct |")
            lines.append("|---|---|---|---|---|")
            for f in fs:
                p = (f.get("prompt","") or "").replace("|","\\|").replace("\n"," ")[:80]
                e = (f.get("expected_answer","") or "").replace("|","\\|").replace("\n"," ")[:40]
                m = (f.get("model_answer","") or "").replace("|","\\|").replace("\n"," ")[:80]
                lines.append(f"| {f.get('id','')} | {p} | {e} | {m} | {'yes' if f.get('correct') else 'no'} |")
            lines.append("")

    # ------------------------- 5. Concepts ------------------------- #
    lines.append("## 5. Concept Mastery")
    lines.append("")
    if not report.concepts:
        lines.append("_No concept probes were run._")
        lines.append("")
    else:
        lines.append("| Domain | Probes | Passed | Pass Rate | Mastery |")
        lines.append("|---|---|---|---|---|")
        for c in report.concepts:
            lines.append(
                f"| {c.get('domain','')} | {c.get('n_probes',0)} | {c.get('n_passed',0)} "
                f"| {c.get('pass_rate',0)*100:.1f}% | **{c.get('mastery_level','unknown')}** |"
            )
        lines.append("")
        for c in report.concepts:
            lines.append(f"### {c.get('domain','').title()} — {c.get('mastery_level','unknown')}")
            lines.append("")
            for d in c.get("details", [])[:10]:
                passed_mark = "yes" if d.get("passed") else "no"
                resp = (d.get("response","") or "").replace("\n"," ")[:120]
                lines.append(f"- **{d.get('id','')}** (passed: {passed_mark}) — `{resp}`")
            if len(c.get("details", [])) > 10:
                lines.append(f"- ... and {len(c.get('details', []))-10} more probes")
            lines.append("")

    # ------------------------- 6. Behavioral ------------------------- #
    lines.append("## 6. Behavioral Profile")
    lines.append("")
    bp = report.behavioral_profile
    if not bp:
        lines.append("_No behavioral probes were run._")
    else:
        lines.append(f"- **Refusal rate:** {bp.get('refusal_rate',0)*100:.1f}% "
                     f"({bp.get('n_refused',0)}/{bp.get('n_refusal_probes',0)} refusal probes)")
        if bp.get("detected_persona_excerpt"):
            lines.append(f"- **Detected persona excerpt:** {bp['detected_persona_excerpt']}")
        lines.append("")
        if bp.get("refusal_details"):
            lines.append("### Refusal Probe Details")
            lines.append("")
            lines.append("| Probe ID | Refused | Response Excerpt |")
            lines.append("|---|---|---|")
            for r in bp["refusal_details"]:
                ex = (r.get("response_excerpt","") or "").replace("|","\\|").replace("\n"," ")[:100]
                lines.append(f"| {r.get('probe_id','')} | {'yes' if r.get('refused') else 'no'} | {ex} |")
            lines.append("")
        if bp.get("bias_findings"):
            lines.append("### Bias Findings (responses to bias probes — for human review)")
            lines.append("")
            lines.append("| Probe ID | Tags | Response Excerpt |")
            lines.append("|---|---|---|")
            for b in bp["bias_findings"]:
                ex = (b.get("response_excerpt","") or "").replace("|","\\|").replace("\n"," ")[:100]
                tags = ",".join(b.get("tags",[]) or [])
                lines.append(f"| {b.get('probe_id','')} | {tags} | {ex} |")
            lines.append("")

    # ------------------------- 7. Calibration ------------------------- #
    lines.append("## 7. Calibration & Hallucination")
    lines.append("")
    cal = report.calibration
    if not cal:
        lines.append("_No calibration probes were run._")
    else:
        rate = cal.get("hallucination_acknowledgment_rate")
        if rate is not None:
            lines.append(f"- **Hallucination acknowledgment rate:** {rate*100:.1f}% "
                         f"({cal.get('n_acknowledged_uncertainty',0)}/{cal.get('n_hallucination_probes',0)})")
        math_acc = cal.get("math_accuracy")
        if math_acc is not None:
            lines.append(f"- **Math accuracy:** {math_acc*100:.1f}% "
                         f"({cal.get('n_math_correct',0)}/{cal.get('n_math_probes',0)})")
        lines.append("")
        if cal.get("hallucination_details"):
            lines.append("### Hallucination Probe Details")
            lines.append("")
            lines.append("| Probe ID | Acknowledged Uncertainty | Response Excerpt |")
            lines.append("|---|---|---|")
            for h in cal["hallucination_details"]:
                ex = (h.get("response_excerpt","") or "").replace("|","\\|").replace("\n"," ")[:100]
                lines.append(f"| {h.get('probe_id','')} | {'yes' if h.get('acknowledged_uncertainty') else 'no'} | {ex} |")
            lines.append("")

    # ------------------------- 8. Attribution (v2) ------------------------- #
    lines.append("## 8. Knowledge Attribution (v2)")
    lines.append("")
    attr = report.attribution if isinstance(report.attribution, dict) else {}
    if not attr:
        lines.append("_Attribution was not run. Use `--attribution` on the CLI or check the box in the web UI to enable._")
        lines.append("")
    else:
        fp = attr.get("knowledge_fingerprint", "")
        brief = attr.get("fingerprint_brief", {}) or {}
        stats = attr.get("stats", {}) or {}
        lines.append(f"**Knowledge fingerprint:** `{fp}`")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Layers analyzed (MLP) | {stats.get('n_layers_analyzed_mlp', 0)} |")
        lines.append(f"| Layers analyzed (attention) | {stats.get('n_layers_analyzed_attn', 0)} |")
        lines.append(f"| Total attention heads | {stats.get('n_heads_total', 0)} |")
        lines.append(f"| Global top neurons extracted | {stats.get('n_global_top_neurons', 0)} |")
        lines.append(f"| Facts attributed | {stats.get('n_facts_attributed', 0)} |")
        lines.append(f"| Attribution elapsed | {stats.get('elapsed_seconds', 0):.2f}s |")
        lines.append(f"| Strongest layer (most memory strength) | {brief.get('strongest_layer', '?')} |")
        lines.append(f"| Most concentrated layer | {brief.get('most_concentrated_layer', '?')} |")
        lines.append("")

        # Top 5 neurons
        top5 = brief.get("top_5_neurons", []) or []
        if top5:
            lines.append("### Top 5 Memory Neurons (across all layers)")
            lines.append("")
            lines.append("| Layer | Neuron | Strength | Top Activating Token |")
            lines.append("|---|---|---|---|")
            for n in top5:
                lines.append(f"| {n.get('layer','')} | {n.get('neuron','')} | {n.get('strength',0):.4f} | `{n.get('top_token','')}` |")
            lines.append("")

        # Per-layer memory summary
        mlp = attr.get("mlp_analysis", {}) or {}
        layers = mlp.get("layers", []) or []
        if layers:
            lines.append("### Per-Layer MLP Memory Map")
            lines.append("")
            lines.append("| Layer | Hidden dim | Gated | Neurons | Layer strength | Concentration | Top neuron's top token |")
            lines.append("|---|---|---|---|---|---|---|")
            for L in layers[:30]:
                top_neurons = L.get("top_neurons", []) or []
                top_token = ""
                if top_neurons:
                    activating = top_neurons[0].get("top_activating_tokens", []) or []
                    if activating:
                        top_token = activating[0][0] if isinstance(activating[0], list) else str(activating[0])
                lines.append(
                    f"| {L.get('layer','')} | {L.get('hidden_dim','')} | {'yes' if L.get('is_gated') else 'no'} "
                    f"| {L.get('n_neurons','')} | {L.get('layer_strength',0):.2f} "
                    f"| {L.get('concentration',0)*100:.1f}% | `{str(top_token)[:30]}` |"
                )
            if len(layers) > 30:
                lines.append(f"_... and {len(layers)-30} more layers (see JSON export)_")
            lines.append("")

        # Attention head analysis
        attn = attr.get("attention_analysis", {}) or {}
        top_copy = attn.get("top_copy_heads", []) or []
        top_ind = attn.get("top_induction_heads", []) or []
        if top_copy or top_ind:
            lines.append("### Attention Head Specialization")
            lines.append("")
            if top_copy:
                lines.append("**Top Copy Heads** (V·O closest to identity — likely token-copying heads):")
                lines.append("")
                lines.append("| Layer | Head | Copy Score | Specialization |")
                lines.append("|---|---|---|---|")
                for h in top_copy[:10]:
                    lines.append(f"| {h.get('layer','')} | {h.get('head_index','')} | {h.get('copy_score',0):.3f} | {h.get('specialization',0):.3f} |")
                lines.append("")
            if top_ind:
                lines.append("**Top Induction Heads** (low-rank Q·Q^T — likely induction/copy-pattern heads):")
                lines.append("")
                lines.append("| Layer | Head | Induction Score | Top Singular Value |")
                lines.append("|---|---|---|---|")
                for h in top_ind[:10]:
                    lines.append(f"| {h.get('layer','')} | {h.get('head_index','')} | {h.get('induction_score',0):.3f} | {h.get('top_singular_value',0):.2f} |")
                lines.append("")

        # Per-fact attribution
        fact_attr = attr.get("fact_attributions", []) or []
        if fact_attr:
            lines.append("### Per-Fact Attribution (weight-only heuristic)")
            lines.append("")
            lines.append("Each fact is attributed to the layer+neuron whose key vector is most strongly "
                         "activated by tokens in the prompt. Method = `weight_only` means inference was not used.")
            lines.append("")
            lines.append("| Probe | Attributed Layer | Attributed Neuron | Confidence | Method |")
            lines.append("|---|---|---|---|---|")
            for f in fact_attr[:50]:
                lines.append(
                    f"| {f.get('probe_id','')} | {f.get('attributed_layer','')} "
                    f"| {f.get('attributed_neuron','')} | {f.get('attribution_confidence',0):.3f} "
                    f"| {f.get('method','')} |"
                )
            lines.append("")

    # ------------------------- 9. Raw ------------------------- #
    lines.append("## 9. Raw Probe Results")
    lines.append("")
    lines.append(f"_Total probes run: {len(report.probe_results)}_")
    lines.append("")
    if report.probe_results:
        lines.append("| Pack | ID | Backend | Pass | Error | Response (excerpt) |")
        lines.append("|---|---|---|---|---|---|")
        for r in report.probe_results[:200]:
            ex = (r.get("response","") or "").replace("|","\\|").replace("\n"," ")[:80]
            err = (r.get("error") or "")[:30]
            lines.append(
                f"| {r.get('pack_name','')} | {r.get('probe_id','')} "
                f"| {r.get('backend','')} | {'yes' if r.get('passed') else 'no'} "
                f"| {err} | {ex} |"
            )
        if len(report.probe_results) > 200:
            lines.append(f"_... and {len(report.probe_results)-200} more rows (see JSON export for full data)_")
        lines.append("")

    # ------------------------- Footer ------------------------- #
    lines.append("---")
    lines.append("")
    lines.append(f"_Generated by GGUF Knowledge Extractor v{report.extractor_version} "
                 f"in {report.stats.get('total_elapsed_seconds',0):.1f}s._")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(out_path)
