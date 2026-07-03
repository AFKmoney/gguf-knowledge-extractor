"""
SQLite exporter
================
Persists the full extraction report into a relational SQLite database
with the following schema:

  - models           (one row per extracted model)
  - tensor_stats     (per-tensor statistics)
  - layer_stats      (per-layer statistics)
  - packs            (probe packs that were run)
  - probes           (probe definitions + results)
  - facts            (extracted fact triples)
  - concepts         (concept mastery summary)
  - behavioral       (behavioral profile per probe)
  - calibration      (calibration metrics)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Union

from ..extractor import KnowledgeReport


SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gguf_path TEXT,
    gguf_filename TEXT,
    name TEXT,
    arch TEXT,
    quantization TEXT,
    vocab_size INTEGER,
    context_length INTEGER,
    embedding_length INTEGER,
    block_count INTEGER,
    file_size_bytes INTEGER,
    extraction_timestamp TEXT,
    extractor_version TEXT,
    backend_used TEXT,
    total_elapsed_seconds REAL,
    n_probes INTEGER,
    n_probes_passed INTEGER,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS tensor_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id),
    name TEXT,
    dtype TEXT,
    shape_json TEXT,
    n_elements INTEGER,
    size_bytes INTEGER,
    mean REAL,
    std REAL,
    min REAL,
    max REAL,
    abs_mean REAL,
    norm REAL,
    sparsity REAL,
    n_outliers INTEGER
);

CREATE TABLE IF NOT EXISTS layer_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id),
    layer_index INTEGER,
    attention_norm REAL,
    mlp_up_norm REAL,
    mlp_down_norm REAL,
    mlp_gate_norm REAL
);

CREATE TABLE IF NOT EXISTS packs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id),
    pack_name TEXT,
    category TEXT,
    domain TEXT,
    n_probes INTEGER,
    n_passed INTEGER,
    n_failed INTEGER,
    n_skipped INTEGER,
    n_errors INTEGER,
    pass_rate REAL,
    avg_latency_seconds REAL
);

CREATE TABLE IF NOT EXISTS probes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id),
    pack_name TEXT,
    pack_category TEXT,
    pack_domain TEXT,
    probe_id TEXT,
    prompt TEXT,
    expected TEXT,
    match_mode TEXT,
    response TEXT,
    passed INTEGER,
    score REAL,
    reason TEXT,
    elapsed_seconds REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    backend TEXT,
    error TEXT,
    tags_json TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id),
    domain TEXT,
    probe_id TEXT,
    prompt TEXT,
    expected_answer TEXT,
    model_answer TEXT,
    correct INTEGER,
    confidence REAL,
    tags_json TEXT,
    source_pack TEXT
);

CREATE TABLE IF NOT EXISTS concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id),
    domain TEXT,
    n_probes INTEGER,
    n_passed INTEGER,
    mastery_level TEXT,
    pass_rate REAL,
    details_json TEXT
);

CREATE TABLE IF NOT EXISTS behavioral (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id),
    refusal_rate REAL,
    n_refusal_probes INTEGER,
    n_refused INTEGER,
    detected_persona_excerpt TEXT,
    refusal_details_json TEXT,
    bias_findings_json TEXT
);

CREATE TABLE IF NOT EXISTS calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id),
    hallucination_ack_rate REAL,
    n_hallucination_probes INTEGER,
    n_acknowledged_uncertainty INTEGER,
    math_accuracy REAL,
    n_math_probes INTEGER,
    n_math_correct INTEGER,
    hallucination_details_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_probes_model ON probes(model_id);
CREATE INDEX IF NOT EXISTS idx_facts_model ON facts(model_id);
CREATE INDEX IF NOT EXISTS idx_tensor_model ON tensor_stats(model_id);

-- v2: Attribution tables
CREATE TABLE IF NOT EXISTS mlp_layers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id),
    layer_index INTEGER,
    hidden_dim INTEGER,
    embed_dim INTEGER,
    is_gated INTEGER,
    n_neurons INTEGER,
    layer_strength REAL,
    concentration REAL
);

CREATE TABLE IF NOT EXISTS mlp_neurons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id),
    layer_index INTEGER,
    neuron_index INTEGER,
    key_norm REAL,
    value_norm REAL,
    memory_strength REAL,
    top_activating_tokens_json TEXT,
    top_output_tokens_json TEXT
);

CREATE TABLE IF NOT EXISTS attention_heads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id),
    layer_index INTEGER,
    head_index INTEGER,
    head_dim INTEGER,
    q_norm REAL,
    v_norm REAL,
    o_norm REAL,
    copy_score REAL,
    induction_score REAL,
    specialization REAL,
    top_singular_value REAL
);

CREATE TABLE IF NOT EXISTS fact_attributions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id),
    probe_id TEXT,
    prompt TEXT,
    expected TEXT,
    model_answer TEXT,
    correct INTEGER,
    attributed_layer INTEGER,
    attributed_neuron INTEGER,
    attribution_confidence REAL,
    method TEXT
);

CREATE TABLE IF NOT EXISTS knowledge_fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id),
    fingerprint TEXT,
    n_layers_analyzed_mlp INTEGER,
    n_layers_analyzed_attn INTEGER,
    n_heads_total INTEGER,
    n_global_top_neurons INTEGER,
    strongest_layer INTEGER,
    most_concentrated_layer INTEGER
);
"""


def export_sqlite(report: KnowledgeReport, out_path: Union[str, Path]) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    conn = sqlite3.connect(str(out_path))
    conn.executescript(SCHEMA)

    cur = conn.cursor()

    md = report.metadata if isinstance(report.metadata, dict) else {}
    cur.execute(
        """INSERT INTO models (
            gguf_path, gguf_filename, name, arch, quantization,
            vocab_size, context_length, embedding_length, block_count,
            file_size_bytes, extraction_timestamp, extractor_version,
            backend_used, total_elapsed_seconds, n_probes, n_probes_passed,
            metadata_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            report.gguf_path, report.gguf_filename,
            md.get("name"), md.get("arch"), md.get("quantization"),
            md.get("vocab_size", 0), md.get("context_length", 0),
            md.get("embedding_length", 0), md.get("block_count", 0),
            md.get("file_size_bytes", 0), report.extraction_timestamp,
            report.extractor_version,
            report.stats.get("backend_used", ""),
            report.stats.get("total_elapsed_seconds", 0),
            report.stats.get("n_probes", 0),
            report.stats.get("n_probes_passed", 0),
            json.dumps(md, default=str),
        ),
    )
    model_id = cur.lastrowid

    # Tensor stats
    wi = report.weight_inspection if isinstance(report.weight_inspection, dict) else {}
    for ts in wi.get("tensor_stats", []):
        cur.execute(
            """INSERT INTO tensor_stats (
                model_id, name, dtype, shape_json, n_elements, size_bytes,
                mean, std, min, max, abs_mean, norm, sparsity, n_outliers
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                model_id, ts.get("name"), ts.get("dtype"),
                json.dumps(ts.get("shape",[])),
                ts.get("n_elements",0), ts.get("size_bytes",0),
                ts.get("mean",0), ts.get("std",0),
                ts.get("min",0), ts.get("max",0),
                ts.get("abs_mean",0), ts.get("norm",0),
                ts.get("sparsity",0), ts.get("n_outliers",0),
            ),
        )

    # Layer stats
    for ls in wi.get("layer_stats", []):
        cur.execute(
            """INSERT INTO layer_stats (
                model_id, layer_index, attention_norm, mlp_up_norm,
                mlp_down_norm, mlp_gate_norm
            ) VALUES (?,?,?,?,?,?)""",
            (
                model_id, ls.get("layer_index", 0),
                (ls.get("attention_norm") or {}).get("norm"),
                (ls.get("mlp_up_norm") or {}).get("norm"),
                (ls.get("mlp_down_norm") or {}).get("norm"),
                (ls.get("mlp_gate_norm") or {}).get("norm"),
            ),
        )

    # Packs
    for s in report.pack_summaries:
        cur.execute(
            """INSERT INTO packs (
                model_id, pack_name, category, domain, n_probes, n_passed,
                n_failed, n_skipped, n_errors, pass_rate, avg_latency_seconds
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                model_id, s.get("pack_name"), s.get("category"), s.get("domain"),
                s.get("n_probes",0), s.get("n_passed",0), s.get("n_failed",0),
                s.get("n_skipped",0), s.get("n_errors",0),
                s.get("pass_rate",0), s.get("avg_latency_seconds",0),
            ),
        )

    # Probes
    for r in report.probe_results:
        cur.execute(
            """INSERT INTO probes (
                model_id, pack_name, pack_category, pack_domain, probe_id,
                prompt, expected, match_mode, response, passed, score, reason,
                elapsed_seconds, prompt_tokens, completion_tokens, backend,
                error, tags_json, metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                model_id, r.get("pack_name"), r.get("pack_category"),
                r.get("pack_domain"), r.get("probe_id"),
                r.get("prompt"), r.get("expected"), r.get("match_mode"),
                r.get("response"), 1 if r.get("passed") else 0,
                r.get("score",0), r.get("reason"),
                r.get("elapsed_seconds",0), r.get("prompt_tokens",0),
                r.get("completion_tokens",0), r.get("backend"),
                r.get("error"),
                json.dumps(r.get("tags",[])),
                json.dumps(r.get("metadata",{})),
            ),
        )

    # Facts
    for f in report.facts:
        cur.execute(
            """INSERT INTO facts (
                model_id, domain, probe_id, prompt, expected_answer,
                model_answer, correct, confidence, tags_json, source_pack
            ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                model_id, f.get("domain"), f.get("id"), f.get("prompt"),
                f.get("expected_answer"), f.get("model_answer"),
                1 if f.get("correct") else 0,
                f.get("confidence",0),
                json.dumps(f.get("tags",[])),
                f.get("source_pack"),
            ),
        )

    # Concepts
    for c in report.concepts:
        cur.execute(
            """INSERT INTO concepts (
                model_id, domain, n_probes, n_passed, mastery_level,
                pass_rate, details_json
            ) VALUES (?,?,?,?,?,?,?)""",
            (
                model_id, c.get("domain"), c.get("n_probes",0),
                c.get("n_passed",0), c.get("mastery_level","unknown"),
                c.get("pass_rate",0),
                json.dumps(c.get("details",[]), default=str)[:65000],
            ),
        )

    # Behavioral
    bp = report.behavioral_profile or {}
    cur.execute(
        """INSERT INTO behavioral (
            model_id, refusal_rate, n_refusal_probes, n_refused,
            detected_persona_excerpt, refusal_details_json, bias_findings_json
        ) VALUES (?,?,?,?,?,?,?)""",
        (
            model_id, bp.get("refusal_rate",0),
            bp.get("n_refusal_probes",0), bp.get("n_refused",0),
            bp.get("detected_persona_excerpt",""),
            json.dumps(bp.get("refusal_details",[]), default=str)[:65000],
            json.dumps(bp.get("bias_findings",[]), default=str)[:65000],
        ),
    )

    # Calibration
    cal = report.calibration or {}
    cur.execute(
        """INSERT INTO calibration (
            model_id, hallucination_ack_rate, n_hallucination_probes,
            n_acknowledged_uncertainty, math_accuracy, n_math_probes,
            n_math_correct, hallucination_details_json
        ) VALUES (?,?,?,?,?,?,?,?)""",
        (
            model_id, cal.get("hallucination_acknowledgment_rate"),
            cal.get("n_hallucination_probes",0),
            cal.get("n_acknowledged_uncertainty",0),
            cal.get("math_accuracy"),
            cal.get("n_math_probes",0), cal.get("n_math_correct",0),
            json.dumps(cal.get("hallucination_details",[]), default=str)[:65000],
        ),
    )

    # v2: Attribution
    attr = report.attribution if isinstance(report.attribution, dict) else {}
    if attr:
        # Fingerprint
        brief = attr.get("fingerprint_brief", {}) or {}
        cur.execute(
            """INSERT INTO knowledge_fingerprints (
                model_id, fingerprint, n_layers_analyzed_mlp,
                n_layers_analyzed_attn, n_heads_total, n_global_top_neurons,
                strongest_layer, most_concentrated_layer
            ) VALUES (?,?,?,?,?,?,?,?)""",
            (
                model_id, attr.get("knowledge_fingerprint",""),
                (attr.get("stats",{}) or {}).get("n_layers_analyzed_mlp",0),
                (attr.get("stats",{}) or {}).get("n_layers_analyzed_attn",0),
                (attr.get("stats",{}) or {}).get("n_heads_total",0),
                (attr.get("stats",{}) or {}).get("n_global_top_neurons",0),
                brief.get("strongest_layer"),
                brief.get("most_concentrated_layer"),
            ),
        )

        # MLP layers
        mlp = attr.get("mlp_analysis", {}) or {}
        for L in mlp.get("layers", []) or []:
            cur.execute(
                """INSERT INTO mlp_layers (
                    model_id, layer_index, hidden_dim, embed_dim, is_gated,
                    n_neurons, layer_strength, concentration
                ) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    model_id, L.get("layer"), L.get("hidden_dim"),
                    L.get("embed_dim"), 1 if L.get("is_gated") else 0,
                    L.get("n_neurons"), L.get("layer_strength",0),
                    L.get("concentration",0),
                ),
            )
            for n in L.get("top_neurons", []) or []:
                cur.execute(
                    """INSERT INTO mlp_neurons (
                        model_id, layer_index, neuron_index, key_norm, value_norm,
                        memory_strength, top_activating_tokens_json, top_output_tokens_json
                    ) VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        model_id, L.get("layer"), n.get("neuron_index"),
                        n.get("key_norm",0), n.get("value_norm",0),
                        n.get("memory_strength",0),
                        json.dumps(n.get("top_activating_tokens",[]), default=str)[:65000],
                        json.dumps(n.get("top_output_tokens",[]), default=str)[:65000],
                    ),
                )

        # Attention heads
        attn = attr.get("attention_analysis", {}) or {}
        for L in attn.get("layers", []) or []:
            for h in L.get("heads", []) or []:
                cur.execute(
                    """INSERT INTO attention_heads (
                        model_id, layer_index, head_index, head_dim, q_norm,
                        v_norm, o_norm, copy_score, induction_score,
                        specialization, top_singular_value
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        model_id, h.get("layer"), h.get("head_index"),
                        h.get("head_dim"), h.get("q_norm",0),
                        h.get("v_norm",0), h.get("o_norm",0),
                        h.get("copy_score",0), h.get("induction_score",0),
                        h.get("specialization",0), h.get("top_singular_value",0),
                    ),
                )

        # Fact attributions
        for f in attr.get("fact_attributions", []) or []:
            cur.execute(
                """INSERT INTO fact_attributions (
                    model_id, probe_id, prompt, expected, model_answer,
                    correct, attributed_layer, attributed_neuron,
                    attribution_confidence, method
                ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    model_id, f.get("probe_id"), f.get("prompt"),
                    f.get("expected"), f.get("model_answer"),
                    1 if f.get("correct") else 0,
                    f.get("attributed_layer"), f.get("attributed_neuron"),
                    f.get("attribution_confidence",0), f.get("method"),
                ),
            )

    conn.commit()
    conn.close()
    return str(out_path)
