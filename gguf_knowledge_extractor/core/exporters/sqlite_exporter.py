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

    conn.commit()
    conn.close()
    return str(out_path)
