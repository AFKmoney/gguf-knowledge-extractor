#!/usr/bin/env python3
"""
GGUF Knowledge Extractor — CLI

Usage:
  python cli.py extract --gguf model.gguf [options]
  python cli.py inspect --gguf model.gguf        (metadata + weights only, no inference)
  python cli.py packs                             (list available probe packs)
  python cli.py backends                          (check inference backend availability)
  python cli.py web                               (start the local web UI)

Examples:
  # Full extraction with default packs
  python cli.py extract --gguf ./models/llama-7b.Q4_K_M.gguf --out ./out/

  # Weights-only (no inference required)
  python cli.py inspect --gguf ./models/model.gguf --out ./out/

  # Specific packs only
  python cli.py extract --gguf model.gguf --packs facts_geography,concepts_programming

  # Use llama.cpp server
  python cli.py extract --gguf model.gguf --prefer server --server-url http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make package importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gguf_knowledge_extractor.core.extractor import KnowledgeExtractor
from gguf_knowledge_extractor.core.inference.base import list_available_backends
from gguf_knowledge_extractor.core.probes.base import list_default_packs
from gguf_knowledge_extractor.core.exporters.json_exporter import export_json
from gguf_knowledge_extractor.core.exporters.markdown_exporter import export_markdown
from gguf_knowledge_extractor.core.exporters.graph_exporter import export_graphml, export_turtle
from gguf_knowledge_extractor.core.exporters.sqlite_exporter import export_sqlite
from gguf_knowledge_extractor.core.causal_tracer import CausalTracer
from gguf_knowledge_extractor.core.rome_editor import RomeEditor, EditRequest
from gguf_knowledge_extractor.core.fingerprint_compare import FingerprintComparator
from gguf_knowledge_extractor.core.model_manager import ModelManager, format_bytes
from gguf_knowledge_extractor.core.gguf_surgeon import GGUFSurgeon, surgery_session
from gguf_knowledge_extractor.core.quant_surgery import QuantSurgeon
from gguf_knowledge_extractor.core.model_merger import ModelMerger
from gguf_knowledge_extractor.core.gguf_diff import GGUFDiffer
from gguf_knowledge_extractor.core.activation_patcher import ActivationPatcher
from gguf_knowledge_extractor.core.imatrix import ImatrixComputer
from gguf_knowledge_extractor.core.quantizer import SmartQuantizer
from gguf_knowledge_extractor.core.knowledge_transplant import KnowledgeTransplanter
from gguf_knowledge_extractor.core.abliterator import Abliterator
from gguf_knowledge_extractor.core.advanced_techniques import (
    MemitEditor, MemitEdit, TaskArithmetic, RepresentationEngineer,
    WandaPruner, SmoothQuantizer, ConceptEraser, DynamicSteerer,
    CausalScrubber, ConstitutionalSurgeon
)


def cmd_extract(args):
    print(f"[extract] GGUF: {args.gguf}")
    packs = list_default_packs()
    if args.packs and args.packs != "all":
        names = {n.strip() for n in args.packs.split(",")}
        packs = [p for p in packs if p.name in names]
        if not packs:
            print(f"[extract] ERROR: no packs matched '{args.packs}'")
            sys.exit(1)
    print(f"[extract] Probe packs: {', '.join(p.name for p in packs)} ({sum(len(p) for p in packs)} probes total)")
    if args.attribution:
        print(f"[extract] Attribution: ENABLED (top {args.attribution_top_k} neurons per layer)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def progress(msg, cur, total):
        if total > 0:
            pct = cur / total * 100
            print(f"[extract] [{cur}/{total}] {pct:.0f}% — {msg}")
        else:
            print(f"[extract] {msg}")

    extractor = KnowledgeExtractor(
        gguf_path=args.gguf,
        server_url=args.server_url,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        prefer_backend=args.prefer,
        progress_cb=progress,
    )
    report = extractor.extract(
        packs=packs,
        do_metadata=True,
        do_weights=not args.no_weights,
        do_probes=not args.no_probes,
        do_attribution=args.attribution,
        attribution_top_k=args.attribution_top_k,
    )

    base = Path(args.gguf).stem
    json_path = export_json(report, out_dir / f"{base}_report.json")
    md_path = export_markdown(report, out_dir / f"{base}_report.md")
    graphml_path = export_graphml(report, out_dir / f"{base}_knowledge.graphml")
    ttl_path = export_turtle(report, out_dir / f"{base}_knowledge.ttl")
    db_path = export_sqlite(report, out_dir / f"{base}_knowledge.db")

    print()
    print(f"[extract] ✓ Done in {report.stats['total_elapsed_seconds']:.1f}s")
    print(f"[extract] Backend used: {report.stats.get('backend_used')}")
    print(f"[extract] Probes run: {report.stats['n_probes']}  |  passed: {report.stats['n_probes_passed']}")
    if args.attribution and report.attribution:
        fp = report.attribution.get("knowledge_fingerprint","")[:16]
        n_layers = (report.attribution.get("stats",{}) or {}).get("n_layers_analyzed_mlp",0)
        n_neurons = (report.attribution.get("stats",{}) or {}).get("n_global_top_neurons",0)
        print(f"[extract] Knowledge fingerprint: {fp}...")
        print(f"[extract] MLP layers analyzed: {n_layers}  |  top neurons extracted: {n_neurons}")
    print(f"[extract] Outputs:")
    print(f"           JSON     : {json_path}")
    print(f"           Markdown : {md_path}")
    print(f"           GraphML  : {graphml_path}")
    print(f"           Turtle   : {ttl_path}")
    print(f"           SQLite   : {db_path}")


def cmd_inspect(args):
    print(f"[inspect] GGUF: {args.gguf}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def progress(msg, cur, total):
        print(f"[inspect] {msg}")

    extractor = KnowledgeExtractor(
        gguf_path=args.gguf,
        prefer_backend="auto",
        progress_cb=progress,
    )
    report = extractor.extract(packs=[], do_metadata=True, do_weights=True, do_probes=False)

    base = Path(args.gguf).stem
    json_path = export_json(report, out_dir / f"{base}_inspect.json")
    md_path = export_markdown(report, out_dir / f"{base}_inspect.md")

    print(f"[inspect] ✓ Done")
    print(f"[inspect] Outputs:")
    print(f"           JSON     : {json_path}")
    print(f"           Markdown : {md_path}")


def cmd_attribute(args):
    """Weights + attribution only — no inference required."""
    print(f"[attribute] GGUF: {args.gguf}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def progress(msg, cur, total):
        if total > 0:
            print(f"[attribute] [{cur}/{total}] {msg}")
        else:
            print(f"[attribute] {msg}")

    extractor = KnowledgeExtractor(
        gguf_path=args.gguf,
        prefer_backend="auto",
        progress_cb=progress,
    )
    report = extractor.extract(
        packs=[],
        do_metadata=True,
        do_weights=True,
        do_probes=False,
        do_attribution=True,
        attribution_top_k=args.top_k,
    )

    base = Path(args.gguf).stem
    json_path = export_json(report, out_dir / f"{base}_attribution.json")
    md_path = export_markdown(report, out_dir / f"{base}_attribution.md")
    graphml_path = export_graphml(report, out_dir / f"{base}_attribution.graphml")
    ttl_path = export_turtle(report, out_dir / f"{base}_attribution.ttl")
    db_path = export_sqlite(report, out_dir / f"{base}_attribution.db")

    print()
    print(f"[attribute] ✓ Done in {report.stats['total_elapsed_seconds']:.1f}s")
    if report.attribution:
        fp = report.attribution.get("knowledge_fingerprint","")
        n_layers = (report.attribution.get("stats",{}) or {}).get("n_layers_analyzed_mlp",0)
        n_neurons = (report.attribution.get("stats",{}) or {}).get("n_global_top_neurons",0)
        n_heads = (report.attribution.get("stats",{}) or {}).get("n_heads_total",0)
        print(f"[attribute] Knowledge fingerprint: {fp}")
        print(f"[attribute] MLP layers analyzed: {n_layers}")
        print(f"[attribute] Attention heads analyzed: {n_heads}")
        print(f"[attribute] Global top neurons extracted: {n_neurons}")
    print(f"[attribute] Outputs:")
    print(f"           JSON     : {json_path}")
    print(f"           Markdown : {md_path}")
    print(f"           GraphML  : {graphml_path}")
    print(f"           Turtle   : {ttl_path}")
    print(f"           SQLite   : {db_path}")


def cmd_packs(args):
    packs = list_default_packs()
    print(f"Found {len(packs)} probe packs:")
    print()
    for p in packs:
        print(f"  • {p.name}  [{p.category}/{p.domain or 'general'}]  ({len(p)} probes)")
        print(f"    {p.description}")
        print()


def cmd_backends(args):
    print(f"Checking backends (server: {args.server_url})...")
    avail = list_available_backends(server_url=args.server_url)
    print()
    for name, ok in avail.items():
        print(f"  {'✓' if ok else '✗'} {name}: {'available' if ok else 'not available'}")
    print()
    if not any(avail.values()):
        print("⚠ No inference backend is available. Probes will be skipped.")
        print("  To enable llama.cpp server: llama-server -m your_model.gguf --port 8080")
        print("  To enable llama-cpp-python: pip install llama-cpp-python")


def cmd_web(args):
    print(f"Starting web UI at http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop.")
    import uvicorn
    from gguf_knowledge_extractor.web.server import app
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


# ---------------------------------------------------------------------- #
# v3 subcommands
# ---------------------------------------------------------------------- #
def cmd_trace(args):
    """v3: Logit-lens causal tracing on fact probes."""
    import gguf
    print(f"[trace] GGUF: {args.gguf}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = gguf.GGUFReader(args.gguf)
    # Load fields
    fields = {}
    for fname in reader.fields.keys():
        try:
            f = reader.get_field(fname)
            if len(f.types) == 1 and f.types[0] == gguf.GGUFValueType.ARRAY:
                arr = f.parts[f.data[0]] if f.data else None
                fields[fname] = arr.tolist() if arr is not None and hasattr(arr, "tolist") else None
            else:
                try:
                    v = f.contents()
                    if hasattr(v, "tolist"):
                        fields[fname] = v.tolist()
                    elif hasattr(v, "item"):
                        fields[fname] = v.item()
                    else:
                        fields[fname] = v
                except Exception:
                    fields[fname] = None
        except Exception:
            pass

    tracer = CausalTracer(reader, fields, top_k=args.top_k)
    if not tracer.is_available():
        print(f"[trace] ERROR: forward pass not available for this GGUF")
        print(f"[trace] (Need Llama-arch with full attn + MLP tensors)")
        sys.exit(1)

    # Load fact probes
    packs = list_default_packs()
    fact_packs = [p for p in packs if p.category == "facts"]
    if args.packs:
        names = {n.strip() for n in args.packs.split(",")}
        fact_packs = [p for p in fact_packs if p.name in names]
    if not fact_packs:
        print(f"[trace] No fact packs found")
        sys.exit(1)

    facts = []
    for pack in fact_packs:
        for probe in pack.probes:
            facts.append({
                "probe_id": probe.id,
                "prompt": probe.prompt,
                "expected": probe.expected,
                "domain": pack.domain,
                "source_pack": pack.name,
            })

    print(f"[trace] Tracing {len(facts)} facts via logit lens...")
    report = tracer.trace_facts(facts)
    print(f"[trace] ✓ Done in {report.stats['elapsed_seconds']:.1f}s")
    print(f"[trace] Forward pass available: {report.stats['forward_pass_available']}")
    print(f"[trace] Facts with logit lens: {report.n_with_logit_lens}/{report.n_facts_traced}")
    print(f"[trace] Correct predictions: {report.n_correct}/{report.n_facts_traced}")
    if report.avg_first_correct_layer is not None:
        print(f"[trace] Avg first-correct layer: {report.avg_first_correct_layer:.1f}")

    # Save
    base = Path(args.gguf).stem
    json_path = out_dir / f"{base}_causal_trace.json"
    with open(json_path, "w") as f:
        json.dump(_to_jsonable_trace(report), f, indent=2, default=str)
    print(f"[trace] Output: {json_path}")


def cmd_edit(args):
    """v3: ROME-style rank-1 fact editing."""
    import gguf, json
    print(f"[edit] GGUF: {args.gguf}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load edit requests
    if args.edits_file:
        with open(args.edits_file) as f:
            edits_data = json.load(f)
        requests = [EditRequest(**e) for e in edits_data]
    elif args.subject and args.prompt and args.target:
        requests = [EditRequest(subject=args.subject, prompt=args.prompt, target_object=args.target)]
    else:
        print("[edit] ERROR: provide --edits-file OR --subject/--prompt/--target")
        sys.exit(1)

    print(f"[edit] {len(requests)} edit request(s)")

    reader = gguf.GGUFReader(args.gguf)
    fields = {}
    for fname in reader.fields.keys():
        try:
            f = reader.get_field(fname)
            if len(f.types) == 1 and f.types[0] == gguf.GGUFValueType.ARRAY:
                arr = f.parts[f.data[0]] if f.data else None
                fields[fname] = arr.tolist() if arr is not None and hasattr(arr, "tolist") else None
            else:
                try:
                    v = f.contents()
                    if hasattr(v, "tolist"):
                        fields[fname] = v.tolist()
                    elif hasattr(v, "item"):
                        fields[fname] = v.item()
                    else:
                        fields[fname] = v
                except Exception:
                    fields[fname] = None
        except Exception:
            pass

    editor = RomeEditor(reader, fields)
    if not editor.is_available():
        print("[edit] ERROR: forward pass not available for this GGUF")
        sys.exit(1)

    output_gguf = out_dir / f"{Path(args.gguf).stem}_edited.gguf"
    report = editor.edit_facts(requests, args.gguf, str(output_gguf))

    print(f"\n[edit] ✓ Done in {report.stats['elapsed_seconds']:.1f}s")
    print(f"[edit] Successful: {report.n_edits_successful}/{report.n_edits_requested}")
    for e in report.edits:
        status = "✓" if e.get("edit_successful") else "✗"
        print(f"  {status} '{e.get('subject','')}' -> '{e.get('target_object','')}' "
              f"(layer {e.get('edited_layer')}, neuron {e.get('edited_neuron')}) "
              f"| pre: '{e.get('pre_edit_prediction','')}' → post: '{e.get('post_edit_prediction','')}'")
    if report.output_gguf_path:
        print(f"[edit] Modified GGUF: {report.output_gguf_path}")

    # Save report
    json_path = out_dir / f"{Path(args.gguf).stem}_edit_report.json"
    with open(json_path, "w") as f:
        json.dump(_to_jsonable_trace(report), f, indent=2, default=str)
    print(f"[edit] Report: {json_path}")


def cmd_compare(args):
    """v3: Cross-model fingerprint comparison."""
    import json
    reports = args.reports
    if len(reports) < 2:
        print("[compare] ERROR: need at least 2 reports to compare")
        sys.exit(1)

    print(f"[compare] Comparing {len(reports)} attribution reports:")
    for r in reports:
        print(f"  - {r}")

    comparator = FingerprintComparator()
    report = comparator.compare_all(reports)

    print(f"\n[compare] ✓ Done")
    print(f"[compare] Models: {report.n_models}")
    print(f"[compare] Pairwise comparisons: {len(report.pairwise)}")
    print(f"[compare] Identical fingerprints: {report.stats['n_identical_fingerprints']}")
    print(f"[compare] Strong lineage (>0.7): {report.stats['n_strong_lineage']}")

    print(f"\n=== Pairwise ===")
    for p in report.pairwise:
        print(f"\n  {p['model_a']}  vs  {p['model_b']}")
        print(f"    Fingerprint match: {p['fingerprint_match']}")
        print(f"    Same arch: {p['same_arch']}")
        print(f"    Top neuron Jaccard: {p['top_neuron_jaccard']:.3f}")
        print(f"    Top token Jaccard: {p['top_token_jaccard']:.3f}")
        print(f"    Concept mastery corr: {p['concept_mastery_correlation']:.3f}")
        print(f"    Behavioral similarity: {p['behavioral_similarity']:.3f}")
        print(f"    Lineage score: {p['lineage_score']:.3f}")
        print(f"    → {p['lineage_hypothesis']}")

    # Save
    out_path = args.out or "./download/comparison_report.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_to_jsonable_trace(report), f, indent=2, default=str)
    print(f"\n[compare] Report saved: {out_path}")


def _to_jsonable_trace(obj):
    """Convert dataclass to JSON-serializable dict."""
    import numpy as np
    from dataclasses import asdict
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable_trace(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable_trace(v) for v in obj]
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


# ---------------------------------------------------------------------- #
# models subcommands (Hugging Face Hub integration)
# ---------------------------------------------------------------------- #
def cmd_models(args):
    """v4: Hugging Face Hub model browser & downloader."""
    mgr = ModelManager(models_dir=args.models_dir, hf_token=args.hf_token)

    if args.models_cmd == "search":
        print(f"[models] Searching HF Hub for '{args.query}' (limit {args.limit})...")
        results = mgr.search(args.query, limit=args.limit, sort=args.sort, gguf_only=not args.all_models)
        if not results:
            print("[models] No models found.")
            return
        print(f"[models] Found {len(results)} models:")
        print()
        print(f"{'Repo ID':<55} {'Downloads':>10} {'Likes':>6} {'GGUF files':>11}")
        print("-" * 90)
        for m in results:
            print(f"{m.repo_id:<55} {m.downloads:>10,} {m.likes:>6} {len(m.gguf_files):>11}")
        print()
        print("Use `python cli.py models info <repo_id>` for details.")

    elif args.models_cmd == "info":
        print(f"[models] Getting info for {args.repo_id}...")
        try:
            info = mgr.get_model_info(args.repo_id)
        except Exception as e:
            print(f"[models] ERROR: {e}")
            sys.exit(1)
        print()
        print(f"  Repo:          {info.repo_id}")
        print(f"  Author:        {info.author}")
        print(f"  Downloads:     {info.downloads:,}")
        print(f"  Likes:         {info.likes}")
        print(f"  Pipeline tag:  {info.pipeline_tag or 'n/a'}")
        print(f"  Gated:         {info.gated}")
        print(f"  Last modified: {info.last_modified}")
        print(f"  Tags:          {', '.join(info.tags[:10])}")
        print()
        print(f"  Files ({len(info.files)}):")
        for f in info.files:
            marker = "[GGUF]" if f.is_gguf else "      "
            size = format_bytes(f.size_bytes) if f.size_bytes else "?"
            print(f"    {marker} {f.filename}  ({size})")

    elif args.models_cmd == "download":
        if not args.filename:
            print("[models] ERROR: --filename is required for download")
            print("[models] Use `models info <repo_id>` to see available files")
            sys.exit(1)
        print(f"[models] Downloading {args.filename} from {args.repo_id}...")

        last_percent = [-1]
        def progress_cb(p):
            curr_pct = int(p.percent)
            if curr_pct != last_percent[0] and curr_pct % 5 == 0:
                last_percent[0] = curr_pct
                print(f"  {p.percent:5.1f}%  {format_bytes(p.bytes_downloaded)} / {format_bytes(p.total_bytes)}  "
                      f"({p.speed_mbps:.1f} MB/s, ETA {p.eta_seconds:.0f}s)")

        result = mgr.download(args.repo_id, args.filename, progress_cb=progress_cb)
        if result.success:
            print()
            print(f"[models] ✓ Downloaded {result.filename}")
            print(f"[models]   Path: {result.local_path}")
            print(f"[models]   Size: {format_bytes(result.size_bytes)}")
            print(f"[models]   Time: {result.elapsed_seconds:.1f}s")
        else:
            print()
            print(f"[models] ✗ FAILED: {result.error}")
            sys.exit(1)

    elif args.models_cmd == "list":
        local = mgr.list_local_models()
        if not local:
            print(f"[models] No local models in {mgr.models_dir}")
            return
        print(f"[models] {len(local)} local models in {mgr.models_dir}:")
        print()
        print(f"{'Filename':<55} {'Size':>10} {'Repo':<40}")
        print("-" * 110)
        for m in local:
            print(f"{m.filename:<55} {format_bytes(m.size_bytes):>10} {(m.repo_id or 'unknown'):<40}")

    elif args.models_cmd == "delete":
        if not args.filename:
            print("[models] ERROR: --filename is required for delete")
            sys.exit(1)
        if mgr.delete_local_model(args.filename):
            print(f"[models] ✓ Deleted {args.filename}")
        else:
            print(f"[models] ✗ File not found: {args.filename}")
            sys.exit(1)

    else:
        print("[models] Available subcommands: search, info, download, list, delete")
        print("  Use --help for details")

    mgr.close()


def cmd_surgery(args):
    """v5: Direct GGUF surgery — modify tensors, metadata, vocab, datasets without retraining."""
    print(f"[surgery] Source: {args.gguf}")
    print(f"[surgery] Output: {args.out}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load operations from JSON file or build from individual flags
    operations = []
    if args.operations_file:
        with open(args.operations_file) as f:
            ops_data = json.load(f)
        if isinstance(ops_data, list):
            operations = ops_data
        elif isinstance(ops_data, dict) and "operations" in ops_data:
            operations = ops_data["operations"]
        else:
            print("[surgery] ERROR: operations file must be a list or {\"operations\": [...]}")
            sys.exit(1)
    else:
        # Build operations from individual flags
        if args.system_prompt:
            operations.append({"op": "bake_system_prompt", "prompt": args.system_prompt})
        if args.chat_template is not None:
            if args.chat_template == "":
                operations.append({"op": "set_chat_template", "template": ""})
            else:
                # Read from file
                with open(args.chat_template) as f:
                    operations.append({"op": "set_chat_template", "template": f.read()})
        if args.inject_dataset:
            # Format: name:path:description
            parts = args.inject_dataset.split(":", 2)
            if len(parts) < 2:
                print("[surgery] ERROR: --inject-dataset format is name:path[:description]")
                sys.exit(1)
            ds_name = parts[0]
            ds_path = parts[1]
            ds_desc = parts[2] if len(parts) > 2 else ""
            with open(ds_path) as f:
                ds_data = json.load(f)
            operations.append({"op": "inject_dataset", "name": ds_name, "data": ds_data, "description": ds_desc})
        if args.add_token:
            # Format: token[:embedding_file]
            parts = args.add_token.split(":", 1)
            token = parts[0]
            embedding = None
            if len(parts) > 1:
                import numpy as np
                embedding = np.load(parts[1]).tolist()
            operations.append({"op": "add_token", "token": token, "embedding": embedding})
        if args.add_steering:
            # Format: layer:name:vector_file[:strength]
            parts = args.add_steering.split(":")
            if len(parts) < 3:
                print("[surgery] ERROR: --add-steering format is layer:name:vector_file[:strength]")
                sys.exit(1)
            import numpy as np
            layer = int(parts[0])
            name = parts[1]
            vector = np.load(parts[2]).tolist()
            strength = float(parts[3]) if len(parts) > 3 else 1.0
            operations.append({"op": "add_steering_vector", "layer": layer, "name": name, "vector": vector, "strength": strength})
        if args.set_meta:
            # Format: key:type:value (type: string|int|float|bool)
            parts = args.set_meta.split(":", 2)
            if len(parts) < 3:
                print("[surgery] ERROR: --set-meta format is key:type:value")
                sys.exit(1)
            key, vtype, value = parts
            if vtype == "int":
                value = int(value)
            elif vtype == "float":
                value = float(value)
            elif vtype == "bool":
                value = value.lower() in ("true", "1", "yes")
            operations.append({"op": "set_metadata", "key": key, "value": value})
        if args.remove_meta:
            operations.append({"op": "remove_metadata", "key": args.remove_meta})

    if not operations:
        print("[surgery] No operations specified. Use --operations-file or individual flags.")
        print("[surgery] Available operations:")
        print("  --system-prompt <text>           Bake a system prompt")
        print("  --chat-template <file>           Set chat template from file")
        print("  --inject-dataset name:path[:desc] Inject a JSON dataset")
        print("  --add-token token[:embed.npy]    Add a new token")
        print("  --add-steering layer:name:vec.npy[:strength]")
        print("                                   Add a steering vector")
        print("  --set-meta key:type:value        Set a metadata field")
        print("  --remove-meta key                Remove a metadata field")
        print("  --operations-file <file>         JSON file with operation list")
        sys.exit(1)

    print(f"[surgery] {len(operations)} operation(s) queued:")
    for i, op in enumerate(operations):
        print(f"  [{i+1}] {op.get('op', '?')}: {', '.join(f'{k}={v}' for k, v in op.items() if k != 'op' and k != 'data' and k != 'vector' and k != 'embedding' and k != 'prompt' and k != 'template')}")

    base_name = Path(args.gguf).stem
    output_path = out_dir / f"{base_name}_surgery.gguf"

    report = surgery_session(args.gguf, str(output_path), operations)
    print(f"\n[surgery] Done in {report.elapsed_seconds:.2f}s")
    print(f"[surgery] Success: {report.success}")
    if report.error:
        print(f"[surgery] Error: {report.error[:500]}")
    else:
        print(f"[surgery] Output: {report.output_path}")
        print(f"[surgery] Tensors: {report.n_tensors}")
        print(f"[surgery] KV pairs: {report.n_kv_pairs}")
        print(f"[surgery] Operations applied: {len(report.operations)}")

    # Save surgery report
    report_path = out_dir / f"{base_name}_surgery_report.json"
    with open(report_path, "w") as f:
        json.dump(_to_jsonable_trace(report), f, indent=2, default=str)
    print(f"[surgery] Report: {report_path}")


def cmd_merge(args):
    """v6: Merge two GGUF models (linear, SLERP, TIES, DARE)."""
    print(f"[merge] Algorithm: {args.algorithm}")
    print(f"[merge] Model A: {args.model_a}")
    print(f"[merge] Model B: {args.model_b}")
    print(f"[merge] Alpha: {args.alpha}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"merged_{args.algorithm}.gguf"

    merger = ModelMerger(args.model_a, args.model_b)
    report = merger.merge(
        str(output_path),
        algorithm=args.algorithm,
        alpha=args.alpha,
        tensor_filter=args.filter,
    )

    print(f"\n[merge] Done in {report.elapsed_seconds:.2f}s")
    print(f"[merge] Success: {report.success}")
    if report.error:
        print(f"[merge] Error: {report.error[:500]}")
    else:
        print(f"[merge] Output: {report.output_path}")
        print(f"[merge] Tensors merged: {report.n_tensors_merged}")
        print(f"[merge] Tensors skipped: {report.n_tensors_skipped}")

    report_path = out_dir / f"merged_{args.algorithm}_report.json"
    with open(report_path, "w") as f:
        json.dump(_to_jsonable_trace(report), f, indent=2, default=str)
    print(f"[merge] Report: {report_path}")


def cmd_diff(args):
    """v6: Diff two GGUF files."""
    print(f"[diff] Model A: {args.model_a}")
    print(f"[diff] Model B: {args.model_b}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    differ = GGUFDiffer(args.model_a, args.model_b)
    report = differ.diff()

    print(f"\n[diff] Done in {report.elapsed_seconds:.2f}s")
    print(f"[diff] File sizes: {format_bytes(report.file_size_a)} vs {format_bytes(report.file_size_b)}")
    print(f"[diff] Tensors: {report.n_tensors_a} vs {report.n_tensors_b}")
    print(f"[diff] Summary:")
    print(f"  Same: {report.summary['n_tensors_same']}")
    print(f"  Modified: {report.summary['n_tensors_modified']}")
    print(f"  Added: {report.summary['n_tensors_added']}")
    print(f"  Removed: {report.summary['n_tensors_removed']}")
    print(f"  Avg cosine similarity: {report.summary['avg_cosine_similarity']:.4f}")
    print(f"  Overall similarity: {report.summary['overall_similarity_percent']:.1f}%")

    # Show top 10 most different tensors
    modified = [t for t in report.tensor_diffs if t["status"] == "modified"]
    modified.sort(key=lambda t: t.get("f32_cosine_sim", 1.0))
    if modified:
        print(f"\n[diff] Top 10 most different tensors:")
        for t in modified[:10]:
            cos = t.get("f32_cosine_sim", 0)
            print(f"  {t['name']}: cos_sim={cos:.4f}, mean_diff={t.get('f32_mean_abs_diff', 0):.4f}")

    report_path = out_dir / "diff_report.json"
    with open(report_path, "w") as f:
        json.dump(_to_jsonable_trace(report), f, indent=2, default=str)
    print(f"\n[diff] Report: {report_path}")


def cmd_mediate(args):
    """v6: Causal mediation analysis (activation patching)."""
    import gguf
    print(f"[mediate] GGUF: {args.gguf}")
    print(f"[mediate] Prompt: {args.prompt}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = gguf.GGUFReader(args.gguf)
    fields = _load_fields_cli(reader)

    patcher = ActivationPatcher(reader, fields)
    if not patcher.is_available():
        print("[mediate] ERROR: forward pass not available")
        sys.exit(1)

    report = patcher.analyze(
        probe_id="cli_mediate",
        prompt=args.prompt,
        expected_answer=args.expected,
        corruption_noise_std=args.noise,
    )

    print(f"\n[mediate] Done in {report.elapsed_seconds:.2f}s")
    print(f"[mediate] Method: {report.method}")
    print(f"[mediate] Clean prediction: '{report.clean_prediction}'")
    print(f"[mediate] Corrupt prediction: '{report.corrupt_prediction}'")
    print(f"[mediate] Best layer: {report.best_layer} (causal effect: {report.best_causal_effect:.6f})")
    print(f"\n[mediate] Per-layer results:")
    for lr in report.layer_results:
        print(f"  L{lr['layer']}: effect={lr['causal_effect']:+.6f}  "
              f"clean={lr['clean_answer_prob_clean']:.4f}  "
              f"corrupt={lr['clean_answer_prob_corrupt']:.4f}  "
              f"restored={lr['clean_answer_prob_restored']:.4f}  "
              f"top1: {lr['clean_top1_token']} -> {lr['corrupt_top1_token']} -> {lr['restored_top1_token']}")

    report_path = out_dir / "mediation_report.json"
    with open(report_path, "w") as f:
        json.dump(_to_jsonable_trace(report), f, indent=2, default=str)
    print(f"\n[mediate] Report: {report_path}")


def _load_fields_cli(reader):
    """Load GGUF metadata fields into a plain dict (shared by trace/mediate)."""
    import gguf
    fields = {}
    for fname in reader.fields.keys():
        if fname.startswith("GGUF."):
            continue
        try:
            f = reader.get_field(fname)
            is_array = (len(f.types) >= 1 and int(f.types[0]) == 9)
            if is_array:
                elem_type = int(f.types[1]) if len(f.types) >= 2 else 8
                value = []
                if elem_type == 8:
                    for i in range(len(f.data)):
                        try:
                            value.append(bytes(f.parts[f.data[i]]).decode("utf-8", errors="replace"))
                        except:
                            pass
                else:
                    if len(f.data) >= 2:
                        arr = f.parts[f.data[1]]
                        try:
                            value = arr.tolist()
                        except:
                            value = list(arr)
            else:
                type_id = int(f.types[0])
                if f.data and len(f.data) > 0:
                    part = f.parts[f.data[0]]
                    if type_id == 8:
                        try:
                            value = bytes(part).decode("utf-8", errors="replace")
                        except:
                            value = str(part)
                    elif part.size == 1:
                        value = part.item()
                    else:
                        try:
                            value = part.tolist()
                        except:
                            value = str(part)
                else:
                    value = None
            fields[fname] = value
        except:
            pass
    return fields


def cmd_imatrix(args):
    """v7: Compute importance matrix for a GGUF model."""
    import gguf
    print(f"[imatrix] GGUF: {args.gguf}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = gguf.GGUFReader(args.gguf)
    fields = _load_fields_cli(reader)
    imputer = ImatrixComputer(reader, fields)
    if not imputer.is_available():
        print("[imatrix] ERROR: forward pass not available")
        sys.exit(1)

    def progress(msg, cur, total):
        if total > 0:
            print(f"[imatrix] [{cur}/{total}] {msg}")

    report = imputer.compute(progress_cb=progress)
    print(f"\n[imatrix] Done in {report.elapsed_seconds:.2f}s")
    print(f"[imatrix] Tensors analyzed: {report.n_tensors_analyzed}")
    print(f"[imatrix] Tokens processed: {report.n_tokens_processed}")
    print(f"\n[imatrix] Top 10 most important tensors:")
    for ti in report.tensor_importances[:10]:
        rec = report.precision_recommendations.get(ti["name"], "?")
        print(f"  {ti['name']:<40} importance={ti['importance_score']:.4f}  -> {rec}")

    base = Path(args.gguf).stem
    json_path = out_dir / f"{base}_imatrix.json"
    with open(json_path, "w") as f:
        json.dump(_to_jsonable_trace(report), f, indent=2, default=str)
    print(f"\n[imatrix] Report: {json_path}")


def cmd_quantize(args):
    """v7: Quantize a GGUF model with optional imatrix."""
    import os
    print(f"[quantize] Source: {args.gguf}")
    print(f"[quantize] Target: {args.qtype}")
    print(f"[quantize] Use imatrix: {args.imatrix}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.gguf).stem
    output_path = out_dir / f"{base}_{args.qtype.lower()}.gguf"

    # Load imatrix if specified
    imatrix_report = None
    if args.imatrix:
        if args.imatrix_file:
            print(f"[quantize] Loading imatrix from {args.imatrix_file}")
            with open(args.imatrix_file) as f:
                imatrix_report = json.load(f)
        else:
            print("[quantize] Computing imatrix from calibration prompts...")
            import gguf
            reader = gguf.GGUFReader(args.gguf)
            fields = _load_fields_cli(reader)
            imputer = ImatrixComputer(reader, fields)
            if imputer.is_available():
                imatrix_report_obj = imputer.compute()
                imatrix_report = _to_jsonable_trace(imatrix_report_obj)

    q = SmartQuantizer(args.gguf, imatrix_report=imatrix_report)
    report = q.quantize(
        str(output_path),
        target_qtype=args.qtype,
        use_imatrix=bool(imatrix_report),
    )

    print(f"\n[quantize] Done in {report.elapsed_seconds:.2f}s")
    print(f"[quantize] Success: {report.success}")
    if report.error:
        print(f"[quantize] Error: {report.error[:500]}")
    else:
        print(f"[quantize] Output: {report.output_path}")
        print(f"[quantize] Input size: {format_bytes(report.input_size_bytes)}")
        print(f"[quantize] Output size: {format_bytes(report.output_size_bytes)}")
        print(f"[quantize] Compression: {report.compression_ratio:.2f}x")
        print(f"[quantize] Tensors quantized: {report.n_tensors_quantized}")
        print(f"[quantize] Tensors kept high precision: {report.n_tensors_kept_high_precision}")
        print(f"[quantize] Avg roundtrip error: {report.avg_roundtrip_error:.6f}")
        print(f"[quantize] Max roundtrip error: {report.max_roundtrip_error:.6f}")

    report_path = out_dir / f"{base}_{args.qtype.lower()}_report.json"
    with open(report_path, "w") as f:
        json.dump(_to_jsonable_trace(report), f, indent=2, default=str)
    print(f"[quantize] Report: {report_path}")


def cmd_transplant(args):
    """v7: Transplant knowledge from source model to target model."""
    print(f"[transplant] Source: {args.source}")
    print(f"[transplant] Target: {args.target}")
    print(f"[transplant] Strategy: {args.strategy}")
    print(f"[transplant] Strength: {args.strength}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{Path(args.target).stem}_transplanted.gguf"

    # Load facts from file or use default
    if args.facts_file:
        with open(args.facts_file) as f:
            facts = json.load(f)
    else:
        # Use default fact probes
        from gguf_knowledge_extractor.core.probes.base import list_default_packs
        packs = list_default_packs()
        facts = []
        for pack in packs:
            if pack.category == "facts":
                for probe in pack.probes:
                    facts.append({
                        "probe_id": probe.id,
                        "prompt": probe.prompt,
                        "expected": probe.expected,
                    })

    print(f"[transplant] {len(facts)} facts to transplant")

    transplanter = KnowledgeTransplanter(
        source_path=args.source,
        target_path=args.target,
        target_output_path=str(output_path),
    )
    if not transplanter.is_available():
        print("[transplant] ERROR: forward pass not available for one or both models")
        sys.exit(1)

    report = transplanter.transplant(
        facts=facts,
        strategy=args.strategy,
        strength=args.strength,
    )

    print(f"\n[transplant] Done in {report.elapsed_seconds:.2f}s")
    print(f"[transplant] Success: {report.success}")
    if report.error:
        print(f"[transplant] Error: {report.error[:500]}")
    else:
        print(f"[transplant] Facts extracted: {report.n_facts_extracted}")
        print(f"[transplant] Facts transplanted: {report.n_facts_transplanted}")
        print(f"[transplant] Successful: {report.n_successful}")
        print(f"[transplant] Output: {report.output_model}")
        print(f"\n[transplant] Layer mapping ({report.mapping_strategy}):")
        for src, tgt in report.layer_mapping.items():
            print(f"  {src} -> L{tgt}")
        print(f"\n[transplant] Results (first 10):")
        for r in report.results[:10]:
            status = "✓" if r.get("transplant_successful") else "✗"
            print(f"  {status} {r.get('probe_id','')}: L{r.get('source_layer','?')}N{r.get('source_neuron','?')} -> L{r.get('target_layer','?')}N{r.get('target_neuron','?')}")

    report_path = out_dir / f"{Path(args.target).stem}_transplant_report.json"
    with open(report_path, "w") as f:
        json.dump(_to_jsonable_trace(report), f, indent=2, default=str)
    print(f"\n[transplant] Report: {report_path}")


def cmd_abliterate(args):
    """v8: Abliterate a model — remove refusal behavior without retraining."""
    print(f"[abliterate] Source: {args.gguf}")
    print(f"[abliterate] Strength: {args.strength}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{Path(args.gguf).stem}_abliterated.gguf"

    abliterator = Abliterator(
        source_path=args.gguf,
        output_path=str(output_path),
    )
    if not abliterator.is_available():
        print("[abliterate] ERROR: forward pass not available (need Llama-arch)")
        sys.exit(1)

    def progress(msg, cur, total):
        if total > 0:
            print(f"[abliterate] [{cur}/{total}] {msg}")

    report = abliterator.abliterate(strength=args.strength, progress_cb=progress)
    print(f"\n[abliterate] Done in {report.elapsed_seconds:.2f}s")
    print(f"[abliterate] Success: {report.success}")
    if report.error:
        print(f"[abliterate] Error: {report.error[:500]}")
    else:
        print(f"[abliterate] Output: {report.output_gguf}")
        print(f"[abliterate] Layers processed: {report.n_layers}")
        print(f"[abliterate] Tensors orthogonalized: {report.n_tensors_orthogonalized}")
        print(f"[abliterate] Refusal directions found: {len(report.refusal_directions)}")

    base = Path(args.gguf).stem
    report_path = out_dir / f"{base}_abliteration_report.json"
    with open(report_path, "w") as f:
        json.dump(_to_jsonable_trace(report), f, indent=2, default=str)
    print(f"[abliterate] Report: {report_path}")


def cmd_advanced(args):
    """v9: Advanced techniques — 10 niche research methods."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    tech = args.technique
    print(f"[advanced] Technique: {tech}")

    if tech == "memit":
        edits_json = json.loads(args.edits) if args.edits else []
        edits = [MemitEdit(**e) for e in edits_json]
        if not edits:
            print("[advanced] ERROR: --edits required for memit (JSON list)")
            sys.exit(1)
        editor = MemitEditor(args.gguf, str(out_dir / "memit_output.gguf"))
        r = editor.edit_batch(edits, target_layer=args.layer)
        print(f"[advanced] {r.n_successful} edits, success={r.success}")

    elif tech == "task_arith":
        if not args.base or not args.finetuned:
            print("[advanced] ERROR: --base and --finetuned required")
            sys.exit(1)
        tv = TaskArithmetic.compute_task_vector(args.base, args.finetuned)
        ta = TaskArithmetic(args.base)
        r = ta.apply(args.operation or "add", [(tv, args.alpha or 0.5)], str(out_dir / "task_arith_output.gguf"))
        print(f"[advanced] success={r.success}")

    elif tech == "repe":
        re = RepresentationEngineer(args.gguf, str(out_dir / "repe_output.gguf"))
        pos = args.positive.split(",") if args.positive else ["I will be honest."]
        neg = args.negative.split(",") if args.negative else ["I will lie."]
        r = re.engineer(args.concept or "honesty", pos, neg, amplification=args.amplification or 1.5)
        print(f"[advanced] modified={r.n_tensors_modified}, success={r.success}")

    elif tech == "wanda":
        wp = WandaPruner(args.gguf, str(out_dir / "wanda_output.gguf"))
        r = wp.prune(target_sparsity=args.sparsity or 0.5)
        print(f"[advanced] sparsity={r.sparsity:.1%}, removed={r.n_weights_removed}, success={r.success}")

    elif tech == "smoothquant":
        sq = SmoothQuantizer(args.gguf, str(out_dir / "smoothquant_output.gguf"))
        r = sq.smooth(alpha=args.alpha or 0.5)
        print(f"[advanced] smoothed={r.n_tensors_smoothed}, success={r.success}")

    elif tech == "erase":
        dim = 64  # default; will use actual dim from model
        direction = np.random.randn(dim).astype(np.float32)
        ce = ConceptEraser(args.gguf, str(out_dir / "erased_output.gguf"))
        r = ce.erase(concept_direction=direction, strength=args.strength or 1.0)
        print(f"[advanced] erased={r.n_tensors_erased}, success={r.success}")

    elif tech == "steer":
        ds = DynamicSteerer(args.gguf, str(out_dir / "dynamic_steer_output.gguf"))
        vecs = [{"layer": args.layer or 0, "name": "steer", "vector": np.random.randn(64).tolist(), "strength": args.strength or 1.0}]
        r = ds.add_steering_vectors(vecs)
        print(f"[advanced] vectors={r.n_vectors}, success={r.success}")

    elif tech == "distill":
        if not args.teacher:
            print("[advanced] ERROR: --teacher required for distill")
            sys.exit(1)
        hd = HiddenStateDistiller.__new__(HiddenStateDistiller)
        from gguf_knowledge_extractor.core.advanced_techniques import HiddenStateDistiller
        hd = HiddenStateDistiller(args.teacher, args.gguf, str(out_dir / "distilled_output.gguf"))
        r = hd.distill(["The capital of France is", "Python is a language"], max_tokens_per_prompt=8)
        print(f"[advanced] layers={r.n_layers_distilled}, success={r.success}")

    elif tech == "scrub":
        cs = CausalScrubber(args.gguf)
        if not cs.is_available():
            print("[advanced] ERROR: forward pass unavailable")
            sys.exit(1)
        r = cs.scrub_test(args.target or "What is the capital of France?", args.control or "What is the capital of Japan?")
        print(f"[advanced] hypotheses={r.n_hypotheses_tested}, success={r.success}")
        for res in r.results:
            print(f"  L{res['layer']}: changed={res['answer_changed']}")

    elif tech == "constitutional":
        cons = ConstitutionalSurgeon(args.gguf, str(out_dir / "constitutional_output.gguf"))
        adjustments = json.loads(args.values) if args.values else {"honesty": 1.5, "helpfulness": 1.2}
        r = cons.adjust_values(adjustments)
        print(f"[advanced] values={r.values_adjusted}, modified={r.n_tensors_modified}, success={r.success}")


def main():
    p = argparse.ArgumentParser(prog="gguf-knowledge-extractor", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # extract
    pe = sub.add_parser("extract", help="Full extraction: metadata + weights + probes")
    pe.add_argument("--gguf", required=True, help="Path to the .gguf file")
    pe.add_argument("--out", default="./download", help="Output directory")
    pe.add_argument("--packs", default="all", help="Comma-separated pack names, or 'all'")
    pe.add_argument("--no-weights", action="store_true", help="Skip weight inspection")
    pe.add_argument("--no-probes", action="store_true", help="Skip inference probes")
    pe.add_argument("--attribution", action="store_true",
                    help="Run v2 knowledge attribution (ROME/MEMIT-style MLP decomposition)")
    pe.add_argument("--attribution-top-k", type=int, default=20,
                    help="Top-K neurons per layer for attribution (default 20)")
    pe.add_argument("--server-url", default="http://127.0.0.1:8080")
    pe.add_argument("--prefer", default="auto", choices=["auto", "server", "python"])
    pe.add_argument("--n-ctx", type=int, default=4096)
    pe.add_argument("--n-gpu-layers", type=int, default=0)
    pe.set_defaults(func=cmd_extract)

    # inspect
    pi = sub.add_parser("inspect", help="Weights + metadata only (no inference)")
    pi.add_argument("--gguf", required=True)
    pi.add_argument("--out", default="./download")
    pi.set_defaults(func=cmd_inspect)

    # attribute (v2)
    pa = sub.add_parser("attribute", help="v2: ROME/MEMIT-style knowledge attribution (no inference required)")
    pa.add_argument("--gguf", required=True)
    pa.add_argument("--out", default="./download")
    pa.add_argument("--top-k", type=int, default=20, help="Top-K neurons per layer")
    pa.set_defaults(func=cmd_attribute)

    # packs
    pp = sub.add_parser("packs", help="List available probe packs")
    pp.set_defaults(func=cmd_packs)

    # backends
    pb = sub.add_parser("backends", help="Check inference backend availability")
    pb.add_argument("--server-url", default="http://127.0.0.1:8080")
    pb.set_defaults(func=cmd_backends)

    # web
    pw = sub.add_parser("web", help="Start local web UI")
    pw.add_argument("--host", default="127.0.0.1")
    pw.add_argument("--port", type=int, default=8000)
    pw.add_argument("--log-level", default="info")
    pw.set_defaults(func=cmd_web)

    # v3: trace
    pt = sub.add_parser("trace", help="v3: Logit-lens causal tracing on fact probes")
    pt.add_argument("--gguf", required=True)
    pt.add_argument("--out", default="./download")
    pt.add_argument("--packs", default="", help="Comma-separated fact pack names (default: all facts)")
    pt.add_argument("--top-k", type=int, default=10)
    pt.set_defaults(func=cmd_trace)

    # v3: edit (ROME)
    ped = sub.add_parser("edit", help="v3: ROME-style rank-1 fact editing")
    ped.add_argument("--gguf", required=True)
    ped.add_argument("--out", default="./download")
    ped.add_argument("--edits-file", help="JSON file with list of {subject, prompt, target_object}")
    ped.add_argument("--subject", help="Single edit: subject")
    ped.add_argument("--prompt", help="Single edit: full prompt")
    ped.add_argument("--target", help="Single edit: target object")
    ped.set_defaults(func=cmd_edit)

    # v3: compare
    pcm = sub.add_parser("compare", help="v3: Cross-model fingerprint comparison")
    pcm.add_argument("reports", nargs="+", help="Two or more *_report.json paths")
    pcm.add_argument("--out", default="./download/comparison_report.json")
    pcm.set_defaults(func=cmd_compare)

    # v4: models (Hugging Face Hub)
    pmd = sub.add_parser("models", help="v4: Browse & download GGUF models from Hugging Face Hub")
    pmd.add_argument("--models-dir", default="/home/z/my-project/models", help="Local models directory")
    pmd.add_argument("--hf-token", default=None, help="Hugging Face API token (for gated models)")
    md_sub = pmd.add_subparsers(dest="models_cmd", required=True)

    md_search = md_sub.add_parser("search", help="Search HF Hub for GGUF models")
    md_search.add_argument("query", help="Search query (e.g. 'llama 3 8b')")
    md_search.add_argument("--limit", type=int, default=20)
    md_search.add_argument("--sort", default="downloads", choices=["downloads", "likes", "lastModified", "createdAt"])
    md_search.add_argument("--all-models", action="store_true", help="Include non-GGUF models in results")

    md_info = md_sub.add_parser("info", help="Get detailed info for a model repo")
    md_info.add_argument("repo_id", help="HF repo ID (e.g. 'hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF')")

    md_dl = md_sub.add_parser("download", help="Download a specific file from a model repo")
    md_dl.add_argument("repo_id", help="HF repo ID")
    md_dl.add_argument("--filename", required=True, help="Filename to download (e.g. 'llama-3.2-1b-instruct-q8_0.gguf')")

    md_list = md_sub.add_parser("list", help="List locally-downloaded models")

    md_del = md_sub.add_parser("delete", help="Delete a local model file")
    md_del.add_argument("filename", help="Filename to delete")

    pmd.set_defaults(func=cmd_models)

    # v5: surgery (direct GGUF modification)
    psu = sub.add_parser("surgery", help="v5: Direct GGUF surgery — modify tensors, metadata, vocab, datasets without retraining")
    psu.add_argument("--gguf", required=True, help="Source GGUF file")
    psu.add_argument("--out", default="./download", help="Output directory")
    psu.add_argument("--operations-file", help="JSON file with list of operations")
    psu.add_argument("--system-prompt", help="Bake a system prompt into the model")
    psu.add_argument("--chat-template", help="Set chat template from file (use '' for empty)")
    psu.add_argument("--inject-dataset", help="Inject dataset: name:path[:description]")
    psu.add_argument("--add-token", help="Add token: token[:embedding.npy]")
    psu.add_argument("--add-steering", help="Add steering vector: layer:name:vector.npy[:strength]")
    psu.add_argument("--set-meta", help="Set metadata: key:type:value (type: string|int|float|bool)")
    psu.add_argument("--remove-meta", help="Remove metadata: key")
    psu.set_defaults(func=cmd_surgery)

    # v6: merge
    pmg = sub.add_parser("merge", help="v6: Merge two GGUF models (linear, SLERP, TIES, DARE)")
    pmg.add_argument("model_a", help="Path to model A (base)")
    pmg.add_argument("model_b", help="Path to model B (fine-tune)")
    pmg.add_argument("--algorithm", default="linear", choices=["linear", "slerp", "ties", "dare"])
    pmg.add_argument("--alpha", type=float, default=0.5, help="Merge weight (0=B, 1=A, 0.5=equal)")
    pmg.add_argument("--filter", help="Regex to filter which tensors to merge (default: all)")
    pmg.add_argument("--out", default="./download")
    pmg.set_defaults(func=cmd_merge)

    # v6: diff
    pdf = sub.add_parser("diff", help="v6: Diff two GGUF files")
    pdf.add_argument("model_a", help="Path to model A")
    pdf.add_argument("model_b", help="Path to model B")
    pdf.add_argument("--out", default="./download")
    pdf.set_defaults(func=cmd_diff)

    # v6: mediate (causal mediation analysis)
    pmd = sub.add_parser("mediate", help="v6: Causal mediation analysis (activation patching)")
    pmd.add_argument("--gguf", required=True)
    pmd.add_argument("--prompt", required=True, help="Fact prompt to analyze")
    pmd.add_argument("--expected", help="Expected answer (for computing causal effect)")
    pmd.add_argument("--noise", type=float, default=1.0, help="Corruption noise std (default: 1.0)")
    pmd.add_argument("--out", default="./download")
    pmd.set_defaults(func=cmd_mediate)

    # v7: imatrix
    pim = sub.add_parser("imatrix", help="v7: Compute importance matrix for quantization calibration")
    pim.add_argument("--gguf", required=True)
    pim.add_argument("--out", default="./download")
    pim.set_defaults(func=cmd_imatrix)

    # v7: quantize
    pqz = sub.add_parser("quantize", help="v7: Quantize GGUF with optional imatrix guidance")
    pqz.add_argument("--gguf", required=True)
    pqz.add_argument("--qtype", default="Q4_0", choices=SmartQuantizer.get_supported_qtypes())
    pqz.add_argument("--imatrix", action="store_true", help="Use imatrix for per-tensor precision")
    pqz.add_argument("--imatrix-file", help="Precomputed imatrix JSON file")
    pqz.add_argument("--out", default="./download")
    pqz.set_defaults(func=cmd_quantize)

    # v7: transplant
    ptp = sub.add_parser("transplant", help="v7: Transplant knowledge from source model to target model")
    ptp.add_argument("--source", required=True, help="Source model (extract knowledge from)")
    ptp.add_argument("--target", required=True, help="Target model (inject knowledge into)")
    ptp.add_argument("--facts-file", help="JSON file with list of {probe_id, prompt, expected}")
    ptp.add_argument("--strategy", default="scaled", choices=["same", "scaled"])
    ptp.add_argument("--strength", type=float, default=1.0, help="0=blend, 1=full overwrite")
    ptp.add_argument("--out", default="./download")
    ptp.set_defaults(func=cmd_transplant)

    # v8: abliterate
    pab = sub.add_parser("abliterate", help="v8: Remove refusal behavior — create uncensored/abliterated model without retraining")
    pab.add_argument("--gguf", required=True)
    pab.add_argument("--strength", type=float, default=1.0, help="0=no change, 1=full abliteration (default: 1.0)")
    pab.add_argument("--out", default="./download")
    pab.set_defaults(func=cmd_abliterate)

    # v9: advanced techniques
    padv = sub.add_parser("advanced", help="v9: 10 advanced techniques (memit, task_arith, repe, wanda, smoothquant, erase, steer, distill, scrub, constitutional)")
    padv.add_argument("technique", choices=["memit", "task_arith", "repe", "wanda", "smoothquant", "erase", "steer", "distill", "scrub", "constitutional"])
    padv.add_argument("--gguf", help="Source GGUF")
    padv.add_argument("--base", help="Base model (for task_arith)")
    padv.add_argument("--finetuned", help="Fine-tuned model (for task_arith)")
    padv.add_argument("--teacher", help="Teacher model (for distill)")
    padv.add_argument("--edits", help="JSON list of edits (for memit)")
    padv.add_argument("--operation", help="add/subtract/combine (for task_arith)")
    padv.add_argument("--concept", help="Concept name (for repe)")
    padv.add_argument("--positive", help="Positive prompts, comma-separated (for repe)")
    padv.add_argument("--negative", help="Negative prompts, comma-separated (for repe)")
    padv.add_argument("--amplification", type=float, help="Amplification factor (for repe)")
    padv.add_argument("--sparsity", type=float, help="Target sparsity 0-1 (for wanda)")
    padv.add_argument("--alpha", type=float, help="Alpha factor (for smoothquant, task_arith)")
    padv.add_argument("--strength", type=float, help="Strength factor (for erase, steer)")
    padv.add_argument("--layer", type=int, help="Target layer")
    padv.add_argument("--target", help="Target prompt (for scrub)")
    padv.add_argument("--control", help="Control prompt (for scrub)")
    padv.add_argument("--values", help="JSON dict of value adjustments (for constitutional)")
    padv.add_argument("--out", default="./download")
    padv.set_defaults(func=cmd_advanced)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
