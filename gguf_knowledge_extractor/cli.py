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

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
