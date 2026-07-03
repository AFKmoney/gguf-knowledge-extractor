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
import os
import sys
from pathlib import Path

# Make package importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gguf_knowledge_extractor.core.extractor import KnowledgeExtractor
from gguf_knowledge_extractor.core.inference.base import list_available_backends
from gguf_knowledge_extractor.core.probes.base import list_default_packs, get_pack_by_name
from gguf_knowledge_extractor.core.exporters.json_exporter import export_json
from gguf_knowledge_extractor.core.exporters.markdown_exporter import export_markdown
from gguf_knowledge_extractor.core.exporters.graph_exporter import export_graphml, export_turtle
from gguf_knowledge_extractor.core.exporters.sqlite_exporter import export_sqlite


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

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
