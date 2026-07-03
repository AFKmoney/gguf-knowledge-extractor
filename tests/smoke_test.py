#!/usr/bin/env python3
"""Smoke test: generate a tiny test GGUF, run every CLI subcommand, verify outputs.

Usage:
    python tests/smoke_test.py

This is the same test the maintainer runs before each release. It exercises
every CLI subcommand end-to-end on a synthetic 3-layer Llama-arch GGUF.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def run(cmd: list[str], desc: str) -> bool:
    print(f"\n[smoke] {desc}")
    print(f"[smoke] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[smoke] FAIL: exit {result.returncode}")
        print(result.stderr[-2000:])
        return False
    print(result.stdout[-500:])
    return True


def main() -> int:
    work = Path("/tmp/gguf_smoke_test")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # 1. Generate test GGUF
    print("\n=== 1. Generate test GGUF ===")
    from make_test_gguf import make_test_gguf
    gguf_path = work / "test_tiny.gguf"
    make_test_gguf(str(gguf_path), vocab_size=128, embed_dim=64, n_layers=3)
    assert gguf_path.exists(), "Test GGUF was not created"
    print(f"[smoke] OK Generated {gguf_path} ({gguf_path.stat().st_size} bytes)")

    cli = ["python3", str(PROJECT_ROOT / "gguf_knowledge_extractor" / "cli.py")]

    # 2. Run each CLI subcommand
    tests = [
        ([*cli, "packs"], "list probe packs"),
        ([*cli, "backends"], "check inference backends"),
        ([*cli, "inspect", "--gguf", str(gguf_path), "--out", str(work)],
         "inspect (metadata + weights)"),
        ([*cli, "attribute", "--gguf", str(gguf_path), "--out", str(work), "--top-k", "5"],
         "v2 attribution (MLP decomposition)"),
        ([*cli, "trace", "--gguf", str(gguf_path), "--out", str(work), "--packs", "facts_geography"],
         "v3 causal tracing (logit lens)"),
        ([*cli, "edit", "--gguf", str(gguf_path), "--out", str(work),
          "--subject", "France", "--prompt", "What is the capital of France?",
          "--target", "Tokyo"],
         "v3 ROME rank-1 editing"),
        ([*cli, "extract", "--gguf", str(gguf_path), "--out", str(work),
          "--packs", "facts_geography", "--attribution"],
         "full extract pipeline"),
        ([*cli, "compare",
          str(work / "test_tiny_attribution.json"),
          str(work / "test_tiny_report.json"),
          "--out", str(work / "comparison.json")],
         "v3 cross-model comparison"),
        ([*cli, "models", "search", "llama", "--limit", "3"],
         "v4 HF Hub search (network required)"),
        ([*cli, "models", "list"],
         "v4 list local models"),
        ([*cli, "surgery", "--gguf", str(gguf_path), "--out", str(work),
          "--system-prompt", "Test surgery prompt",
          "--set-meta", "general.custom:string:tested"],
         "v5 GGUF surgery (system prompt + metadata)"),
    ]

    failures = 0
    for cmd, desc in tests:
        if not run(cmd, desc):
            failures += 1

    # 3. Verify all output files are valid
    print("\n=== 3. Verify outputs ===")
    json_files = list(work.glob("*.json"))
    db_files = list(work.glob("*.db"))
    edited_gguf = work / "test_tiny_edited.gguf"

    for jf in json_files:
        try:
            json.load(open(jf))
            print(f"[smoke] OK {jf.name} is valid JSON")
        except Exception as e:
            print(f"[smoke] FAIL {jf.name}: {e}")
            failures += 1

    for db in db_files:
        try:
            c = sqlite3.connect(db)
            n_tables = len(c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
            c.close()
            print(f"[smoke] OK {db.name}: {n_tables} tables")
        except Exception as e:
            print(f"[smoke] FAIL {db.name}: {e}")
            failures += 1

    if edited_gguf.exists():
        import gguf
        r = gguf.GGUFReader(str(edited_gguf))
        print(f"[smoke] OK {edited_gguf.name} loads ({len(r.tensors)} tensors)")
    else:
        print(f"[smoke] FAIL {edited_gguf.name} missing")
        failures += 1

    # 4. Final summary
    print(f"\n=== Summary ===")
    if failures == 0:
        print(f"[smoke] OK All tests passed!")
        return 0
    else:
        print(f"[smoke] FAIL {failures} failure(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
