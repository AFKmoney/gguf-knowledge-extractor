#!/usr/bin/env python3
"""Assemble staged plain sources and restore forward_pass from gzip loader + patch."""
from __future__ import annotations
import base64, gzip, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def assemble_cli() -> None:
    parts_dir = ROOT / ".restore_plain"
    parts = sorted(parts_dir.glob("cli_part_*.txt"))
    if not parts:
        print("no cli parts; skip cli")
        return
    text = "".join(p.read_text(encoding="utf-8") for p in parts)
    out = ROOT / "gguf_knowledge_extractor" / "cli.py"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} chars) from {len(parts)} parts")
    for p in parts:
        p.unlink()

def restore_forward_pass() -> None:
    path = ROOT / "gguf_knowledge_extractor" / "core" / "forward_pass.py"
    src = path.read_text(encoding="utf-8")
    if "gzip.decompress" in src and "base64.b64decode" in src:
        m = re.search(r"base64\.b64decode\('([^']+)'\)", src, re.S)
        if not m:
            raise SystemExit("could not find base64 payload in forward_pass.py")
        plain = gzip.decompress(base64.b64decode(m.group(1))).decode("utf-8")
        path.write_text(plain, encoding="utf-8")
        print(f"decoded gzip loader -> plain ({len(plain)} chars)")
    patch = ROOT / ".restore_plain" / "forward_pass.patch"
    if patch.exists():
        r = subprocess.run(["patch", "-p1", "--forward", "--reject-file=-"], cwd=ROOT, input=patch.read_text(encoding="utf-8"), text=True, capture_output=True)
        print(r.stdout)
        print(r.stderr)
        if r.returncode not in (0, 1):
            # 1 can mean already applied
            print("patch return", r.returncode)
            if r.returncode > 1:
                raise SystemExit(r.returncode)
        print("applied forward_pass.patch")
    head = path.read_text(encoding="utf-8")[:80]
    if "Numpy Llama Forward Pass" not in head:
        raise SystemExit(f"forward_pass.py still wrong: {head!r}")
    if "gzip.decompress" in path.read_text(encoding="utf-8")[:500]:
        raise SystemExit("forward_pass.py still looks like a loader")
    print("forward_pass.py OK")

def cleanup() -> None:
    d = ROOT / ".restore_plain"
    if d.exists():
        for p in d.iterdir():
            p.unlink()
        d.rmdir()
        print("removed .restore_plain")

if __name__ == "__main__":
    assemble_cli()
    restore_forward_pass()
    cleanup()
    print("DONE")
