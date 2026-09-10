#!/usr/bin/env python3
"""Assemble docs/_restore/*.partNN into target source paths; delete staging."""
from pathlib import Path
import hashlib

RESTORE = Path("docs/_restore")
TARGETS = {
    "rome_editor.py": "gguf_knowledge_extractor/core/rome_editor.py",
    "forward_pass.py": "gguf_knowledge_extractor/core/forward_pass.py",
    "cli.py": "gguf_knowledge_extractor/cli.py",
    "model_manager.py": "gguf_knowledge_extractor/core/model_manager.py",
    "LABORATORY_REPORT.md": "docs/LABORATORY_REPORT.md",
}
EXPECTED = {
    "forward_pass.py": {
        "sha256": "14772f7396e8faba55ef32da30b528593e4396becbad8e9a30e3cd0ab6d99d38",
        "min_lines": 600,
        "must_contain": "Numpy Llama Forward",
        "must_not_start": "import gzip",
    }
}

def parts_for(name: str):
    files = sorted(RESTORE.glob(f"{name}.part*"))
    if not files:
        raise SystemExit(f"missing parts for {name}")
    return files

def main():
    for name, dest in TARGETS.items():
        blobs = [p.read_text() for p in parts_for(name)]
        text = "".join(blobs)
        if name in EXPECTED:
            exp = EXPECTED[name]
            if exp["must_contain"] not in text:
                raise SystemExit(f"{name} missing marker")
            if text.lstrip().startswith(exp["must_not_start"]):
                raise SystemExit(f"{name} looks like gzip stub")
            lines = text.count("\n") + (0 if text.endswith("\n") else 1)
            if lines < exp["min_lines"]:
                raise SystemExit(f"{name} too few lines: {lines}")
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest != exp["sha256"]:
                raise SystemExit(f"{name} sha mismatch {digest}")
        Path(dest).write_text(text)
        print(f"wrote {dest} ({len(text)} chars)")
    for p in list(RESTORE.glob("*")):
        if p.name != "assemble_restore.py":
            p.unlink()
    for extra in [Path("docs/_size_smoke.txt"), Path("docs/VALIDATION_STAGING.md")]:
        if extra.exists():
            extra.unlink()
            print("removed", extra)
    # remove self last
    self = Path("docs/_restore/assemble_restore.py")
    if self.exists():
        self.unlink()
    if RESTORE.exists() and not any(RESTORE.iterdir()):
        RESTORE.rmdir()
    print("OK")

if __name__ == "__main__":
    main()
