#!/usr/bin/env python3
import base64, hashlib, shutil
from pathlib import Path
RESTORE = Path("docs/_restore")
TARGETS = {
  "rome_editor.py": ("gguf_knowledge_extractor/core/rome_editor.py", "42b4a0afbe71b8c0"),
  "forward_pass.py": ("gguf_knowledge_extractor/core/forward_pass.py", "14772f7396e8faba"),
  "cli.py": ("gguf_knowledge_extractor/cli.py", "3c0cfe71467c591d"),
}
for name, (dest, prefix) in TARGETS.items():
    parts = sorted(RESTORE.glob(f"{name}.b64.*"))
    if not parts:
        raise SystemExit(f"missing {name} parts")
    data = base64.b64decode("".join(p.read_text().strip() for p in parts))
    digest = hashlib.sha256(data).hexdigest()
    if not digest.startswith(prefix):
        raise SystemExit(f"{name} sha {digest} != {prefix}*")
    Path(dest).write_bytes(data)
    text = data.decode()
    if name == "forward_pass.py":
        if text.lstrip().startswith("import gzip") or "Numpy Llama Forward" not in text:
            raise SystemExit("forward_pass invalid")
    print(f"wrote {dest} {len(data)} sha={digest[:16]}")
for p in [Path("docs/_size_smoke.txt"), Path("docs/VALIDATION_STAGING.md")]:
    if p.exists():
        p.unlink(); print("removed", p)
if RESTORE.exists():
    shutil.rmtree(RESTORE); print("removed docs/_restore")
print("OK")
