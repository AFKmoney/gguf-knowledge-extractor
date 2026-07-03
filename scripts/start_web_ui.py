#!/usr/bin/env python3
"""Launch the GGUF Knowledge Extractor web UI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn
from gguf_knowledge_extractor.web.server import app

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"\n  GGUF Knowledge Extractor — Web UI")
    print(f"  → http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
