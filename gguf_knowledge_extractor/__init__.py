"""
GGUF Knowledge Extractor
========================
Extract knowledge from GGUF model files without retraining.

Hybrid approach:
  - Direct weight/tensor inspection (gguf-py + numpy)
  - Inference-based probing (llama.cpp server or llama-cpp-python)

Outputs:
  - JSON
  - Markdown report
  - Knowledge graph (GraphML + RDF/Turtle)
  - SQLite database

Author: built with GLM/Z.ai
"""

__version__ = "1.0.0"
__all__ = ["core", "web"]
