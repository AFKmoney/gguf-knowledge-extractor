# GGUF Knowledge Extractor — Work Log

---
Task ID: 1
Agent: GLM (main agent)
Task: Build a software tool that extracts all knowledge from a GGUF model file without retraining. Hybrid approach (weight inspection + inference probing), local web UI, all output formats (JSON/Markdown/Knowledge Graph/SQLite), YAML-configurable probe packs, support both llama.cpp server and llama-cpp-python backends.

Work Log:
- Asked the user 8 clarifying questions covering extraction goal, technique, interface, output formats, runtime, probe scope, ship speed, and free-text remarks. User selected: Concepts & skills + "all of it" in remarks, Hybrid, Local web UI, all 4 output formats, Both backends, User-configurable (YAML), Full v1.
- Installed Python dependencies (gguf, fastapi, uvicorn, pydantic, pyyaml, jinja2, networkx, httpx). Attempted llama-cpp-python install but it timed out (compilation required) — designed graceful fallback.
- Created project structure under /home/z/my-project/gguf_knowledge_extractor/ with core/, web/, probe_packs/, scripts/.
- Built GGUFParser class wrapping gguf.GGUFReader, exposing structured metadata (arch, vocab, layers, quantization, tokenizer, tensors).
- Built WeightInspector computing per-tensor stats (mean/std/norm/sparsity/outliers), embedding-space analysis (centroid norm, pairwise cosine, outlier tokens), layer grouping, and capacity estimate (bits/tokens/words per parameter).
- Built inference engine abstraction: ServerBackend (HTTP, OpenAI-compatible), PythonBackend (llama-cpp-python in-process), AutoBackend (auto-select with fallback). All gracefully degrade when unavailable.
- Built YAML-based probe pack system with 8 default packs (94 probes total) covering facts (geography/science/history), concepts (programming/domains), behavioral (refusals/biases), and calibration (hallucination/math/reasoning).
- Built KnowledgeExtractor orchestrator that runs the 3-stage pipeline (parse → inspect → probe) and aggregates results into facts, concepts, behavioral_profile, and calibration structures.
- Built 4 exporters: JSON (full report), Markdown (human-readable), GraphML (NetworkX graph), RDF/Turtle (semantic triples), SQLite (9-table relational schema).
- Built FastAPI web UI with dark-mode frontend (HTML/CSS/JS) for drag-and-drop GGUF upload, pack selection, backend configuration, live progress polling, and 5-format download buttons.
- Wrote CLI entry point with subcommands: extract, inspect (weights-only), packs, backends, web.
- Created test GGUF generator (scripts/make_test_gguf.py) that builds a tiny synthetic llama-arch GGUF with F32 embeddings + F16 layer weights for validation.
- Debugged and fixed: tensor shape reversal (GGUF stores C-order, no reversal needed), quantization inference for mixed-precision models, embedding analysis dimension order.
- End-to-end validated: launched web UI, submitted test GGUF via /api/extract, verified all 5 output formats download correctly with proper content (JSON 36KB, MD 6KB, GraphML 10KB, Turtle 6KB, SQLite 65KB with all 9 tables populated).
- Verified probes are correctly marked "skipped" with reason "no_backend" when no inference engine is available (graceful degradation).

Stage Summary:
- Delivered a working v1.0.0 of the GGUF Knowledge Extractor at /home/z/my-project/gguf_knowledge_extractor/
- All requested features implemented: hybrid extraction, web UI, 5 output formats, YAML probe packs, both backends supported, user-configurable scope
- Tested end-to-end with synthetic GGUF; all exporters produce valid output
- Project structure: 13 Python source files (~2500 LOC), 8 YAML probe packs (94 probes), 1 frontend (HTML/CSS/JS)
- CLI: `python gguf_knowledge_extractor/cli.py extract --gguf model.gguf`
- Web UI: `python scripts/start_web_ui.py 8000` → http://127.0.0.1:8000
- README.md documents architecture, usage, probe pack authoring, and limitations
