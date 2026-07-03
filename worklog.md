# GGUF Knowledge Extractor — Work Log

---
Task ID: 1
Agent: GLM (main agent)
Task: Build v1 — extract all knowledge from a GGUF file without retraining. Hybrid (weight inspection + inference probing), local web UI, all 5 output formats, YAML probe packs, both backends supported.

Work Log:
- Asked user 8 clarifying questions; user selected: Concepts & skills + "all of it" in remarks, Hybrid, Local web UI, all 4 output formats, Both backends, User-configurable (YAML), Full v1.
- Installed Python deps (gguf, fastapi, uvicorn, pydantic, pyyaml, jinja2, networkx, httpx). llama-cpp-python timed out (compilation) — designed graceful fallback.
- Built GGUFParser (gguf-py wrapper), WeightInspector (numpy stats + embedding analysis + capacity estimate).
- Built inference engine: ServerBackend (HTTP) + PythonBackend (llama-cpp-python) + AutoBackend (fallback).
- Built 8 YAML probe packs (94 probes total): facts (geography/science/history), concepts (programming/domains), behavioral (refusals/biases), calibration.
- Built KnowledgeExtractor orchestrator, 4 exporters (JSON, Markdown, GraphML+Turtle, SQLite 9-table schema), FastAPI web UI with drag-drop frontend, CLI with 5 subcommands.
- Built test GGUF generator (synthetic llama-arch with F32 emb + F16 layers).
- Debugged: tensor shape reversal (GGUF stores C-order, no reversal), quantization inference for mixed-precision, embedding analysis dim order.
- End-to-end validated: web UI submits, all 5 outputs produced, all formats download correctly.

Stage Summary:
- Delivered working v1.0.0 with all requested features. 13 Python files (~2500 LOC), 8 YAML probe packs, full web UI.

---
Task ID: 2 (v2 upgrade)
Agent: GLM (main agent)
Task: Build v2 — ROME/MEMIT-style direct knowledge attribution: MLP memory decomposition, attention head analysis, per-fact attribution, knowledge fingerprint.

Work Log:
- Built core/mlp_analyzer.py: For each transformer block, decomposes the MLP into "memory neurons" (key × value rank-1 components). For each top-K neuron, finds top-K vocabulary tokens that activate it (via embedding_matrix @ key_vector) and top-K tokens whose embedding is most aligned with the value vector (cosine similarity). Computes memory strength = ||key|| × ||value|| per neuron, layer strength, and concentration.
- Built core/attention_analyzer.py: For each attention head, computes copy_score (how close V·O is to identity), induction_score (low-rank Q·Qᵀ via SVD entropy), specialization (SVD concentration), top singular value. Detects copy heads and induction heads per Anthropic's interpretability heuristics.
- Built core/knowledge_attribution.py: Orchestrates MLP + attention analysis. Produces per-fact attribution via weight-only heuristic (finds the layer+neuron whose key vector is most strongly activated by tokens in the prompt). Computes a SHA-256 knowledge fingerprint from the global top-50 neurons.
- Updated core/extractor.py: Added `do_attribution` parameter to extract() pipeline. New stage 5 runs the attributor after probes complete. KnowledgeReport dataclass gets new `attribution` field.
- Updated core/exporters/markdown_exporter.py: New Section 8 "Knowledge Attribution (v2)" with fingerprint banner, per-layer MLP memory map, top copy/induction heads tables, per-fact attribution table.
- Updated core/exporters/sqlite_exporter.py: Added 5 new tables: mlp_layers, mlp_neurons, attention_heads, fact_attributions, knowledge_fingerprints. Total now 14 tables.
- Updated core/exporters/graph_exporter.py: Added attribution nodes (fingerprint, neurons, attributions) and edges (has_fingerprint, contains_neuron, attributed_by, located_at) to both GraphML and RDF/Turtle outputs.
- Updated gguf_knowledge_extractor/cli.py: Added --attribution and --attribution-top-k flags to extract subcommand. Added new `attribute` subcommand that runs weights+attribution only (no inference needed).
- Updated web/server.py + web/static/{index.html,app.js}: New "v2: Knowledge attribution" checkbox in extract form, new attribution_top_k field, server passes both to background worker. Job view renders full attribution visualization: fingerprint banner, stats grid, top 5 neurons table, per-layer MLP memory map, top copy/induction heads tables, per-fact attribution table.
- Updated scripts/make_test_gguf.py: Now generates a complete transformer block with attn_q, attn_k, attn_v, attn_output, ffn_up, ffn_down per layer (was previously missing attn_k/v/output). Added tokenizer scores array.
- Tested end-to-end with synthetic GGUF: 3 MLP layers analyzed, 30 top neurons extracted (10/layer), 6 attention heads analyzed, 15 facts attributed (weight_only method since no inference backend), knowledge fingerprint computed (SHA-256). All 5 output formats verified to contain attribution data. SQLite has 5 new tables populated correctly.

Stage Summary:
- Delivered v2 with ROME/MEMIT-style MLP decomposition, attention head specialization detection, per-fact attribution, and knowledge fingerprint.
- Total codebase now: 16 Python files (~3500 LOC), 14-table SQLite schema, full web UI with attribution visualization.
- v2 attribution works WITHOUT inference (weight-only path) — every model can be attributed even without a backend.
- New CLI subcommand: `attribute` (no inference required).
- New CLI flag: `--attribution` on `extract` to enable v2 alongside v1.
- End-to-end validated on synthetic GGUF; all 5 output formats contain attribution data.
