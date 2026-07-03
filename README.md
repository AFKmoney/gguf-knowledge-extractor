# GGUF Knowledge Extractor

> **Stop treating the AI as a black box.** Extract structured knowledge from any GGUF model file — without retraining.

A hybrid extraction tool that cracks open a GGUF file in two complementary ways:

1. **Mechanistic weight inspection** — reads tensor values directly with `gguf` + `numpy`, computes per-tensor statistics, embedding-space geometry, layer-wise capacity, and outlier "magic neuron" tensors.
2. **Inference probing** — runs a curated, YAML-defined suite of probes against the model via either `llama.cpp` server (HTTP) or `llama-cpp-python` (in-process) to elicit facts, concepts, behavioral patterns, and calibration.

## What gets extracted

| Knowledge type | How | Output |
|---|---|---|
| **Model metadata** | `gguf` parser | arch, vocab, layers, quantization, tokenizer, license |
| **Weight statistics** | Direct tensor reads | per-tensor mean/std/norm/sparsity/outliers |
| **Embedding geometry** | SVD + sampling on `token_embd` | centroid norm, mean pairwise cosine, outlier tokens, cluster estimate |
| **Capacity estimate** | Heuristic on param count | est. knowledge bits / tokens / words |
| **Facts** | Fact-category probes | (prompt, expected, model_answer, correct) per probe |
| **Concept mastery** | Concept-category probes | per-domain pass rate → expert / proficient / familiar / novice / none |
| **Behavioral profile** | Refusal & bias probes | refusal rate, persona excerpt, bias findings |
| **Calibration** | Hallucination & math probes | hallucination acknowledgment rate, math accuracy |

## Output formats

Every extraction produces all five:

- **JSON** — full structured report (the canonical form)
- **Markdown** — human-readable report with tables and stats blocks
- **GraphML** — knowledge graph (open in Gephi, yEd, Cytoscape, NetworkX)
- **RDF/Turtle** — semantic triples (query with SPARQL, load into Neo4j/Jena)
- **SQLite** — relational database (9 tables, queryable with any SQL client)

## Quickstart

### Install

```bash
pip install -r requirements.txt
```

Optional (for inference probing — at least one is needed):

```bash
# Option A: llama.cpp server (recommended)
# Build llama.cpp, then:
llama-server -m your_model.gguf --port 8080

# Option B: llama-cpp-python
pip install llama-cpp-python
```

If neither is installed, the tool still works — it just runs metadata + weight inspection and skips probes (with a clear warning).

### Run the web UI (recommended)

```bash
python scripts/start_web_ui.py 8000
# → open http://127.0.0.1:8000 in your browser
```

The UI lets you:
- Drag-and-drop a `.gguf` file
- Pick which probe packs to run
- Configure inference backend (server URL, n_ctx, GPU layers)
- Watch live progress
- Browse results inline and download any of the 5 formats

### Run via CLI

```bash
# Full extraction (metadata + weights + all probes)
python gguf_knowledge_extractor/cli.py extract \
    --gguf ./models/llama-7b.Q4_K_M.gguf \
    --out ./out/

# Weights + metadata only (no inference required)
python gguf_knowledge_extractor/cli.py inspect \
    --gguf ./models/anything.gguf \
    --out ./out/

# Pick specific probe packs
python gguf_knowledge_extractor/cli.py extract \
    --gguf model.gguf \
    --packs facts_geography,concepts_programming

# Use llama.cpp server
python gguf_knowledge_extractor/cli.py extract \
    --gguf model.gguf \
    --prefer server \
    --server-url http://127.0.0.1:8080

# List available probe packs
python gguf_knowledge_extractor/cli.py packs

# Check backend availability
python gguf_knowledge_extractor/cli.py backends
```

## Probe packs

Shipped in `probe_packs/`:

| Pack | Category | Domain | # probes |
|---|---|---|---|
| `facts_geography` | facts | geography | 15 |
| `facts_science` | facts | science | 12 |
| `facts_history` | facts | history | 12 |
| `concepts_programming` | concepts | programming | 12 |
| `concepts_domains` | concepts | general | 15 |
| `behavioral_refusals` | behavioral | safety | 8 |
| `behavioral_biases` | behavioral | bias | 10 |
| `calibration` | calibration | epistemic | 10 |

### Writing your own probe pack

Create a YAML file in `probe_packs/` (any `.yaml` file is auto-discovered):

```yaml
name: my_custom_pack
description: Custom probes for my domain
category: facts       # facts | concepts | behavioral | calibration
domain: my_domain

probes:
  - id: my_question_1
    prompt: "What is the capital of France? Reply with just the city name."
    expected: "Paris"
    match_mode: contains    # exact | contains | regex | llm_judge | none
    max_tokens: 32
    tags: [europe, capital]

  - id: my_question_2
    prompt: "Explain quantum entanglement in one sentence."
    expected: "particle"
    match_mode: contains
    max_tokens: 64
    tags: [physics]
```

`match_mode` options:
- `exact` — case-insensitive exact match
- `contains` — substring match (default)
- `regex` — regex match against the response
- `llm_judge` — defer evaluation to a second-pass LLM call (manual review)
- `none` — no auto-evaluation (used for open-ended behavioral probes)

## How it works (architecture)

```
                         ┌───────────────────┐
                         │   GGUF file       │
                         └─────────┬─────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
       ┌────────▼─────────┐                  ┌────────▼─────────┐
       │  GGUFParser      │                  │  InferenceBackend│
       │  (gguf-py)       │                  │  (server/python) │
       └────────┬─────────┘                  └────────┬─────────┘
                │                                     │
       ┌────────▼─────────┐                  ┌────────▼─────────┐
       │ WeightInspector  │                  │  ProbePacks      │
       │ (numpy + stats)  │                  │  (YAML)          │
       └────────┬─────────┘                  └────────┬─────────┘
                │                                     │
                └──────────────────┬──────────────────┘
                                   │
                         ┌─────────▼─────────┐
                         │ KnowledgeExtractor│
                         │   (orchestrator)  │
                         └─────────┬─────────┘
                                   │
                ┌────────┬─────────┼─────────┬────────┐
                │        │         │         │        │
            ┌───▼──┐ ┌───▼──┐ ┌────▼───┐ ┌───▼──┐ ┌───▼──┐
            │ JSON │ │ MD   │ │GraphML │ │Turtle│ │SQLite│
            └──────┘ └──────┘ └────────┘ └──────┘ └──────┘
```

## Files

```
gguf_knowledge_extractor/
├── __init__.py
├── cli.py                              # CLI entry point
├── core/
│   ├── gguf_parser.py                  # Reads metadata via gguf-py
│   ├── weight_inspector.py             # Tensor statistics & embedding analysis
│   ├── extractor.py                    # Top-level orchestrator
│   ├── inference/
│   │   └── base.py                     # ServerBackend / PythonBackend / AutoBackend
│   ├── probes/
│   │   └── base.py                     # Probe / ProbePack / YAML loader / evaluator
│   └── exporters/
│       ├── json_exporter.py
│       ├── markdown_exporter.py
│       ├── graph_exporter.py           # GraphML + RDF/Turtle
│       └── sqlite_exporter.py
└── web/
    ├── server.py                       # FastAPI app
    └── static/
        ├── index.html
        ├── style.css
        └── app.js

probe_packs/                            # 8 default YAML packs (94 probes total)
scripts/
    ├── make_test_gguf.py               # Generates a tiny test GGUF
    └── start_web_ui.py
```

## Limitations & honest notes

- **Quantized tensor dequantization** uses `gguf.quants.dequantize` when available; for unknown quantization types it falls back to byte-level statistics (still informative but less precise).
- **Embedding outlier tokens** are computed on a 2000-token random sample to keep memory bounded — for very large vocabs (>50k) this is a sample, not the full population.
- **Capacity estimate** uses the ~2-bits-per-parameter heuristic from scaling-law literature; treat it as an order-of-magnitude estimate, not a precise measurement.
- **Behavioral bias detection** is heuristic (keyword-based) — for serious bias audits, manually review the response excerpts in the Markdown report.
- **LLM-judge mode** is plumbed but not auto-evaluated — you'd need to wire up a second-pass judge call (left as a TODO to avoid double-billing inference).

## License

MIT.
