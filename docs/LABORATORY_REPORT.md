# Independent Laboratory Report — AFKmoney/gguf-knowledge-extractor

**Role:** Skeptical independent validation (not author self-assessment)  
**Sources:** `/workspace/validation/reports/00`–`08` only (esp. `07` ROME). Metrics not measured elsewhere are marked **INCONCLUSIVE**.  
**Scientific success (end-to-end ROME on this model/config):** **No.**

---

## A. Environment

| Field | Value |
|-------|-------|
| Host | Linux VM (`Linux 6.12.94+` x86_64) |
| CPUs | 8 |
| RAM | ~15 GiB (no swap); peak RSS during ROME ≈ 10.9 GiB |
| GPU | None observed (CPU-only) |
| Python | 3.13.5 |
| Venv | `/workspace/validation/.venv` |
| numpy | 2.5.3 |
| gguf | 0.19.0 |
| llama-cpp-python | 0.3.35 (built from source after `apt install g++`; no cp313 prebuilt wheel) |
| Commit under test (upstream tip) | `40adcae1ff9869c8f62ac030931ea018accab593` — *Merge PR #1 — Scientific correctness pass…* |
| Validation branch | `validation/forward-axis-fix` |
| Validation commits (in order) | `729ffb8` → `9d80e1b` → `8cd9635` → `4f9239f` |

Unit tests on `40adcae`: **12 passed, 1 failed** (MEMIT assert bug). After `4f9239f`: **13 passed, 0 failed**.  
Smoke on `40adcae`: exit 1 — 2 failures on hardcoded `/home/z/my-project/models` (`PermissionError`); other smoke steps did not crash. After `8cd9635`: `cli models list` exits 0.

---

## B. Model

| Field | Value |
|-------|-------|
| File | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` |
| Path | `/workspace/validation/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` |
| Source | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` (huggingface_hub; pristine on disk) |
| Architecture | **llama** |
| Quant label | Q4_K_M; tensor mix **Q4_K (135) + Q6_K (21) + F32 (45)** |
| Size | 668,788,096 bytes |
| SHA-256 | `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0` |
| n_embd / n_layer / n_head / n_kv_head | 2048 / 22 / 32 / 4 |
| ffn / vocab / context | 5632 / 32000 (inferred) / 2048 |
| rope_freq_base / rope_dim / norm_eps | 10000.0 / 64 / ≈1e-5 |
| Tensor count | 201 |

Forward and in-memory ROME operate on **dequantized float32** copies; on-disk GGUF was not modified by successful experiments.

---

## C. Baseline — pre-edit behavior (France capital prompts)

Measured on NumPy forward **after** axis+dequant fixes (`9d80e1b`), tokenizer = llama.cpp SPM + BOS (report `07`).

| Prompt | top-1 | Paris rank / p | Tokyo rank / p |
|--------|-------|----------------|----------------|
| `The capital of France is` | ` the` | 3 / 0.0664 | 628 / 0.0001 |
| `France's capital is` | ` Paris` | 1 / 0.1716 | 558 / 0.0001 |
| `The capital city of France is` | ` a` | 23 / 0.0031 | 972 / ~0 |
| chat: capital of France → `The capital of France is` | ` the` | 3 / 0.0843 | 1204 / 0.0001 |
| `The capital of Germany is` (locality) | ` the` | 116 / 0.0006 | 378 / 0.0002 |
| `The capital of Japan is` (locality) | ` Tokyo` | 313 / 0.0002 | 1 / 0.3206 |
| `The color of the sky is` (locality) | ` blue` | 2864 / ~0 | 6173 / ~0 |

**Fact choice for ROME:** subject France, main prompt `The capital of France is`, target object ` Tokyo` (id 20377). Model does **not** already top-1 Paris on the main prompt (Paris rank 3); Tokyo is a clear wrong object (rank 628, p≈7e-5).

**Pre-fix baseline (reports 03–05b):** NumPy forward either crashed or produced constant top-1 ` >>` (id 5099) with cosine ≈ 0 vs llama.cpp — not usable as a belief model. Those pre-fix logits are **not** claimed as model behavior.

Independent llama.cpp next-token on the same France prompt (05b): top-1 ` Paris` (p≈0.38). Post-`9d80e1b` NumPy still often prefers ` the` (Paris in top-5, rank 2) — residual fp32/Q4 ranking gap, not treated as edit success.

---

## D. Experiment 1 — Forward axis / dequant bugs + fixes

### Hypothesis
Real llama.cpp GGUFs expose weight axis order incompatible with `NumpyLlamaForward` assumptions (`embd[tok]` + `x @ W.T` with `[vocab,dim]` / `[out,in]`). Synthetic smoke GGUFs used the **opposite** layout → false positive.

### Config
- Model: TinyLlama Q4_K_M above  
- API: `GGUFParser` + `NumpyLlamaForward.forward_with_lens`  
- Baseline commit: `40adcae`  
- Branch: `validation/forward-axis-fix`

### Results

| Stage | Verdict | Evidence |
|-------|---------|----------|
| Pre-fix forward (`40adcae`) | **FAIL** | `ValueError` broadcast `(3,32000)` vs `(2048,)` in `_rms_norm`; `dim=32000`, `head_dim=1000` (reports 03–04) |
| Axis normalize only (`729ffb8`) | **PARTIAL** | Forward completes; logits finite; shapes correct — but top-1 stuck on ` >>`; cosine vs llama.cpp ≈ 0.000–0.004; top-5 ∩ = 0 (05, 05b) |
| Dequant layout fix (`9d80e1b`) | **PARTIAL (conditional go)** | Cosine ≈ 0.96–0.996; top-5 ∩ = 1–5; top-1 matches llama.cpp on short prompts; residual gaps on longer prompts attributed to fp32 NumPy vs ggml/Q4_K_M (06) |

**Root cause (06):** `gguf.quants.dequantize` already returns numpy-order arrays; code then `reshape(t.shape)` (ggml order) **scrambled** non-square weights. Axis `.T` restored shapes but not element order. Fix: keep dequantize/GGUFReader layout as-is.

ROME readiness after Experiment 1: **conditional** — usable for experiments needing context-sensitive next tokens; **not** bit-identical to llama.cpp; **not** paper-faithful ROME proof.

---

## E. Experiment 2+ — ROME multi-layer, aggressive SPSA, quant write-back (from 07)

**Method variant:** `rome_rank1_full_matrix_target_spsa` — documented **experimental / not paper-equivalent**.  
**Path:** `Q* GGUF → dequantize → float32 in-memory edit`; write-back rejects non-F16/F32/BF16.  
**Seed:** 0. Default SPSA: iters=12, step=0.05, probe_scale=0.01, kl_weight=0.01, l2_weight=1e-4; ridge=0.01. Layers: 4, 8, 11, 14, 17, 20. Calibration: 8 prompts.

### E0 — Quantized write-back
- **FAIL:** `ValueError: ROME full-matrix patch requires F32/F16/BF16, got Q6_K` on `blk.0.ffn_down.weight` (on-disk type Q6_K). Pristine GGUF not overwritten.

### E1 — Default multi-layer
All layers: `edit_successful` / top-1 Tokyo = **False**. Tokyo Δp ~1e-5; ‖ΔW‖_F ≲ 0.6; constraint residual ‖(W+Δ)k−v*‖ ≈ 1e-7…1e-6 with ridge. Paraphrase top-1: **0/3**. Locality preserved only because edits are near-noop (loc Δarg 0/3). Best weak layer: **4**; worst: **20**.

### E1b/E1c — Aggressive SPSA + H2 + α-scale (layer 4)
| Config | Tokyo p / rank | top-1 | para | locality |
|--------|----------------|-------|------|----------|
| aggressive (iters=32, step=5) | 5.0e-3 / 24 | False | 0/3 | **3/3 flip** |
| capacity_noreg | 4.0e-3 / 21 | False | 0/3 | 3/3 |
| single_key matched to SPSA (α=1) | **8.3e-3 / 10** (best) | False | 0/3 | 3/3 |
| α-scale up to 32 | p falls; pred → `ensa` | False | 0/3 | 3/3 |

Scaling α>1 does **not** yield Tokyo top-1; locality remains broken whenever the edit is large enough to matter.

### Scientific success?
**No.** No run meets the contract (target top-1 or strong controlled edit **and** paraphrase **and** locality). What worked: float in-memory path, rank-1 constraint residual with ridge, reproducible H1/H2, weak probability movement under aggressive SPSA.

---

## F. Failures

| Failure | Where observed | Outcome |
|---------|----------------|---------|
| NumPy forward crash (axis) | 03–04 @ `40adcae` | Broadcast / OOB; ROME blocked |
| Constant garbage top-1 ` >>` | 05–05b @ `729ffb8` | Shape-OK but uncorrelated with llama.cpp |
| Weight scramble after dequant reshape | 06 @ `729ffb8` | Cosine ≈ 0; fixed in `9d80e1b` |
| Singular C / ROME constraint | 07 H1 | Default `C=kkᵀ` without ridge: `ROME constraint is numerically singular`; editor `edit_successful=False` |
| Q write-back | 07 E0 | Hard reject on Q6_K `ffn_down` |
| Constant / weak edit under default SPSA | 07 E1 | Near-noop; never top-1 |
| Aggressive edit destroys locality | 07 E1b/E1c | 3/3 unrelated prompts flip; KL≈4–7 |
| MEMIT unit-test bad assert | 00, 08 | Solver OK; assert algebraically wrong → fixed `4f9239f` |
| ModelManager hardcoded path | 00, 08 | `PermissionError` on `/home/z/...` → fixed `8cd9635` |
| Synthetic smoke false positive | 04 | Synth layout opposite of real GGUF |

Crashes of the process during post-fix ROME runs: **not reported** as process aborts; failures are logical (singular C, reject, unsuccessful edit metrics).

---

## G. Diagnostics — causes with evidence

1. **Axis mismatch (CONFIRMED, 04):** Real `token_embd` via GGUFReader ggml shape `[2048,32000]`; code assumed `[vocab,dim]`. Synth `make_test_gguf.py` writes opposite order → smoke false positive.

2. **Dequant reshape scramble (CONFIRMED primary, 06):** `dequantize` returns correct numpy layout; `reshape(t.shape)` permutes elements on non-square tensors. Flat weight cosine vs correct load ≈ 0 after axis-only fix. Explains constant ` >>` despite finite logits.

3. **H1 covariance singularity (CONFIRMED, 07):** Single-key `C=kkᵀ` rank-deficient; without ridge `compute_edit` fails. With 8 calibration keys, `C` rank_est=8 ≪ 5632; ridge=1e-2 conditions the solve. Constraint residual then PASS (~1e-7…1e-6).

4. **H2 C-weighted vs SPSA single-key (CONFIRMED, 07):** Same aggressive `v*`: C-weighted apply Tokyo p≈0.005 (rank 24) vs single-key matched p≈0.008 (rank 10).

5. **Tokenizer mismatch (CONFIRMED, 07):** CausalTracer ids omit BOS vs SPM `[1, …]`. Eval used SPM+BOS.

6. **Quant persistence impossible by construction (CONFIRMED, 06_quantization_pathway + 07 E0):** In-memory float edit ≠ quantized GGUF patch; `RomeEditor._write_modified_gguf` hard-rejects Q*.

7. **SPSA ≠ paper ROME (documented):** Target optimization is SPSA, not autograd/Adam as in the paper; method flag `paper_equivalent: False`.

8. **MEMIT test bug (CONFIRMED, 00/08):** Assert `delta@A ≈ residual + delta@A` iff residual≈0 — not the regularized normal equation. Independent check: normal equation holds (~4.8e-7); residual≈0 does not (~0.11).

9. **Residual NumPy↔llama.cpp top-1 gaps (PARTIAL, 06):** After fix, cosine high but some longer prompts disagree on argmax — consistent with fp32 vs Q4_K_M kernels; no second minimal fix applied without stronger evidence.

---

## H. Corrections

Only the following commits are in scope for this validation branch (justified by measured failures above):

| Commit | Message | Justification |
|--------|---------|---------------|
| `729ffb83a68d02cc6b99bfc7fbc3fcf167777de3` | Fix GGUF forward weight axis orientation for real llama.cpp layouts | Stops crash; restores `dim=2048`, `head_dim=64`, embd `[vocab,dim]`, non-square `[out,in]`. Necessary but **insufficient** alone (05b: still garbage logits). |
| `9d80e1b94abce0e246478bb6b32e2c5ab26f42d9` | Fix GGUF tensor load: keep dequantize/GGUFReader layout | Removes scramble; restores cosine ≈ 0.96–0.996 vs llama.cpp. Required for any honest next-token / ROME experiment. |
| `8cd9635aa82eecb9677bbff54acf6f0af8fda2dc` | Fix ModelManager default models dir to portable `./models` | Removes non-portable `/home/z/...` smoke/CLI failure. Hygiene only; does not affect ROME math. |
| `4f9239f3278c9213d24c163bbd9b70efbcb00683` | Fix MEMIT unit test: drop algebraically wrong residual assert | Test correctness only; solver unchanged. pytest 13/13. Does **not** upgrade MEMIT to paper-complete multi-layer validation. |

No other commits are claimed as part of this independent fix set.

---

## I. Final validation

| Criterion | Status | Brief explanation |
|-----------|--------|-------------------|
| Equation-level ROME math | **PASS** | Unit tests: rank-1 constraint / update / reference solver (00, 08). With ridge, ‖(W+Δ)k−v*‖ ≈ 1e-7…1e-6 on real TinyLlama (07 E3). |
| Equation-level MEMIT math | **PARTIAL** | Regularized normal equation holds under independent check; unit test fixed (`4f9239f`). Multi-layer residual propagation / paper-complete MEMIT **not** exercised → do not claim full MEMIT validation. |
| NumPy forward correctness vs llama.cpp | **PARTIAL** | Post-`9d80e1b`: cosine ≈ 0.96–0.996, top-5 ∩ 1–5, short-prompt top-1 match. Not bit-identical; longer prompts can disagree on argmax. Pre-fix: **FAIL**. |
| In-memory ROME edit success | **PARTIAL** | Path runs; aggressive SPSA moves Tokyo to rank ~10 / p≈8e-3; **never top-1** on target prompt. Default SPSA ≈ noop. |
| Paraphrase generalization | **FAIL** | 0/3 paraphrase top-1 across all measured settings (07). |
| Locality | **FAIL** | Preserved only when edit is near-noop; collapses 3/3 when ‖ΔW‖ ≳ 30–100 (07). |
| Paper-faithfulness / SPSA | **PARTIAL** | Rank-1 algebra + residual OK with ridge; SPSA target opt ≠ paper autograd; H1/H2 gaps; `paper_equivalent: False`. |
| Quantized GGUF persistence | **FAIL** | Hard reject on Q6_K `ffn_down` (07 E0). Float workspace edit ≠ quantized file edit. |
| Smoke / CLI hygiene (ModelManager) | **PASS** | After `8cd9635`, `models list` works with `./models`. Pre-fix path was environment FAIL. |

### Bottom line

This laboratory does **not** upgrade ambiguous outcomes to success. End-to-end scientific success for ROME on TinyLlama Q4_K_M with the shipped `rome_rank1_full_matrix_target_spsa` pipeline is **No**: no top-1 rewrite with paraphrase and locality; quantized persistence impossible; method not paper-equivalent. The validation branch **does** establish (1) real-GGUF forward can be made shape-correct and largely correlated with llama.cpp after two load bugs, (2) equation-level ROME algebra with ridge is numerically sound, and (3) behavioral editing on this config remains weak or unlocalized under the tested hyperparameters.

---

*Artifacts:* reports `00`–`08` under `/workspace/validation/reports/`; ROME JSON `07_rome_experiments.json`; branch `validation/forward-axis-fix`.
