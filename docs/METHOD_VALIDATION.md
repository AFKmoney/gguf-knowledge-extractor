# Scientific Method Validation Contract

This project now treats a method name as a reproducibility claim, not a UI label.
A method is **validated** only when its implementation matches the defining
mathematical intervention/update and has deterministic equation-level tests plus
model-level behavioral tests.

## Reference contracts

| Method | Defining requirement | Current gate |
|---|---|---|
| ROME | rank-one constrained least-squares update using key second moment `C = E[kk^T]`; target value `v*` must be defined independently of a single neuron-column overwrite | equation test + model edit success/locality test |
| MEMIT | batch update across selected MLP layers, with calibration covariance and propagated target residuals | batch linear-solve test + multi-layer edit benchmark |
| Task Arithmetic | task vector is `theta_ft - theta_base`; arithmetic is elementwise in a common parameterization | exact vector identity test |
| TIES | trim low-magnitude deltas, elect sign, then merge only sign-consistent entries | deterministic merge test |
| DARE | Bernoulli drop on task deltas followed by `1/(1-p)` rescaling | seeded reproducibility + expectation test |
| Wanda | importance `|W_ij| E|x_j|`, pruning performed per output row | row-wise mask test + perplexity benchmark |
| SmoothQuant | compute channel scales from activation/weight maxima and apply the paired transforms `W' = W diag(s)`, `x' = diag(s)^-1 x` | algebraic equivalence test + quantization benchmark |
| LEACE | closed-form concept-erasure operator fitted from representation/concept statistics | requires covariance/label data and classifier leakage evaluation |
| RepE | estimate a representation direction from a specified dataset and intervene on activations or an explicitly derived equivalent weight transform | direction reproducibility + steering benchmark |
| Dynamic steering | vector must actually be consumed by the runtime at the specified layer; GGUF metadata alone is not execution | runtime integration test |
| Hidden-state distillation | teacher/student activations must be aligned with compatible tokenization/architecture and regression objective | hidden-state error + downstream evaluation |
| Causal scrubbing | interventions must implement the stated causal graph/hypothesis and use an outcome metric, not a binary prediction flip alone | resampling/intervention suite |
| Constitutional surgery | value direction must be measured from a defined preference/constitution dataset and the intervention evaluated for intended and collateral behavior | paired benchmark |

## Hard rule

A feature that cannot satisfy its gate is reported as **prototype / heuristic**
and is not described as an implementation of the paper method.

This distinction matters especially for model editing: causal localization does
not by itself prove that editing the localized layer will succeed. Empirical work
has found weak correlation between causal-tracing localization and edit success,
so both must be measured independently.

## Reproducibility requirements

Every stochastic method must expose a seed. Reports must include:

- method and exact variant;
- model architecture and tensor dtypes;
- calibration dataset identity/hash and sample count;
- seed;
- hyperparameters;
- target metric and baseline metric;
- edit success, specificity/locality, and generalization where applicable;
- numerical residuals for closed-form constraints;
- whether quantization changed during the operation.

## Primary references

- Meng et al., **Locating and Editing Factual Associations in GPT** (ROME), arXiv:2202.05262.
- Meng et al., **Mass-Editing Memory in a Transformer** (MEMIT), arXiv:2210.07229.
- Ilharco et al., **Editing Models with Task Arithmetic**, arXiv:2212.04089 / ICLR 2023.
- Yadav et al., **TIES-Merging: Resolving Interference When Merging Models**, arXiv:2306.01708 / NeurIPS 2023.
- Sun et al., **A Simple and Effective Pruning Approach for Large Language Models** (Wanda), arXiv:2306.11695.
- Xiao et al., **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**, arXiv:2211.10438.
- Belrose et al., **LEACE: Perfect Linear Concept Erasure in Closed Form**, arXiv:2306.03819.

The repository's `core/research_math.py` contains the equation-level primitives
for the methods whose defining algebra can be tested independently of GGUF I/O.
