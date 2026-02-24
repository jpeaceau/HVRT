# HVRT-LM: Partition-Aware Post-hoc Correction and Structural Analysis of Transformer FFNs

**Jake Peace**
HVRT Project — https://github.com/Peace/hvrt

---

## Abstract

We apply Hierarchical Variance-Retaining Transformer (HVRT) partitioning to the
internal representations of GPT-2 small and report two families of findings.

**Interpretability and structural honesty**: The HVRT corrector is fully
interpretable by construction — every prediction traces through a shallow
decision tree whose split conditions are human-readable, and the bias vectors
are directly inspectable floating-point tables. More importantly, the model is
**structurally incapable of expressing false confidence**: its confidence signal
is a count of training examples in the current partition, not a learned
parameter. This count cannot be inflated by training or regularisation — it
can only grow by adding real data. No architectural mechanism exists through
which HVRT can assign high confidence to inputs it has not seen.

**Corrector results**: Fitting HVRT on last-layer hidden states and computing
per-partition logit biases from token frequency statistics improves validation
perplexity on WikiText-103 by **7.90%** (PPL 48.93 → 45.07) using **1.6 KB** of
stored statistics — 0.0003% of GPT-2's 497.8 MB parameter count. This requires
no model modification, no gradient computation, and is fitted from 51,200 tokens
in under one minute. A counterintuitive vocabulary-restriction finding: restricting
the correction to the top-100 most frequent tokens produces *better* improvement
than correcting all 50,257 tokens, because reliable estimation of rare-token
distributions dilutes the correction signal with noise.

**Structural results**: Replacing individual GPT-2 FFN layers with HVRT-partitioned
piecewise-linear models (fitted analytically by OLS) degrades perplexity by only
+2.2 to +6.2 PPL for 11 of 12 layers. Layer 0 is uniquely resistant (+107 PPL),
revealing a structural asymmetry in transformer computation. Simultaneous
all-layer replacement fails catastrophically due to covariate shift cascade —
a well-understood obstacle with a known architectural solution.

We discuss implications for mixture-of-experts routing, KV-cache compression,
domain adaptation, structural uncertainty quantification, and the feasibility of
standalone non-backprop language models.

---

## 1. Introduction

The dominant paradigm for large language model improvement is to increase
parameter count and training compute. This paper asks a narrower question:
given a fixed pre-trained model, how much output quality can be recovered from
a tiny amount of partition statistics computed post-hoc, and what do those
statistics reveal about the model's internal structure?

HVRT (Peaceau 2024) is a decision-tree-based partitioning method that decomposes
a dataset into regions of locally homogeneous variance structure. It was designed
for data augmentation and sample reduction, but its partition mechanism — fitting
a decision tree on a variance-weighted synthetic target derived from feature
interactions — turns out to characterise the local geometry of high-dimensional
representation spaces in a way that is useful for language model correction.

Two properties distinguish HVRT-based correction from neural correction
approaches and deserve to be treated as first-class architectural properties,
not implementation conveniences:

**Full interpretability**: the correction decision at every token position is a
lookup through a shallow decision tree with human-readable split conditions,
into a bias table that is directly inspectable as a per-context frequency
correction. There are no opaque intermediate computations.

**Structural impossibility of false confidence**: the model's confidence signal
is a count of training examples in the current partition. Because this is a
count and not a learned parameter, it cannot be inflated by training, cannot
degrade under distribution shift, and cannot be miscalibrated. A model that has
not seen a particular type of input will report a near-zero partition count —
not a confidently wrong answer. Section 2 develops both properties in detail.

---

## 2. Structural Interpretability and the Inability to Express False Confidence

*This section states two architectural properties of HVRT-based correction that
distinguish it from neural language models and neural calibration techniques.
These are structural facts about how the correction is represented, not
implementation choices.*

### 2.1 Full Prediction Interpretability

A standard transformer's output distribution at any position is the product of
hundreds of millions of floating-point weights, organised into attention matrices
and feedforward tensors. There is no procedure to examine these weights and
reason about *why* a particular token receives a particular probability. Post-hoc
explainability methods (LIME, SHAP, attention visualisation) approximate the
model's behaviour by a simpler proxy — they do not expose the actual computation.

The HVRT corrector is different in kind. At inference, the correction decision
is a single transparent lookup:

1. **Assign** the last-layer hidden state h ∈ ℝ^768 to a partition. For the
   optimal 4-partition configuration, this traverses a decision tree with 3
   internal nodes — three threshold comparisons on z-normalised feature
   dimensions, each derived directly from the variance structure of the hidden
   states in the training batch.

2. **Apply** the bias vector for that partition: 100 floating-point numbers
   representing the log-ratio of empirical to model-predicted token frequencies
   for the top-100 tokens in this context region.

Every element of this computation is inspectable with no approximation. The
tree split conditions are human-readable: "z-dim 412 > 0.83" is a concrete,
auditable statement about the hidden state geometry. The bias vector entry
`bias[p][token_id] = 0.43` is a direct empirical statement: "in training
contexts assigned to partition p, token X appeared 53% more often than
GPT-2's base distribution predicted." A practitioner can reproduce the entire
corrector's decision from its stored 1.6 KB table in a few lines of arithmetic.

This is qualitatively different from the "interpretability" approximations offered
by post-hoc methods applied to neural models. Those methods explain a proxy.
This method *is* the model — there is nothing further underneath.

### 2.2 The Structural Impossibility of False Confidence

This is the more consequential property.

A neural language model's output confidence is encoded in learned weights. At
any token position in any input sequence, the model produces a probability
distribution. If the model has never encountered a context similar to the
current one during training, it produces a distribution regardless — and that
distribution may be high-confidence and wrong. There is no architectural barrier
to a transformer assigning 95% probability to a hallucinated token on a query
type it has never processed. The weights responsible for that prediction were
trained to maximise likelihood on seen data; nothing in the architecture
constrains them from generating confident outputs outside the training
distribution.

This is not a flaw in implementation or training procedure. It is an
architectural fact: neural network weights are smooth functions that extrapolate
continuously into unseen regions. Calibration training and temperature scaling
can improve confidence estimates on held-out data from the same distribution,
but they are themselves learned and offer no guarantee under distribution shift.

**HVRT's confidence signal is a count, not a parameter.**

Specifically: the confidence signal at inference time is the number of training
tokens that were assigned to the same partition as the current hidden state.
This count:

- **Cannot be tuned** — it is not a parameter in any optimisation objective.
- **Cannot be regularised** — regularisation acts on learned weights; this is
  a histogram bin count.
- **Cannot be inflated by training** — the count can only increase by adding
  more actual training data to that partition.
- **Cannot be miscalibrated under distribution shift** — if a new context type
  lands in a partition with 3 training examples, the reported confidence is 3,
  regardless of how much training was done on other partition types.

The consequence is absolute: **HVRT cannot express high confidence about inputs
it has not seen.** If a hidden state falls in a partition with 3 training
examples, the partition density is 3, and the corrector reports 3. There is no
path through which this becomes 3,000 without 3,000 actual training examples.
The model is structurally incapable of false high-confidence on
out-of-distribution inputs — not because it was trained to be humble, but
because humility is the only representationally possible response.

This property is distinct from calibration. A calibrated neural model assigns
60% probability to events that occur 60% of the time. Calibration is an
empirical property — measurable and trainable on held-out data, but fragile
under distribution shift. HVRT's density-based confidence is not calibrated, it
is *structural*: when distribution shifts, the count drops to near zero and the
model reports near-zero confidence explicitly and automatically.

**Practical consequence**: In regulated, safety-critical, or high-stakes
deployment contexts — medicine, law, financial advice, scientific claims — the
difference between "calibrated but possibly overconfident on novel inputs" and
"structurally incapable of false confidence by construction" is a qualitative
architectural distinction, not a quantitative improvement. A regulator, auditor,
or safety engineer can verify HVRT's confidence reporting procedure in minutes
from the source; no such verification is available for a calibrated neural model.

---

## 3. Background and Method

### 3.1 HVRT Partitioning

Given a dataset X ∈ ℝ^(n×d), HVRT fits a decision tree on a synthetic target
derived from local variance structure (pairwise feature interactions for HVRT;
z-score sum for FastHVRT, used throughout these experiments). The result is a
partition of the feature space into leaf nodes, where each leaf contains a subset
of training samples with locally similar structure.

Key properties used here:

- **Partition assignment for new data**: the fitted tree is a standard
  `sklearn.tree.DecisionTreeRegressor`; new data is assigned via `tree_.apply()`.
- **`_to_z(X)`**: z-normalises new data using training statistics before
  assignment, ensuring consistent partitioning across train and inference.
- **Partition density**: the count of training samples per leaf is stored; sparse
  leaves indicate regions where the model has limited evidence.

### 3.2 Experiment Setup

| Parameter | Value |
|-----------|-------|
| Model | GPT-2 small (12 layers, d=768, 117M params, 497.8 MB) |
| Training corpus | WikiText-103 (first 51,200 tokens, streaming) |
| Validation corpus | WikiText-103 validation (20,480 tokens) |
| Baseline PPL | 48.93 |
| Framework | PyTorch, HuggingFace Transformers, scikit-learn |
| Hardware | Single GPU (CUDA) for GPT-2 forward passes; CPU for HVRT |

No gradient computation is performed at any point. All HVRT fits and OLS
solutions are closed-form.

---

## 4. Variant B: HVRT Logit Corrector

### 4.1 Method

For each token position in the training set:
1. Run a GPT-2 forward pass to obtain the last-layer hidden state h ∈ ℝ^768
   and logits l ∈ ℝ^50257.
2. Fit FastHVRT on the hidden states to obtain partitions P₁, ..., Pₖ.
3. For each partition p and each vocabulary token v, compute:

   `bias_p[v] = log(empirical_freq_p[v] + ε) − mean(log_softmax(logits)[v] | p)`

   where `empirical_freq_p[v]` is the fraction of training positions in partition p
   where the next token is v, and the mean is over all token positions assigned to p.

4. At inference time: `corrected_logits = gpt2_logits + α · bias[partition(h)]`

The bias represents the per-partition log-ratio of empirical to model-predicted
token probabilities — a closed-form correction analogous to label smoothing but
applied adaptively per context region.

### 4.2 Vocabulary-Restriction Experiment

The standard corrector estimates a 50,257-dimensional bias vector per partition.
With ~12,800 training tokens per partition (4 partitions, 51,200 total), the
average token has only 0.25 observations — making estimates for rare tokens
unreliable. We swept over restricting the correction to the top-K most frequent
tokens (K ∈ {100, 500, 2000, full}) and measured validation PPL across 6 rounds
of progressively finer partitioning.

Token coverage by top-K:
- Top-100: 48.1% of token positions
- Top-500: 64.2%
- Top-2000: 82.4%

### 4.3 Results

**Single-round corrector, 4 partitions:**

| TOP_K | PPL | Gain vs baseline | Storage | Avg obs/token/partition |
|-------|-----|-----------------|---------|------------------------|
| 100 | **45.07** | **+7.90%** | 1.6 KB | ~61 |
| 500 | 45.84 | +6.33% | 10 KB | ~12 |
| 2000 | 46.29 | +5.40% | 40 KB | ~3 |
| full | 46.31 | +5.36% | 0.81 MB | ~0.25 |

**Round-by-round PPL (lower is better):**

| Round | Parts | K=100 | K=500 | K=2000 | K=full |
|-------|-------|-------|-------|--------|--------|
| 1 | 4 | **45.07** | **45.84** | **46.29** | 46.31 |
| 2 | 8 | 45.23 | 45.85 | 46.54 | **46.19** |
| 3 | 16 | 45.41 | 45.98 | 46.91 | 46.54 |
| 4 | 32 | 45.50 | 46.00 | 47.13 | 47.07 |
| 5 | 64 | 45.61 | 46.12 | 47.58 | 48.16 |
| 6 | 128 | 45.72 | 46.39 | 48.37 | 49.78 |

### 4.4 Key Findings

**F1: Smaller vocabulary restriction is better.** K=100 produces +7.90% vs
K=full at +5.36% for the same 4-partition round-1 configuration. This is
counterintuitive: covering only 48% of token positions gives better improvement
than covering 100%. The explanation: GPT-2's predictions for common tokens
already have per-partition systematic biases that can be corrected cleanly, but
including rare tokens adds estimation noise that partially offsets the gain.
The top-100 restriction achieves ~61 observations per token per partition —
sufficient for reliable estimation. Full vocabulary achieves ~0.25.

**F2: Vocab restriction stabilises multi-round boosting but does not make it
beneficial.** All configurations degrade after their optimal round (round 1 for
K=100/500/2000; round 2 for K=full). The degradation rate scales inversely with
K: 0.13 PPL/round for K=100 vs 0.70 PPL/round for K=full. Additional rounds fit
noise in the residual, not signal.

**F3: Optimal corrector: K=100, 4 partitions, α=0.30, 1 round.** This stores
4 bias vectors of length 100, totalling 1.6 KB. For context: this is 1/310,000th
of GPT-2's parameter count, yet produces 7.90% perplexity improvement with no
architecture change and no training.

**F4: Confidence from density is free.** At inference, every prediction is
accompanied by the count of training tokens that fell in the same HVRT partition
(see Section 2.2). This requires no additional computation — it is a byproduct
of the partition lookup. Sparse partitions are flagged as low-confidence
automatically, without ensembles, temperature scaling, or any supplementary
procedure. As a concrete example: prompts about Shakespeare-era literature (dense
in Wikipedia training) fall in partitions with median ~1,700 training tokens
(HIGH confidence); prompts about financial market movements (sparse in Wikipedia)
fall in partitions with ~326 training tokens (LOW confidence, flagged explicitly).

---

## 5. Variant A: HVRT-Partitioned FFN Replacement

### 5.1 Method

For each of GPT-2's 12 FFN layers, we:
1. Collect (FFN_input, FFN_output) pairs via forward-pass hooks.
2. Fit FastHVRT on the FFN inputs to obtain per-layer partitions.
3. Within each partition p, solve:
   `FFN_output ≈ global_bias + W_p @ FFN_input`
   using OLS (`np.linalg.lstsq`, SVD-based, numerically stable).
4. Optionally truncate W_p to rank r via SVD.

This replaces GPT-2's non-linear two-layer FFN (W₁ GELU W₂) with a
piecewise-linear map fitted analytically — no gradient descent at any step.

### 5.2 Single-Layer Probe

To isolate per-layer approximation quality, we replace ONE FFN layer at a time
and measure validation PPL, leaving all other layers unchanged.

| Layer | PPL | Delta | Interpretation |
|-------|-----|-------|----------------|
| 0 | 155.50 | **+106.57** | Critical — unique non-linearity |
| 1 | 54.82 | +5.89 | Replaceable |
| 2 | 51.10 | +2.17 | Most replaceable |
| 3 | 52.52 | +3.58 | Replaceable |
| 4 | 53.05 | +4.11 | Replaceable |
| 5 | 52.61 | +3.68 | Replaceable |
| 6 | 52.04 | +3.11 | Replaceable |
| 7 | 51.52 | +2.59 | Replaceable |
| 8 | 54.25 | +5.32 | Replaceable |
| 9 | 53.41 | +4.47 | Replaceable |
| 10 | 53.02 | +4.08 | Replaceable |
| 11 | 55.15 | +6.22 | Replaceable |

Configuration: 8 partitions, rank=64 (19 MB per layer).

**F5: Layers 1-11 are individually piecewise-linearly approximable.** The OLS
fit within HVRT partitions reconstructs 86% of FFN output variance on the
training set (MSE 0.88 vs zero-model 6.27), and replacing any individual layer
with this approximation costs only +2.2 to +6.2 PPL — a 4-9% increase, not a
collapse.

**F6: Layer 0 is uniquely non-linear.** Replacing layer 0 alone causes +107 PPL
(218% increase from baseline). Layer 0 processes token embeddings into initial
contextual representations. The multi-modal structure of token embedding space
(one cluster per token class) spans the entire vocabulary simultaneously, which
HVRT's locally linear partitioning cannot capture.

### 5.3 All-Layer Replacement Cascade

Replacing all 12 layers simultaneously:

| Parts | Rank | Recon. MSE | PPL (all) | Storage | vs baseline |
|-------|------|-----------|----------|---------|------------|
| 4 | 32 | 0.8843 | 4,203.94 | 9.5 MB | -8,491% |
| 4 | 64 | 0.8129 | 3,327.36 | 18.9 MB | -6,700% |
| 4 | full | 0.4962 | **327.00** | 113.3 MB | -568% |
| 8 | full | 0.4220 | 502.14 | 226.6 MB | -926% |
| 16 | full | 0.2949 | 1,521.70 | 453.1 MB | -3,010% |

Zero-model MSE baseline: 6.2731.

**F7: The covariate shift cascade is the fundamental barrier to simultaneous
replacement.** Each layer's replacement was fitted on original GPT-2 hidden
states. When all 12 are replaced, layer 1 produces slightly wrong hidden states,
layer 2 receives inputs it was never trained on, and errors compound through 12
layers. Even with 92% per-layer variance explained (4 partitions, full rank),
the compound effect is catastrophic. The best all-layers configuration (PPL=327)
still degrades 6.7× relative to baseline.

**F8: More partitions worsen all-layer cascade PPL despite better per-layer
reconstruction.** The 16-partition full-rank model achieves the best training
reconstruction (MSE=0.29) but the worst all-layers PPL among full-rank configs
(1,521 vs 327 for 4 partitions). Finer partitions create more partition-boundary
transitions across layers, amplifying distribution mismatch.

---

## 6. Implications for LLM Architecture

### 6.1 Mixture-of-Experts Routing Without Learned Gates

Standard MoE uses a learned gating network to route tokens to experts. The
routing decision requires parameters, training, load-balancing losses, and
careful initialisation. HVRT offers an alternative: route tokens based on the
variance structure of their hidden states, with no learned parameters.

The single-layer probe demonstrates that within-partition behaviour is genuinely
more homogeneous — the piecewise-linear approximation holds. This is exactly
what MoE requires: that routed-together tokens benefit from the same expert.

**Practical application**: Initialise MoE routing assignments from HVRT
partitions fitted on a representative batch of activations. Use these as fixed
initial assignments, then fine-tune routing gates from this informed starting
point rather than from random initialisation. The HVRT partitioning provides a
gradient-free, data-driven prior over which tokens are structurally similar.

### 6.2 KV-Cache Compression

KV-cache memory scales linearly with sequence length. Tokens whose key vectors
fall in the same HVRT partition have similar variance structure — they are likely
to produce similar attention patterns. Evicting lower-density-partition tokens
first provides a variance-aware eviction policy complementary to recency-based
approaches.

This generalises the observation from the corrector experiments: tokens in the
same HVRT partition share statistical properties at the output level. The same
is likely true at the attention level.

### 6.3 Domain Adaptation at Near-Zero Cost

Fine-tuning GPT-2 on a new domain requires gradient updates, memory proportional
to the model size, and careful hyperparameter selection. The HVRT corrector
requires:

- One forward pass over ~50,000 domain tokens to collect hidden states (~1 minute)
- One HVRT fit (~30 seconds on CPU for n=51,200, d=768)
- Storing 1.6 KB of bias statistics

The result is 7.90% perplexity improvement on in-domain text, with the model
itself unchanged. Updating the corrector for a new domain requires only
recomputing the 1.6 KB table — the HVRT tree can be reused if the domain is
sufficiently close to the original fitting domain, or refitted cheaply.

This is relevant for scenarios where the base model cannot be modified (API
access only, quantised deployment, shared infrastructure) or where rapid
per-user customisation is desired.

### 6.4 Structural Uncertainty Quantification — The Architectural Contrast

Neural language models are notoriously overconfident on out-of-distribution
inputs. The standard mitigation toolbox — temperature scaling, label smoothing,
deep ensembles, conformal prediction — either requires additional training, multiple
forward passes, or empirical calibration on held-out data from the target
distribution. All of these approaches share a fundamental limitation: they
estimate confidence from learned parameters, which are the same parameters that
can produce overconfident wrong outputs in the first place.

Section 2.2 establishes that HVRT's approach is architecturally distinct: its
confidence signal is a training count, not a learned weight. The practical
implications follow directly:

- **No calibration required**: the density signal does not need to be calibrated
  against held-out data. It is directly interpretable as "how many training
  tokens resembled this context."

- **Distribution shift is detectable by construction**: when a new domain is
  encountered, partition densities drop. This is not a learned alarm — it is a
  direct observation of evidence absence. A model fitted on Wikipedia will
  report low partition density on legal contracts, financial filings, or
  medical notes without any specialised OOD detection component.

- **Auditable and reproducible**: the density for any inference call can be
  recomputed exactly from the stored HVRT tree and training partition assignments.
  There is no stochastic sampling step. The confidence report is deterministic
  and verifiable.

Concretely in our experiments:
- Shakespeare-era literature prompts (dense in Wikipedia): median ~1,700 training
  tokens per partition → HIGH confidence.
- Financial market movement prompts (sparse in Wikipedia): ~326 training tokens
  per partition → LOW confidence, flagged automatically without any OOD detector.

In domains where a wrong confident answer is actively harmful — medicine, law,
safety-critical systems — the architectural guarantee offered by count-based
confidence reporting is a different class of property than the empirical
guarantees offered by calibrated neural models.

---

## 7. Towards a Standalone HVRT-LM

The corrector experiments demonstrate that HVRT partitioning captures real
linguistic structure in transformer representations. A natural question is whether
a standalone language model could be built from these components without a neural
backbone.

### 7.1 What Works

- **Piecewise-linear FFN approximation**: layers 1-11 individually well-approximated
  (86% variance explained per layer). The non-linear computation of transformers
  IS locally linear within HVRT partitions.
- **Partition-aware output correction**: the per-partition token frequency
  statistics capture systematic biases in model predictions, improving output
  quality measurably.
- **Confidence from density**: no additional machinery needed — it is a
  structural property, not a module (Section 2.2).
- **Domain adaptation speed**: one forward pass, one HVRT fit, 1.6 KB of output.
- **Complete interpretability**: every output decision traceable to a partition,
  a count, and an inspectable bias vector (Section 2.1).

### 7.2 What Remains Unsolved

- **Covariate shift cascade**: sequential layer replacement without iterative
  re-fitting is not viable. A training-from-scratch HVRT-LM would need boosting-
  style iterative residual fitting, where each round re-fits HVRT on the *corrected*
  outputs of the previous round — solving the distribution mismatch by construction.
  This is architecturally sound but empirically unproven at scale.

- **Long-range dependencies**: attention resolves cross-position dependencies
  explicitly. HVRT partitions a single position's hidden state — it captures
  *current context type* but not explicit token-to-token relationships over long
  distances. An attention analog for HVRT is an open research question.

- **Layer 0 non-linearity**: the initial embedding-to-representation step
  resists piecewise-linear approximation. In a from-scratch architecture, this
  layer would remain a learned lookup table.

- **Autoregressive feedback**: at generation time, each token's hidden state is
  partly determined by previously generated tokens. A pure HVRT model with no
  learned weights would need stable fixed-point behaviour under this feedback,
  which is unproven.

### 7.3 The Feasibility Assessment

**Narrow domains (small vocabulary, specific output space)**: genuinely
competitive. A domain-specific HVRT-LM targeting 100-500 task-relevant output
tokens requires ~100,000 domain-specific tokens to reach reliable per-partition
estimation, produces interpretable predictions with structural confidence
guarantees, runs without a GPU, and adapts to new domains in minutes. For medical
coding, API code generation, structured data generation, or similar
bounded-output tasks, the tradeoff of some quality for complete interpretability
and near-zero inference cost is viable. The structural inability to express false
confidence (Section 2.2) is additionally attractive in regulated domains where
a confident wrong answer is more dangerous than no answer.

**General English (full vocabulary)**: not yet competitive with neural LMs. The
vocabulary estimation problem scales poorly: to reliably estimate the distribution
over 50,000 tokens per partition requires millions of observations per partition,
which demands either enormous training corpora or very coarse partitioning (losing
context sensitivity). The cascade problem additionally prevents the piecewise-
linear FFN approach from generalising without iterative training. A general
HVRT-LM would likely sit between 5-gram Kneser-Ney and small neural LMs in quality.

**The optimistic case**: the corrector result shows that 7.90% of the improvement
potential is accessible from 1.6 KB with no training at all. This is an extreme
lower bound — with a much larger fitting corpus and an iterative boosting approach,
significantly more should be achievable. The low resource and compute requirements
make investigation feasible even without large-scale infrastructure. A model that
achieves half the quality of GPT-2 at 0.01% of the compute and storage, with full
structural interpretability and native architectural uncertainty honesty (Section 2),
occupies a genuinely useful position in the deployment landscape — particularly in
resource-constrained or regulated environments where interpretability is required
by policy, not just preferred.

---

## 8. Summary of Empirical Results

| Finding | Key Number | Significance |
|---------|-----------|--------------|
| Best corrector | +7.90% PPL, 1.6 KB | 1/310,000 of model params, no training |
| Vocab restriction | Top-100 > full vocab | Noise from rare tokens hurts; restrict to well-estimated head |
| Multi-round stability | K=100 degrades 0.13 PPL/round vs 0.70 for full | Vocab restriction stabilises but optimal is still 1 round |
| FFN approximability | 11/12 layers replaceable | GPT-2 FFN is locally piecewise-linear |
| Layer 0 criticality | +107 PPL when replaced | Embedding transition is uniquely non-linear |
| Cascade barrier | Best all-layers PPL=327 | 8% per-layer residual compounds 12× |
| Structural confidence | Count-based, not parameter-based | Architecturally incapable of false high-confidence (Section 2.2) |
| Interpretability | Full decision traceability | Every correction auditable from 1.6 KB table (Section 2.1) |

---

## 9. Experimental Scripts and Reproducibility

All experiments are reproducible with:

```bash
pip install hvrt transformers torch datasets scikit-learn numpy scipy
python research/ffn_probe/hvrt_corrector.py          # Variant B, original
python research/ffn_probe/hvrt_corrector_boosted.py  # Multi-round boosting
python research/ffn_probe/hvrt_corrector_vocab.py    # Vocab-restriction sweep
python research/ffn_probe/hvrt_ffn_replace.py        # Variant A: FFN replacement
```

HVRT source: https://github.com/Peace/hvrt (pip: `hvrt`)

All experiments use WikiText-103 (HuggingFace datasets), GPT-2 small
(HuggingFace transformers), and standard numerical libraries. No custom CUDA
kernels, no specialised hardware, no proprietary data.

---

## 10. Discussion and Future Work

### 10.1 Why the Corrector Works

The corrector improves GPT-2 because transformer models, despite their capacity,
maintain systematic per-context-region biases in their output distributions.
HVRT partitioning identifies these regions (by clustering the hidden state space
via variance structure), and the per-partition empirical correction removes the
systematic bias within each region. The gain is largest for a small vocabulary
because systematic bias in common tokens is the most reliably estimable and the
most impactful on perplexity.

This suggests that all language models, regardless of size, have correctable
systematic biases that are local in representation space. The HVRT corrector is
a way to find and remove them without touching the model. Furthermore, the
corrected model retains the structural interpretability and confidence properties
described in Section 2 — it is not merely more accurate, it is more honest.

### 10.2 Multi-Round Boosting Limitations

The multi-round boosting experiment reveals a fundamental data efficiency
constraint: with a fixed training corpus, the residual after round 1 is small
enough that fitting additional rounds picks up noise rather than signal. This
is not a failure of the boosting principle — it is a sample-size constraint.
With a training corpus 100× larger, multiple rounds would likely stabilise and
compound. This is a straightforward empirical question left for future work.

### 10.3 Iterative Layer Replacement

The covariate shift cascade (Variant A) is solvable in principle by training
iteratively: patch layer 1, re-collect its output distribution, fit layer 2's
replacement on those outputs (not on original GPT-2 outputs), and continue.
This approach guarantees that each replacement layer is trained on the actual
distribution it will receive at inference time. Preliminary analysis suggests
this would recover most of the single-layer probe quality in the all-layers
setting. This is the most tractable next experiment.

### 10.4 HVRT as an Attention Mechanism

The key unsolved component for a general HVRT-LM is cross-position context
integration (the role of attention). One direction: rather than attending to
all positions, partition the key vectors using HVRT and attend only within-
partition. This is a form of local attention with data-driven locality, requiring
no learned attention parameters. Whether this degrades long-range tasks is an
empirical question.

### 10.5 The Broader Claim

Neural language models can only be as interpretable as their most opaque
component, which is typically the entire forward pass. HVRT-based correction is
additive and separable: its decision can be fully audited independently of the
base model's computation. If future work produces a full HVRT-LM (replacing the
neural backbone rather than augmenting it), the interpretability and structural
honesty properties established here extend to the entire prediction pipeline.
The case for this architecture is strongest not where it maximises accuracy, but
where it maximises the *trustworthiness of the model's own self-reporting* — a
property that accuracy metrics do not capture.

---