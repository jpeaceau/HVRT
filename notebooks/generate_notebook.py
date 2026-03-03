"""Generate notebooks/hvrt_llm_probe.ipynb.

Run from repo root:
    python notebooks/generate_notebook.py
"""
import json, uuid, os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, 'hvrt_llm_probe.ipynb')


def uid():
    return uuid.uuid4().hex[:8]


def to_lines(src):
    src = src.lstrip('\n').rstrip('\n')
    if not src:
        return []
    lines = src.split('\n')
    return [l + '\n' for l in lines[:-1]] + [lines[-1]]


def md(src):
    return {'cell_type': 'markdown', 'id': uid(), 'metadata': {},
            'source': to_lines(src)}


def code(src):
    return {'cell_type': 'code', 'id': uid(), 'metadata': {},
            'source': to_lines(src), 'execution_count': None, 'outputs': []}


# ─────────────────────────────────────────────────────────────────────────────
# Cell 1 — Title
# ─────────────────────────────────────────────────────────────────────────────

C01 = md("""
# HVRT x GPT-2: Semantic Partitioning Without Labels

**[HVRT](https://pypi.org/project/hvrt/)** (Hierarchical Variance-Retaining Transformer)
partitions a dataset into variance-homogeneous regions using a purpose-built decision tree.
It was built for tabular data — dimensionality reduction and synthetic data generation.

**This notebook asks a different question:**

> *Can HVRT discover semantically coherent topic clusters by partitioning raw GPT-2
> hidden-state vectors — with no labels, no training, and no dimensionality reduction?*

The corpus: **60 sentences** across **6 topics** (science, history, cooking, sports,
technology, emotions).  Random-chance partition purity = 1/6 ≈ **0.167**.

| # | Experiment | Key idea |
|---|---|---|
| 1 | **Flat baseline** | Single-pass HVRT on each GPT-2 layer |
| 2 | **Cross-layer conditioning** | Anchor on an early layer, refine on target layer |
| 3 | **Multi-round sweep** | How many refinement rounds until saturation? |
| — | **Try your own** | Drop in any topics and sentences |

No GPU required — GPT-2 small runs on CPU in ~10 s.  With CUDA it is faster.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 2 — Install + setup + corpus
# ─────────────────────────────────────────────────────────────────────────────

C02 = code("""
# ── Install ──────────────────────────────────────────────────────────────────
!pip install hvrt transformers torch --quiet

%matplotlib inline
import os, warnings, time
import numpy as np
import matplotlib.pyplot as plt

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM']       = 'false'
warnings.filterwarnings('ignore')

from hvrt import FastHVRT

# ── Corpus: 60 prompts, 6 topics ─────────────────────────────────────────────
PROMPTS = {
    "science": [
        "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
        "DNA is a double helix structure that encodes genetic information in four bases.",
        "Quantum entanglement allows particles to correlate regardless of separation distance.",
        "The standard model of particle physics describes all known fundamental particles.",
        "Newton's third law states that every action has an equal and opposite reaction.",
        "The human brain contains approximately 86 billion neurons connected by synapses.",
        "Black holes are regions of spacetime where escape velocity exceeds the speed of light.",
        "Enzymes are biological catalysts that lower the activation energy of reactions.",
        "The periodic table organises elements by atomic number and chemical properties.",
    ],
    "history": [
        "The Roman Empire fell in 476 AD when the last Western Roman Emperor was deposed.",
        "The French Revolution began in 1789 with the storming of the Bastille prison.",
        "World War II ended in 1945 with the unconditional surrender of Germany and Japan.",
        "The Industrial Revolution transformed manufacturing from the 1760s onward in Britain.",
        "Christopher Columbus reached the Caribbean in 1492 while sailing west from Spain.",
        "The Great Wall of China was built over many centuries to protect northern borders.",
        "Ancient Egypt developed hieroglyphic writing around 3200 BCE along the Nile.",
        "The Apollo 11 mission successfully landed humans on the Moon in July 1969.",
        "Gutenberg's printing press, invented in the 1440s, transformed the spread of knowledge.",
        "The Renaissance was a cultural rebirth that began in Florence in the 14th century.",
    ],
    "cooking": [
        "Pasta should be cooked in heavily salted boiling water until al dente texture.",
        "Caramelisation occurs when sugars are heated above their melting point temperature.",
        "Emulsification combines two immiscible liquids such as oil and vinegar.",
        "Bread rises because yeast ferments sugars and produces carbon dioxide gas.",
        "The Maillard reaction creates brown crust on meat when cooked at high temperatures.",
        "Tempering chocolate involves cycling temperature to stabilise cocoa butter crystals.",
        "Beurre blanc is a classic French butter sauce emulsified with white wine reduction.",
        "Sourdough starter contains wild yeasts and lactic acid bacteria in symbiosis.",
        "Sushi rice is seasoned with a mixture of rice vinegar, sugar, and salt.",
        "A roux is made by cooking equal parts butter and flour together until smooth.",
    ],
    "sports": [
        "A soccer match consists of two 45-minute halves plus referee-added injury time.",
        "The Tour de France is a gruelling multi-stage bicycle race held annually in July.",
        "Basketball was invented by Canadian James Naismith in 1891 in Massachusetts.",
        "A cricket innings ends when ten of the eleven batting side wickets have fallen.",
        "The Olympics are held every four years, alternating between summer and winter games.",
        "Tennis scoring uses love, fifteen, thirty, and deuce within a single game.",
        "An American football touchdown scores six points plus an opportunity to add more.",
        "The marathon distance of 42.195 km commemorates the legendary run from Marathon.",
        "Chess is played on an eight-by-eight board with sixteen pieces on each side.",
        "Swimming events include four strokes: freestyle, backstroke, breaststroke, butterfly.",
    ],
    "technology": [
        "Machine learning algorithms iteratively improve performance by training on data.",
        "The internet transmits data using the TCP/IP protocol suite across global networks.",
        "Blockchain provides a decentralised immutable ledger secured by cryptographic hashing.",
        "Solid-state drives are faster than hard disk drives because they have no moving parts.",
        "Cloud computing delivers on-demand computing resources via the internet at scale.",
        "Natural language processing enables computers to understand and generate human text.",
        "Encryption protects sensitive data by converting it into an unreadable ciphertext.",
        "Transistors are the fundamental switching elements in all modern electronic devices.",
        "Git is a distributed version control system created by Linus Torvalds in 2005.",
        "Convolutional neural networks learn spatial hierarchies of features from images.",
    ],
    "emotions": [
        "I feel an overwhelming sense of joy whenever I reunite with my family after travel.",
        "The grief experienced after losing a loved one can persist and reshape identity.",
        "Anxiety manifests as persistent rumination about future events that may not occur.",
        "Nostalgia is a bittersweet longing for the past that blends happiness with sadness.",
        "Empathy allows us to share and understand the inner emotional states of others.",
        "Fear activates the fight-or-flight response via the amygdala in the limbic system.",
        "Gratitude journalling has been shown to measurably improve psychological well-being.",
        "Loneliness is distinct from solitude and involves a painful sense of disconnection.",
        "Awe arises when encountering something vast that challenges existing mental frameworks.",
        "Curiosity is an intrinsic motivational force that drives exploration and discovery.",
    ],
}

TOPIC_NAMES     = list(PROMPTS.keys())
TOPIC_COLORS    = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
N_TOPICS        = len(TOPIC_NAMES)
RANDOM_BASELINE = 1.0 / N_TOPICS
ANCHOR_LAYER    = 2   # best base layer for cross-layer conditioning

all_prompts, all_labels = [], []
for i, (topic, texts) in enumerate(PROMPTS.items()):
    all_prompts.extend(texts)
    all_labels.extend([i] * len(texts))
labels = np.array(all_labels, dtype=np.int64)

print(f"Corpus  : {len(all_prompts)} prompts  |  {N_TOPICS} topics")
print(f"Topics  : {TOPIC_NAMES}")
print(f"Random-chance purity baseline: {RANDOM_BASELINE:.3f}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 3 — GPT-2 extraction header
# ─────────────────────────────────────────────────────────────────────────────

C03 = md("""
## Step 1: Extract GPT-2 Hidden States

GPT-2 small has **12 transformer blocks** plus an embedding layer — 13 layers total.
We extract mean-pooled hidden-state vectors from *every* layer for all 60 prompts,
producing 13 matrices of shape `(60, 768)`.

The model is used **frozen in eval mode** — nothing is trained or fine-tuned.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 4 — Load GPT-2 + extract activations
# ─────────────────────────────────────────────────────────────────────────────

C04 = code("""
import torch
from transformers import GPT2Model, GPT2Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device : {device}")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2Model.from_pretrained('gpt2').to(device)
model.eval()

N_LAYERS = model.config.n_layer   # 12
D_MODEL  = model.config.n_embd    # 768
n_params = sum(p.numel() for p in model.parameters())
print(f"Model  : GPT-2 small  |  {N_LAYERS} blocks, d={D_MODEL}  |  {n_params:,} params")
print(f"Extracting {N_LAYERS + 1} layers x {len(all_prompts)} prompts...")

t0   = time.time()
_raw = [[] for _ in range(N_LAYERS + 1)]

with torch.no_grad():
    for prompt in all_prompts:
        enc = tokenizer(prompt, return_tensors='pt',
                        truncation=True, max_length=64).to(device)
        out = model(**enc, output_hidden_states=True)
        for i, h in enumerate(out.hidden_states):
            _raw[i].append(h.squeeze(0).mean(0).cpu().float().numpy())

layer_acts  = [np.array(a, dtype=np.float32) for a in _raw]
anchor_acts = layer_acts[ANCHOR_LAYER]

del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Done in {time.time()-t0:.1f}s  --  {len(layer_acts)} layers x {layer_acts[0].shape}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 5 — Experiment 1 header
# ─────────────────────────────────────────────────────────────────────────────

C05 = md("""
## Experiment 1: Flat HVRT Baseline

A single-pass `FastHVRT` with **8 partitions** is fitted independently on each layer's
activations.  **Weighted partition purity** measures how well the discovered partitions
align with the true topic labels — without those labels ever being seen during fitting.

$$\\text{purity} = \\sum_{k} \\frac{|P_k|}{n} \\cdot \\frac{\\max_t \\text{count}(t, P_k)}{|P_k|}$$

Random chance = 0.167.  Values above that mean HVRT found real geometric structure.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 6 — Flat HVRT table
# ─────────────────────────────────────────────────────────────────────────────

C06 = code("""
def weighted_purity(labels, partition_ids, n_topics=N_TOPICS):
    total = len(labels)
    wp    = 0.0
    for pid in np.unique(partition_ids):
        mask   = partition_ids == pid
        n      = int(mask.sum())
        counts = np.bincount(labels[mask], minlength=n_topics)
        wp    += (counts.max() / n) * (n / total)
    return float(wp), int(np.unique(partition_ids).size)


N_PARTS  = 8
MIN_LEAF = 4

print(f"{'Lyr':>4}  {'Parts':>5}  {'Purity':>7}  {'vs baseline':>12}")
print(f"{'-'*4}  {'-'*5}  {'-'*7}  {'-'*12}")

flat_purities = []
best_so_far   = -1.0
for li, act in enumerate(layer_acts):
    m = FastHVRT(n_partitions=N_PARTS, min_samples_leaf=MIN_LEAF,
                 auto_tune=False, n_jobs=-1, random_state=42)
    m.fit(act)
    p, n_p = weighted_purity(labels, m.partition_ids_)
    flat_purities.append(p)
    d   = p - RANDOM_BASELINE
    tag = ''
    if p > best_so_far:
        best_so_far = p
        tag = '  <-- new best'
    print(f"{li:>4}  {n_p:>5}  {p:>7.4f}  {d:>+12.4f}{tag}")

flat_purities = np.array(flat_purities)
best_fl       = int(np.argmax(flat_purities))
print(f"\\nBest layer : {best_fl}  |  purity={flat_purities[best_fl]:.4f}  "
      f"({flat_purities[best_fl]/RANDOM_BASELINE:.1f}x above random)")
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 7 — Flat HVRT chart
# ─────────────────────────────────────────────────────────────────────────────

C07 = code("""
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(len(layer_acts)), flat_purities, 'o-', color='#2196F3',
        lw=2, ms=7, label='Flat HVRT (8 partitions, 1 round)')
ax.axhline(RANDOM_BASELINE, color='gray', ls='--', lw=1.5,
           label=f'Random baseline ({RANDOM_BASELINE:.3f})')
ax.axvline(best_fl, color='#2196F3', ls=':', alpha=0.4, lw=1.5)

ax.set_xlabel('GPT-2 Layer  (0 = embedding)', fontsize=12)
ax.set_ylabel('Weighted Partition Purity', fontsize=12)
ax.set_title('HVRT Partition Purity Across GPT-2 Layers  --  No Labels Used',
             fontsize=13, fontweight='bold')
ax.set_xticks(range(len(layer_acts)))
ax.set_xticklabels(['emb'] + [str(i) for i in range(1, len(layer_acts))], fontsize=9)
ax.legend(fontsize=10)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Peak purity {flat_purities[best_fl]:.3f} at layer {best_fl} -- "
      f"{flat_purities[best_fl]/RANDOM_BASELINE:.1f}x the random baseline.")
print("HVRT finds real semantic structure from raw activation geometry alone.")
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 8 — Experiment 2 header
# ─────────────────────────────────────────────────────────────────────────────

C08 = md("""
## Experiment 2: Cross-Layer Conditioning

Early GPT-2 layers (L2-L4) produce stable, category-level representations.
Later layers specialise but become increasingly anisotropic (all vectors cluster in
a narrow cone), making direct partitioning noisy.

**Idea:** anchor the *first* partitioning pass on the early stable layer, then refine
each group using the *target* layer's geometry.  This is directly analogous to
gradient boosting — each round makes a small, focused geometric refinement:

```
Round 1:  partition ANCHOR layer (L2)         -->  coarse topic groups
Round 2:  within each group, partition TARGET -->  finer sub-structure
```

Three strategies compared across all 13 layers:

| Strategy | Description |
|---|---|
| **Flat** | Single pass on target layer (Experiment 1 baseline) |
| **Hierarchical** | Two passes on the *same* target layer |
| **GeoXGB** | Round 1 on anchor L2, round 2 on target layer |
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 9 — GeoXGB implementation + comparison table
# ─────────────────────────────────────────────────────────────────────────────

C09 = code("""
def multi_round_geoxgb(anchor_acts, target_acts, n_rounds, splits_per_round,
                        max_depth=None, min_leaf=4, seed=42):
    \"\"\"
    Multi-round GeoXGB-inspired partitioning.

    Round 1: split the ANCHOR layer (semantically stable).
    Round 2+: within each existing group, split the TARGET layer.

    Parameters
    ----------
    anchor_acts    : (n, d) activations from the anchor layer.
    target_acts    : (n, d) activations from the layer being probed.
    n_rounds       : number of refinement rounds.
    splits_per_round: partitions per round (learning-rate analogue).
    \"\"\"
    n        = len(target_acts)
    groups   = [(np.arange(n), True)]   # (global_indices, use_anchor)
    compound = np.full(n, -1, dtype=np.int64)
    cid      = 0

    for round_idx in range(n_rounds):
        next_groups = []
        for grp_indices, use_anchor in groups:
            n_sub = len(grp_indices)
            if n_sub < min_leaf * splits_per_round:
                compound[grp_indices] = cid
                cid += 1
                continue
            effective = min(splits_per_round, n_sub // min_leaf)
            fit_acts  = (anchor_acts[grp_indices] if use_anchor
                         else target_acts[grp_indices])
            try:
                m = FastHVRT(n_partitions=effective, min_samples_leaf=min_leaf,
                             max_depth=max_depth, auto_tune=False, n_jobs=1,
                             random_state=seed + round_idx * 1000 + cid)
                m.fit(fit_acts)
                for spid in np.unique(m.partition_ids_):
                    sub_global = grp_indices[m.partition_ids_ == spid]
                    if round_idx < n_rounds - 1:
                        next_groups.append((sub_global, False))
                    else:
                        compound[sub_global] = cid
                        cid += 1
            except Exception:
                compound[grp_indices] = cid
                cid += 1
        groups = next_groups

    for grp_indices, _ in groups:
        compound[grp_indices] = cid
        cid += 1
    return compound


# Run all 3 strategies
print("Comparing strategies across all layers...")
hier_purities, geo_purities = [], []

for li, act in enumerate(layer_acts):
    # Hierarchical: 2 rounds on the SAME layer
    ids = multi_round_geoxgb(act, act, n_rounds=2, splits_per_round=3, min_leaf=4)
    p, _ = weighted_purity(labels, ids)
    hier_purities.append(p)

    # GeoXGB: anchor=L2, then target layer
    ids = multi_round_geoxgb(anchor_acts, act, n_rounds=2, splits_per_round=3, min_leaf=4)
    p, _ = weighted_purity(labels, ids)
    geo_purities.append(p)

hier_purities = np.array(hier_purities)
geo_purities  = np.array(geo_purities)

print(f"\\n{'Strategy':<26}  {'Peak':>6}  {'Best Layer':>10}  {'vs Flat':>8}")
print(f"{'-'*26}  {'-'*6}  {'-'*10}  {'-'*8}")
for name, purities in [('Flat (1 round)', flat_purities),
                        ('Hierarchical (2 rnd, same lyr)', hier_purities),
                        ('GeoXGB (2 rnd, L2 + target)', geo_purities)]:
    pk   = purities.max()
    bl   = int(purities.argmax())
    diff = pk - flat_purities.max()
    print(f"{name:<26}  {pk:>6.4f}  {bl:>10}  {diff:>+8.4f}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 10 — Strategy comparison chart
# ─────────────────────────────────────────────────────────────────────────────

C10 = code("""
layers = list(range(len(layer_acts)))
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(layers, flat_purities,  'o-',  color='#2196F3', lw=2,   ms=6,
        label='Flat (1 round, target layer)')
ax.plot(layers, hier_purities,  's--', color='#FF9800', lw=2,   ms=6,
        label='Hierarchical (2 rounds, same layer)')
ax.plot(layers, geo_purities,   '^-',  color='#4CAF50', lw=2.5, ms=7,
        label='GeoXGB (anchor L2 + target, 2 rounds)')
ax.axhline(RANDOM_BASELINE, color='gray', ls=':', lw=1.5, label='Random baseline')

ax.set_xlabel('GPT-2 Layer', fontsize=12)
ax.set_ylabel('Weighted Partition Purity', fontsize=12)
ax.set_title('Strategy Comparison: Flat vs Hierarchical vs Cross-Layer Conditioning',
             fontsize=13, fontweight='bold')
ax.set_xticks(layers)
ax.set_xticklabels(['emb'] + [str(i) for i in range(1, len(layer_acts))], fontsize=9)
ax.legend(fontsize=9)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 11 — Experiment 3 header
# ─────────────────────────────────────────────────────────────────────────────

C11 = md("""
## Experiment 3: How Many Rounds?

**`n_rounds`** controls how many sequential partitioning passes are applied.
Think of it as a step-count in gradient descent: more rounds = finer resolution,
at the cost of needing enough samples in each group to split further.

**`splits_per_round`** is the branching factor per round:
- `2` = binary splits (conservative, low learning rate)
- `3` = ternary splits (faster, higher learning rate)

We sweep `splits in {2, 3}` x `rounds in {1, 2, 3}` across all 13 layers
(6 configs x 13 layers = 78 FastHVRT fits, takes ~1 second).
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 12 — Sweep
# ─────────────────────────────────────────────────────────────────────────────

C12 = code("""
SWEEP_SPLITS = [2, 3]
SWEEP_ROUNDS = [1, 2, 3]
configs      = [(s, r) for s in SWEEP_SPLITS for r in SWEEP_ROUNDS]
sweep_grid   = np.zeros((len(configs), len(layer_acts)))

t0 = time.time()
for ci, (splits, rounds) in enumerate(configs):
    for li, act in enumerate(layer_acts):
        ids = multi_round_geoxgb(anchor_acts, act, n_rounds=rounds,
                                  splits_per_round=splits, min_leaf=4, seed=42)
        p, _ = weighted_purity(labels, ids)
        sweep_grid[ci, li] = p

print(f"Sweep done in {time.time()-t0:.1f}s  "
      f"({len(configs)} configs x {len(layer_acts)} layers)")

print(f"\\n{'splits':>6}  {'rounds':>6}  {'peak':>7}  {'best layer':>10}  {'vs flat-8':>9}")
print(f"{'-'*6}  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*9}")
for ci, (splits, rounds) in enumerate(configs):
    pk   = sweep_grid[ci].max()
    bl   = int(sweep_grid[ci].argmax())
    diff = pk - flat_purities.max()
    print(f"{splits:>6}  {rounds:>6}  {pk:>7.4f}  {bl:>10}  {diff:>+9.4f}")

best_ci = int(np.unravel_index(sweep_grid.argmax(), sweep_grid.shape)[0])
best_li = int(sweep_grid[best_ci].argmax())
print(f"\\nOverall best : splits={configs[best_ci][0]}, rounds={configs[best_ci][1]}, "
      f"layer={best_li}, purity={sweep_grid.max():.4f}")
print(f"Gain over flat-8 single-pass : {sweep_grid.max() - flat_purities.max():+.4f}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 13 — Sweep charts
# ─────────────────────────────────────────────────────────────────────────────

C13 = code("""
cmap = plt.get_cmap('tab10')
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# ── Left: purity vs layer for each config ────────────────────────────────────
ax = axes[0]
for ci, (splits, rounds) in enumerate(configs):
    ls    = '-'  if splits == 3 else '--'
    color = cmap(rounds - 1)
    ax.plot(range(len(layer_acts)), sweep_grid[ci], ls=ls, color=color,
            lw=1.8, alpha=0.85, label=f'splits={splits}, rounds={rounds}')
ax.axhline(RANDOM_BASELINE, color='gray', ls=':', lw=1.5, label='Random baseline')
ax.set_xlabel('GPT-2 Layer', fontsize=11)
ax.set_ylabel('Weighted Purity', fontsize=11)
ax.set_title('All configs: purity vs layer', fontsize=12, fontweight='bold')
ax.set_xticks(list(range(len(layer_acts))))
ax.set_xticklabels(['e'] + [str(i) for i in range(1, len(layer_acts))], fontsize=8)
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

# ── Right: peak purity by n_rounds ───────────────────────────────────────────
ax = axes[1]
for splits in SWEEP_SPLITS:
    idxs  = [ci for ci, (s, r) in enumerate(configs) if s == splits]
    peaks = [sweep_grid[i].max() for i in idxs]
    rnds  = [configs[i][1] for i in idxs]
    ax.plot(rnds, peaks, 'o-', lw=2.5, ms=10, label=f'splits={splits}')
ax.axhline(flat_purities.max(), color='#2196F3', ls='--', lw=1.5,
           label='Flat-8 peak (1 round)')
ax.axhline(RANDOM_BASELINE, color='gray', ls=':', lw=1.5, label='Random baseline')
ax.set_xlabel('Number of Rounds', fontsize=11)
ax.set_ylabel('Peak Purity  (across all layers)', fontsize=11)
ax.set_title('n_rounds is the dominant axis', fontsize=12, fontweight='bold')
ax.set_xticks(SWEEP_ROUNDS)
ax.legend(fontsize=10)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 14 — Partition anatomy header
# ─────────────────────────────────────────────────────────────────────────────

C14 = md("""
## What Is Inside a Partition?

The chart below shows the **topic composition** of each partition produced by
the best sweep configuration at the best layer.  The number above each bar is that
partition's purity score.

A pure partition means HVRT's variance-geometry split naturally landed on a topic
boundary — purely from the shape of the activation cloud, no label supervision.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 15 — Stacked bar of partition composition
# ─────────────────────────────────────────────────────────────────────────────

C15 = code("""
best_splits, best_rounds = configs[best_ci]
ids_best = multi_round_geoxgb(anchor_acts, layer_acts[best_li],
                               n_rounds=best_rounds, splits_per_round=best_splits,
                               min_leaf=4, seed=42)

# Sort partitions by dominant topic then purity for a readable chart
def part_sort_key(pid):
    mask   = ids_best == pid
    counts = np.bincount(labels[mask], minlength=N_TOPICS)
    return (int(counts.argmax()), -counts.max() / mask.sum())

unique_parts  = sorted(np.unique(ids_best), key=part_sort_key)
n_parts_best  = len(unique_parts)

fig, ax = plt.subplots(figsize=(max(8, n_parts_best), 4))
bottoms = np.zeros(n_parts_best)

for ti, (topic, color) in enumerate(zip(TOPIC_NAMES, TOPIC_COLORS)):
    heights = []
    for pid in unique_parts:
        mask   = ids_best == pid
        counts = np.bincount(labels[mask], minlength=N_TOPICS)
        heights.append(counts[ti] / mask.sum())
    ax.bar(range(n_parts_best), heights, bottom=bottoms,
           color=color, label=topic, edgecolor='white', lw=0.5)
    bottoms += np.array(heights)

for xi, pid in enumerate(unique_parts):
    mask   = ids_best == pid
    counts = np.bincount(labels[mask], minlength=N_TOPICS)
    purity = counts.max() / mask.sum()
    n_in   = int(mask.sum())
    ax.text(xi,  1.02, f'{purity:.2f}', ha='center', fontsize=8.5, fontweight='bold')
    ax.text(xi, -0.08, f'n={n_in}',    ha='center', fontsize=7.5, color='gray')

ax.set_xlabel('Partition  (sorted by dominant topic)', fontsize=11)
ax.set_ylabel('Topic Fraction', fontsize=11)
ax.set_title(
    f'Partition Composition  |  Layer {best_li}  |  '
    f'splits={best_splits}, rounds={best_rounds}  |  '
    f'purity={sweep_grid[best_ci, best_li]:.3f}',
    fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, ncol=2)
ax.set_xticks(range(n_parts_best))
ax.set_xticklabels([str(pid) for pid in unique_parts], fontsize=9)
ax.set_ylim(-0.12, 1.10)
ax.axhline(1, color='black', lw=0.5, alpha=0.3)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 16 — Try your own header
# ─────────────────────────────────────────────────────────────────────────────

C16 = md("""
## Try Your Own Topics

Replace `MY_PROMPTS` below with any topics and sentences you like.

- At least **4 sentences per topic** (3 is the hard minimum)
- At least **2 topics**  (more topics = harder separation task)
- More semantically distinct topics are easier for HVRT to separate

This cell reloads GPT-2 so it runs independently — you do not need to re-run
the earlier cells.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 17 — Custom prompts
# ─────────────────────────────────────────────────────────────────────────────

C17 = code("""
# ── Edit this section ──────────────────────────────────────────────────────
MY_PROMPTS = {
    "climate": [
        "Global average temperatures have risen by over 1.1 degrees since pre-industrial times.",
        "Melting Arctic sea ice accelerates warming through the ice-albedo feedback mechanism.",
        "Carbon capture technology aims to remove CO2 directly from the atmosphere.",
        "Renewable energy from wind and solar now outcompetes new fossil fuel plants on cost.",
        "Rising sea levels threaten coastal cities and low-lying island nations worldwide.",
    ],
    "music": [
        "The blues scale forms the harmonic backbone of jazz and rock guitar improvisation.",
        "Polyphony allows multiple independent melodic voices to coexist in counterpoint.",
        "Synthesisers generate sound electronically by shaping oscillator waveforms.",
        "Rhythm and metre organise musical time into patterns of stressed and unstressed beats.",
        "The circle of fifths maps harmonic relationships between all twelve musical keys.",
    ],
    "medicine": [
        "Antibiotics target bacterial cell walls and protein synthesis rather than viruses.",
        "The blood-brain barrier restricts which molecules can enter the central nervous system.",
        "CRISPR-Cas9 enables precise edits to DNA sequences guided by complementary RNA.",
        "Vaccines prime the immune system by presenting harmless antigens for later recognition.",
        "Stem cells can differentiate into specialised cell types for regenerative therapies.",
    ],
}
# ── No changes needed below ────────────────────────────────────────────────

import torch
from transformers import GPT2Model, GPT2Tokenizer

my_topic_names = list(MY_PROMPTS.keys())
my_n_topics    = len(my_topic_names)
my_baseline    = 1.0 / my_n_topics
my_colors      = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3'][:my_n_topics]

my_prompts_list, my_labels_list = [], []
for i, (topic, texts) in enumerate(MY_PROMPTS.items()):
    my_prompts_list.extend(texts)
    my_labels_list.extend([i] * len(texts))
my_labels_arr = np.array(my_labels_list, dtype=np.int64)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok2   = GPT2Tokenizer.from_pretrained('gpt2')
tok2.pad_token = tok2.eos_token
mdl2   = GPT2Model.from_pretrained('gpt2').to(device)
mdl2.eval()

_raw2 = [[] for _ in range(mdl2.config.n_layer + 1)]
with torch.no_grad():
    for prompt in my_prompts_list:
        enc = tok2(prompt, return_tensors='pt', truncation=True, max_length=64).to(device)
        out = mdl2(**enc, output_hidden_states=True)
        for i, h in enumerate(out.hidden_states):
            _raw2[i].append(h.squeeze(0).mean(0).cpu().float().numpy())

my_layer_acts = [np.array(a, dtype=np.float32) for a in _raw2]
del mdl2

# Flat HVRT across all layers
my_flat = []
for act in my_layer_acts:
    m = FastHVRT(n_partitions=my_n_topics * 2, min_samples_leaf=3,
                 auto_tune=False, n_jobs=-1, random_state=42)
    m.fit(act)
    p, _ = weighted_purity(my_labels_arr, m.partition_ids_, n_topics=my_n_topics)
    my_flat.append(p)
my_flat = np.array(my_flat)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(len(my_layer_acts)), my_flat, 'o-', color='#9C27B0', lw=2, ms=7,
        label='Flat HVRT')
ax.axhline(my_baseline, color='gray', ls='--', lw=1.5,
           label=f'Random baseline ({my_baseline:.3f})')
ax.set_xlabel('GPT-2 Layer', fontsize=12)
ax.set_ylabel('Weighted Partition Purity', fontsize=12)
ax.set_title('Custom Topics -- HVRT Partition Purity', fontsize=13, fontweight='bold')
ax.set_xticks(range(len(my_layer_acts)))
ax.set_xticklabels(['emb'] + [str(i) for i in range(1, len(my_layer_acts))], fontsize=9)
ax.legend(fontsize=10)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

best_p = my_flat.max()
print(f"Topics         : {my_topic_names}")
print(f"Peak purity    : {best_p:.4f} at layer {int(my_flat.argmax())}")
print(f"vs random      : {best_p / my_baseline:.1f}x  (baseline={my_baseline:.3f})")
""")

# ─────────────────────────────────────────────────────────────────────────────
# Assemble notebook
# ─────────────────────────────────────────────────────────────────────────────

cells = [C01, C02, C03, C04, C05, C06, C07, C08, C09, C10,
         C11, C12, C13, C14, C15, C16, C17]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {OUT}")
print(f"Cells  : {len(cells)}")
