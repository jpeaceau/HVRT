"""
HVRT LLM Activation Probe — Hyperparameter Sweep
=================================================

Systematically searches three axes of the GeoXGB-inspired cross-layer
partitioning approach:

  splits_per_round  [2, 3]         Learning-rate analogue.
                                   2 = binary, conservative (low LR).
                                   3 = ternary, faster (high LR).

  max_depth         [2, 3, 4, None] HVRT tree depth per round.
                                   Shallow = coarser, more robust.
                                   None = unlimited.

  n_rounds          [1, 2, 3]      How many sequential refinement rounds.
                                   Round 1 = anchor (Layer 2).
                                   Rounds 2+ = residual on target layer.

Total configurations: 4 x 3 x 2 = 24.
Each applied across all 13 GPT-2 layers => 312 FastHVRT fits.

Output
------
  1. Full sweep table: for each config, best layer and peak purity.
  2. Best config detail: layer-by-layer purity table.
  3. Per-layer best: which config wins at each layer.

Usage
-----
    python research/llm_probe/hvrt_sweep_probe.py
"""

from __future__ import annotations

import itertools
import time
import numpy as np

# ============================================================
# Configuration
# ============================================================

MODEL_NAME       = 'gpt2'
DEVICE           = 'auto'
ANCHOR_LAYER     = 2       # best base layer from baseline run
MIN_SAMPLES_LEAF = 4       # hard floor — no partition smaller than this

SWEEP_SPLITS  = [2, 3]
SWEEP_DEPTH   = [2, 3, 4, None]
SWEEP_ROUNDS  = [1, 2, 3]

# ============================================================
# Prompt corpus
# ============================================================

PROMPTS: dict[str, list[str]] = {
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

# ============================================================
# Model loading & activation extraction
# ============================================================

def load_and_extract(model_name, device_pref, prompts, max_length=64):
    import torch
    from transformers import GPT2Model, GPT2Tokenizer

    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
             if device_pref == 'auto' else device_pref

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(model_name).to(device)
    model.eval()

    n_layers = model.config.n_layer
    hidden_size = model.config.n_embd
    print(f"  GPT-2 small: {n_layers} layers, d={hidden_size}, device={device}")

    all_hidden = None
    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors='pt', truncation=True,
                            max_length=max_length, padding=False).to(device)
            out = model(**enc, output_hidden_states=True)
            if all_hidden is None:
                all_hidden = [[] for _ in out.hidden_states]
            for i, h in enumerate(out.hidden_states):
                all_hidden[i].append(h.squeeze(0).mean(0).cpu().float().numpy())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [np.array(layer, dtype=np.float32) for layer in all_hidden]

# ============================================================
# Multi-round GeoXGB-inspired partitioning
# ============================================================

def multi_round_geoxgb(
    anchor_acts: np.ndarray,
    target_acts: np.ndarray,
    n_rounds: int,
    splits_per_round: int,
    max_depth: int | None,
    min_leaf: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Generalised multi-round GeoXGB-inspired partitioning.

    Round 1: fit FastHVRT on anchor_acts with splits_per_round partitions
             and max_depth depth limit.
    Round k>1: within each partition from round k-1, fit FastHVRT on the
               corresponding rows of target_acts.

    This implements the "reduced learning rate" idea: by using only 2 or 3
    splits per round (rather than 8 in one pass), each round makes a small,
    stable geometric refinement.  Multiple rounds then accumulate into a
    fine-grained partition structure.

    Parameters
    ----------
    anchor_acts    : activations from the anchor layer (semantically stable).
    target_acts    : activations from the target layer being probed.
    n_rounds       : number of refinement rounds (1 = single flat pass).
    splits_per_round: partitions created per round (learning rate analogue).
    max_depth      : HVRT tree max depth per round.
    min_leaf       : minimum samples per leaf; prevents over-splitting.
    seed           : base random seed.

    Returns
    -------
    compound_ids : ndarray (n_samples,) of integer partition assignments.
    """
    from hvrt import FastHVRT

    n = len(target_acts)
    # Each element: (global_indices_into_target, source_acts_for_this_round)
    # Round 1 uses anchor_acts; rounds 2+ use target_acts.
    groups: list[tuple[np.ndarray, np.ndarray]] = [
        (np.arange(n), anchor_acts)
    ]

    compound = np.full(n, -1, dtype=np.int64)
    cid = 0

    for round_idx in range(n_rounds):
        next_groups: list[tuple[np.ndarray, np.ndarray]] = []
        use_target = round_idx > 0   # round 1 uses anchor; rest use target

        for grp_indices, grp_acts in groups:
            n_sub = len(grp_indices)
            min_for_split = min_leaf * splits_per_round

            if n_sub < min_for_split:
                # Too small to split further — assign and done
                compound[grp_indices] = cid
                cid += 1
                continue

            effective_splits = min(splits_per_round, n_sub // min_leaf)
            fit_acts = (target_acts[grp_indices] if use_target
                        else grp_acts[grp_indices]
                        if grp_acts is not anchor_acts
                        else anchor_acts[grp_indices])

            try:
                m = FastHVRT(
                    n_partitions=effective_splits,
                    min_samples_leaf=min_leaf,
                    max_depth=max_depth,
                    auto_tune=False,
                    n_jobs=1,
                    random_state=seed + round_idx * 1000 + cid,
                )
                m.fit(fit_acts)
                sub_ids = m.partition_ids_

                for spid in np.unique(sub_ids):
                    sub_mask = sub_ids == spid
                    sub_global = grp_indices[sub_mask]
                    if round_idx < n_rounds - 1:
                        # Not the last round: queue for further splitting
                        next_groups.append((sub_global, anchor_acts))
                    else:
                        # Last round: assign final partition
                        compound[sub_global] = cid
                        cid += 1

            except Exception:
                compound[grp_indices] = cid
                cid += 1

        groups = next_groups

    # Any groups remaining from intermediate rounds (shouldn't happen, but safe)
    for grp_indices, _ in groups:
        compound[grp_indices] = cid
        cid += 1

    return compound

# ============================================================
# Purity metric
# ============================================================

def weighted_purity(labels: np.ndarray, partition_ids: np.ndarray,
                    n_topics: int) -> tuple[float, int]:
    total = len(labels)
    wp = 0.0
    for pid in np.unique(partition_ids):
        mask = partition_ids == pid
        n = int(mask.sum())
        counts = np.bincount(labels[mask], minlength=n_topics)
        wp += (counts.max() / n) * (n / total)
    return wp, int(np.unique(partition_ids).size)

# ============================================================
# Main sweep
# ============================================================

def main():
    # --- Corpus ---
    all_prompts, all_labels = [], []
    topic_names = list(PROMPTS.keys())
    n_topics = len(topic_names)
    for idx, (_, texts) in enumerate(PROMPTS.items()):
        all_prompts.extend(texts)
        all_labels.extend([idx] * len(texts))
    labels = np.array(all_labels, dtype=np.int64)
    random_baseline = 1.0 / n_topics

    print("\nHVRT LLM Probe — Hyperparameter Sweep")
    print("=" * 72)
    print(f"  Model        : {MODEL_NAME}")
    print(f"  Anchor layer : {ANCHOR_LAYER}")
    print(f"  Sweep axes   : splits_per_round={SWEEP_SPLITS}  "
          f"max_depth={SWEEP_DEPTH}  n_rounds={SWEEP_ROUNDS}")
    n_configs = len(SWEEP_SPLITS) * len(SWEEP_DEPTH) * len(SWEEP_ROUNDS)
    print(f"  Configurations: {n_configs}  x  13 layers  =  {n_configs * 13} fits")
    print(f"  Random baseline purity: {random_baseline:.4f}\n")

    # --- Extract activations ---
    print("Loading model and extracting activations...")
    t0 = time.perf_counter()
    layer_acts = load_and_extract(MODEL_NAME, DEVICE, all_prompts)
    print(f"  Done in {time.perf_counter() - t0:.1f}s  "
          f"({len(layer_acts)} layers x {layer_acts[0].shape})\n")

    anchor_acts = layer_acts[ANCHOR_LAYER]
    n_layers = len(layer_acts)

    # --------------------------------------------------------
    # Full sweep
    # --------------------------------------------------------
    configs = list(itertools.product(SWEEP_SPLITS, SWEEP_DEPTH, SWEEP_ROUNDS))
    # shape: (n_configs, n_layers) -> purity
    purity_grid  = np.zeros((len(configs), n_layers))
    nparts_grid  = np.zeros((len(configs), n_layers), dtype=int)

    print("Running sweep...")
    t_sweep = time.perf_counter()
    for ci, (splits, depth, rounds) in enumerate(configs):
        for li, act in enumerate(layer_acts):
            ids = multi_round_geoxgb(
                anchor_acts, act,
                n_rounds=rounds,
                splits_per_round=splits,
                max_depth=depth,
                min_leaf=MIN_SAMPLES_LEAF,
                seed=42,
            )
            p, np_ = weighted_purity(labels, ids, n_topics)
            purity_grid[ci, li] = p
            nparts_grid[ci, li] = np_
    print(f"  Sweep done in {time.perf_counter() - t_sweep:.1f}s\n")

    # --------------------------------------------------------
    # Summary table — one row per config
    # --------------------------------------------------------
    print("=" * 72)
    print("Sweep results — sorted by peak purity (desc)")
    print(f"  Random baseline: {random_baseline:.4f}")
    print("=" * 72)
    print(f"  {'splits':>6}  {'depth':>6}  {'rounds':>6}  "
          f"{'peak purity':>11}  {'best layer':>10}  "
          f"{'mean purity':>11}  {'trend':>18}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*11}  {'-'*10}  {'-'*11}  {'-'*18}")

    summary = []
    for ci, (splits, depth, rounds) in enumerate(configs):
        row_purities = purity_grid[ci]
        peak = float(row_purities.max())
        best_l = int(row_purities.argmax())
        mean_p = float(row_purities.mean())
        half = n_layers // 2
        early_mean = float(row_purities[:half].mean())
        late_mean  = float(row_purities[half:].mean())
        trend = "increasing" if late_mean > early_mean else "flat/decr"
        summary.append((ci, splits, depth, rounds, peak, best_l,
                         mean_p, early_mean, late_mean, trend))

    summary.sort(key=lambda x: -x[4])  # sort by peak purity desc

    for ci, splits, depth, rounds, peak, best_l, mean_p, em, lm, trend in summary:
        depth_str = str(depth) if depth is not None else "None"
        print(f"  {splits:>6}  {depth_str:>6}  {rounds:>6}  "
              f"{peak:>11.4f}  {best_l:>10}  "
              f"{mean_p:>11.4f}  {trend:>18}")

    # --------------------------------------------------------
    # Best config detail: layer-by-layer table
    # --------------------------------------------------------
    best_ci, best_splits, best_depth, best_rounds, best_peak, best_peak_layer, \
        best_mean, *_ = summary[0]

    print(f"\n{'=' * 72}")
    print(f"Best config: splits={best_splits}, depth={best_depth}, "
          f"rounds={best_rounds}  (peak={best_peak:.4f} at layer {best_peak_layer})")
    print(f"{'=' * 72}\n")
    print(f"  {'Lyr':>4}  {'Purity':>8}  {'Parts':>6}  "
          f"{'vs baseline':>11}  {'vs flat-8':>10}")

    # Flat-8 baseline (rounds=1, splits=8 → single pass 8 partitions)
    flat_purities = np.zeros(n_layers)
    for li, act in enumerate(layer_acts):
        from hvrt import FastHVRT
        m = FastHVRT(n_partitions=8, min_samples_leaf=MIN_SAMPLES_LEAF,
                     auto_tune=False, n_jobs=-1, random_state=42)
        m.fit(act)
        flat_purities[li], _ = weighted_purity(labels, m.partition_ids_, n_topics)

    print(f"  {'-'*4}  {'-'*8}  {'-'*6}  {'-'*11}  {'-'*10}")
    for li in range(n_layers):
        p = purity_grid[best_ci, li]
        np_ = nparts_grid[best_ci, li]
        vs_base = p - random_baseline
        vs_flat = p - flat_purities[li]
        vs_base_str = f"+{vs_base:.4f}" if vs_base >= 0 else f"{vs_base:.4f}"
        vs_flat_str = f"+{vs_flat:.4f}" if vs_flat >= 0 else f"{vs_flat:.4f}"
        marker = " <-- peak" if li == best_peak_layer else ""
        print(f"  {li:>4}  {p:>8.4f}  {np_:>6}  "
              f"{vs_base_str:>11}  {vs_flat_str:>10}{marker}")

    # --------------------------------------------------------
    # Per-layer best config
    # --------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("Per-layer winning configuration")
    print(f"{'=' * 72}\n")
    print(f"  {'Lyr':>4}  {'Best purity':>11}  {'splits':>6}  "
          f"{'depth':>6}  {'rounds':>6}  {'Parts':>6}")
    print(f"  {'-'*4}  {'-'*11}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

    for li in range(n_layers):
        col = purity_grid[:, li]
        winner_ci = int(col.argmax())
        w_splits, w_depth, w_rounds = configs[winner_ci]
        w_purity = float(col[winner_ci])
        w_parts  = int(nparts_grid[winner_ci, li])
        depth_str = str(w_depth) if w_depth is not None else "None"
        anchor_tag = f"  <anchor L{ANCHOR_LAYER}>" if li == ANCHOR_LAYER else ""
        print(f"  {li:>4}  {w_purity:>11.4f}  {w_splits:>6}  "
              f"{depth_str:>6}  {w_rounds:>6}  {w_parts:>6}{anchor_tag}")

    # --------------------------------------------------------
    # Learning-rate axis summary
    # --------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("Learning-rate axis: splits=2 (low LR) vs splits=3 (high LR)")
    print(f"{'=' * 72}\n")
    for splits in SWEEP_SPLITS:
        idxs = [i for i, (s, d, r) in enumerate(configs) if s == splits]
        peak_purities = purity_grid[idxs].max(axis=1)
        lr_label = "low  (binary)" if splits == 2 else "high (ternary)"
        print(f"  splits={splits}  [{lr_label}]  "
              f"best={peak_purities.max():.4f}  "
              f"mean-of-bests={peak_purities.mean():.4f}")

    print(f"\n{'=' * 72}")
    print("Depth axis: effect of max_depth on peak purity")
    print(f"{'=' * 72}\n")
    for depth in SWEEP_DEPTH:
        idxs = [i for i, (s, d, r) in enumerate(configs) if d == depth]
        peak_purities = purity_grid[idxs].max(axis=1)
        depth_str = str(depth) if depth is not None else "None (unlimited)"
        print(f"  max_depth={depth_str:<12}  "
              f"best={peak_purities.max():.4f}  "
              f"mean-of-bests={peak_purities.mean():.4f}")

    print(f"\n{'=' * 72}")
    print("Rounds axis: effect of n_rounds on peak purity")
    print(f"{'=' * 72}\n")
    for rounds in SWEEP_ROUNDS:
        idxs = [i for i, (s, d, r) in enumerate(configs) if r == rounds]
        peak_purities = purity_grid[idxs].max(axis=1)
        lr_note = ("(single pass, no boosting)" if rounds == 1 else
                   f"({rounds} rounds of refinement)")
        print(f"  n_rounds={rounds}  {lr_note:<32}  "
              f"best={peak_purities.max():.4f}  "
              f"mean-of-bests={peak_purities.mean():.4f}")

    print(f"\n  Overall sweep peak  : {purity_grid.max():.4f}")
    print(f"  Flat-8 peak         : {flat_purities.max():.4f}")
    print(f"  Random baseline     : {random_baseline:.4f}")
    print(f"  Gain over flat-8    : {purity_grid.max() - flat_purities.max():+.4f}")
    print()


if __name__ == "__main__":
    main()
