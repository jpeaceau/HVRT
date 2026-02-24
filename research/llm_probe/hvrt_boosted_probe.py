"""
HVRT LLM Activation Probe — Boosted / GeoXGB-Inspired Edition
=============================================================

Compares three partitioning strategies applied to GPT-2 hidden states:

  1. Flat HVRT        — single pass, n_partitions=8  (baseline, same as
                        hvrt_llm_probe.py with FastHVRT)

  2. Hierarchical     — two rounds on the *same* layer's activations.
                        Round 1: 4 coarse partitions.
                        Round 2: within each coarse partition, 2 finer
                        sub-partitions (max 8 final, matching flat count).
                        Asks: does iterative refinement on a single layer
                        outperform a single flat pass?

  3. GeoXGB-inspired  — round 1 on Layer 2 (the proven semantic anchor from
                        the baseline experiment), round 2 on the *target*
                        layer's activations within those anchors.
                        Asks: does conditioning on early-layer coarse structure
                        reveal finer semantic sub-structure in deeper layers
                        that a flat pass misses?

The GeoXGB philosophy: identify geometric subregions with an early/cheap
learner, then train independent residual learners within each region.
Here the "residual learner" is a second HVRT fit on the deeper layer's
activations restricted to the coarse partition's members.

Parameters
----------
  ANCHOR_LAYER      Layer used as base for the cross-layer approach.
                    Set to the best-purity layer from the baseline run.
  R1_PARTS          Round-1 partition count (both hierarchical & cross-layer).
  R2_PARTS          Round-2 sub-partition count per round-1 group.
  MIN_SAMPLES_LEAF  Minimum leaf size; round-2 is skipped for groups too small.

Usage
-----
    python research/llm_probe/hvrt_boosted_probe.py
"""

from __future__ import annotations

import time
import numpy as np

# ============================================================
# Configuration
# ============================================================

MODEL_NAME       = 'gpt2'
DEVICE           = 'auto'
ANCHOR_LAYER     = 2       # best purity layer from the FastHVRT baseline run
R1_PARTS         = 4       # round-1 coarse partitions
R2_PARTS         = 2       # round-2 sub-partitions per coarse group
MIN_SAMPLES_LEAF = 4       # minimum partition size

# Flat baseline uses the same total budget for fair comparison
FLAT_PARTS = R1_PARTS * R2_PARTS   # = 8

# ============================================================
# Prompt corpus  (identical to hvrt_llm_probe.py)
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
# Activation extraction  (shared with hvrt_llm_probe.py)
# ============================================================

def load_model(model_name: str, device: str):
    import torch
    from transformers import GPT2Model, GPT2Tokenizer
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer, device


def extract_all_layers(prompts, model, tokenizer, device, max_length=64):
    import torch
    with torch.no_grad():
        all_hidden = None
        for prompt in prompts:
            enc = tokenizer(
                prompt, return_tensors='pt', truncation=True,
                max_length=max_length, padding=False,
            ).to(device)
            out = model(**enc, output_hidden_states=True)
            if all_hidden is None:
                all_hidden = [[] for _ in out.hidden_states]
            for i, h in enumerate(out.hidden_states):
                all_hidden[i].append(h.squeeze(0).mean(0).cpu().float().numpy())
    return [np.array(layer, dtype=np.float32) for layer in all_hidden]

# ============================================================
# Partitioning strategies
# ============================================================

def flat_hvrt(activations: np.ndarray, n_parts: int,
              min_leaf: int, seed: int) -> np.ndarray:
    """Single-pass HVRT — baseline."""
    from hvrt import FastHVRT
    m = FastHVRT(n_partitions=n_parts, min_samples_leaf=min_leaf,
                 auto_tune=False, n_jobs=-1, random_state=seed)
    m.fit(activations)
    return m.partition_ids_


def hierarchical_hvrt(activations: np.ndarray, r1_parts: int, r2_parts: int,
                      min_leaf: int, seed: int) -> np.ndarray:
    """
    Two-round HVRT on the same layer.

    Round 1 splits the full set into r1_parts coarse groups.
    Round 2 splits each coarse group into up to r2_parts finer groups,
    skipping groups too small to split (< min_leaf * r2_parts samples).
    """
    from hvrt import FastHVRT

    m1 = FastHVRT(n_partitions=r1_parts, min_samples_leaf=min_leaf,
                  auto_tune=False, n_jobs=-1, random_state=seed)
    m1.fit(activations)
    r1_ids = m1.partition_ids_

    compound = np.full(len(activations), -1, dtype=np.int64)
    cid = 0
    for r1_pid in np.unique(r1_ids):
        mask = r1_ids == r1_pid
        n_sub = int(mask.sum())
        X_sub = activations[mask]

        if n_sub < min_leaf * r2_parts:
            compound[mask] = cid
            cid += 1
            continue

        n_r2 = min(r2_parts, n_sub // min_leaf)
        try:
            m2 = FastHVRT(n_partitions=n_r2, min_samples_leaf=min_leaf,
                          auto_tune=False, n_jobs=1,
                          random_state=seed + int(r1_pid))
            m2.fit(X_sub)
            r2_ids = m2.partition_ids_
            for r2_pid in np.unique(r2_ids):
                sub_mask = r2_ids == r2_pid
                compound[np.where(mask)[0][sub_mask]] = cid
                cid += 1
        except Exception:
            compound[mask] = cid
            cid += 1

    return compound


def geoxgb_inspired(anchor_activations: np.ndarray,
                    target_activations: np.ndarray,
                    r1_parts: int, r2_parts: int,
                    min_leaf: int, seed: int) -> np.ndarray:
    """
    GeoXGB-inspired cross-layer partitioning.

    Round 1 (base learner): fit FastHVRT on anchor_activations (early
    layer with proven semantic signal) to obtain coarse anchors.

    Round 2 (residual learner): within each anchor region, fit a
    second FastHVRT on target_activations — the deeper layer's
    activations for that same subset of prompts.

    The hypothesis: early layers define coarse topic boundaries;
    deeper layers contain finer within-topic structure that is only
    visible once you isolate each topic's geometric neighbourhood.
    """
    from hvrt import FastHVRT

    # Base learner on anchor layer
    m_anchor = FastHVRT(n_partitions=r1_parts, min_samples_leaf=min_leaf,
                        auto_tune=False, n_jobs=-1, random_state=seed)
    m_anchor.fit(anchor_activations)
    anchor_ids = m_anchor.partition_ids_

    compound = np.full(len(target_activations), -1, dtype=np.int64)
    cid = 0
    for apid in np.unique(anchor_ids):
        mask = anchor_ids == apid
        n_sub = int(mask.sum())
        X_target_sub = target_activations[mask]

        if n_sub < min_leaf * r2_parts:
            compound[mask] = cid
            cid += 1
            continue

        n_r2 = min(r2_parts, n_sub // min_leaf)
        try:
            m_res = FastHVRT(n_partitions=n_r2, min_samples_leaf=min_leaf,
                             auto_tune=False, n_jobs=1,
                             random_state=seed + int(apid))
            m_res.fit(X_target_sub)
            res_ids = m_res.partition_ids_
            for rpid in np.unique(res_ids):
                sub_mask = res_ids == rpid
                compound[np.where(mask)[0][sub_mask]] = cid
                cid += 1
        except Exception:
            compound[mask] = cid
            cid += 1

    return compound

# ============================================================
# Analysis helpers
# ============================================================

def partition_purity(labels: np.ndarray, partition_ids: np.ndarray,
                     n_topics: int) -> tuple[float, int]:
    """Return (weighted_purity, n_final_partitions)."""
    unique_parts = np.unique(partition_ids)
    total = len(labels)
    wp = 0.0
    for pid in unique_parts:
        mask = partition_ids == pid
        part_labels = labels[mask]
        n = int(mask.sum())
        counts = np.bincount(part_labels, minlength=n_topics)
        wp += (counts.max() / n) * (n / total)
    return wp, len(unique_parts)


def breakdown(labels: np.ndarray, partition_ids: np.ndarray,
              topic_names: list[str]) -> list[dict]:
    """Per-partition composition details, sorted by purity desc."""
    n_topics = len(topic_names)
    rows = []
    for pid in np.unique(partition_ids):
        mask = partition_ids == pid
        part_labels = labels[mask]
        n = int(mask.sum())
        counts = np.bincount(part_labels, minlength=n_topics)
        dominant = topic_names[int(np.argmax(counts))]
        purity = float(counts.max() / n)
        composition = '  '.join(
            f"{topic_names[i][:4]}:{counts[i]}"
            for i in np.argsort(-counts) if counts[i] > 0
        )
        rows.append({'pid': int(pid), 'n': n, 'purity': purity,
                     'dominant': dominant, 'composition': composition})
    return sorted(rows, key=lambda r: -r['purity'])

# ============================================================
# Main
# ============================================================

def main():
    # --- Corpus ---
    all_prompts, all_labels = [], []
    topic_names = list(PROMPTS.keys())
    n_topics = len(topic_names)
    for idx, (topic, texts) in enumerate(PROMPTS.items()):
        all_prompts.extend(texts)
        all_labels.extend([idx] * len(texts))
    labels = np.array(all_labels, dtype=np.int64)
    n_total = len(all_prompts)
    random_baseline = 1.0 / n_topics

    print("\nHVRT LLM Activation Probe — GeoXGB-Inspired Edition")
    print("=" * 72)
    print(f"  Model        : {MODEL_NAME}")
    print(f"  Prompts      : {n_total}  ({n_topics} topics x {n_total // n_topics} each)")
    print(f"  Anchor layer : {ANCHOR_LAYER}  (best-purity layer from baseline)")
    print(f"  Strategies   : Flat({FLAT_PARTS}p)  |  "
          f"Hierarchical({R1_PARTS}x{R2_PARTS})  |  "
          f"GeoXGB({R1_PARTS}anchor+{R2_PARTS}residual)")
    print(f"  Random baseline purity: {random_baseline:.4f}")

    # --- Load & extract ---
    print()
    model, tokenizer, device = load_model(MODEL_NAME, DEVICE)
    n_layers = model.config.n_layer
    print(f"  Loaded GPT-2  ({n_layers} layers, d={model.config.n_embd}, device={device})")

    print(f"\nExtracting activations ({n_total} prompts, {n_layers + 1} layers)...")
    t0 = time.perf_counter()
    layer_acts = extract_all_layers(all_prompts, model, tokenizer, device)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    anchor_acts = layer_acts[ANCHOR_LAYER]

    # --- Layer-by-layer comparison ---
    print(f"\n{'=' * 72}")
    print("Layer-by-layer purity comparison")
    print(f"  (Flat={FLAT_PARTS}p  |  Hier={R1_PARTS}x{R2_PARTS}  |  "
          f"GeoXGB=anchor@L{ANCHOR_LAYER}+residual@L)")
    print(f"{'=' * 72}\n")
    print(f"  {'Lyr':>4}  "
          f"{'Flat':>7}  {'Np':>3}  "
          f"{'Hier':>7}  {'Np':>3}  "
          f"{'GeoXGB':>7}  {'Np':>3}  "
          f"{'Best':>8}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*3}  {'-'*7}  {'-'*3}  {'-'*7}  {'-'*3}  {'-'*8}")

    results = []
    best_geo_layer, best_geo_purity = 0, -1.0

    for layer_idx, act in enumerate(layer_acts):
        seed = 42

        ids_flat  = flat_hvrt(act, FLAT_PARTS, MIN_SAMPLES_LEAF, seed)
        ids_hier  = hierarchical_hvrt(act, R1_PARTS, R2_PARTS, MIN_SAMPLES_LEAF, seed)
        ids_geo   = geoxgb_inspired(anchor_acts, act, R1_PARTS, R2_PARTS,
                                     MIN_SAMPLES_LEAF, seed)

        p_flat,  n_flat  = partition_purity(labels, ids_flat,  n_topics)
        p_hier,  n_hier  = partition_purity(labels, ids_hier,  n_topics)
        p_geo,   n_geo   = partition_purity(labels, ids_geo,   n_topics)

        best_tag = max(
            [('flat', p_flat), ('hier', p_hier), ('geo', p_geo)],
            key=lambda x: x[1]
        )[0]

        results.append({
            'layer': layer_idx,
            'flat': p_flat,  'n_flat': n_flat,
            'hier': p_hier,  'n_hier': n_hier,
            'geo':  p_geo,   'n_geo':  n_geo,
            'best': best_tag,
        })

        anchor_tag = f'<anchor L{ANCHOR_LAYER}>' if layer_idx == ANCHOR_LAYER else ''

        # Bold-ish marker for winner
        def fmt(p, tag):
            s = f"{p:.4f}"
            return f"[{s}]" if tag == best_tag else f" {s} "

        print(f"  {layer_idx:>4}  "
              f"{fmt(p_flat,'flat'):>8}  {n_flat:>3}  "
              f"{fmt(p_hier,'hier'):>8}  {n_hier:>3}  "
              f"{fmt(p_geo,'geo'):>8}  {n_geo:>3}  "
              f"  {best_tag:<8}{anchor_tag}")

        if p_geo > best_geo_purity:
            best_geo_purity = p_geo
            best_geo_layer = layer_idx

    # --- Win counts ---
    wins = {'flat': 0, 'hier': 0, 'geo': 0}
    for r in results:
        wins[r['best']] += 1

    print(f"\n  Win counts  (across {len(results)} layers):")
    print(f"    Flat HVRT({FLAT_PARTS}p)    : {wins['flat']:>2} layers")
    print(f"    Hierarchical({R1_PARTS}x{R2_PARTS}) : {wins['hier']:>2} layers")
    print(f"    GeoXGB-inspired  : {wins['geo']:>2} layers")

    # --- GeoXGB breakdown at best geo layer ---
    ids_geo_best = geoxgb_inspired(anchor_acts, layer_acts[best_geo_layer],
                                    R1_PARTS, R2_PARTS, MIN_SAMPLES_LEAF, 42)
    rows = breakdown(labels, ids_geo_best, topic_names)

    print(f"\n{'=' * 72}")
    print(f"GeoXGB partition breakdown — best geo layer: Layer {best_geo_layer} "
          f"(purity={best_geo_purity:.4f})")
    print(f"{'=' * 72}\n")
    print(f"  {'Part':>5}  {'Size':>5}  {'Purity':>7}  "
          f"{'Dominant':<16}  Composition")
    print(f"  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*16}  {'-'*38}")
    for r in rows:
        print(f"  {r['pid']:>5}  {r['n']:>5}  {r['purity']:>7.3f}  "
              f"{r['dominant']:<16}  {r['composition']}")

    # --- Trend analysis ---
    n_layers_total = len(results)
    half = n_layers_total // 2

    def trend(key):
        early = np.mean([r[key] for r in results[:half]])
        late  = np.mean([r[key] for r in results[half:]])
        direction = "increasing" if late > early else "flat/decreasing"
        return early, late, direction

    e_f, l_f, d_f = trend('flat')
    e_h, l_h, d_h = trend('hier')
    e_g, l_g, d_g = trend('geo')

    print(f"\n{'=' * 72}")
    print("Trend analysis  (early-layer mean vs late-layer mean purity)")
    print(f"{'=' * 72}\n")
    print(f"  {'Strategy':<24}  {'Early mean':>10}  {'Late mean':>10}  Trend")
    print(f"  {'-'*24}  {'-'*10}  {'-'*10}  {'-'*18}")
    print(f"  {'Flat HVRT':<24}  {e_f:>10.4f}  {l_f:>10.4f}  {d_f}")
    print(f"  {'Hierarchical HVRT':<24}  {e_h:>10.4f}  {l_h:>10.4f}  {d_h}")
    print(f"  {'GeoXGB-inspired':<24}  {e_g:>10.4f}  {l_g:>10.4f}  {d_g}")

    print(f"\n{'=' * 72}")
    print("Summary")
    print(f"{'=' * 72}\n")

    best_geo_row = max(results, key=lambda r: r['geo'])
    best_flat_row = max(results, key=lambda r: r['flat'])
    best_hier_row = max(results, key=lambda r: r['hier'])
    print(f"  Peak purity — Flat     : {best_flat_row['flat']:.4f}  at layer {best_flat_row['layer']}")
    print(f"  Peak purity — Hier     : {best_hier_row['hier']:.4f}  at layer {best_hier_row['layer']}")
    print(f"  Peak purity — GeoXGB   : {best_geo_row['geo']:.4f}  at layer {best_geo_row['layer']}")
    print(f"  Random baseline        : {random_baseline:.4f}")

    geo_gain = best_geo_row['geo'] - best_flat_row['flat']
    sign = '+' if geo_gain >= 0 else ''
    print(f"\n  GeoXGB vs flat peak    : {sign}{geo_gain:.4f}  "
          f"({'improvement' if geo_gain >= 0 else 'regression'})")

    if d_g == "increasing":
        print(f"\n  GeoXGB trend is INCREASING — deeper layers do contain finer")
        print(f"  sub-structure once coarse anchors from Layer {ANCHOR_LAYER} are applied.")
    else:
        print(f"\n  GeoXGB trend is flat/decreasing — finer structure is not")
        print(f"  progressively revealed by cross-layer conditioning alone.")
    print()


if __name__ == "__main__":
    main()
