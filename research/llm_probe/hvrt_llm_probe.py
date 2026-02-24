"""
HVRT LLM Activation Probe — Research Experiment
================================================

Tests whether HVRT's variance-aware hierarchical partitioning applied to
transformer hidden states produces semantically coherent clusters without
any supervision signal.

Hypothesis
----------
Later transformer layers encode more semantically discriminative
representations.  HVRT partitions fitted on those layers should therefore
cluster prompts of the same topic more tightly than partitions fitted on
early layers.  We measure this via:

1. **Weighted partition purity** — fraction of each partition occupied by
   its dominant topic; weighted by partition size.  Random baseline = 1/n_topics.
2. **Intra / inter cosine ratio** — mean cosine similarity within HVRT
   partitions divided by the global mean cosine similarity.  > 1 means
   HVRT groups geometrically similar activations together.

Important notes
---------------
- No LLM training required: GPT-2 is loaded frozen in eval mode.
- HVRT auto-tuner is disabled here.  It is designed for tabular data where
  d << n.  GPT-2 hidden size d=768 >> our prompt corpus n, so the auto-tuner
  would set min_samples_leaf > n and collapse the tree to a single leaf.
  Use explicit n_partitions / min_samples_leaf instead.
- FastHVRT is used instead of HVRT because d=768 makes the pairwise
  interaction computation in standard HVRT unnecessary (O(d²) vs O(d)).

Dependencies
------------
    pip install transformers torch
    (hvrt already installed in your environment)

Usage
-----
    python research/llm_probe/hvrt_llm_probe.py

Optional flags (set at top of file):
    MODEL_NAME    — any HuggingFace causal / encoder model name
    DEVICE        — 'cuda', 'cpu', or 'auto'
    N_PARTITIONS  — number of HVRT leaf partitions
"""

from __future__ import annotations

import time
import numpy as np

# ============================================================
# Configuration
# ============================================================

MODEL_NAME    = 'gpt2'        # HuggingFace model name
DEVICE        = 'auto'        # 'auto' selects CUDA if available
N_PARTITIONS  = 8             # HVRT target partition count
MIN_SAMPLES_LEAF = 4          # HVRT min samples per leaf
MAX_DEPTH     = None          # HVRT tree max depth (None = unlimited)

# ============================================================
# Prompt corpus  (10 prompts × 6 topics = 60 samples total)
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
# Activation extraction
# ============================================================

def load_model(model_name: str, device: str):
    """Load a frozen GPT-2 (or compatible) model in eval mode."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Install transformers and torch:\n"
            "    pip install transformers torch"
        ) from exc

    import torch

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"  Loading '{model_name}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def extract_all_layers(
    prompts: list[str],
    model,
    tokenizer,
    device: str,
    max_length: int = 64,
) -> list[np.ndarray]:
    """
    Extract mean-pooled hidden states from every transformer layer.

    Returns
    -------
    layer_activations : list of ndarray, length n_layers+1
        Each element has shape (n_prompts, hidden_size).
        Index 0 = embedding output; index k = output of transformer block k.
    """
    import torch

    with torch.no_grad():
        all_hidden: list[list[np.ndarray]] = []  # [layer][prompt]

        for i, prompt in enumerate(prompts):
            enc = tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=False,
            ).to(device)

            out = model(**enc)
            # hidden_states: tuple of (n_layers+1,) tensors, each (1, seq_len, d)
            hidden_states = out.hidden_states

            if i == 0:
                # Initialise list-of-lists on first prompt
                all_hidden = [[] for _ in hidden_states]

            for layer_idx, h in enumerate(hidden_states):
                # Mean-pool across sequence dimension
                pooled = h.squeeze(0).mean(dim=0).cpu().float().numpy()
                all_hidden[layer_idx].append(pooled)

    return [np.array(layer, dtype=np.float32) for layer in all_hidden]


# ============================================================
# Analysis helpers
# ============================================================

def partition_purity(
    labels: np.ndarray,
    partition_ids: np.ndarray,
    n_topics: int,
) -> dict:
    """
    Measure how well HVRT partition assignments align with topic labels.

    Returns
    -------
    dict with keys:
        weighted_purity   — partition-size-weighted average purity
        per_partition     — {pid: purity}
        sizes             — {pid: n_samples}
    """
    unique_parts = np.unique(partition_ids)
    purities: dict[int, float] = {}
    sizes: dict[int, int] = {}

    for pid in unique_parts:
        mask = partition_ids == pid
        part_labels = labels[mask]
        n = int(mask.sum())
        counts = np.bincount(part_labels, minlength=n_topics)
        purities[int(pid)] = float(counts.max() / n)
        sizes[int(pid)] = n

    total = sum(sizes.values())
    weighted = sum(purities[p] * sizes[p] for p in purities) / total
    return {'weighted_purity': weighted, 'per_partition': purities, 'sizes': sizes}


def cosine_ratio(
    activations: np.ndarray,
    partition_ids: np.ndarray,
    n_global_pairs: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute mean within-partition cosine similarity and a global baseline.

    Returns (intra_mean, inter_mean).  intra/inter > 1 means HVRT groups
    geometrically similar activations.
    """
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    A = (activations / norms).astype(np.float32)

    unique_parts = np.unique(partition_ids)
    within_sims: list[float] = []
    for pid in unique_parts:
        part = A[partition_ids == pid]
        if len(part) < 2:
            continue
        sim = part @ part.T
        n = len(part)
        idx = np.triu_indices(n, k=1)
        within_sims.extend(sim[idx].tolist())

    rng = np.random.default_rng(seed)
    n_total = len(activations)
    n_pairs = min(n_global_pairs, n_total * (n_total - 1) // 2)
    ii = rng.integers(0, n_total, size=n_pairs)
    jj = rng.integers(0, n_total, size=n_pairs)
    valid = ii != jj
    global_sims = (A[ii[valid]] * A[jj[valid]]).sum(axis=1).tolist()

    return float(np.mean(within_sims)) if within_sims else 0.0, float(np.mean(global_sims))


# ============================================================
# Main experiment
# ============================================================

def main():
    from hvrt import HVRT

    # --- Flatten corpus ---
    all_prompts: list[str] = []
    all_labels: list[int] = []
    topic_names = list(PROMPTS.keys())
    for label_idx, (topic, texts) in enumerate(PROMPTS.items()):
        all_prompts.extend(texts)
        all_labels.extend([label_idx] * len(texts))
    labels = np.array(all_labels, dtype=np.int64)
    n_total = len(all_prompts)
    n_topics = len(topic_names)
    random_baseline = 1.0 / n_topics

    print("\nHVRT LLM Activation Probe")
    print("=" * 72)
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Prompts     : {n_total}  ({n_topics} topics × {n_total // n_topics} each)")
    print(f"  Topics      : {topic_names}")
    print(f"  HVRT        : HVRT (pairwise O(d^2)), n_partitions={N_PARTITIONS}, "
          f"min_samples_leaf={MIN_SAMPLES_LEAF}")
    print(f"  Random purity baseline: {random_baseline:.4f}")

    # --- Load model ---
    print()
    model, tokenizer, device = load_model(MODEL_NAME, DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    try:
        n_layers = model.config.n_layer
        hidden_size = model.config.n_embd
    except AttributeError:
        # Fallback for non-GPT2 configs
        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
    print(f"  Parameters  : {n_params:,}")
    print(f"  Layers      : {n_layers}, hidden size: {hidden_size}")

    # --- Extract activations ---
    print(f"\nExtracting hidden states from {n_layers + 1} layers "
          f"({n_total} prompts × {hidden_size}d)...")
    t0 = time.perf_counter()
    layer_activations = extract_all_layers(all_prompts, model, tokenizer, device)
    print(f"  Done in {time.perf_counter() - t0:.1f}s  "
          f"— {len(layer_activations)} layers × {layer_activations[0].shape}")

    # Free GPU memory now that we have all activations as numpy arrays
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # --- Layer-by-layer HVRT analysis ---
    print(f"\n{'=' * 72}")
    print("HVRT Partition Analysis — per layer")
    print(f"{'=' * 72}\n")
    print(f"  {'Lyr':>4}  {'Parts':>6}  {'Purity':>8}  {'dBaseline':>10}"
          f"  {'IntraCos':>9}  {'InterCos':>9}  {'Ratio':>6}  {'FitTime':>8}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*6}  {'-'*8}")

    results = []
    best_purity_layer = 0
    best_purity = -1.0

    for layer_idx, act in enumerate(layer_activations):
        t_fit = time.perf_counter()
        hvrt = HVRT(
            n_partitions=N_PARTITIONS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_depth=MAX_DEPTH,
            auto_tune=False,
            n_jobs=-1,
            random_state=42,
        )
        hvrt.fit(act)
        fit_time = time.perf_counter() - t_fit
        part_ids = hvrt.partition_ids_

        purity_info = partition_purity(labels, part_ids, n_topics)
        intra, inter = cosine_ratio(act, part_ids)
        ratio = intra / max(inter, 1e-10)
        wp = purity_info['weighted_purity']
        n_parts = len(np.unique(part_ids))

        results.append({
            'layer': layer_idx,
            'n_parts': n_parts,
            'weighted_purity': wp,
            'intra_cos': intra,
            'inter_cos': inter,
            'ratio': ratio,
            'fit_time': fit_time,
        })

        delta = wp - random_baseline
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"  {layer_idx:>4}  {n_parts:>6}  {wp:>8.4f}  {delta_str:>10}"
              f"  {intra:>9.4f}  {inter:>9.4f}  {ratio:>6.3f}  {fit_time:>7.2f}s")

        if wp > best_purity:
            best_purity = wp
            best_purity_layer = layer_idx

    # --- Detailed breakdown at best layer ---
    best_act = layer_activations[best_purity_layer]
    hvrt_best = HVRT(
        n_partitions=N_PARTITIONS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_depth=MAX_DEPTH,
        auto_tune=False,
        n_jobs=-1,
        random_state=42,
    )
    hvrt_best.fit(best_act)
    part_ids_best = hvrt_best.partition_ids_
    purity_best = partition_purity(labels, part_ids_best, n_topics)

    print(f"\n{'=' * 72}")
    print(f"Partition breakdown — Layer {best_purity_layer} "
          f"(best purity: {best_purity:.4f})")
    print(f"{'=' * 72}\n")
    print(f"  {'Part':>5}  {'Size':>5}  {'Purity':>7}  Dominant topic   Composition")
    print(f"  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*16}  {'-'*36}")

    for pid, purity in sorted(
        purity_best['per_partition'].items(), key=lambda x: -x[1]
    ):
        mask = part_ids_best == pid
        part_labels = labels[mask]
        counts = np.bincount(part_labels, minlength=n_topics)
        dominant = topic_names[int(np.argmax(counts))]
        size = purity_best['sizes'][pid]
        composition = '  '.join(
            f"{topic_names[i][:4]}:{counts[i]}"
            for i in np.argsort(-counts)
            if counts[i] > 0
        )
        print(f"  {pid:>5}  {size:>5}  {purity:>7.3f}  {dominant:<16}  {composition}")

    # --- Summary ---
    print(f"\n{'=' * 72}")
    print("Summary")
    print(f"{'=' * 72}\n")

    # Layer with highest purity
    print(f"  Best purity layer : {best_purity_layer}  (purity={best_purity:.4f}, "
          f"baseline={random_baseline:.4f}, "
          f"d={best_purity - random_baseline:+.4f})")

    # Layer with highest intra/inter ratio
    best_ratio_row = max(results, key=lambda r: r['ratio'])
    print(f"  Best cosine ratio : layer {best_ratio_row['layer']}  "
          f"(ratio={best_ratio_row['ratio']:.3f})")

    # Trend: does purity increase across layers?
    purities_by_layer = [r['weighted_purity'] for r in results]
    first_half_mean = float(np.mean(purities_by_layer[: len(purities_by_layer) // 2]))
    second_half_mean = float(np.mean(purities_by_layer[len(purities_by_layer) // 2:]))
    trend = "increasing" if second_half_mean > first_half_mean else "flat/decreasing"
    print(f"  Purity trend      : {trend}  "
          f"(early layers mean={first_half_mean:.4f}, "
          f"late layers mean={second_half_mean:.4f})")

    print(f"\n  Interpretation:")
    if best_purity > random_baseline + 0.05:
        print(f"  HVRT partitions are semantically coherent above chance. "
              f"Partition purity ({best_purity:.3f}) exceeds random ({random_baseline:.3f}) "
              f"by {best_purity - random_baseline:+.3f}.")
    elif best_purity > random_baseline:
        print(f"  Weak above-chance semantic coherence. Consider more prompts "
              f"or larger n_partitions to resolve finer structure.")
    else:
        print(f"  No above-chance semantic coherence detected. "
              f"The corpus may be too small or N_PARTITIONS too large.")
    print()


if __name__ == "__main__":
    main()
