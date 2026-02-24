"""
HVRT Boosting Corrector — GPT-2 Perplexity Experiment
======================================================

Variant B: fit HVRT on GPT-2's last-layer hidden states and use
variance-homogeneous partitions to learn logit corrections.
No backpropagation. No model surgery. GPT-2 is frozen throughout.

Algorithm
---------
1. Run GPT-2 on a WikiText-103 train subset.
   Collect last-layer hidden states + true next-tokens.
2. Fit FastHVRT on the hidden states.
   The tree uses model.tree_.apply() to assign new tokens at inference.
3. Per partition: compute a logit-bias vector from the log-ratio of
   (empirical token frequency in partition) vs (GPT-2 expected frequency).
4. At inference: corrected_logits = gpt2_logits + alpha * bias[partition]
5. Tune alpha on a held-out slice of the training split.
6. Evaluate perplexity on the WikiText-103 validation set.

Sweep: n_partitions ∈ {4, 8, 16, 32, 64, 128}

Key question: how much of GPT-2's systematic error is geometrically
structured in hidden-state space — and how cheaply can we correct it?

Usage
-----
    pip install datasets
    python research/ffn_probe/hvrt_corrector.py
"""

from __future__ import annotations

import math
import time
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_NAME       = "gpt2"
N_FIT_TOKENS     = 51_200     # train tokens used to fit corrector
N_VAL_TOKENS     = 20_480     # val tokens used to evaluate perplexity
SEQ_LEN          = 128        # tokens per sequence
BATCH_SIZE       = 8          # sequences per forward pass (= 1024 tokens)
ALPHA_GRID       = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
PARTITIONS_SWEEP = [4, 8, 16, 32, 64, 128]
MIN_LEAF         = 16         # hard floor on partition size
VAL_FRAC         = 0.15       # fraction of FIT data held out for alpha tuning

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_tokens(tokenizer, split: str, n_tokens: int) -> list[int]:
    """
    Load and tokenize WikiText-103 incrementally up to n_tokens.
    Streams article-by-article so the full corpus is never held in RAM.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install datasets:  pip install datasets")

    ds  = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    ids: list[int] = []
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        ids.extend(tokenizer.encode(text))
        if len(ids) >= n_tokens + 1:
            break
    return ids[: n_tokens + 1]   # +1 because targets are shifted by 1


# -----------------------------------------------------------------------------
# Activation collection
# -----------------------------------------------------------------------------

def collect_activations(
    model, tokenizer, token_ids: list[int], device: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run GPT-2 and collect (last_hidden_state, true_next_token) pairs.

    Returns
    -------
    hidden : (n_tokens, d_model)  float32
    labels : (n_tokens,)          int32   true next-token IDs
    """
    model.eval()
    all_ids   = torch.tensor(token_ids, dtype=torch.long)
    n_seqs    = (len(all_ids) - 1) // SEQ_LEN
    inputs    = all_ids[: n_seqs * SEQ_LEN].reshape(n_seqs, SEQ_LEN)
    targets   = all_ids[1: n_seqs * SEQ_LEN + 1].reshape(n_seqs, SEQ_LEN)

    hidden_list, label_list = [], []

    with torch.no_grad():
        for i in range(0, n_seqs, BATCH_SIZE):
            inp = inputs[i : i + BATCH_SIZE].to(device)
            tgt = targets[i : i + BATCH_SIZE].to(device)

            out = model(inp, output_hidden_states=True)
            h   = out.hidden_states[-1]   # (b, seq, d)

            hidden_list.append(h.reshape(-1, h.shape[-1]).cpu().float().numpy())
            label_list.append(tgt.reshape(-1).cpu().numpy().astype(np.int32))

    return np.vstack(hidden_list), np.concatenate(label_list)


# -----------------------------------------------------------------------------
# Partition-assignment on new data
# -----------------------------------------------------------------------------

def assign_partitions(hvrt_model, X_new: np.ndarray) -> np.ndarray:
    """
    Use the fitted HVRT tree to assign new hidden states to partition IDs.

    Relies on:
      hvrt_model._to_z(X)       — apply the same z-normalisation used in fit()
      hvrt_model.tree_.apply(Xz) — sklearn DecisionTreeRegressor leaf IDs
    """
    X_z = hvrt_model._to_z(X_new.astype(np.float64))
    return hvrt_model.tree_.apply(X_z).astype(np.int32)


# -----------------------------------------------------------------------------
# Bias-vector computation
# -----------------------------------------------------------------------------

def compute_bias_vectors(
    model,
    tokenizer,
    hvrt_model,
    hidden_train: np.ndarray,
    labels_train: np.ndarray,
    device: str,
    vocab_size: int,
) -> dict[int, np.ndarray]:
    """
    For each partition p, compute a logit-bias vector:

        bias_p[v] = log(empirical_freq_p[v] + eps)
                  - log_softmax(mean_logits_p)[v]

    The first term is the empirical next-token distribution within the partition.
    The second term is what GPT-2 expects on average within the partition.
    Their log-difference is the direction we should push the logits.

    Implementation uses two cheap passes over the training data:
      Pass 1 (done): hidden states + true tokens already collected.
      Pass 2: recompute GPT-2 logits in batches; accumulate per-partition
              mean logits WITHOUT storing all logits simultaneously.
    """
    # -- Partition assignments -------------------------------------------------
    pids         = assign_partitions(hvrt_model, hidden_train)
    unique_parts = np.unique(pids)
    n_parts      = len(unique_parts)
    eps          = 1e-8

    # -- Empirical frequency per partition ------------------------------------
    # Shape: (n_parts, vocab_size) — counts of true next tokens
    part_to_idx = {int(p): i for i, p in enumerate(unique_parts)}
    token_counts = np.zeros((n_parts, vocab_size), dtype=np.float64)
    for i, pid in enumerate(pids):
        token_counts[part_to_idx[int(pid)], labels_train[i]] += 1.0
    empirical_freq = token_counts / (token_counts.sum(axis=1, keepdims=True) + eps)

    # -- Mean GPT-2 logits per partition (second pass) -------------------------
    # Streaming accumulation — only 1 batch of logits in memory at a time.
    mean_logits = np.zeros((n_parts, vocab_size), dtype=np.float64)
    part_counts = np.zeros(n_parts, dtype=np.float64)

    model.eval()
    n = len(hidden_train)
    n_seqs   = n // SEQ_LEN
    # Reconstruct the same token-sequence layout used in collect_activations
    # We need the INPUT token IDs, not the targets.
    # Since we don't re-store them, we rerun a forward pass in token order.
    # hidden_train is already ordered token-by-token, so pids[i] gives
    # the partition for token position i. We need logits for those same positions.
    #
    # Strategy: run the model on sequential (SEQ_LEN,) chunks in batch.
    # The logit at position t within a chunk corresponds to hidden_train[chunk*SEQ_LEN + t].

    hidden_flat_idx = 0
    with torch.no_grad():
        for chunk_start in range(0, n_seqs * BATCH_SIZE, BATCH_SIZE):
            # We don't have the original input_ids here, so we re-extract logits
            # by running the model on SEQ_LEN-long sequences.
            # Use the hidden states themselves as a proxy to find the right batch:
            chunk_h = hidden_train[
                chunk_start * SEQ_LEN: (chunk_start + BATCH_SIZE) * SEQ_LEN
            ]
            if len(chunk_h) == 0:
                break
            # We can't recompute logits without the original token IDs.
            # Fall back to: logit = lm_head @ hidden (exactly what GPT-2 does).
            # This avoids a second full forward pass.
            h_tensor = torch.from_numpy(chunk_h.astype(np.float32)).to(device)
            logits   = model.lm_head(model.transformer.ln_f(h_tensor))  # (n_tok, vocab)
            logits_np = logits.float().cpu().numpy()

            for j in range(len(chunk_h)):
                gi  = chunk_start * SEQ_LEN + j
                if gi >= n:
                    break
                pid  = pids[gi]
                pidx = part_to_idx[int(pid)]
                mean_logits[pidx] += logits_np[j]
                part_counts[pidx] += 1.0

    mean_logits /= np.maximum(part_counts[:, None], 1.0)

    # -- Log-ratio bias --------------------------------------------------------
    # log_softmax(mean_logits_p)
    log_gpt2 = mean_logits - np.log(
        np.exp(mean_logits - mean_logits.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)
    ) - mean_logits.max(axis=1, keepdims=True)
    # numerically stable log-softmax: log_softmax(x) = x - logsumexp(x)
    # recompute properly
    log_gpt2_stable = (
        mean_logits
        - (np.log(np.exp(
            mean_logits - mean_logits.max(axis=1, keepdims=True)
        ).sum(axis=1, keepdims=True))
        + mean_logits.max(axis=1, keepdims=True))
    )

    log_empirical = np.log(empirical_freq + eps)

    # bias[pidx, v] = log_empirical[v] - log_gpt2[v]  (log-ratio)
    bias_dict: dict[int, np.ndarray] = {}
    for i, p in enumerate(unique_parts):
        bias_dict[int(p)] = (log_empirical[i] - log_gpt2_stable[i]).astype(np.float32)

    return bias_dict, pids, unique_parts


# -----------------------------------------------------------------------------
# Perplexity evaluation
# -----------------------------------------------------------------------------

def evaluate_perplexity(
    model,
    token_ids: list[int],
    device: str,
    hvrt_model=None,
    bias_dict: dict | None = None,
    alpha: float = 0.0,
    vocab_size: int = 50257,
) -> float:
    """
    Compute perplexity on a flat token-ID list.

    If hvrt_model and bias_dict are provided, apply the logit correction.
    """
    model.eval()
    all_ids = torch.tensor(token_ids, dtype=torch.long)
    n_seqs  = (len(all_ids) - 1) // SEQ_LEN
    if n_seqs == 0:
        return float("nan")
    inputs  = all_ids[: n_seqs * SEQ_LEN].reshape(n_seqs, SEQ_LEN)
    targets = all_ids[1 : n_seqs * SEQ_LEN + 1].reshape(n_seqs, SEQ_LEN)

    total_nll = 0.0
    total_n   = 0

    fallback_bias = np.zeros(vocab_size, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n_seqs, BATCH_SIZE):
            inp = inputs[i : i + BATCH_SIZE].to(device)
            tgt = targets[i : i + BATCH_SIZE].to(device)

            out     = model(inp, output_hidden_states=(hvrt_model is not None))
            logits  = out.logits  # (b, seq, vocab)

            if hvrt_model is not None and bias_dict is not None and alpha > 0:
                # Assign each token to a partition and apply the bias
                last_h  = out.hidden_states[-1]  # (b, seq, d)
                h_flat  = last_h.reshape(-1, last_h.shape[-1]).cpu().float().numpy()
                p_ids   = assign_partitions(hvrt_model, h_flat)

                logits_np = logits.reshape(-1, vocab_size).cpu().float().numpy()
                for j, pid in enumerate(p_ids):
                    bias = bias_dict.get(int(pid), fallback_bias)
                    logits_np[j] += alpha * bias
                logits_corrected = torch.from_numpy(
                    logits_np.reshape(logits.shape)
                ).to(device)
            else:
                logits_corrected = logits

            nll = F.cross_entropy(
                logits_corrected.reshape(-1, vocab_size),
                tgt.reshape(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_n   += tgt.numel()

    return math.exp(total_nll / total_n)


# -----------------------------------------------------------------------------
# Storage accounting
# -----------------------------------------------------------------------------

def storage_mb(hvrt_model, bias_dict: dict) -> tuple[float, float]:
    """
    Estimate storage of the HVRT tree + bias vectors in MB.
    Returns (tree_mb, bias_mb).
    """
    import pickle
    tree_bytes = len(pickle.dumps(hvrt_model.tree_))
    bias_bytes = sum(v.nbytes for v in bias_dict.values())
    return tree_bytes / 1e6, bias_bytes / 1e6


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    import warnings
    warnings.filterwarnings("ignore")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from hvrt import FastHVRT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nHVRT Boosting Corrector — GPT-2 Perplexity Experiment")
    print("=" * 72)
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Device      : {device}")
    print(f"  Fit tokens  : {N_FIT_TOKENS:,}  ({int(N_FIT_TOKENS*(1-VAL_FRAC)):,} fit / "
          f"{int(N_FIT_TOKENS*VAL_FRAC):,} alpha-tune)")
    print(f"  Val tokens  : {N_VAL_TOKENS:,}")
    print(f"  Partitions  : {PARTITIONS_SWEEP}")
    print(f"  Alpha grid  : {ALPHA_GRID}")

    # -- Load model ------------------------------------------------------------
    print("\nLoading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model     = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    vocab_size = model.config.vocab_size
    d_model    = model.config.n_embd
    n_params   = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M params  |  d={d_model}  |  vocab={vocab_size}")

    # -- Load tokens -----------------------------------------------------------
    print("\nLoading WikiText-103...")
    t0 = time.time()
    fit_ids = load_tokens(tokenizer, "train",      N_FIT_TOKENS)
    val_ids = load_tokens(tokenizer, "validation", N_VAL_TOKENS)
    print(f"  Loaded in {time.time()-t0:.1f}s  "
          f"(fit={len(fit_ids):,} tokens, val={len(val_ids):,} tokens)")

    # -- Baseline perplexity ---------------------------------------------------
    print("\nComputing GPT-2 baseline perplexity on validation set...")
    t0 = time.time()
    base_ppl = evaluate_perplexity(model, val_ids, device, vocab_size=vocab_size)
    print(f"  GPT-2 baseline PPL = {base_ppl:.2f}  ({time.time()-t0:.1f}s)")

    # -- Collect training activations ------------------------------------------
    n_fit_use = int(N_FIT_TOKENS * (1 - VAL_FRAC))
    n_tune    = N_FIT_TOKENS - n_fit_use

    print(f"\nCollecting hidden states from {n_fit_use:,} fit tokens...")
    t0 = time.time()
    hidden_all, labels_all = collect_activations(model, tokenizer, fit_ids, device)
    hidden_fit = hidden_all[:n_fit_use]
    labels_fit = labels_all[:n_fit_use]
    hidden_tune = hidden_all[n_fit_use:]
    labels_tune = labels_all[n_fit_use:]
    print(f"  Done in {time.time()-t0:.1f}s  —  "
          f"hidden shape: {hidden_fit.shape}  ({hidden_fit.nbytes/1e6:.0f} MB)")

    # -- Precompute tune-set baseline NLL -------------------------------------
    tune_ids = fit_ids[n_fit_use : n_fit_use + n_tune + 1]

    # -- Sweep -----------------------------------------------------------------
    results = []

    for n_parts in PARTITIONS_SWEEP:
        print(f"\n{'-'*60}")
        print(f"  n_partitions = {n_parts}")

        # Fit HVRT
        t0 = time.time()
        hvrt = FastHVRT(
            n_partitions=n_parts,
            min_samples_leaf=MIN_LEAF,
            auto_tune=False,
            n_jobs=-1,
            random_state=42,
        )
        hvrt.fit(hidden_fit.astype(np.float64))
        actual_parts = len(np.unique(hvrt.partition_ids_))
        print(f"  HVRT fit in {time.time()-t0:.1f}s  —  "
              f"actual partitions: {actual_parts}")

        # Compute bias vectors
        print(f"  Computing bias vectors...")
        t0 = time.time()
        bias_dict, _, _ = compute_bias_vectors(
            model, tokenizer, hvrt, hidden_fit, labels_fit, device, vocab_size
        )
        print(f"  Bias computation: {time.time()-t0:.1f}s")

        # Alpha tuning on held-out fit slice
        best_alpha, best_tune_ppl = 0.0, float("inf")
        for alpha in ALPHA_GRID:
            tune_ppl = evaluate_perplexity(
                model, tune_ids, device,
                hvrt_model=hvrt, bias_dict=bias_dict,
                alpha=alpha, vocab_size=vocab_size,
            )
            if tune_ppl < best_tune_ppl:
                best_tune_ppl = tune_ppl
                best_alpha    = alpha

        # Evaluate on validation set
        print(f"  Evaluating on val (alpha={best_alpha})...")
        t0 = time.time()
        val_ppl = evaluate_perplexity(
            model, val_ids, device,
            hvrt_model=hvrt, bias_dict=bias_dict,
            alpha=best_alpha, vocab_size=vocab_size,
        )
        eval_time = time.time() - t0

        tree_mb, bias_mb = storage_mb(hvrt, bias_dict)
        total_mb  = tree_mb + bias_mb
        gain_pct  = 100 * (base_ppl - val_ppl) / base_ppl

        print(f"  Val PPL: {val_ppl:.2f}  (baseline={base_ppl:.2f}, "
              f"gain={gain_pct:+.2f}%,  alpha={best_alpha})")
        print(f"  Storage: {tree_mb:.2f} MB tree  +  {bias_mb:.1f} MB bias  "
              f"=  {total_mb:.1f} MB total  ({eval_time:.1f}s)")

        results.append({
            "n_parts":    n_parts,
            "actual":     actual_parts,
            "alpha":      best_alpha,
            "base_ppl":   base_ppl,
            "val_ppl":    val_ppl,
            "gain_pct":   gain_pct,
            "tree_mb":    tree_mb,
            "bias_mb":    bias_mb,
            "total_mb":   total_mb,
        })

    # -- Summary table ---------------------------------------------------------
    gpt2_mb = sum(p.numel() * 4 for p in model.parameters()) / 1e6

    print(f"\n{'='*72}")
    print("Results Summary")
    print(f"{'='*72}")
    print(f"  GPT-2 total size     : {gpt2_mb:.1f} MB")
    print(f"  GPT-2 baseline PPL   : {base_ppl:.2f}")
    print(f"  WikiText-103 val tokens : {N_VAL_TOKENS:,}")
    print()
    print(f"  {'Parts':>6}  {'Actual':>6}  {'alpha':>5}  "
          f"{'PPL':>7}  {'Gain%':>6}  {'Storage':>10}  {'% of GPT-2':>11}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*5}  "
          f"{'-'*7}  {'-'*6}  {'-'*10}  {'-'*11}")

    for r in results:
        pct_of_gpt2 = 100 * r["total_mb"] / gpt2_mb
        print(f"  {r['n_parts']:>6}  {r['actual']:>6}  {r['alpha']:>5.2f}  "
              f"{r['val_ppl']:>7.2f}  {r['gain_pct']:>+6.2f}%  "
              f"{r['total_mb']:>8.1f} MB  {pct_of_gpt2:>9.2f}%")

    best = min(results, key=lambda x: x["val_ppl"])
    print(f"\n  Best: {best['n_parts']} partitions -> "
          f"PPL {best['val_ppl']:.2f}  ({best['gain_pct']:+.2f}%),  "
          f"{best['total_mb']:.1f} MB corrector  "
          f"({100*best['total_mb']/gpt2_mb:.1f}% of GPT-2 size)")
    print()


if __name__ == "__main__":
    main()
