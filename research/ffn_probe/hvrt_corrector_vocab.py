"""
HVRT Vocab-Restricted Corrector -- Multi-Round Stability Test
==============================================================

The multi-round boosting corrector degraded after round 2 because reliable
per-partition estimation of a 50,257-token distribution requires far more
training data than 51,200 tokens can provide at fine partition granularities.

This script tests whether restricting the correction to the TOP-K most
frequent tokens solves that instability.

The logic is simple:
  - Top-K tokens by frequency follow Zipf's law: a small fraction covers
    most of the probability mass.
  - Top 100 tokens cover ~40-50% of token occurrences in Wikipedia text.
  - Top 500 tokens cover ~60-65%.
  - Top 2000 tokens cover ~75-80%.
  - Restricting to top-K makes the estimation problem K/vocab_size times
    easier.  For K=100 with 4 partitions: ~3840 observations per partition
    distributed over 100 tokens = ~38 avg per token.  Reliable.

Bias formula (restricted to top-K tokens):
    bias_p[k] = log(empirical_freq_p[top_k_ids[k]] + eps)
                - mean_log_prob_p[top_k_ids[k]]

where mean_log_prob_p is from the FULL distribution log-softmax (correctly
normalised) and the bias is applied only to the top-K logits at inference.
This shifts the relative probabilities among top-K tokens and between
top-K vs the tail — exactly where the correction signal is strongest.

Sweep: TOP_K in {100, 500, 2000, full}, up to N_ROUNDS rounds each.

Usage
-----
    python research/ffn_probe/hvrt_corrector_vocab.py
"""

from __future__ import annotations
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME       = "gpt2"
N_FIT_TOKENS     = 51_200
N_VAL_TOKENS     = 20_480
SEQ_LEN          = 128
BATCH_SIZE       = 8
N_ROUNDS         = 6
PARTS_PER_ROUND  = [4, 8, 16, 32, 64, 128]
ALPHA_GRID       = [0.05, 0.1, 0.2, 0.3, 0.5]
MIN_PART_SAMPLES = 20

# Top-K vocab sizes to sweep.  None = unrestricted (full 50,257).
TOP_K_SWEEP = [100, 500, 2000, None]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tokens(tokenizer, split: str, n_tokens: int) -> list[int]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install datasets:  pip install datasets")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    ids: list[int] = []
    for row in ds:
        if not row["text"].strip():
            continue
        ids.extend(tokenizer.encode(row["text"]))
        if len(ids) >= n_tokens + 1:
            break
    return ids[: n_tokens + 1]


# ---------------------------------------------------------------------------
# HVRT partition assignment
# ---------------------------------------------------------------------------
def assign_partitions(hvrt_model, X_new: np.ndarray) -> np.ndarray:
    X_z = hvrt_model._to_z(X_new.astype(np.float64))
    return hvrt_model.tree_.apply(X_z).astype(np.int32)


# ---------------------------------------------------------------------------
# Training data collection (hidden states + targets only -- no logit cache)
# ---------------------------------------------------------------------------
def collect_data(model, token_ids: list[int], device: str):
    model.eval()
    all_ids  = torch.tensor(token_ids, dtype=torch.long)
    n_seqs   = (len(all_ids) - 1) // SEQ_LEN
    inputs_t = all_ids[: n_seqs * SEQ_LEN].reshape(n_seqs, SEQ_LEN)
    targets_t= all_ids[1 : n_seqs * SEQ_LEN + 1].reshape(n_seqs, SEQ_LEN)

    hidden_list, target_list = [], []
    with torch.no_grad():
        for i in range(0, n_seqs, BATCH_SIZE):
            inp = inputs_t[i : i + BATCH_SIZE].to(device)
            tgt = targets_t[i : i + BATCH_SIZE]
            out = model(inp, output_hidden_states=True)
            d   = out.hidden_states[-1].shape[-1]
            hidden_list.append(
                out.hidden_states[-1].reshape(-1, d).cpu().float().numpy()
            )
            target_list.append(tgt.reshape(-1).numpy().astype(np.int32))

    return (
        np.vstack(hidden_list),
        np.concatenate(target_list),
        inputs_t,
    )


# ---------------------------------------------------------------------------
# Apply corrections (vocab-restricted or full)
# ---------------------------------------------------------------------------
def apply_corrections(
    logits_np: np.ndarray,
    hidden_np: np.ndarray,
    correctors: list,
) -> np.ndarray:
    """
    Each corrector entry: (hvrt, bias_dict, alpha, top_k_ids | None)
    top_k_ids  array of token IDs where the bias is defined (None = full vocab).
    bias_dict  {pid: bias_vector}  length = len(top_k_ids) or vocab_size.
    """
    out = logits_np.copy()
    for hvrt, bias_dict, alpha, top_k_ids in correctors:
        pids = assign_partitions(hvrt, hidden_np)
        for pid in np.unique(pids):
            mask    = pids == pid
            pid_key = int(pid)
            if pid_key not in bias_dict:
                continue
            if top_k_ids is None:
                out[mask] += alpha * bias_dict[pid_key]
            else:
                # np.ix_ avoids the chained-fancy-index copy trap:
                # out[mask][:, top_k_ids] += ...  silently drops the write
                rows = np.where(mask)[0]
                out[np.ix_(rows, top_k_ids)] += alpha * bias_dict[pid_key]
    return out


# ---------------------------------------------------------------------------
# Fit one boosting round (vocab-restricted, streaming)
# ---------------------------------------------------------------------------
def fit_one_round(
    model,
    device: str,
    inputs_t: torch.Tensor,
    hidden_states: np.ndarray,
    next_tokens: np.ndarray,
    correctors: list,
    n_partitions: int,
    vocab_size: int,
    top_k_ids,           # np.ndarray of length K, or None for full vocab
) -> tuple[object, dict, dict]:
    from hvrt import FastHVRT

    K = len(top_k_ids) if top_k_ids is not None else vocab_size

    # Fit HVRT on hidden states
    hvrt = FastHVRT(
        n_partitions=n_partitions,
        min_samples_leaf=16,
        auto_tune=False,
        n_jobs=-1,
        random_state=42,
    )
    hvrt.fit(hidden_states.astype(np.float64))
    pids        = assign_partitions(hvrt, hidden_states)
    unique_pids = np.unique(pids)

    # Empirical token counts (only within top-K or full vocab)
    part_counts: dict[int, int]        = {}
    emp_counts:  dict[int, np.ndarray] = {}
    for pid in unique_pids:
        mask = pids == pid
        n_in = int(mask.sum())
        part_counts[int(pid)] = n_in
        targets_p = next_tokens[mask]
        full_counts = np.bincount(targets_p, minlength=vocab_size).astype(np.float64)
        if top_k_ids is not None:
            emp_counts[int(pid)] = full_counts[top_k_ids]
        else:
            emp_counts[int(pid)] = full_counts

    # Streaming accumulation of mean log-probs (K per partition)
    lp_sums: dict[int, np.ndarray] = {
        int(pid): np.zeros(K, dtype=np.float64) for pid in unique_pids
    }

    n_seqs  = inputs_t.shape[0]
    tok_idx = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, n_seqs, BATCH_SIZE):
            inp = inputs_t[i : i + BATCH_SIZE].to(device)
            out = model(inp, output_hidden_states=True)
            n   = inp.shape[0] * SEQ_LEN
            d   = out.hidden_states[-1].shape[-1]
            h   = out.hidden_states[-1].reshape(n, d).cpu().float().numpy()
            lg  = out.logits.reshape(n, vocab_size).cpu().float().numpy()

            h_batch   = hidden_states[tok_idx : tok_idx + n]
            pids_batch = pids[tok_idx : tok_idx + n]

            # Apply previous corrections (batch-sized, not corpus-sized)
            lg_corr = apply_corrections(lg, h_batch, correctors)

            # Full log-softmax (required for correctly normalised log-probs)
            lp = (
                torch.from_numpy(lg_corr).float().log_softmax(dim=-1).numpy()
            )

            # Extract top-K columns (or all if unrestricted)
            lp_k = lp[:, top_k_ids] if top_k_ids is not None else lp

            for pid in np.unique(pids_batch):
                mask = pids_batch == pid
                lp_sums[int(pid)] += lp_k[mask].sum(axis=0)

            tok_idx += n

    # Build bias vectors
    bias_dict: dict[int, np.ndarray] = {}
    eps = 1e-8
    for pid in unique_pids:
        n_in = part_counts[int(pid)]
        if n_in < MIN_PART_SAMPLES:
            continue

        emp = emp_counts[int(pid)]

        # UNCONDITIONAL probability of each token in the full partition
        # (NOT normalised within the top-K subset — that would produce
        # conditional probs that cancel the mean_lp renormalization).
        emp_prob = emp / n_in     # shape (K,)  sums to <= 1 (< 1 if top-K < full vocab)

        # Mean log-prob from the FULL distribution softmax (no renormalization).
        # lp_sums contains sum of log_softmax(full_logits)[:, top_k_ids]
        # so mean_lp[k] = E[log P_model(token=top_k_ids[k] | context)]
        # which is in the same probability space as emp_prob.
        mean_lp = lp_sums[int(pid)] / n_in   # shape (K,)

        # Bias: log(empirical_prob) - mean_log_prob
        # Both quantities are in the same "full distribution" scale, so
        # their difference is the per-token correction signal.
        bias_dict[int(pid)] = (
            np.log(emp_prob + eps) - mean_lp
        ).astype(np.float32)

    return hvrt, bias_dict, part_counts


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------
def evaluate_ppl(
    model,
    token_ids: list[int],
    device: str,
    vocab_size: int,
    correctors: list | None = None,
) -> float:
    model.eval()
    all_ids = torch.tensor(token_ids, dtype=torch.long)
    n_seqs  = (len(all_ids) - 1) // SEQ_LEN
    if n_seqs == 0:
        return float("nan")
    inputs  = all_ids[: n_seqs * SEQ_LEN].reshape(n_seqs, SEQ_LEN)
    targets = all_ids[1 : n_seqs * SEQ_LEN + 1].reshape(n_seqs, SEQ_LEN)

    total_nll, total_n = 0.0, 0
    with torch.no_grad():
        for i in range(0, n_seqs, BATCH_SIZE):
            inp = inputs[i : i + BATCH_SIZE].to(device)
            tgt = targets[i : i + BATCH_SIZE].to(device)

            if correctors:
                out = model(inp, output_hidden_states=True)
                d   = out.hidden_states[-1].shape[-1]
                h   = out.hidden_states[-1].reshape(-1, d).cpu().float().numpy()
                lg  = out.logits.reshape(-1, vocab_size).cpu().float().numpy()
                lg_c = apply_corrections(lg, h, correctors)
                logits_t = torch.from_numpy(lg_c).to(device)
            else:
                out = model(inp)
                logits_t = out.logits.reshape(-1, vocab_size)

            nll = F.cross_entropy(logits_t, tgt.reshape(-1), reduction="sum")
            total_nll += nll.item()
            total_n   += tgt.numel()

    return math.exp(total_nll / total_n)


# ---------------------------------------------------------------------------
# Alpha grid search
# ---------------------------------------------------------------------------
def search_alpha(
    model, val_ids, device, vocab_size,
    correctors_prev, hvrt_new, bias_new, alpha_grid, top_k_ids,
) -> tuple[float, float]:
    best_alpha, best_ppl = alpha_grid[0], float("inf")
    for alpha in alpha_grid:
        candidate = correctors_prev + [(hvrt_new, bias_new, alpha, top_k_ids)]
        ppl       = evaluate_ppl(model, val_ids, device, vocab_size, candidate)
        if ppl < best_ppl:
            best_alpha, best_ppl = alpha, ppl
    return best_alpha, best_ppl


# ---------------------------------------------------------------------------
# Run one complete TOP_K experiment
# ---------------------------------------------------------------------------
def run_top_k_experiment(
    model,
    device: str,
    tokenizer,
    val_ids: list[int],
    hidden_states: np.ndarray,
    next_tokens: np.ndarray,
    inputs_t: torch.Tensor,
    vocab_size: int,
    base_ppl: float,
    top_k: int | None,
) -> list[dict]:
    """Run N_ROUNDS of boosting with a given vocabulary restriction."""
    label = f"TOP_K={top_k}" if top_k is not None else "TOP_K=full"

    # Select top-K token IDs by training frequency
    if top_k is not None:
        freq       = np.bincount(next_tokens, minlength=vocab_size)
        top_k_ids  = np.argsort(freq)[::-1][:top_k].astype(np.int64)
        top_k_coverage = freq[top_k_ids].sum() / freq.sum()
        print(f"\n  {label}  (covers {top_k_coverage:.1%} of training token positions)")
    else:
        top_k_ids = None
        print(f"\n  {label}")

    print(f"  {'Rnd':>3}  {'Parts':>5}  {'Alpha':>5}  {'Val PPL':>8}  "
          f"{'vs base':>8}  {'Cum.MB':>7}")
    print(f"  {'-'*3}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*7}")

    correctors: list = []
    rows: list[dict] = []
    import pickle

    for r in range(N_ROUNDS):
        n_parts = PARTS_PER_ROUND[r]

        hvrt, bias_dict, pcounts = fit_one_round(
            model, device, inputs_t,
            hidden_states, next_tokens,
            correctors, n_parts, vocab_size, top_k_ids,
        )

        best_alpha, best_ppl = search_alpha(
            model, val_ids, device, vocab_size,
            correctors, hvrt, bias_dict, ALPHA_GRID, top_k_ids,
        )
        correctors.append((hvrt, bias_dict, best_alpha, top_k_ids))

        # Storage
        total_mb = sum(
            len(pickle.dumps(h.tree_)) / 1e6
            + sum(v.nbytes for v in b.values()) / 1e6
            for h, b, _, _ in correctors
        )
        gain = 100 * (base_ppl - best_ppl) / base_ppl

        print(f"  {r+1:>3}  {n_parts:>5}  {best_alpha:>5.2f}  "
              f"{best_ppl:>8.2f}  {gain:>+7.2f}%  {total_mb:>7.2f}")

        rows.append({
            "top_k":  top_k if top_k is not None else vocab_size,
            "round":  r + 1,
            "parts":  n_parts,
            "alpha":  best_alpha,
            "ppl":    best_ppl,
            "gain":   gain,
            "mb":     total_mb,
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import warnings
    warnings.filterwarnings("ignore")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nHVRT Vocab-Restricted Corrector Sweep")
    print("=" * 72)
    print(f"  Device        : {device}")
    print(f"  Fit tokens    : {N_FIT_TOKENS:,}")
    print(f"  Val tokens    : {N_VAL_TOKENS:,}")
    print(f"  N rounds      : {N_ROUNDS}")
    print(f"  Parts / round : {PARTS_PER_ROUND}")
    print(f"  TOP_K sweep   : {TOP_K_SWEEP}")

    # Load
    print("\nLoading GPT-2...")
    tokenizer  = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model      = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    vocab_size = model.config.vocab_size
    gpt2_mb    = sum(p.numel() * 4 for p in model.parameters()) / 1e6

    print("\nLoading WikiText-103...")
    t0       = time.time()
    fit_ids  = load_tokens(tokenizer, "train",      N_FIT_TOKENS)
    val_ids  = load_tokens(tokenizer, "validation", N_VAL_TOKENS)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Baseline
    print("\nBaseline perplexity...")
    base_ppl = evaluate_ppl(model, val_ids, device, vocab_size)
    print(f"  Baseline PPL = {base_ppl:.2f}")

    # Collect hidden states once
    print(f"\nCollecting training activations ({N_FIT_TOKENS:,} tokens)...")
    t0 = time.time()
    hidden_states, next_tokens, inputs_t = collect_data(model, fit_ids, device)
    print(f"  Done in {time.time()-t0:.1f}s  |  hidden={hidden_states.shape}")

    # Token frequency distribution info
    freq    = np.bincount(next_tokens, minlength=vocab_size)
    for k in [100, 500, 2000]:
        top_ids  = np.argsort(freq)[::-1][:k]
        coverage = freq[top_ids].sum() / freq.sum()
        print(f"  Top-{k:<5} tokens cover {coverage:.1%} of training positions")

    # Run sweep
    print(f"\n{'='*72}")
    print("Sweep: TOP_K vs Rounds")
    print(f"{'='*72}")
    all_rows: list[dict] = []

    for top_k in TOP_K_SWEEP:
        rows = run_top_k_experiment(
            model, device, tokenizer, val_ids,
            hidden_states, next_tokens, inputs_t,
            vocab_size, base_ppl, top_k,
        )
        all_rows.extend(rows)

    # Summary table: best PPL per TOP_K (across all rounds)
    print(f"\n{'='*72}")
    print("Best PPL per TOP_K (optimal round)")
    print(f"{'='*72}")
    print(f"  {'TOP_K':>6}  {'Best PPL':>8}  {'Best gain':>9}  "
          f"{'At round':>8}  {'Storage':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}")

    seen_topk: dict[int, dict] = {}
    for row in all_rows:
        k = row["top_k"]
        if k not in seen_topk or row["ppl"] < seen_topk[k]["ppl"]:
            seen_topk[k] = row

    for k in sorted(seen_topk.keys()):
        r = seen_topk[k]
        label = str(k) if k != vocab_size else "full"
        print(f"  {label:>6}  {r['ppl']:>8.2f}  {r['gain']:>+8.2f}%  "
              f"  {r['round']:>6}  {r['mb']:>7.2f}MB")

    print(f"\n  Baseline PPL : {base_ppl:.2f}")
    print(f"  GPT-2 size   : {gpt2_mb:.1f} MB")

    # Per-round stability comparison (does vocab restriction help?)
    print(f"\n{'='*72}")
    print("Round-by-round PPL: does vocab restriction stabilise boosting?")
    print(f"{'='*72}")
    unique_topk = sorted({r["top_k"] for r in all_rows})
    headers = ["Rnd"] + [
        str(k) if k != vocab_size else "full" for k in unique_topk
    ]
    print("  " + "  ".join(f"{h:>8}" for h in headers))
    print("  " + "  ".join("-" * 8 for _ in headers))
    for rnd in range(1, N_ROUNDS + 1):
        row_cells = [f"{rnd:>8}"]
        for k in unique_topk:
            match = [r for r in all_rows if r["top_k"] == k and r["round"] == rnd]
            if match:
                row_cells.append(f"{match[0]['ppl']:>8.2f}")
            else:
                row_cells.append(f"{'':>8}")
        print("  " + "  ".join(row_cells))

    print(f"\n  (lower PPL = better; does the vocab-restricted column peak and"
          f" hold rather than degrade?)")


if __name__ == "__main__":
    main()
