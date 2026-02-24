"""
HVRT Boosted Corrector -- Multi-Round Iterative Correction of GPT-2
====================================================================

Each round fits a new HVRT on the last-layer hidden states and computes
a per-partition bias on the LOG-PROBABILITY RESIDUAL left by all previous
rounds.  The approach is analogous to gradient boosting but entirely
closed-form: no gradients, no backprop -- just partition averages.

After N rounds the corrected distribution is:

    log_p(v | x) = log_softmax(gpt2_logits(x))[v]
                   + sum_r  alpha_r * bias_r[ partition_r(hidden(x)), v ]

where each bias_r is the mean residual log-probability deficit in that
partition after all prior rounds.

Confidence reporting (built-in, for free)
------------------------------------------
Every HVRT partition has an explicit training density (n_tokens seen).
At inference time the model can report which partition each input fell in,
how many training examples it saw there, and flag the correction as
uncertain when that count is low.  This is structurally impossible in
a standard neural network without special calibration techniques.

Usage
-----
    pip install datasets transformers torch
    python research/ffn_probe/hvrt_corrector_boosted.py
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
MODEL_NAME           = "gpt2"
N_FIT_TOKENS         = 51_200
N_VAL_TOKENS         = 20_480
SEQ_LEN              = 128
BATCH_SIZE           = 8
N_ROUNDS             = 5
PARTS_PER_ROUND      = [4, 8, 16, 32, 64]   # progressively finer each round
ALPHA_GRID           = [0.05, 0.1, 0.2, 0.3, 0.5]
MIN_PART_SAMPLES     = 20    # skip bias for partitions with fewer tokens than this
CONFIDENCE_THRESHOLD = 100   # flag partition as LOW-CONFIDENCE below this count

DEMO_PROMPTS = [
    "The quick brown fox",
    "In the beginning of the universe",
    "The neural network was trained on",
    "Scientists have recently discovered that",
    "The stock market fell sharply after",
    "To be or not to be, that is",
]


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
# Training data collection — hidden states + next tokens only.
# We deliberately do NOT cache logits: at vocab=50,257, storing all
# N_FIT_TOKENS logits at once requires ~10 GB.  Logits are computed
# on-the-fly per batch inside fit_one_round.
# ---------------------------------------------------------------------------
def collect_data(
    model, token_ids: list[int], device: str
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    hidden_states : (N, d_model)  float32 numpy  last transformer layer
    next_tokens   : (N,)          int32   numpy   ground-truth next token
    inputs_t      : (n_seqs, SEQ_LEN) long  tensor  (for per-round forward pass)
    targets_t     : (n_seqs, SEQ_LEN) long  tensor
    """
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
        targets_t,
    )


# ---------------------------------------------------------------------------
# Apply stacked corrections (operates on a single batch, not the full corpus)
# ---------------------------------------------------------------------------
def apply_corrections(
    logits_np: np.ndarray,
    hidden_np: np.ndarray,
    correctors: list,
) -> np.ndarray:
    """
    correctors: list of (hvrt_model, bias_dict, alpha)
    logits_np / hidden_np should be a batch (not the full training corpus).
    Returns corrected logits array (same shape as logits_np).
    """
    out = logits_np.copy()
    for hvrt, bias_dict, alpha in correctors:
        pids = assign_partitions(hvrt, hidden_np)
        for pid in np.unique(pids):
            mask    = pids == pid
            pid_key = int(pid)
            if pid_key in bias_dict:
                out[mask] += alpha * bias_dict[pid_key]
    return out


# ---------------------------------------------------------------------------
# Fit one boosting round (streaming -- never materialises the full logit matrix)
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
) -> tuple[object, dict, dict]:
    """
    Fit one round on the log-probability residual after all previous rounds.

    Processes the training data in BATCH_SIZE chunks so that at most
    BATCH_SIZE * SEQ_LEN * vocab * 4 bytes (~200 MB) of logits are live
    at any one time.

    Returns
    -------
    hvrt            fitted FastHVRT
    bias_dict       {partition_id: bias_vector (vocab,)  float32}
    part_counts     {partition_id: n_training_tokens}
    """
    from hvrt import FastHVRT

    # Fit a new HVRT partition on hidden states
    hvrt = FastHVRT(
        n_partitions=n_partitions,
        min_samples_leaf=16,
        auto_tune=False,
        n_jobs=-1,
        random_state=42,
    )
    hvrt.fit(hidden_states.astype(np.float64))
    pids = assign_partitions(hvrt, hidden_states)
    unique_pids = np.unique(pids)

    # Empirical token counts (cheap — no logits needed)
    part_counts: dict[int, int] = {}
    emp_counts:  dict[int, np.ndarray] = {}
    for pid in unique_pids:
        mask = pids == pid
        part_counts[int(pid)] = int(mask.sum())
        emp_counts[int(pid)]  = np.bincount(
            next_tokens[mask], minlength=vocab_size
        ).astype(np.float64)

    # Accumulate mean corrected log-probs per partition (streaming batches)
    lp_sums:   dict[int, np.ndarray] = {
        int(pid): np.zeros(vocab_size, dtype=np.float64) for pid in unique_pids
    }

    model.eval()
    n_seqs = inputs_t.shape[0]
    tok_idx = 0
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

            # Apply all previous corrections (only n tokens, not 51 k)
            lg_corr = apply_corrections(lg, h_batch, correctors)

            # Log-softmax (torch is numerically stable)
            lp = (
                torch.from_numpy(lg_corr).float().log_softmax(dim=-1).numpy()
            )

            for pid in np.unique(pids_batch):
                mask = pids_batch == pid
                lp_sums[int(pid)] += lp[mask].sum(axis=0)

            tok_idx += n

    # Build bias vectors
    bias_dict: dict[int, np.ndarray] = {}
    eps = 1e-8
    for pid in unique_pids:
        n_in = part_counts[int(pid)]
        if n_in < MIN_PART_SAMPLES:
            continue

        empirical = emp_counts[int(pid)]
        empirical /= empirical.sum()

        mean_lp = lp_sums[int(pid)] / n_in
        lse     = mean_lp.max() + np.log(np.exp(mean_lp - mean_lp.max()).sum())
        mean_lp = mean_lp - lse

        bias_dict[int(pid)] = (
            np.log(empirical + eps) - mean_lp
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
# Alpha grid search (on validation set)
# ---------------------------------------------------------------------------
def search_alpha(
    model, val_ids, device, vocab_size,
    correctors_prev, hvrt_new, bias_new, alpha_grid,
) -> tuple[float, float]:
    best_alpha, best_ppl = alpha_grid[0], float("inf")
    for alpha in alpha_grid:
        candidate = correctors_prev + [(hvrt_new, bias_new, alpha)]
        ppl       = evaluate_ppl(model, val_ids, device, vocab_size, candidate)
        if ppl < best_ppl:
            best_alpha, best_ppl = alpha, ppl
    return best_alpha, best_ppl


# ---------------------------------------------------------------------------
# Confidence-aware prediction demo
# ---------------------------------------------------------------------------
def demo_confidence(
    model,
    tokenizer,
    correctors: list,
    part_counts_list: list[dict],
    device: str,
    prompts: list[str],
) -> None:
    """
    For each prompt, report the corrected next-token distribution AND
    explicit confidence signals derived from partition density.

    This is the key interpretability property: every prediction comes with
    "I have seen N similar contexts in training" -- something a standard
    neural network cannot express without external calibration.
    """
    print(f"\n{'='*72}")
    print("Confidence-Aware Prediction Demo")
    print(f"{'='*72}")
    print("  Partition density = fraction of training tokens in that partition.")
    print("  Sparse partitions flag the correction as LOW-CONFIDENCE.\n")

    model.eval()
    for prompt in prompts:
        enc = tokenizer.encode(prompt)
        inp = torch.tensor([enc], dtype=torch.long).to(device)

        with torch.no_grad():
            out = model(inp, output_hidden_states=True)

        d       = out.hidden_states[-1].shape[-1]
        last_h  = out.hidden_states[-1][0, -1:].cpu().float().numpy()  # (1, d)
        logits  = out.logits[0, -1:].cpu().float().numpy()             # (1, vocab)

        # --- Confidence per round ---
        min_density = 1.0
        round_reports = []
        for r, ((hvrt, bias_dict, alpha), pcounts) in enumerate(
                zip(correctors, part_counts_list)):
            pid     = int(assign_partitions(hvrt, last_h)[0])
            count   = pcounts.get(pid, 0)
            total   = sum(pcounts.values())
            density = count / total if total > 0 else 0.0
            min_density = min(min_density, density)
            flag    = "OK " if count >= CONFIDENCE_THRESHOLD else "LOW"
            round_reports.append(f"R{r+1}[n={count} {flag}]")

        # --- Corrected predictions ---
        corr_logits = apply_corrections(logits, last_h, correctors)
        top_ids     = np.argsort(corr_logits[0])[::-1][:5]
        top_toks    = [tokenizer.decode([t]) for t in top_ids]
        top_probs   = torch.softmax(
            torch.from_numpy(corr_logits[0]).float(), dim=-1
        ).numpy()[top_ids]

        conf_label = (
            "HIGH"                    if min_density > 0.05 else
            "MODERATE"                if min_density > 0.01 else
            "LOW -- correction uncertain"
        )

        print(f"Prompt     : {prompt!r}")
        print(f"Confidence : {conf_label}  (min partition density = {min_density:.4f})")
        print(f"Partitions : {' | '.join(round_reports)}")
        print(f"Top-5 next tokens (corrected):")
        for tok, prob in zip(top_toks, top_probs):
            bar = "#" * max(1, int(prob * 40))
            print(f"  {repr(tok):20s}  {prob:.4f}  {bar}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import warnings
    warnings.filterwarnings("ignore")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nHVRT Boosted Corrector -- GPT-2")
    print("=" * 72)
    print(f"  Device         : {device}")
    print(f"  Fit tokens     : {N_FIT_TOKENS:,}")
    print(f"  Val tokens     : {N_VAL_TOKENS:,}")
    print(f"  Rounds         : {N_ROUNDS}")
    print(f"  Parts / round  : {PARTS_PER_ROUND}")
    print(f"  Alpha grid     : {ALPHA_GRID}")

    # Load model
    print("\nLoading GPT-2...")
    tokenizer  = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model      = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    vocab_size = model.config.vocab_size
    gpt2_mb    = sum(p.numel() * 4 for p in model.parameters()) / 1e6
    print(f"  GPT-2: {gpt2_mb:.1f} MB,  vocab={vocab_size:,}")

    # Load tokens
    print("\nLoading WikiText-103...")
    t0      = time.time()
    fit_ids = load_tokens(tokenizer, "train",      N_FIT_TOKENS)
    val_ids = load_tokens(tokenizer, "validation", N_VAL_TOKENS)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Baseline PPL
    print("\nBaseline perplexity...")
    base_ppl = evaluate_ppl(model, val_ids, device, vocab_size)
    print(f"  Baseline PPL = {base_ppl:.2f}")

    # Collect training activations ONCE (reused every round).
    # Logits are NOT cached here -- too large (51200 x 50257 x 4 = 9.6 GB).
    # Each round streams its own batched forward pass for logit statistics.
    print(f"\nCollecting training activations ({N_FIT_TOKENS:,} tokens)...")
    t0 = time.time()
    hidden_states, next_tokens, inputs_t, targets_t = collect_data(
        model, fit_ids, device
    )
    print(f"  Done in {time.time()-t0:.1f}s  |  "
          f"hidden={hidden_states.shape}")

    # Multi-round boosting
    correctors:      list       = []
    part_counts_list: list[dict] = []
    total_mb = 0.0

    print(f"\n{'='*72}")
    print("Boosting Rounds")
    print(f"{'='*72}")
    print(f"  {'Rnd':>3}  {'Parts':>5}  {'Alpha':>5}  {'Val PPL':>8}  "
          f"{'vs base':>8}  {'Cum. MB':>7}  {'Time':>6}")
    print(f"  {'-'*3}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*6}")

    for r in range(N_ROUNDS):
        n_parts = PARTS_PER_ROUND[r]
        t0 = time.time()

        hvrt, bias_dict, pcounts = fit_one_round(
            model, device, inputs_t,
            hidden_states, next_tokens,
            correctors, n_parts, vocab_size,
        )
        part_counts_list.append(pcounts)

        best_alpha, best_ppl = search_alpha(
            model, val_ids, device, vocab_size,
            correctors, hvrt, bias_dict, ALPHA_GRID,
        )
        correctors.append((hvrt, bias_dict, best_alpha))

        # Cumulative storage
        import pickle
        total_mb = sum(
            len(pickle.dumps(h.tree_)) / 1e6
            + sum(v.nbytes for v in b.values()) / 1e6
            for h, b, _ in correctors
        )
        gain = 100 * (base_ppl - best_ppl) / base_ppl

        print(f"  {r+1:>3}  {n_parts:>5}  {best_alpha:>5.2f}  "
              f"{best_ppl:>8.2f}  {gain:>+7.2f}%  {total_mb:>7.2f}  "
              f"{time.time()-t0:>5.1f}s")

    # Summary
    final_ppl  = best_ppl
    total_gain = 100 * (base_ppl - final_ppl) / base_ppl
    print(f"\n{'='*72}")
    print("Summary")
    print(f"{'='*72}")
    print(f"  Baseline PPL   : {base_ppl:.2f}")
    print(f"  Final PPL      : {final_ppl:.2f}  ({total_gain:+.2f}% total gain)")
    print(f"  Total storage  : {total_mb:.2f} MB  "
          f"({total_mb / gpt2_mb * 100:.3f}% of GPT-2)")

    # Per-round marginal gain
    print(f"\n  Per-round breakdown:")
    prev_ppl = base_ppl
    for r, (hvrt, bias_dict, alpha) in enumerate(correctors):
        ppl_r = evaluate_ppl(
            model, val_ids, device, vocab_size, correctors[: r + 1]
        )
        delta = prev_ppl - ppl_r
        print(f"    Round {r+1}: PPL={ppl_r:.2f}  (delta={delta:+.2f}, "
              f"parts={PARTS_PER_ROUND[r]}, alpha={alpha})")
        prev_ppl = ppl_r

    # Partition density statistics
    print(f"\n  Partition density (training coverage):")
    for r, (pcounts, n_parts) in enumerate(zip(part_counts_list, PARTS_PER_ROUND)):
        counts = list(pcounts.values())
        n_conf = sum(1 for c in counts if c >= CONFIDENCE_THRESHOLD)
        total  = sum(counts)
        print(f"    Round {r+1} ({n_parts} parts): "
              f"min={min(counts):,}  median={int(np.median(counts)):,}  "
              f"confident={n_conf}/{len(counts)} partitions  "
              f"(>= {CONFIDENCE_THRESHOLD} tokens)")

    # Confidence demo
    demo_confidence(
        model, tokenizer, correctors, part_counts_list, device, DEMO_PROMPTS
    )


if __name__ == "__main__":
    main()
