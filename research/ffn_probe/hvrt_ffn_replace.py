"""
HVRT FFN Replacement — GPT-2 Perplexity Experiment
====================================================

Variant A: replace GPT-2's Feed-Forward Network (FFN) layers with
HVRT-partitioned linear models, fitted analytically (no backprop).

Architecture
------------
GPT-2 FFN (per layer):
    input  (768)  →  W1 (768×3072)  →  GELU  →  W2 (3072×768)  →  output (768)
    Parameters per layer: 768*3072 + 3072*768 = 4,718,592
    Total FFN (12 layers): 56.6M parameters  ≈  226 MB

HVRT-FFN replacement:
    For each partition p: fit  W_p  (d_in → d_out, optionally low-rank)
    At inference: h_out = W[partition(h_in)] @ h_in
    Parameters: n_partitions × rank × 2 × d_model per layer
    With P=16, r=32:  16 × 32 × 2 × 768 × 12 = 9.4M parameters  ≈  38 MB

The within-partition linear model is fitted by ordinary least squares (OLS).
Optionally truncated to rank r via SVD for further compression.

Sweep
-----
n_partitions ∈ {4, 8, 16, 32}  ×  rank ∈ {32, 64, 128, None=full}
Reports perplexity and compression ratio for each combination.

Usage
-----
    pip install datasets
    python research/ffn_probe/hvrt_ffn_replace.py
"""

from __future__ import annotations

import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_NAME       = "gpt2"
N_FIT_TOKENS     = 25_600     # tokens per FFN layer for fitting (reduced for speed)
N_VAL_TOKENS     = 20_480     # val tokens for evaluation
SEQ_LEN          = 128
BATCH_SIZE       = 8
MIN_LEAF         = 16

PARTITIONS_SWEEP = [4, 8, 16]
RANK_SWEEP       = [32, 64, None]   # None = full-rank OLS

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_tokens(tokenizer, split: str, n_tokens: int) -> list[int]:
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
    return ids[: n_tokens + 1]


# -----------------------------------------------------------------------------
# FFN activation collection
# -----------------------------------------------------------------------------

def collect_ffn_io(model, token_ids: list[int], device: str,
                   n_layers: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Hook into every FFN layer and collect (ffn_input, ffn_output) pairs.

    Returns
    -------
    ffn_in  : list of (n_tokens, d_model)  arrays, one per layer
    ffn_out : list of (n_tokens, d_model)  arrays, one per layer
    """
    ffn_in_buffers  = [[] for _ in range(n_layers)]
    ffn_out_buffers = [[] for _ in range(n_layers)]
    hooks = []

    for layer_idx in range(n_layers):
        mlp = model.transformer.h[layer_idx].mlp

        def make_hook(li):
            def hook(module, inp, out):
                # inp is a tuple; inp[0] is the FFN input hidden state
                ffn_in_buffers[li].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu().float().numpy())
                ffn_out_buffers[li].append(out.detach().reshape(-1, out.shape[-1]).cpu().float().numpy())
            return hook

        h = mlp.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    model.eval()
    all_ids = torch.tensor(token_ids, dtype=torch.long)
    n_seqs  = (len(all_ids) - 1) // SEQ_LEN
    inputs  = all_ids[: n_seqs * SEQ_LEN].reshape(n_seqs, SEQ_LEN)

    with torch.no_grad():
        for i in range(0, n_seqs, BATCH_SIZE):
            model(inputs[i : i + BATCH_SIZE].to(device))

    for h in hooks:
        h.remove()

    ffn_in  = [np.vstack(b) for b in ffn_in_buffers]
    ffn_out = [np.vstack(b) for b in ffn_out_buffers]
    return ffn_in, ffn_out


# -----------------------------------------------------------------------------
# HVRT partition assignment (new data)
# -----------------------------------------------------------------------------

def assign_partitions(hvrt_model, X_new: np.ndarray) -> np.ndarray:
    X_z = hvrt_model._to_z(X_new.astype(np.float64))
    return hvrt_model.tree_.apply(X_z).astype(np.int32)


# -----------------------------------------------------------------------------
# Within-partition OLS (optionally low-rank)
# -----------------------------------------------------------------------------

def fit_partition_linear(X: np.ndarray, Y: np.ndarray, rank: int | None):
    """
    Fit a linear map  Y ≈ X @ W  within a partition using least squares.

    Uses np.linalg.lstsq (SVD-based) which is numerically stable for both
    overdetermined (n >> d) and near-singular systems.  The rcond threshold
    discards directions where the data has no signal, preventing overfitting.

    Returns
    -------
    If rank is None : W  (d_in, d_out)  — full OLS weight matrix
    If rank given   : (U, V)  where U=(d_in, r), V=(r, d_out)
                      prediction: (X @ U) @ V
    """
    # lstsq with rcond: discard singular values < rcond * sigma_max
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=1e-3)
    W = W.astype(np.float32)    # (d_in, d_out)

    if rank is None or rank >= min(W.shape):
        return W, None

    # Low-rank via SVD: W ≈ U S V^T → truncate to rank r
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    r  = min(rank, len(s))
    U  = (U[:, :r] * s[:r]).astype(np.float32)   # (d_in, r)  absorb s into U
    Vt = Vt[:r, :].astype(np.float32)             # (r, d_out)
    return U, Vt


# -----------------------------------------------------------------------------
# HVRT-FFN replacement module
# -----------------------------------------------------------------------------

class HVRTLinearFFN(nn.Module):
    """
    Replaces a GPT-2 MLP layer with HVRT-partitioned linear models.

    At inference time:
      1. Assign the input hidden state to its HVRT partition (centroid lookup).
      2. Apply the within-partition linear (or low-rank) model.
    """

    def __init__(self, hvrt_model, partition_models: dict, bias: np.ndarray):
        """
        Parameters
        ----------
        hvrt_model      : fitted FastHVRT instance (for partition assignment)
        partition_models: {partition_id: (W,) or (U, V)}
                           W  = full-rank (d_in, d_out)
                           U,V = low-rank factors
        bias            : global bias (d_out,) = mean of all FFN outputs
        """
        super().__init__()
        self.hvrt    = hvrt_model
        self.models  = partition_models
        self.bias_t  = torch.from_numpy(bias).float()
        self.unique_parts = sorted(partition_models.keys())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat     = x.reshape(-1, x.shape[-1])          # (n, d)
        n, d       = x_flat.shape

        device     = x_flat.device
        x_np       = x_flat.cpu().float().numpy()
        pids       = assign_partitions(self.hvrt, x_np)  # (n,)

        out = torch.zeros(n, self.bias_t.shape[0], dtype=torch.float32)

        for pid in self.unique_parts:
            mask = pids == pid
            if not mask.any():
                continue
            x_sub = x_flat[mask].cpu().float()
            val   = self.models[pid]
            if isinstance(val, tuple):
                U, V = val
                U_t  = torch.from_numpy(U)
                V_t  = torch.from_numpy(V)
                sub_out = (x_sub @ U_t) @ V_t
            else:
                W_t = torch.from_numpy(val)
                sub_out = x_sub @ W_t
            out[mask] = sub_out

        out = out + self.bias_t
        return out.to(device).reshape(orig_shape)


# -----------------------------------------------------------------------------
# Fitting a single layer's HVRT-FFN
# -----------------------------------------------------------------------------

def fit_hvrt_ffn_layer(
    ffn_in_layer: np.ndarray,
    ffn_out_layer: np.ndarray,
    n_partitions: int,
    rank: int | None,
) -> HVRTLinearFFN:
    """Fit HVRT + within-partition OLS for one FFN layer."""
    from hvrt import FastHVRT

    hvrt = FastHVRT(
        n_partitions=n_partitions,
        min_samples_leaf=MIN_LEAF,
        auto_tune=False,
        n_jobs=-1,
        random_state=42,
    )
    hvrt.fit(ffn_in_layer.astype(np.float64))
    pids         = hvrt.partition_ids_
    unique_parts = np.unique(pids)

    bias      = ffn_out_layer.mean(axis=0).astype(np.float32)
    # Centre OUTPUT only (keeps bias interpretable as the global mean).
    # Do NOT centre input — the W matrices must act on the raw (uncentred)
    # hidden states at inference time, so fitting and inference must use
    # the same representation.
    ffn_out_c = ffn_out_layer - bias

    partition_models: dict[int, tuple | np.ndarray] = {}
    for pid in unique_parts:
        mask  = pids == pid
        X_sub = ffn_in_layer[mask]   # raw input, no centring
        Y_sub = ffn_out_c[mask]
        if len(X_sub) < 4:
            partition_models[int(pid)] = np.zeros(
                (X_sub.shape[1], Y_sub.shape[1]), dtype=np.float32
            )
            continue
        W, V = fit_partition_linear(X_sub, Y_sub, rank)
        partition_models[int(pid)] = (W, V) if V is not None else W

    return HVRTLinearFFN(hvrt, partition_models, bias)


# -----------------------------------------------------------------------------
# Perplexity evaluation
# -----------------------------------------------------------------------------

def evaluate_perplexity(model, token_ids: list[int], device: str,
                        vocab_size: int) -> float:
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
            inp    = inputs[i : i + BATCH_SIZE].to(device)
            tgt    = targets[i : i + BATCH_SIZE].to(device)
            logits = model(inp).logits
            nll    = F.cross_entropy(logits.reshape(-1, vocab_size),
                                     tgt.reshape(-1), reduction="sum")
            total_nll += nll.item()
            total_n   += tgt.numel()
    return math.exp(total_nll / total_n)


# -----------------------------------------------------------------------------
# Storage accounting
# -----------------------------------------------------------------------------

def ffn_replacement_storage_mb(hvrt_ffn_layers: list[HVRTLinearFFN]) -> float:
    """Estimate total MB for all replacement layers (trees + matrices + bias)."""
    import pickle
    total = 0
    for layer in hvrt_ffn_layers:
        total += len(pickle.dumps(layer.hvrt.tree_))
        for val in layer.models.values():
            if isinstance(val, tuple):
                total += sum(v.nbytes for v in val)
            else:
                total += val.nbytes
        total += layer.bias_t.numpy().nbytes
    return total / 1e6


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    import warnings
    warnings.filterwarnings("ignore")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nHVRT FFN Replacement — GPT-2 Perplexity Experiment")
    print("=" * 72)
    print(f"  Model          : {MODEL_NAME}")
    print(f"  Device         : {device}")
    print(f"  Fit tokens/lyr : {N_FIT_TOKENS:,}")
    print(f"  Val tokens     : {N_VAL_TOKENS:,}")
    print(f"  Partitions     : {PARTITIONS_SWEEP}")
    print(f"  Ranks          : {RANK_SWEEP}")

    # -- Load model ------------------------------------------------------------
    print("\nLoading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model     = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    n_layers   = model.config.n_layer
    d_model    = model.config.n_embd
    vocab_size = model.config.vocab_size
    gpt2_mb    = sum(p.numel() * 4 for p in model.parameters()) / 1e6
    ffn_mb     = (n_layers * 2 * d_model * (4 * d_model) * 4) / 1e6
    print(f"  GPT-2 total    : {gpt2_mb:.1f} MB  |  FFN-only: {ffn_mb:.1f} MB")
    print(f"  Layers         : {n_layers}, d_model={d_model}")

    # -- Load tokens -----------------------------------------------------------
    print("\nLoading WikiText-103...")
    t0 = time.time()
    fit_ids = load_tokens(tokenizer, "train",      N_FIT_TOKENS)
    val_ids = load_tokens(tokenizer, "validation", N_VAL_TOKENS)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # -- Baseline perplexity ---------------------------------------------------
    print("\nBaseline GPT-2 perplexity...")
    base_ppl = evaluate_perplexity(model, val_ids, device, vocab_size)
    print(f"  Baseline PPL = {base_ppl:.2f}")

    # -- Collect FFN activations -----------------------------------------------
    print(f"\nCollecting FFN activations ({n_layers} layers × {N_FIT_TOKENS:,} tokens)...")
    t0 = time.time()
    ffn_in_all, ffn_out_all = collect_ffn_io(model, fit_ids, device, n_layers)
    print(f"  Done in {time.time()-t0:.1f}s  —  "
          f"each layer: {ffn_in_all[0].shape}  "
          f"({ffn_in_all[0].nbytes/1e6:.0f} MB)")

    # -- Reconstruction MSE baseline (zero-model per layer) --------------------
    print("\nMSE of zero-model (predict global mean) per layer:")
    zero_mse = []
    for li in range(n_layers):
        mean_out = ffn_out_all[li].mean(axis=0, keepdims=True)
        mse = float(((ffn_out_all[li] - mean_out)**2).mean())
        zero_mse.append(mse)
    print(f"  Mean zero-model MSE across layers: {np.mean(zero_mse):.4f}")

    # -- Fit HVRT once per (n_partitions, layer); cache and reuse per rank ------
    # This avoids re-running the expensive HVRT tree fit for every rank value.
    print(f"\nPre-fitting HVRT trees for all partition counts and layers...")
    hvrt_cache: dict[tuple[int, int], object] = {}  # (n_parts, layer) -> hvrt model
    for n_parts in PARTITIONS_SWEEP:
        print(f"  Fitting trees: n_partitions={n_parts} across {n_layers} layers...")
        t0 = time.time()
        for li in range(n_layers):
            from hvrt import FastHVRT
            hvrt = FastHVRT(n_partitions=n_parts, min_samples_leaf=MIN_LEAF,
                            auto_tune=False, n_jobs=-1, random_state=42)
            hvrt.fit(ffn_in_all[li].astype(np.float64))
            hvrt_cache[(n_parts, li)] = hvrt
        print(f"    Done in {time.time()-t0:.1f}s")

    # -- Single-layer probe: replace ONE layer at a time, measure PPL impact ---
    # This isolates the per-layer approximation quality independently of
    # covariate-shift cascades when all layers are replaced simultaneously.
    print(f"\n{'='*72}")
    print("Single-layer probe: per-layer PPL when only that FFN is replaced")
    print(f"  (n_partitions=8, rank=64)")
    print(f"{'='*72}")
    single_results = []
    probe_parts, probe_rank = 8, 64
    for li in range(n_layers):
        hvrt = hvrt_cache[(probe_parts, li)]
        repl = HVRTLinearFFN.__new__(HVRTLinearFFN)
        # Build directly from cached hvrt to avoid re-fitting
        nn.Module.__init__(repl)
        repl.hvrt   = hvrt
        repl.models = {}
        pids        = hvrt.partition_ids_
        bias        = ffn_out_all[li].mean(axis=0).astype(np.float32)
        repl.bias_t = torch.from_numpy(bias).float()
        repl.unique_parts = list(np.unique(pids))
        ffn_out_c   = ffn_out_all[li] - bias
        for pid in np.unique(pids):
            mask  = pids == pid
            X_sub = ffn_in_all[li][mask]
            Y_sub = ffn_out_c[mask]
            W, V  = fit_partition_linear(X_sub, Y_sub, probe_rank)
            repl.models[int(pid)] = (W, V) if V is not None else W

        patched = copy.deepcopy(model).cpu()
        patched.transformer.h[li].mlp = repl
        ppl = evaluate_perplexity(patched, val_ids, "cpu", vocab_size)
        delta = ppl - base_ppl
        print(f"  Layer {li:>2}: PPL={ppl:.2f}  ({delta:>+7.2f} vs baseline)")
        single_results.append({"layer": li, "ppl": ppl, "delta": delta})
        del patched

    best_single = min(single_results, key=lambda x: x["ppl"])
    worst_single = max(single_results, key=lambda x: x["ppl"])
    print(f"  Most replaceable layer : {best_single['layer']}  "
          f"(PPL={best_single['ppl']:.2f}, delta={best_single['delta']:+.2f})")
    print(f"  Most critical layer    : {worst_single['layer']}  "
          f"(PPL={worst_single['ppl']:.2f}, delta={worst_single['delta']:+.2f})")

    # -- All-layer replacement sweep -------------------------------------------
    results = []

    for n_parts in PARTITIONS_SWEEP:
        for rank in RANK_SWEEP:
            rank_str = str(rank) if rank else "full"
            print(f"\n{'-'*60}")
            print(f"  n_partitions={n_parts}, rank={rank_str}")

            # Build replacement layers from cached HVRT trees
            t0 = time.time()
            hvrt_ffn_layers = []
            layer_mses = []
            for li in range(n_layers):
                hvrt = hvrt_cache[(n_parts, li)]
                repl = HVRTLinearFFN.__new__(HVRTLinearFFN)
                nn.Module.__init__(repl)
                repl.hvrt   = hvrt
                repl.models = {}
                pids        = hvrt.partition_ids_
                bias        = ffn_out_all[li].mean(axis=0).astype(np.float32)
                repl.bias_t = torch.from_numpy(bias).float()
                repl.unique_parts = list(np.unique(pids))
                ffn_out_c   = ffn_out_all[li] - bias
                for pid in np.unique(pids):
                    mask  = pids == pid
                    X_sub = ffn_in_all[li][mask]
                    Y_sub = ffn_out_c[mask]
                    W, V  = fit_partition_linear(X_sub, Y_sub, rank)
                    repl.models[int(pid)] = (W, V) if V is not None else W

                # Reconstruction MSE on training data
                with torch.no_grad():
                    x_torch = torch.from_numpy(ffn_in_all[li].astype(np.float32))
                    pred    = repl(x_torch).numpy()
                mse = float(((pred - ffn_out_all[li])**2).mean())
                layer_mses.append(mse)
                hvrt_ffn_layers.append(repl)

            fit_time = time.time() - t0
            mean_mse = float(np.mean(layer_mses))
            print(f"  Build time: {fit_time:.1f}s  |  "
                  f"Mean reconstruction MSE: {mean_mse:.4f}  "
                  f"(zero-model: {np.mean(zero_mse):.4f})")

            # Swap all 12 FFN layers simultaneously
            patched = copy.deepcopy(model).cpu()
            for li, repl in enumerate(hvrt_ffn_layers):
                patched.transformer.h[li].mlp = repl

            print(f"  Evaluating all-layers-replaced perplexity...")
            t0 = time.time()
            patched_ppl = evaluate_perplexity(patched, val_ids, "cpu", vocab_size)
            eval_time   = time.time() - t0

            repl_mb  = ffn_replacement_storage_mb(hvrt_ffn_layers)
            gain_pct = 100 * (base_ppl - patched_ppl) / base_ppl
            compress = ffn_mb / repl_mb if repl_mb > 0 else float("inf")

            print(f"  PPL (all replaced): {patched_ppl:.2f}  "
                  f"(baseline={base_ppl:.2f}, gain={gain_pct:+.2f}%)  "
                  f"[{eval_time:.1f}s]")
            print(f"  Storage: {repl_mb:.1f} MB  "
                  f"({compress:.1f}x compression of FFN-only {ffn_mb:.1f} MB)")

            results.append({
                "n_parts":  n_parts,
                "rank":     rank_str,
                "mean_mse": mean_mse,
                "base_ppl": base_ppl,
                "ppl":      patched_ppl,
                "gain_pct": gain_pct,
                "repl_mb":  repl_mb,
                "compress": compress,
            })
            del patched

    # -- Summary ---------------------------------------------------------------
    print(f"\n{'='*72}")
    print("Results Summary")
    print(f"{'='*72}")
    print(f"  GPT-2 total   : {gpt2_mb:.1f} MB  |  FFN-only: {ffn_mb:.1f} MB")
    print(f"  Baseline PPL  : {base_ppl:.2f}")
    print()
    print(f"  {'Parts':>5}  {'Rank':>4}  {'MSE':>7}  "
          f"{'PPL (all)':>9}  {'Gain%':>6}  {'MB':>6}  {'Compress':>8}")
    print(f"  {'-'*5}  {'-'*4}  {'-'*7}  "
          f"{'-'*9}  {'-'*6}  {'-'*6}  {'-'*8}")
    for r in results:
        print(f"  {r['n_parts']:>5}  {r['rank']:>4}  {r['mean_mse']:>7.4f}  "
              f"{r['ppl']:>9.2f}  {r['gain_pct']:>+6.2f}%  "
              f"{r['repl_mb']:>6.1f}  {r['compress']:>7.1f}x")
    print()


if __name__ == "__main__":
    main()
