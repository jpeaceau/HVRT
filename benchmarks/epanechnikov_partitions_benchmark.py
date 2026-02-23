#!/usr/bin/env python3
"""
Epanechnikov Kernel × Partition-Count Benchmark
================================================

Tests the hypothesis: "Epanechnikov's regression performance degradation is caused
by insufficient partition granularity, not by the kernel itself."

In the bandwidth benchmark, Epanechnikov won classification tasks convincingly but
lost on regression.  The product kernel (per-feature independent sampling) breaks
inter-feature correlations — but if partitions are fine enough, each partition
becomes small enough that the within-partition correlation structure is weak,
making the independence assumption of the product kernel less harmful.

Method
------
- Datasets: regression only (housing, multimodal, emergence_divergence,
  emergence_bifurcation)
- Candidates: h=0.10 (prior overall winner), epanechnikov (challenger),
  default KDE h=0.10 (control for the n_partitions sweep)
- Partition counts: auto-tuned default, then 30 / 50 / 75 / 100 / 150 / 200
- Ratios: 5× and 10× (where the regression gap was largest in the prior benchmark)
- CV: 5-fold × 3-repeat = 15 evaluations per condition

Metrics (same as bandwidth_benchmark.py)
-----------------------------------------
  tstr_delta  ML utility; primary.  Higher → better.
  corr_mae    Correlation structure preservation.  Lower → better.
  mw1         Marginal fidelity (Wasserstein-1).  Lower → better.

Usage
-----
    python benchmarks/epanechnikov_partitions_benchmark.py
    python benchmarks/epanechnikov_partitions_benchmark.py --quick
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

if hasattr(sys.stdout, 'buffer') and \
        getattr(sys.stdout, 'encoding', '').lower().replace('-', '') not in \
        ('utf8', 'utf-8'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )

from hvrt import HVRT, epanechnikov
from hvrt.benchmarks.datasets import BENCHMARK_DATASETS


# ─────────────────────────────────────────────────────────────────────────────
# Candidates: (label, expand-kwargs)
# n_partitions is injected separately per sweep step
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATES = {
    'h=0.10':       {'bandwidth': 0.10, 'adaptive_bandwidth': False},
    'h=0.30':       {'bandwidth': 0.30, 'adaptive_bandwidth': False},
    'epanechnikov': {'generation_strategy': epanechnikov},
}


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers (shared with bandwidth_benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────

def marginal_w1(X_real: np.ndarray, X_synth: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance
    return float(np.mean([
        wasserstein_distance(X_real[:, j], X_synth[:, j])
        for j in range(X_real.shape[1])
    ]))


def correlation_mae(X_real: np.ndarray, X_synth: np.ndarray) -> float:
    d = X_real.shape[1]
    if d < 2:
        return 0.0
    idx = np.triu_indices(d, k=1)
    return float(np.mean(np.abs(
        np.corrcoef(X_real.T)[idx] - np.corrcoef(X_synth.T)[idx]
    )))


def tstr_scores(X_tr, y_tr, X_te, y_te, X_synth, y_synth, random_state=42):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        trtr_m = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        trtr_m.fit(X_tr, y_tr)
        trtr = float(r2_score(y_te, trtr_m.predict(X_te)))

        tstr_m = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        tstr_m.fit(X_synth, y_synth)
        tstr = float(r2_score(y_te, tstr_m.predict(X_te)))

    return trtr, tstr


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one(
    X_tr, y_tr, X_te, y_te,
    n_synth: int,
    bw_kwargs: dict,
    n_partitions_override,
    seed: int,
) -> dict:
    _nan = {k: float('nan') for k in ('mw1', 'corr_mae', 'trtr', 'tstr_delta', 'actual_n_parts')}

    try:
        XY_tr = np.column_stack([X_tr, y_tr.reshape(-1, 1).astype(float)])

        model = HVRT(random_state=seed)
        model.fit(XY_tr)

        bw  = bw_kwargs.get('bandwidth')
        ada = bw_kwargs.get('adaptive_bandwidth', False)
        gs  = bw_kwargs.get('generation_strategy')

        XY_s = model.expand(
            n=n_synth,
            bandwidth=bw,
            adaptive_bandwidth=ada,
            generation_strategy=gs,
            n_partitions=n_partitions_override,
        )

        # Actual partition count used (may differ from override due to min_samples_leaf)
        actual_n_parts = int(model.n_partitions_)

        X_s = XY_s[:, :-1]
        y_s = XY_s[:, -1]

        trtr, tstr = tstr_scores(X_tr, y_tr, X_te, y_te, X_s, y_s, random_state=seed)

        return {
            'mw1':        marginal_w1(X_tr, X_s),
            'corr_mae':   correlation_mae(X_tr, X_s),
            'trtr':       trtr,
            'tstr_delta': tstr - trtr,
            'actual_n_parts': actual_n_parts,
        }

    except Exception as exc:
        warnings.warn(f'evaluate_one failed: {exc}', stacklevel=2)
        return _nan


def run_condition(
    X_full, y_full, exp_ratio, bw_kwargs, n_partitions_override,
    n_splits, n_repeats, random_state, max_n,
) -> dict:
    from sklearn.model_selection import RepeatedKFold

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    accum = defaultdict(list)

    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X_full)):
        X_tr, X_te = X_full[tr_idx], X_full[te_idx]
        y_tr, y_te = y_full[tr_idx], y_full[te_idx]

        if len(X_tr) > max_n:
            rng = np.random.RandomState(random_state + fold_i)
            sel = rng.choice(len(X_tr), size=max_n, replace=False)
            X_tr, y_tr = X_tr[sel], y_tr[sel]

        n_synth = max(4, int(len(X_tr) * exp_ratio))
        seed = random_state + fold_i * 13

        res = evaluate_one(
            X_tr, y_tr, X_te, y_te,
            n_synth, bw_kwargs, n_partitions_override, seed,
        )
        for k in res:
            accum[k].append(res[k])

    def _ms(k):
        a = np.array(accum[k], dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a))

    out = {}
    for k in ('mw1', 'corr_mae', 'trtr', 'tstr_delta', 'actual_n_parts'):
        m, s = _ms(k)
        out[f'{k}_mean'] = m
        out[f'{k}_std']  = s
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_partition_sweep_table(
    ds_name: str,
    ratio: float,
    sweep_results: dict,   # {n_parts_label: {cand_name: metrics}}
    n_splits: int,
    n_repeats: int,
):
    """
    Rows = partition-count variants.
    Columns = candidates × metrics (TSTR Δ and Corr.MAE).
    """
    cand_names = list(CANDIDATES.keys())
    n_cands = len(cand_names)

    # Column layout: label(16) + per-candidate: TSTR(12) Corr(10)
    col_w = 24
    row_label_w = 16
    W = row_label_w + n_cands * col_w
    W = max(W, 80)

    print(f'╔{"═" * W}╗')
    title = (f'  {ds_name}  |  ratio={ratio:.0f}x  |  partition sweep  |  '
             f'{n_splits}-fold × {n_repeats}-repeat')
    print(f'║{title:<{W}}║')
    print(f'╠{"═" * W}╣')

    hdr = f'  {"n_partitions":<14}'
    for c in cand_names:
        hdr += f'{c + ":TSTR":>13}{c + ":Corr":>11}'
    print(f'║{hdr:<{W}}║')
    print(f'╠{"─" * W}╣')

    for n_label, cand_results in sweep_results.items():
        # Find best TSTR and best Corr across candidates for this row
        tstr_vals = {c: cand_results[c]['tstr_delta_mean'] for c in cand_names
                     if not np.isnan(cand_results[c]['tstr_delta_mean'])}
        corr_vals = {c: cand_results[c]['corr_mae_mean'] for c in cand_names
                     if not np.isnan(cand_results[c]['corr_mae_mean'])}
        best_tstr = max(tstr_vals, key=tstr_vals.__getitem__) if tstr_vals else None
        best_corr = min(corr_vals, key=corr_vals.__getitem__) if corr_vals else None

        # Get actual n_parts from first candidate (they all use the same tree)
        any_cand = list(cand_results.values())[0]
        actual = int(round(any_cand.get('actual_n_parts_mean', 0)))
        row = f'  {n_label:<10}(≈{actual:3d})'

        for c in cand_names:
            r = cand_results[c]
            tstr_m = r['tstr_delta_mean']
            corr_m = r['corr_mae_mean']

            tstr_s = f'{tstr_m:+.4f}' if not np.isnan(tstr_m) else '—'
            corr_s = f'{corr_m:.4f}'  if not np.isnan(corr_m) else '—'

            tstr_mk = '◆' if c == best_tstr else ' '
            corr_mk = '◆' if c == best_corr else ' '

            row += f' {tstr_s + tstr_mk:>12} {corr_s + corr_mk:>10}'

        print(f'║{row:<{W}}║')

    print(f'╚{"═" * W}╝')
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Epanechnikov × partitions benchmark')
    parser.add_argument('--ratios', nargs='+', type=float, default=[5.0, 10.0])
    parser.add_argument('--max-n', type=int, default=500)
    parser.add_argument('--quick', action='store_true',
                        help='Faster run: fewer folds, fewer partition counts')
    args = parser.parse_args()

    if args.quick:
        n_splits, n_repeats = 3, 1
        PARTITION_SWEEP = {'auto': None, '50': 50, '100': 100}
    else:
        n_splits, n_repeats = 5, 3
        PARTITION_SWEEP = {
            'auto':  None,
            '30':    30,
            '50':    50,
            '75':    75,
            '100':   100,
            '150':   150,
            '200':   200,
        }

    # Load all datasets and keep only regression ones
    all_ds = {}
    for ds_name, gen_fn in BENCHMARK_DATASETS.items():
        X_full, y_full, _ = gen_fn(random_state=42)
        X_full = np.asarray(X_full, dtype=float)
        y_full = np.asarray(y_full, dtype=float)
        is_cls = len(np.unique(y_full)) <= 20
        if not is_cls:
            all_ds[ds_name] = (X_full, y_full)

    print()
    print('=' * 72)
    print('  HVRT — Epanechnikov × Partition-Count Benchmark')
    print('=' * 72)
    print(f'  Hypothesis : finer partitions close the Epanechnikov regression gap')
    print(f'  Datasets   : {", ".join(all_ds)}')
    print(f'  Ratios     : {args.ratios}')
    print(f'  Partitions : {list(PARTITION_SWEEP.keys())}')
    print(f'  Candidates : {list(CANDIDATES.keys())}')
    print(f'  max_n      : {args.max_n}')
    print(f'  CV         : {n_splits}-fold × {n_repeats}-repeat = '
          f'{n_splits * n_repeats} evals per condition')
    print()

    for ds_name, (X_full, y_full) in all_ds.items():

        print(f'  [{ds_name}]  n={len(X_full)}, d={X_full.shape[1]}')

        for ratio in args.ratios:
            sweep_results = {}

            for n_label, n_parts_override in PARTITION_SWEEP.items():
                cand_results = {}
                for cand_name, bw_kwargs in CANDIDATES.items():
                    print(f'    [{ratio:.0f}x  {n_label:<6}  {cand_name:<14}] ...',
                          end=' ', flush=True)

                    res = run_condition(
                        X_full, y_full, ratio, bw_kwargs,
                        n_parts_override,
                        n_splits, n_repeats,
                        random_state=42, max_n=args.max_n,
                    )
                    cand_results[cand_name] = res
                    print(
                        f'TSTR Δ={res["tstr_delta_mean"]:+.4f}  '
                        f'Corr={res["corr_mae_mean"]:.4f}  '
                        f'(n_parts≈{int(round(res["actual_n_parts_mean"]))})'
                    )

                sweep_results[n_label] = cand_results

            print()
            print_partition_sweep_table(ds_name, ratio, sweep_results, n_splits, n_repeats)


if __name__ == '__main__':
    main()
