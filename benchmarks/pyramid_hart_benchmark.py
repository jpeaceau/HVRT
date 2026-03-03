#!/usr/bin/env python3
"""
PyramidHART Quality Benchmark
==============================

Compares PyramidHART .reduce() and .expand() quality against the HVRT model
family (HVRT, HART, FastHART) and a simple Random/Bootstrap baseline, using
the same datasets, metrics, and evaluation protocol as the main benchmark suite.

Reduction ratios tested : 0.5, 0.4, 0.3, 0.2, 0.1
  — 40 % and 50 % retention are explicitly included in line with the
    specification; the existing default suite only tests 0.5, 0.3, 0.2, 0.1.

Expansion ratios tested : 1×, 2×, 5×

Metrics (reduction)
-------------------
  marginal_fidelity      1 − mean normalised Wasserstein-1  (↑ better)
  correlation_fidelity   Frobenius similarity of correlation matrices  (↑ better)
  tail_preservation      geometric mean of percentile-range ratios  (↑ better)
  ml_delta               ML-utility Δ vs full-data TRTR baseline  (↑ better)

Metrics (expansion)
-------------------
  marginal_fidelity      1 − mean normalised Wasserstein-1  (↑ better)
  discriminator_accuracy classifier accuracy; target ≈ 0.5  (closer = better)
  tail_preservation      geometric mean of percentile-range ratios  (↑ better)
  ml_delta               TSTR − TRTR ML-utility delta  (↑ better)

Usage
-----
    python benchmarks/pyramid_hart_benchmark.py
    python benchmarks/pyramid_hart_benchmark.py --quick
    python benchmarks/pyramid_hart_benchmark.py --tasks reduce
    python benchmarks/pyramid_hart_benchmark.py --tasks expand
    python benchmarks/pyramid_hart_benchmark.py --max-n-expand 200
"""

from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# UTF-8 stdout for Windows terminals that default to cp1252 / similar
if hasattr(sys.stdout, 'buffer') and \
        getattr(sys.stdout, 'encoding', '').lower().replace('-', '') not in ('utf8', 'utf-8'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )

import numpy as np

from hvrt.benchmarks import (
    run_reduction_benchmark,
    run_expansion_benchmark,
    print_results_table,
)


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

# 40 % and 50 % retention explicitly requested; extend the standard four ratios
REDUCTION_RATIOS = [0.5, 0.4, 0.3, 0.2, 0.1]
EXPANSION_RATIOS = [1.0, 2.0, 5.0]

# PyramidHART vs the HART/HVRT family + simple baselines
REDUCTION_METHODS = [
    'PyramidHART-size', 'PyramidHART-var',
    'HVRT-size',        'HVRT-var',
    'HART-size',        'HART-var',
    'FastHART-size',    'FastHART-var',
    'Random',
]

EXPANSION_METHODS = [
    'PyramidHART-size',      'PyramidHART-var',
    'PyramidHART-ARejection','PyramidHART-SignEpan', 'PyramidHART-MST',
    'HVRT-size',             'HVRT-var',
    'HART-size',             'HART-var',
    'FastHART-size',         'FastHART-var',
    'Bootstrap-Noise',
]

QUICK_DATASETS = ['multimodal', 'housing']


# ---------------------------------------------------------------------------
# Win-count summary table
# ---------------------------------------------------------------------------

def _print_win_table(results, task, methods):
    """
    Aggregate win counts across all (dataset × ratio) conditions.

    A method 'wins' a condition when it achieves the best value for that
    metric in that condition.  For discriminator_accuracy the target is 0.5,
    so the winner minimises |accuracy − 0.5|.
    """
    param_key = 'ratio' if task == 'reduce' else 'expansion_ratio'

    if task == 'reduce':
        metric_defs = [
            ('marginal_fidelity',    'Marg.Fid ↑',   'max'),
            ('correlation_fidelity', 'Corr.Fid ↑',   'max'),
            ('tail_preservation',    'Tail.Pres ↑',  'max'),
            ('ml_delta',             'ML Δ ↑',        'max'),
        ]
    else:
        metric_defs = [
            ('marginal_fidelity',      'Marg.Fid ↑',  'max'),
            ('discriminator_accuracy', 'Disc.Acc →0.5', 'disc'),
            ('tail_preservation',      'Tail.Pres ↑', 'max'),
            ('ml_delta',               'ML Δ ↑',       'max'),
        ]

    rows = [r for r in results if r['task'] == task and not r.get('published_only')]
    conditions = sorted({
        (r['dataset'], r['params'].get(param_key)) for r in rows
    })

    wins = {m: {key: 0 for key, *_ in metric_defs} for m in methods}
    n_cond = 0

    for ds, pv in conditions:
        cond = {
            r['method']: r['metrics']
            for r in rows
            if r['dataset'] == ds and r['params'].get(param_key) == pv
               and r['method'] in methods
        }
        if not cond:
            continue
        n_cond += 1

        for key, _, rule in metric_defs:
            vals = {m: cond[m].get(key, float('nan')) for m in methods if m in cond}
            valid = {m: v for m, v in vals.items() if v == v}
            if not valid:
                continue
            if rule == 'disc':
                best = min(valid, key=lambda m: abs(valid[m] - 0.5))
            elif rule == 'max':
                best = max(valid, key=valid.__getitem__)
            else:
                best = min(valid, key=valid.__getitem__)
            wins[best][key] += 1

    col_w = max(16, max(len(m) for m in methods) + 2)
    W = 22 + len(methods) * col_w

    print(f'\n╔{"═" * W}╗')
    print(f'║  {"WIN-COUNT  (" + str(n_cond) + " conditions: dataset × ratio)":<{W - 3}}║')
    print(f'╠{"═" * W}╣')
    hdr = f'  {"Metric":<20}' + ''.join(f'{m:>{col_w}}' for m in methods)
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    for key, label, _ in metric_defs:
        row = f'  {label:<20}' + ''.join(f'{wins[m][key]:>{col_w}}' for m in methods)
        print(f'║{row}║')

    print(f'╠{"─" * W}╣')
    totals = {m: sum(wins[m].values()) for m in methods}
    max_t = max(totals.values()) if totals else 0
    total_row = f'  {"Total wins":<20}' + ''.join(
        f'{str(totals[m]) + ("◆" if totals[m] == max_t else ""):{col_w}}'
        for m in methods
    )
    print(f'║{total_row}║')
    print(f'╚{"═" * W}╝')
    print()


# ---------------------------------------------------------------------------
# Per-ratio breakdown: mean metric value across datasets
# ---------------------------------------------------------------------------

def _print_ratio_breakdown(results, task, methods, ratios_list):
    """
    For each ratio, print a one-line mean summary of the primary metric
    (marginal_fidelity for both tasks; ml_delta as the ML signal).
    """
    param_key = 'ratio' if task == 'reduce' else 'expansion_ratio'
    rows = [r for r in results if r['task'] == task and not r.get('published_only')]

    print(f'\n  Mean per-ratio summary  ({"↑ higher = better" if task == "reduce" else "MargFid ↑  ML Δ ↑"})')
    print(f'  {"ratio":<8}', end='')
    for m in methods:
        print(f'{m:>18}', end='')
    print()
    print('  ' + '─' * (8 + 18 * len(methods)))

    for pv in ratios_list:
        pv_rows = [r for r in rows if r['params'].get(param_key) == pv]
        if not pv_rows:
            continue

        label = f'{pv:.0%}' if task == 'reduce' else f'{pv:.0f}×'
        print(f'  {label:<8}', end='')

        for m in methods:
            m_rows = [r for r in pv_rows if r['method'] == m]
            if not m_rows:
                print(f'{"—":>18}', end='')
                continue
            mf_vals = [r['metrics'].get('marginal_fidelity', float('nan')) for r in m_rows]
            mld_vals = [r['metrics'].get('ml_delta', float('nan')) for r in m_rows]
            mf_mean  = np.nanmean(mf_vals) if mf_vals else float('nan')
            mld_mean = np.nanmean(mld_vals) if mld_vals else float('nan')
            cell = f'{mf_mean:.3f}/{mld_mean:+.3f}'
            print(f'{cell:>18}', end='')
        print()
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--tasks', nargs='+', choices=['reduce', 'expand'], default=['reduce', 'expand'],
        help='Which benchmark tasks to run (default: both)',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--quick', action='store_true',
        help='2 datasets only (multimodal + housing) for faster development checks',
    )
    parser.add_argument(
        '--max-n-expand', type=int, default=500, metavar='N',
        help='Training-set cap for expansion benchmarks (default: 500)',
    )
    args = parser.parse_args()

    ds_arg = QUICK_DATASETS if args.quick else 'all'
    ds_label = 'quick (multimodal, housing)' if args.quick else \
        'adult, fraud, housing, multimodal, emergence_divergence, emergence_bifurcation'

    print()
    print('═' * 72)
    print('  PyramidHART Quality Benchmark  (v2.11.0)')
    print('═' * 72)
    print(f'  Datasets         : {ds_label}')
    print(f'  Reduction ratios : {REDUCTION_RATIOS}')
    print(f'  Expansion ratios : {EXPANSION_RATIOS}')
    print(f'  max_n (expand)   : {args.max_n_expand}')
    print(f'  seed             : {args.seed}')
    print()

    all_reduce = []
    all_expand = []

    # ------------------------------------------------------------------
    # Reduction
    # ------------------------------------------------------------------
    if 'reduce' in args.tasks:
        print('─' * 72)
        print('  REDUCTION BENCHMARK')
        print('─' * 72)
        all_reduce = run_reduction_benchmark(
            datasets=ds_arg,
            methods=REDUCTION_METHODS,
            ratios=REDUCTION_RATIOS,
            random_state=args.seed,
            verbose=True,
        )
        print()
        print_results_table(all_reduce, task='reduce')
        _print_ratio_breakdown(all_reduce, 'reduce', REDUCTION_METHODS, REDUCTION_RATIOS)
        _print_win_table(all_reduce, 'reduce', REDUCTION_METHODS)

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------
    if 'expand' in args.tasks:
        print('─' * 72)
        print(f'  EXPANSION BENCHMARK  (max_n={args.max_n_expand})')
        print('─' * 72)
        all_expand = run_expansion_benchmark(
            datasets=ds_arg,
            methods=EXPANSION_METHODS,
            expansion_ratios=EXPANSION_RATIOS,
            random_state=args.seed,
            max_n=args.max_n_expand,
            verbose=True,
            include_references=False,
        )
        print()
        print_results_table(all_expand, task='expand')
        _print_ratio_breakdown(all_expand, 'expand', EXPANSION_METHODS, EXPANSION_RATIOS)
        _print_win_table(all_expand, 'expand', EXPANSION_METHODS)

    print('Done.')


if __name__ == '__main__':
    main()
