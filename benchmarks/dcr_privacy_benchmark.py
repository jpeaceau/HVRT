#!/usr/bin/env python3
"""
HVRT Privacy–Fidelity Trade-off Benchmark
==========================================

Sweeps `bandwidth` and `n_partitions` to characterise the Privacy DCR ↔
fidelity trade-off, enabling parameter selection based on privacy requirements.

A second sweep varies `adaptive_bandwidth` at expansion ratios 1×/2×/5× to
show how bandwidth auto-scaling shifts DCR at generation time.

All runs use continuous-feature datasets only (fraud, housing, multimodal);
the adult dataset is excluded because its near-duplicate categorical records
collapse the real→real NN distance to ≈ 0, making DCR unreliable there.

Metrics
-------
  privacy_dcr   Distance-to-Closest-Record ratio.  See README §Privacy evaluation.
  novelty_min   Min distance from any synthetic sample to any real sample.
  mf            Marginal fidelity (1 = perfect, ↑ better).
  disc_err      |discriminator accuracy − 0.50|  (↓ better; 0 = indistinguishable).
  tstr_delta    TSTR score minus TRTR baseline (↑ better; 0 = matches real data).

Decision matrix
---------------
Results are grouped into four privacy profiles after aggregation:
  Tight      DCR < 0.40   (augmentation / ML utility priority)
  Moderate   0.40 – 0.70  (balanced)
  High       0.70 – 1.00  (privacy-aware)
  Maximum    DCR ≥ 1.00   (anonymisation / regulatory contexts)

Usage
-----
    python benchmarks/dcr_privacy_benchmark.py
    python benchmarks/dcr_privacy_benchmark.py --datasets housing multimodal
    python benchmarks/dcr_privacy_benchmark.py --ratio 5.0
    python benchmarks/dcr_privacy_benchmark.py --no-adaptive
    python benchmarks/dcr_privacy_benchmark.py --output benchmarks/results/dcr_privacy.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# UTF-8 stdout on Windows
if hasattr(sys.stdout, 'buffer') and \
        getattr(sys.stdout, 'encoding', '').lower().replace('-', '') \
        not in ('utf8', 'utf-8'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )

from hvrt import HVRT
from hvrt.benchmarks.datasets import BENCHMARK_DATASETS
from hvrt.benchmarks.metrics import (
    marginal_fidelity,
    discriminator_accuracy,
    privacy_dcr,
    novelty_min,
    ml_utility_tstr,
)
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Continuous datasets only (DCR is well-defined; no categorical near-duplicates)
DATASETS = ['fraud', 'housing', 'multimodal']

BANDWIDTH_GRID    = ['auto', 0.10, 0.30, 0.50, 1.00, 'scott']
N_PARTITIONS_GRID = [None, 20, 10, 5]           # None = auto-tune
ADAPTIVE_RATIOS   = [1.0, 2.0, 5.0]            # expansion ratios for adaptive sweep

DEFAULT_EXPANSION_RATIO = 2.0
MAX_N        = 500
RANDOM_STATE = 42

# Privacy profile thresholds (applied to mean DCR across datasets)
PROFILES = [
    ('Tight',    0.00, 0.40),
    ('Moderate', 0.40, 0.70),
    ('High',     0.70, 1.00),
    ('Maximum',  1.00, float('inf')),
]


# ─────────────────────────────────────────────────────────────────────────────
# Core measurement
# ─────────────────────────────────────────────────────────────────────────────

def _proxy_y(X_train, y_train, X_synth):
    """Assign synthetic labels via a GBM proxy trained on real data."""
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    is_cls = len(np.unique(y_train)) <= 20
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        proxy = (GradientBoostingClassifier(n_estimators=50, random_state=42)
                 if is_cls
                 else GradientBoostingRegressor(n_estimators=50, random_state=42))
        proxy.fit(X_train, y_train)
    return proxy.predict(X_synth)


def run_one(
    X_train, y_train, X_test, y_test,
    n_partitions, bandwidth, adaptive_bandwidth, expansion_ratio,
    random_state=RANDOM_STATE,
):
    """
    Fit HVRT, expand, and return all privacy + fidelity metrics.

    Returns
    -------
    dict  metric name → float, plus 'fit_time' and 'expand_time'
    """
    n_synth = max(2, int(len(X_train) * expansion_ratio))

    model = HVRT(
        n_partitions=n_partitions,
        bandwidth=bandwidth,
        random_state=random_state,
    )
    t0 = time.perf_counter()
    model.fit(X_train)
    t_fit = time.perf_counter() - t0

    t1 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X_synth = model.expand(
            n=n_synth,
            adaptive_bandwidth=adaptive_bandwidth,
        )
    t_expand = time.perf_counter() - t1

    y_synth = _proxy_y(X_train, y_train, X_synth)

    trtr = ml_utility_tstr(X_train, y_train, X_test, y_test)
    tstr = ml_utility_tstr(X_synth, y_synth, X_test, y_test)

    return {
        'privacy_dcr':      privacy_dcr(X_train, X_synth),
        'novelty_min':      novelty_min(X_train, X_synth),
        'mf':               marginal_fidelity(X_train, X_synth),
        'disc_err':         abs(discriminator_accuracy(X_train, X_synth) - 0.50),
        'ml_utility_trtr':  float(trtr),
        'ml_utility_tstr':  float(tstr),
        'tstr_delta':       float(tstr - trtr),
        'fit_time':         round(t_fit, 4),
        'expand_time':      round(t_expand, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hr(char='─', width=88):
    return char * width


def _profile_label(dcr):
    for name, lo, hi in PROFILES:
        if lo <= dcr < hi:
            return name
    return 'Maximum'


def print_grid_table(grid_results):
    """Print per-dataset and aggregate grid results."""
    # Aggregate across datasets
    agg = defaultdict(lambda: defaultdict(list))
    for r in grid_results:
        key = (r['n_partitions'], r['bandwidth'])
        for m, v in r['metrics'].items():
            agg[key][m].append(v)

    rows = []
    for (n_parts, bw), metrics in agg.items():
        rows.append({
            'n_parts':   n_parts,
            'bw':        bw,
            'dcr':       float(np.mean(metrics['privacy_dcr'])),
            'nov':       float(np.mean(metrics['novelty_min'])),
            'mf':        float(np.mean(metrics['mf'])),
            'disc_err':  float(np.mean(metrics['disc_err'])),
            'trtr':      float(np.mean(metrics['ml_utility_trtr'])),
            'tstr':      float(np.mean(metrics['ml_utility_tstr'])),
            'delta':     float(np.mean(metrics['tstr_delta'])),
        })
    rows.sort(key=lambda r: r['dcr'])

    print('\n' + _hr('═'))
    print('  BANDWIDTH × PARTITIONS  — Privacy–Fidelity Grid')
    print('  (mean across continuous datasets; expansion ratio fixed)')
    print(_hr('═'))
    header = (
        f"  {'Profile':<10} {'n_parts':<9} {'bandwidth':<11}"
        f" {'DCR':>7} {'Nov.Min':>9} {'Marg.F':>7} {'Disc.Err':>9}"
        f" {'TRTR':>7} {'TSTR':>7} {'TSTR Δ':>8}"
    )
    print(header)
    print('  ' + _hr('─', len(header) - 2))

    prev_profile = None
    for r in rows:
        profile = _profile_label(r['dcr'])
        n_str = str(r['n_parts']) if r['n_parts'] is not None else 'auto'
        if profile != prev_profile:
            if prev_profile is not None:
                print('  ' + _hr('·', len(header) - 2))
            prev_profile = profile
        print(
            f"  {profile:<10} {n_str:<9} {str(r['bw']):<11}"
            f" {r['dcr']:>7.3f} {r['nov']:>9.4f} {r['mf']:>7.3f}"
            f" {r['disc_err']:>9.4f}"
            f" {r['trtr']:>7.4f} {r['tstr']:>7.4f} {r['delta']:>+8.3f}"
        )

    print(_hr('═'))


def print_adaptive_table(adaptive_results):
    """Print adaptive-bandwidth results grouped by expansion ratio."""
    from itertools import groupby

    print('\n' + _hr('═'))
    print('  ADAPTIVE BANDWIDTH  — DCR vs Expansion Ratio')
    print('  (n_parts=auto, bandwidth=auto, mean across continuous datasets)')
    print(_hr('═'))

    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in adaptive_results:
        for key in ('privacy_dcr', 'mf', 'ml_utility_trtr', 'ml_utility_tstr', 'tstr_delta'):
            agg[r['expansion_ratio']][r['adaptive_bandwidth']][key].append(
                r['metrics'][key]
            )

    header = (
        f"  {'Ratio':<8} {'Adaptive':<12} {'DCR':>7}"
        f" {'Profile':<12} {'Marg.F':>7} {'TRTR':>7} {'TSTR':>7} {'TSTR Δ':>8}"
    )
    print(header)
    print('  ' + _hr('─', len(header) - 2))

    for ratio in sorted(agg.keys()):
        for adaptive in [False, True]:
            if adaptive not in agg[ratio]:
                continue
            m = agg[ratio][adaptive]
            dcr  = float(np.mean(m['privacy_dcr']))
            mf   = float(np.mean(m['mf']))
            trtr = float(np.mean(m['ml_utility_trtr']))
            tstr = float(np.mean(m['ml_utility_tstr']))
            delta = float(np.mean(m['tstr_delta']))
            print(
                f"  {ratio:<8.1f} {str(adaptive):<12} {dcr:>7.3f}"
                f" {_profile_label(dcr):<12} {mf:>7.3f}"
                f" {trtr:>7.4f} {tstr:>7.4f} {delta:>+8.3f}"
            )

    print(_hr('═'))


def print_decision_matrix(grid_results, adaptive_results, expansion_ratio):
    """
    Summarise the best parameter choice for each privacy profile.
    Selects the configuration with highest DCR within each profile
    that does not degrade TSTR below −0.05.
    """
    agg = defaultdict(lambda: defaultdict(list))
    for r in grid_results:
        key = (r['n_partitions'], r['bandwidth'])
        for m, v in r['metrics'].items():
            agg[key][m].append(v)

    rows = []
    for (n_parts, bw), metrics in agg.items():
        dcr  = float(np.mean(metrics['privacy_dcr']))
        rows.append({
            'n_parts': n_parts,
            'bw':      bw,
            'dcr':     dcr,
            'mf':      float(np.mean(metrics['mf'])),
            'disc_err':float(np.mean(metrics['disc_err'])),
            'trtr':    float(np.mean(metrics['ml_utility_trtr'])),
            'tstr':    float(np.mean(metrics['ml_utility_tstr'])),
            'delta':   float(np.mean(metrics['tstr_delta'])),
            'profile': _profile_label(dcr),
        })

    print('\n' + _hr('═'))
    print('  PRIVACY–FIDELITY DECISION MATRIX')
    print(f'  (expansion ratio {expansion_ratio}×, continuous data, TSTR Δ ≥ −0.05 filter)')
    print(_hr('═'))
    print(
        f"  {'Profile':<11} {'DCR target':<14} {'Recommended parameters':<32}"
        f" {'DCR':>7} {'Marg.F':>7} {'Disc.Err':>9} {'TRTR':>7} {'TSTR':>7} {'TSTR Δ':>8}"
    )
    print('  ' + _hr('─', 106))

    for profile, lo, hi in PROFILES:
        candidates = [
            r for r in rows
            if lo <= r['dcr'] < hi and r['delta'] >= -0.05
        ]
        if not candidates:
            candidates = [r for r in rows if lo <= r['dcr'] < hi]
        if not candidates:
            print(f"  {profile:<11} [{lo:.2f}, {hi:.2f})    {'—':>32}")
            continue

        best = max(candidates, key=lambda r: r['mf'])

        n_str = str(best['n_parts']) if best['n_parts'] is not None else 'None (auto)'
        params = f"n_partitions={n_str}, bandwidth={best['bw']!r}"
        dcr_range = f"[{lo:.2f}, {hi if hi < 99 else '∞'})"
        print(
            f"  {profile:<11} {dcr_range:<14} {params:<38}"
            f" {best['dcr']:>7.3f} {best['mf']:>7.3f}"
            f" {best['disc_err']:>9.4f}"
            f" {best['trtr']:>7.4f} {best['tstr']:>7.4f} {best['delta']:>+8.3f}"
        )

    print(_hr('═'))
    print('  Notes:')
    print('    • DCR < 0.10 = synthetic lies very close to real records (near-copy risk).')
    print('    • DCR 0.40–0.70: HVRT default — realistic samples for ML augmentation.')
    print('    • DCR ≥ 1.00: samples more dispersed than real neighbours.')
    print('    • TSTR Δ < −0.05 means downstream ML utility drops meaningfully.')
    print('    • For categorical/mixed data, compute DCR on continuous columns only.')
    print(_hr('═'))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HVRT Privacy–Fidelity Trade-off Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        help='Datasets to benchmark (default: fraud housing multimodal)')
    parser.add_argument('--ratio', type=float, default=DEFAULT_EXPANSION_RATIO,
                        help='Expansion ratio for the main grid sweep (default: 2.0)')
    parser.add_argument('--no-adaptive', action='store_true',
                        help='Skip the adaptive-bandwidth ratio sweep')
    parser.add_argument('--output', default='benchmarks/results/dcr_privacy.json',
                        help='Path to save JSON results')
    args = parser.parse_args()

    print(f'\nHVRT Privacy–Fidelity Benchmark')
    print(f'  Datasets         : {", ".join(args.datasets)}')
    print(f'  Grid ratio       : {args.ratio}×')
    print(f'  Training cap     : {MAX_N} samples')
    print(f'  Adaptive sweep   : {"disabled" if args.no_adaptive else "enabled"}')

    all_results = []

    # ── Main grid ────────────────────────────────────────────────────────────
    grid_results = []
    print(f'\n{"─" * 70}')
    print('  SWEEP 1: bandwidth × n_partitions')
    print(f'{"─" * 70}')

    for ds_name in args.datasets:
        X, y, _ = BENCHMARK_DATASETS[ds_name](random_state=RANDOM_STATE)
        X, y = X[:MAX_N], y[:MAX_N]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        for n_parts in N_PARTITIONS_GRID:
            for bw in BANDWIDTH_GRID:
                n_str = str(n_parts) if n_parts is not None else 'auto'
                label = f'[{ds_name}] n_parts={n_str:<5} bw={str(bw):<8}'
                print(f'  {label}', end=' ', flush=True)
                try:
                    m = run_one(
                        X_tr, y_tr, X_te, y_te,
                        n_partitions=n_parts,
                        bandwidth=bw,
                        adaptive_bandwidth=False,
                        expansion_ratio=args.ratio,
                    )
                    print(
                        f"dcr={m['privacy_dcr']:.3f}  "
                        f"mf={m['mf']:.3f}  "
                        f"disc_err={m['disc_err']:.3f}  "
                        f"tstr_delta={m['tstr_delta']:+.3f}"
                    )
                    entry = {
                        'sweep':             'bw_x_parts',
                        'dataset':           ds_name,
                        'n_partitions':      n_parts,
                        'bandwidth':         str(bw),
                        'adaptive_bandwidth':False,
                        'expansion_ratio':   args.ratio,
                        'metrics':           m,
                    }
                    grid_results.append(entry)
                    all_results.append(entry)
                except Exception as exc:
                    print(f'ERROR: {exc}')

    # ── Adaptive sweep ───────────────────────────────────────────────────────
    adaptive_results = []
    if not args.no_adaptive:
        print(f'\n{"─" * 70}')
        print('  SWEEP 2: adaptive_bandwidth × expansion ratio  (n_parts=auto, bw=auto)')
        print(f'{"─" * 70}')

        for ds_name in args.datasets:
            X, y, _ = BENCHMARK_DATASETS[ds_name](random_state=RANDOM_STATE)
            X, y = X[:MAX_N], y[:MAX_N]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )

            for ratio in ADAPTIVE_RATIOS:
                for adaptive in [False, True]:
                    label = (
                        f'[{ds_name}] ratio={ratio:.0f}×  '
                        f'adaptive={str(adaptive):<6}'
                    )
                    print(f'  {label}', end=' ', flush=True)
                    try:
                        m = run_one(
                            X_tr, y_tr, X_te, y_te,
                            n_partitions=None,
                            bandwidth='auto',
                            adaptive_bandwidth=adaptive,
                            expansion_ratio=ratio,
                        )
                        print(
                            f"dcr={m['privacy_dcr']:.3f}  "
                            f"mf={m['mf']:.3f}  "
                            f"tstr_delta={m['tstr_delta']:+.3f}"
                        )
                        entry = {
                            'sweep':             'adaptive',
                            'dataset':           ds_name,
                            'n_partitions':      None,
                            'bandwidth':         'auto',
                            'adaptive_bandwidth':adaptive,
                            'expansion_ratio':   ratio,
                            'metrics':           m,
                        }
                        adaptive_results.append(entry)
                        all_results.append(entry)
                    except Exception as exc:
                        print(f'ERROR: {exc}')

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as fh:
        json.dump(all_results, fh, indent=2)
    print(f'\nResults saved to {args.output}')

    # ── Print tables ─────────────────────────────────────────────────────────
    if grid_results:
        print_grid_table(grid_results)
    if adaptive_results:
        print_adaptive_table(adaptive_results)
    if grid_results:
        print_decision_matrix(grid_results, adaptive_results, args.ratio)


if __name__ == '__main__':
    main()
