#!/usr/bin/env python3
"""
HVRT Benchmark Report Generator
================================
Reads benchmark_results.json (produced by run_benchmarks.py) and generates:

  1. Console text report:
     - Metric interpretation guide
     - Aggregate summary tables (mean ± std across all datasets and conditions)
     - Method rankings per metric
     - Per-dataset breakdowns

  2. Saved figures (--output-dir, default: benchmarks/results/report/):
     - reduction_overview.png   Bar charts for all reduction metrics
     - expansion_overview.png   Bar charts for all expansion metrics
     - heatmap_reduction.png    Methods × metrics normalised heatmap (reduction)
     - heatmap_expansion.png    Methods × metrics normalised heatmap (expansion)
     - scatter_efficiency.png   Quality vs speed scatter (both tasks)
     - radar_comparison.png     Radar chart — HVRT family vs best competitors

  3. report.txt saved alongside figures (full text report)

Usage
-----
    python benchmarks/report_results.py
    python benchmarks/report_results.py --input  benchmarks/results/benchmark_results.json
    python benchmarks/report_results.py --output-dir benchmarks/results/report
    python benchmarks/report_results.py --no-plots
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from math import pi

import numpy as np

# Ensure the package is importable when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ─────────────────────────────────────────────────────────────────────────────
# Metric metadata
# ─────────────────────────────────────────────────────────────────────────────

METRIC_INFO = {
    # ── Reduction ───────────────────────────────────────────────────────────
    'marginal_fidelity': {
        'display':     'Marginal Fidelity',
        'range':       '[0, 1]',
        'target':      'maximize  (1.0 = perfect)',
        'better':      'higher',
        'direction':   'higher',
        'description': (
            "1 − mean normalised 1-D Wasserstein distance across all features.\n"
            "Measures how faithfully the per-feature marginal distributions are\n"
            "preserved after reduction or synthesis.  A value ≥ 0.95 is strong;\n"
            "below 0.90 indicates meaningful distributional shift."
        ),
    },
    'correlation_fidelity': {
        'display':     'Correlation Fidelity',
        'range':       '(−∞, 1]',
        'target':      'maximize  (1.0 = identical correlation structure)',
        'better':      'higher',
        'direction':   'higher',
        'description': (
            "1 − Frobenius(C_transformed − C_original) / Frobenius(C_original).\n"
            "Measures preservation of pairwise feature correlations.  Critical for\n"
            "datasets where inter-feature relationships drive downstream models."
        ),
    },
    'tail_preservation': {
        'display':     'Tail Preservation',
        'range':       '(0, ∞)',
        'target':      '1.0  (exact tail match)',
        'better':      'nearest to 1.0',
        'direction':   'target_1',
        'description': (
            "Geometric mean of per-feature 5th/95th percentile-range ratios.\n"
            "< 1: tail shrinkage (extremes discarded).  > 1: tail amplification.\n"
            "|value − 1.0| is the tail error; target = 0.  Especially important\n"
            "for anomaly detection, fraud modelling, and risk analysis."
        ),
    },
    'ml_utility_retention': {
        'display':     'ML Utility Retention',
        'range':       '[0, 1] for F1 ;  (−∞, 1] for R²',
        'target':      'maximize  (1.0 = full-data performance)',
        'better':      'higher',
        'direction':   'higher',
        'description': (
            "Train-on-Reduced, Test-on-Real (TRTR).\n"
            "F1 weighted (classification) or R² (regression).\n"
            "Measures whether a reduced set trains models as well as the full set.\n"
            "≥ 0.95 implies ≤5 % predictive performance loss."
        ),
    },
    'emergence_score': {
        'display':     'Emergence Score',
        'range':       '[0, 1]',
        'target':      'maximize  (1.0 = identical conditional structure)',
        'better':      'higher',
        'direction':   'higher',
        'description': (
            "Proportion of samples whose decision-tree leaf assignment agrees\n"
            "between models fitted on original vs transformed data.\n"
            "Captures preservation of conditional structure (emergence).\n"
            "Only computed on the dedicated emergence benchmark datasets."
        ),
    },
    # ── Expansion ───────────────────────────────────────────────────────────
    'discriminator_accuracy': {
        'display':     'Discriminator Accuracy',
        'range':       '[0, 1]',
        'target':      '0.50  (indistinguishable from real)',
        'better':      'nearest to 0.50',
        'direction':   'target_05',
        'description': (
            "Logistic regression accuracy classifying real vs synthetic samples.\n"
            "50 % = classifier is at chance → synthetic is indistinguishable from real.\n"
            "Above 60 % suggests detectable artefacts; above 70 % is poor quality.\n"
            "Primary quality metric for synthetic data generation."
        ),
    },
    'privacy_dcr': {
        'display':     'Privacy DCR',
        'range':       '[0, ∞)',
        'target':      '> 1.0  (privacy-preserving)',
        'better':      'higher',
        'direction':   'higher',
        'description': (
            "Distance-to-Closest-Record ratio.\n"
            "= median(synth→real distance) / median(real→real LOO distance).\n"
            "> 1.0: synthetic samples are further from training data than real\n"
            "samples are from each other → privacy-preserving.\n"
            "< 1.0: risk of memorisation / near-copy of training data."
        ),
    },
    'novelty_min': {
        'display':     'Novelty (Min Distance)',
        'range':       '[0, ∞)',
        'target':      '> 0  (no exact copies)',
        'better':      'higher',
        'direction':   'higher',
        'description': (
            "Minimum Euclidean distance from any synthetic sample to any real sample.\n"
            "0 = at least one synthetic point is an exact copy of real data.\n"
            "Higher confirms genuinely novel generation.\n"
            "Complementary to privacy_dcr; captures worst-case privacy leakage."
        ),
    },
    'ml_utility_tstr': {
        'display':     'ML Utility (TSTR)',
        'range':       '[0, 1] for F1 ;  (−∞, 1] for R²',
        'target':      'maximize  (match real-data performance)',
        'better':      'higher',
        'direction':   'higher',
        'description': (
            "Train-on-Synthetic, Test-on-Real (TSTR).\n"
            "F1 weighted (classification) or R² (regression).\n"
            "Measures whether synthetic data is a useful model-training substitute.\n"
            "Close to real-data baseline confirms downstream utility."
        ),
    },
    # ── Timing ──────────────────────────────────────────────────────────────
    'train_time_seconds': {
        'display':     'Fit Time (s)',
        'range':       '[0, ∞)',
        'target':      'minimize',
        'better':      'lower',
        'direction':   'lower',
        'description': "Wall-clock time for model fitting (tree + KDE construction).",
    },
    'operation_time_seconds': {
        'display':     'Operation Time (s)',
        'range':       '[0, ∞)',
        'target':      'minimize',
        'better':      'lower',
        'direction':   'lower',
        'description': "Wall-clock time for the reduce / expand operation itself.",
    },
}

REDUCE_COLS  = ['marginal_fidelity', 'correlation_fidelity', 'tail_preservation',
                'ml_utility_retention', 'operation_time_seconds']
EXPAND_COLS  = ['marginal_fidelity', 'discriminator_accuracy', 'tail_preservation',
                'privacy_dcr', 'novelty_min', 'ml_utility_tstr', 'operation_time_seconds']

# Metrics used in the radar chart (normalised to [0,1] where 1 = best)
RADAR_REDUCE = ['marginal_fidelity', 'correlation_fidelity', 'tail_preservation',
                'ml_utility_retention']
RADAR_EXPAND = ['marginal_fidelity', 'discriminator_accuracy', 'tail_preservation',
                'privacy_dcr', 'ml_utility_tstr']

HVRT_COLOUR     = '#1565C0'   # deep blue — HVRT family
FASTHVRT_COLOUR = '#42A5F5'   # light blue — FastHVRT family
COMP_COLOUR     = '#9E9E9E'   # grey — competitors
TARGET_COLOUR   = '#E53935'   # red — target reference lines


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _method_colour(name):
    if name.startswith('FastHVRT'):
        return FASTHVRT_COLOUR
    if name.startswith('HVRT'):
        return HVRT_COLOUR
    return COMP_COLOUR


def _score_for_rank(metric, value):
    """Return a sortable rank-score where higher rank-score = better."""
    d = METRIC_INFO.get(metric, {}).get('direction', 'higher')
    if d == 'higher':
        return value
    if d == 'lower':
        return -value
    if d == 'target_1':
        return -abs(value - 1.0)
    if d == 'target_05':
        return -abs(value - 0.5)
    return value


def _normalise_for_heatmap(metric, values):
    """Normalise a list of metric values to [0, 1] where 1 = best."""
    arr = np.array(values, dtype=float)
    d = METRIC_INFO.get(metric, {}).get('direction', 'higher')

    if d == 'higher':
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-10:
            return np.ones_like(arr) * 0.5
        return (arr - lo) / (hi - lo)

    if d == 'lower':
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-10:
            return np.ones_like(arr) * 0.5
        return (hi - arr) / (hi - lo)

    if d == 'target_1':
        err = np.abs(arr - 1.0)
        max_err = err.max()
        if max_err < 1e-10:
            return np.ones_like(arr)
        return 1.0 - err / max_err

    if d == 'target_05':
        err = np.abs(arr - 0.5)
        max_err = err.max()
        if max_err < 1e-10:
            return np.ones_like(arr)
        return 1.0 - err / max_err

    return arr


def aggregate_by_method(results, task):
    """
    Aggregate metric values per method across all datasets and param conditions.

    Returns
    -------
    dict: method → {metric: {'mean': float, 'std': float, 'n': int, 'values': list}}
    """
    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r['task'] != task:
            continue
        method = r['method']
        for k, v in r['metrics'].items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                data[method][k].append(float(v))

    agg = {}
    for method, metrics in data.items():
        agg[method] = {}
        for k, vals in metrics.items():
            agg[method][k] = {
                'mean':   float(np.mean(vals)),
                'std':    float(np.std(vals)),
                'median': float(np.median(vals)),
                'min':    float(np.min(vals)),
                'max':    float(np.max(vals)),
                'n':      len(vals),
                'values': vals,
            }
    return agg


def aggregate_by_dataset_method(results, task):
    """Aggregate per (dataset, method) pair — mean across param conditions."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        if r['task'] != task:
            continue
        for k, v in r['metrics'].items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                data[r['dataset']][r['method']][k].append(float(v))

    agg = {}
    for ds, methods in data.items():
        agg[ds] = {}
        for method, metrics in methods.items():
            agg[ds][method] = {k: float(np.mean(v)) for k, v in metrics.items()}
    return agg


def rank_methods(agg, metric):
    """
    Return methods sorted best→worst for the given metric.
    Only includes methods that have values for this metric.
    """
    candidates = [
        (m, agg[m][metric]['mean'])
        for m in agg
        if metric in agg[m]
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda x: _score_for_rank(metric, x[1]), reverse=True)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────

def _hr(char='═', width=80):
    return char * width


def _fmt(v, metric):
    if isinstance(v, float) and np.isnan(v):
        return '     N/A'
    d = METRIC_INFO.get(metric, {}).get('direction', 'higher')
    if 'time' in metric:
        return f'{v:>8.3f}s'
    return f'{v:>8.4f}'


def print_metric_guide(out):
    print(_hr(), file=out)
    print('  METRIC INTERPRETATION GUIDE', file=out)
    print(_hr(), file=out)

    sections = [
        ('REDUCTION METRICS', ['marginal_fidelity', 'correlation_fidelity',
                                'tail_preservation', 'ml_utility_retention',
                                'emergence_score']),
        ('EXPANSION METRICS', ['marginal_fidelity', 'discriminator_accuracy',
                                'tail_preservation', 'privacy_dcr',
                                'novelty_min', 'ml_utility_tstr']),
        ('TIMING METRICS',    ['train_time_seconds', 'operation_time_seconds']),
    ]

    seen = set()
    for section_title, keys in sections:
        print(f'\n  {section_title}', file=out)
        print('  ' + _hr('─', 76), file=out)
        for k in keys:
            if k in seen:
                continue
            seen.add(k)
            info = METRIC_INFO.get(k, {})
            print(f'\n  {info.get("display", k)}  [{k}]', file=out)
            print(f'    Range  : {info.get("range", "?")}', file=out)
            print(f'    Target : {info.get("target", "?")}', file=out)
            for line in info.get('description', '').splitlines():
                print(f'    {line}', file=out)


def print_aggregate_table(agg, cols, task_label, out):
    method_w = max(18, max(len(m) for m in agg) + 2)
    col_w    = 16
    header   = f'  {"Method":<{method_w}}' + ''.join(
        f'{METRIC_INFO.get(c, {}).get("display", c)[:col_w-1]:>{col_w}}'
        for c in cols
    )
    sep = _hr('─', len(header))

    print(_hr(), file=out)
    print(f'  {task_label}  —  AGGREGATE SUMMARY  (mean ± std across all datasets & conditions)', file=out)
    print(_hr(), file=out)
    print(header, file=out)
    print(f'  {sep}', file=out)

    for method in sorted(agg.keys()):
        row = f'  {method:<{method_w}}'
        for c in cols:
            if c in agg[method]:
                s = agg[method][c]
                if 'time' in c:
                    row += f'{s["mean"]:>{col_w}.3f}'
                else:
                    row += f'{s["mean"]:>{col_w - 6}.4f} ±{s["std"]:<5.3f}'
            else:
                row += f'{"N/A":>{col_w}}'
        print(row, file=out)

    print(f'  {sep}', file=out)


def print_rankings(agg, cols, task_label, out):
    print(f'\n  {task_label} — RANKINGS  (best → worst per metric)', file=out)
    print('  ' + _hr('─', 76), file=out)

    for c in cols:
        ranked = rank_methods(agg, c)
        if not ranked:
            continue
        info   = METRIC_INFO.get(c, {})
        target = info.get('target', '')
        line   = f'  {info.get("display", c):<30}  target: {target:<35}'
        print(line, file=out)
        parts  = []
        for i, (m, v) in enumerate(ranked, 1):
            fmt = _fmt(v, c)
            parts.append(f'    {i}. {m} ({fmt.strip()})')
        print('\n'.join(parts[:8]), file=out)
        print(file=out)


def print_per_dataset_table(ds_agg, cols, task_label, out):
    print(_hr(), file=out)
    print(f'  {task_label}  —  PER-DATASET BREAKDOWN', file=out)
    print(_hr(), file=out)

    col_w    = 12
    for ds_name in sorted(ds_agg.keys()):
        methods = ds_agg[ds_name]
        method_w = max(18, max(len(m) for m in methods) + 2)
        print(f'\n  Dataset: {ds_name}', file=out)
        header = f'    {"Method":<{method_w}}' + ''.join(
            f'{METRIC_INFO.get(c, {}).get("display", c)[:col_w-1]:>{col_w}}'
            for c in cols
        )
        print(header, file=out)
        print('    ' + _hr('─', len(header) - 4), file=out)

        # Sort methods by primary metric
        primary = cols[0]
        def sort_key(item):
            v = item[1].get(primary, float('nan'))
            if np.isnan(v):
                return -np.inf
            return _score_for_rank(primary, v)

        for method, mets in sorted(methods.items(), key=sort_key, reverse=True):
            row = f'    {method:<{method_w}}'
            for c in cols:
                v = mets.get(c, float('nan'))
                if isinstance(v, float) and np.isnan(v):
                    row += f'{"N/A":>{col_w}}'
                elif 'time' in c:
                    row += f'{v:>{col_w}.3f}'
                else:
                    row += f'{v:>{col_w}.4f}'
            print(row, file=out)


def generate_text_report(results, out=None):
    if out is None:
        out = sys.stdout

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    print(_hr('═'), file=out)
    print('  HVRT BENCHMARK REPORT', file=out)
    print(f'  Generated : {ts}', file=out)
    n_reduce = sum(1 for r in results if r['task'] == 'reduce')
    n_expand = sum(1 for r in results if r['task'] == 'expand')
    print(f'  Results   : {n_reduce} reduction runs,  {n_expand} expansion runs', file=out)
    print(_hr('═'), file=out)

    # ── Metric guide ───────────────────────────────────────────────────────
    print_metric_guide(out)

    # ── Reduction aggregate ────────────────────────────────────────────────
    if n_reduce:
        agg_r  = aggregate_by_method(results, 'reduce')
        ds_r   = aggregate_by_dataset_method(results, 'reduce')
        print(file=out)
        print_aggregate_table(agg_r, REDUCE_COLS, 'REDUCTION', out)
        print_rankings(agg_r, REDUCE_COLS, 'REDUCTION', out)
        print_per_dataset_table(ds_r, REDUCE_COLS, 'REDUCTION', out)

    # ── Expansion aggregate ────────────────────────────────────────────────
    if n_expand:
        agg_e = aggregate_by_method(results, 'expand')
        ds_e  = aggregate_by_dataset_method(results, 'expand')
        print(file=out)
        print_aggregate_table(agg_e, EXPAND_COLS, 'EXPANSION', out)
        print_rankings(agg_e, EXPAND_COLS, 'EXPANSION', out)
        print_per_dataset_table(ds_e, EXPAND_COLS, 'EXPANSION', out)

    print(f'\n{_hr("═")}', file=out)
    print('  END OF REPORT', file=out)
    print(_hr('═'), file=out)


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar_ax(ax, agg, metric, title, show_target=None):
    """Fill a single Axes with a horizontal bar chart for one metric."""
    methods = sorted(agg.keys(), key=lambda m: _score_for_rank(metric, agg[m].get(metric, {}).get('mean', 0)), reverse=True)
    means   = [agg[m].get(metric, {}).get('mean', np.nan) for m in methods]
    stds    = [agg[m].get(metric, {}).get('std',  0.0)    for m in methods]
    colours = [_method_colour(m) for m in methods]

    y_pos = np.arange(len(methods))
    bars  = ax.barh(y_pos, means, xerr=stds, capsize=3,
                    color=colours, edgecolor='white', linewidth=0.4, error_kw={'linewidth': 0.8})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=7)
    ax.set_title(title, fontsize=8, fontweight='bold')
    info = METRIC_INFO.get(metric, {})
    ax.set_xlabel(info.get('display', metric), fontsize=7)

    if show_target is not None:
        ax.axvline(show_target, color=TARGET_COLOUR, linestyle='--',
                   linewidth=1.0, alpha=0.8, label=f'target = {show_target}')
        ax.legend(fontsize=6)

    # Annotate the best bar
    finite_means = [m for m in means if not np.isnan(m)]
    if finite_means:
        ax.tick_params(axis='both', labelsize=7)


def _make_legend(fig):
    """Add a shared colour legend to a figure."""
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=HVRT_COLOUR,     label='HVRT'),
        Patch(facecolor=FASTHVRT_COLOUR, label='FastHVRT'),
        Patch(facecolor=COMP_COLOUR,     label='Competitors'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 & 2 — Overview bar charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_reduction_overview(results):
    import matplotlib.pyplot as plt

    agg  = aggregate_by_method(results, 'reduce')
    cols = [c for c in REDUCE_COLS if any(c in agg[m] for m in agg)]
    if not cols:
        return None

    ncols = 3
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()

    targets = {'tail_preservation': 1.0}

    for i, col in enumerate(cols):
        _bar_ax(axes[i], agg, col,
                METRIC_INFO.get(col, {}).get('display', col),
                show_target=targets.get(col))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('HVRT Benchmark — Reduction Performance Overview\n'
                 '(mean ± std across all datasets & ratios)',
                 fontsize=11, fontweight='bold')
    _make_legend(fig)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig


def plot_expansion_overview(results):
    import matplotlib.pyplot as plt

    agg  = aggregate_by_method(results, 'expand')
    cols = [c for c in EXPAND_COLS if any(c in agg[m] for m in agg)]
    if not cols:
        return None

    ncols = 3
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()

    targets = {
        'discriminator_accuracy': 0.50,
        'tail_preservation':      1.0,
        'privacy_dcr':            1.0,
    }

    for i, col in enumerate(cols):
        _bar_ax(axes[i], agg, col,
                METRIC_INFO.get(col, {}).get('display', col),
                show_target=targets.get(col))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('HVRT Benchmark — Expansion Performance Overview\n'
                 '(mean ± std across all datasets & expansion ratios)',
                 fontsize=11, fontweight='bold')
    _make_legend(fig)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 & 4 — Normalised heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def _heatmap_fig(agg, metrics, title):
    """
    Build a normalised heatmap: methods (rows) × metrics (cols).
    Each cell is normalised to [0,1] where 1 = best across methods for that metric.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    methods = sorted(agg.keys())
    cols    = [c for c in metrics if any(c in agg[m] for m in agg)]
    if not cols or not methods:
        return None

    # Build raw and normalised matrices
    raw_mat  = np.full((len(methods), len(cols)), np.nan)
    norm_mat = np.full((len(methods), len(cols)), np.nan)

    for ci, col in enumerate(cols):
        vals = [agg[m].get(col, {}).get('mean', np.nan) for m in methods]
        finite = [v for v in vals if not np.isnan(v)]
        if not finite:
            continue
        norm_vals = _normalise_for_heatmap(col, [v if not np.isnan(v) else np.nanmean(finite) for v in vals])
        for ri, (raw, nv) in enumerate(zip(vals, norm_vals)):
            raw_mat[ri, ci]  = raw
            norm_mat[ri, ci] = nv if not np.isnan(raw) else np.nan

    fig_h = max(3.5, 0.45 * len(methods) + 1.5)
    fig_w = max(8,   1.4  * len(cols)    + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(norm_mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    # Annotate cells with raw values
    for ri in range(len(methods)):
        for ci in range(len(cols)):
            v = raw_mat[ri, ci]
            if np.isnan(v):
                ax.text(ci, ri, 'N/A', ha='center', va='center', fontsize=7, color='#888')
            else:
                direction = METRIC_INFO.get(cols[ci], {}).get('direction', 'higher')
                fmt = '.3f' if 'time' not in cols[ci] else '.3f'
                ax.text(ci, ri, f'{v:{fmt}}', ha='center', va='center',
                        fontsize=7.5, fontweight='bold',
                        color='#111' if 0.3 < norm_mat[ri, ci] < 0.85 else '#fff')

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(
        [METRIC_INFO.get(c, {}).get('display', c) for c in cols],
        rotation=30, ha='right', fontsize=8,
    )
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)

    # Highlight HVRT-family rows
    for ri, m in enumerate(methods):
        if m.startswith('HVRT') or m.startswith('FastHVRT'):
            ax.add_patch(plt.Rectangle(
                (-0.5, ri - 0.5), len(cols), 1,
                fill=False, edgecolor=HVRT_COLOUR, linewidth=1.8, zorder=3,
            ))

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Normalised score  (1 = best)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(title + '\n(colour = normalised rank;  number = raw value;  blue border = HVRT family)',
                 fontsize=9, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_heatmap_reduction(results):
    agg = aggregate_by_method(results, 'reduce')
    return _heatmap_fig(agg, REDUCE_COLS,
                        'HVRT Benchmark — Reduction  ×  All Metrics')


def plot_heatmap_expansion(results):
    agg = aggregate_by_method(results, 'expand')
    return _heatmap_fig(agg, EXPAND_COLS,
                        'HVRT Benchmark — Expansion  ×  All Metrics')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Quality vs speed scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_efficiency(results):
    """
    Scatter: operation_time_seconds (x) vs primary quality metric (y).
    Reduction → marginal_fidelity.  Expansion → 1 − |discriminator − 0.50|.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    tasks_cfg = [
        ('reduce', 'marginal_fidelity', 'Marginal Fidelity',
         'Reduction: Quality vs Speed', axes[0]),
        ('expand', 'discriminator_accuracy', 'Discriminator Accuracy  (target=0.50)',
         'Expansion: Quality vs Speed', axes[1]),
    ]

    for task, quality_metric, q_label, title, ax in tasks_cfg:
        agg = aggregate_by_method(results, task)
        if not agg:
            ax.set_visible(False)
            continue

        for method, stats in agg.items():
            if quality_metric not in stats or 'operation_time_seconds' not in stats:
                continue
            x = stats['operation_time_seconds']['mean']
            y = stats[quality_metric]['mean']
            ye = stats[quality_metric]['std']
            xe = stats['operation_time_seconds']['std']
            c  = _method_colour(method)

            ax.errorbar(x, y, xerr=xe, yerr=ye,
                        fmt='o', color=c, markersize=7,
                        elinewidth=0.8, capsize=3, alpha=0.85)
            ax.annotate(method, (x, y),
                        textcoords='offset points', xytext=(5, 3),
                        fontsize=6.5, color=c)

        if task == 'expand':
            ax.axhline(0.50, color=TARGET_COLOUR, linestyle='--',
                       linewidth=1.0, alpha=0.7, label='target 0.50')
            ax.legend(fontsize=7)

        ax.set_xscale('symlog', linthresh=0.001)
        ax.set_xlabel('Operation time (s)  [symlog scale]', fontsize=8)
        ax.set_ylabel(q_label, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    _make_legend(fig)
    fig.suptitle('Quality vs Computational Cost  (error bars = ±1 std across datasets/conditions)',
                 fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Radar / spider charts
# ─────────────────────────────────────────────────────────────────────────────

def _radar_ax(ax, methods_data, metric_labels, title):
    """
    Draw a radar chart on ax.

    methods_data : list of (method_name, [normalised_value per metric])
    metric_labels: list of str
    """
    n = len(metric_labels)
    if n < 3:
        ax.set_visible(False)
        return

    angles = [k / float(n) * 2 * pi for k in range(n)]
    angles += angles[:1]

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=7)

    # Concentric reference rings
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [r] * (n + 1), color='#cccccc', linewidth=0.5, linestyle=':')

    for method, vals in methods_data:
        vals_closed = list(vals) + [vals[0]]
        c = _method_colour(method)
        ax.plot(angles, vals_closed, 'o-', linewidth=1.5, color=c,
                markersize=3, label=method, alpha=0.9)
        ax.fill(angles, vals_closed, alpha=0.06, color=c)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=5, color='#888')
    ax.set_title(title, fontsize=9, fontweight='bold', pad=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12),
              fontsize=6.5, framealpha=0.85)


def plot_radar_comparison(results):
    """
    Two radar charts side-by-side: one for reduction, one for expansion.
    Normalises each metric so that the best-performing method scores 1.0.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw={'projection': 'polar'})

    for task, radar_cols, title, ax in [
        ('reduce', RADAR_REDUCE, 'Reduction',  axes[0]),
        ('expand', RADAR_EXPAND, 'Expansion', axes[1]),
    ]:
        agg = aggregate_by_method(results, task)
        if not agg:
            ax.set_visible(False)
            continue

        # Only keep metrics present in at least one method
        cols = [c for c in radar_cols if any(c in agg[m] for m in agg)]
        if len(cols) < 3:
            ax.set_visible(False)
            continue

        # Normalise per metric
        norm_scores = {}
        for m in agg:
            norm_scores[m] = []
            for c in cols:
                raw = agg[m].get(c, {}).get('mean', np.nan)
                norm_scores[m].append(raw if not np.isnan(raw) else 0.0)

        all_vals_per_col = {
            c: [agg[m].get(c, {}).get('mean', np.nan) for m in agg]
            for c in cols
        }
        norm_by_col = {
            c: _normalise_for_heatmap(c, [v if not np.isnan(v) else 0 for v in all_vals_per_col[c]])
            for c in cols
        }
        methods_sorted = sorted(agg.keys())
        methods_data = [
            (m, [norm_by_col[c][i] for i, mc in enumerate(methods_sorted) if mc == m][0]
             if m in methods_sorted else [0] * len(cols))
            for m in methods_sorted
        ]
        # Rebuild properly
        methods_data = []
        for i, m in enumerate(methods_sorted):
            row = [float(norm_by_col[c][i]) for c in cols]
            methods_data.append((m, row))

        labels = [METRIC_INFO.get(c, {}).get('display', c) for c in cols]
        _radar_ax(ax, methods_data, labels,
                  f'{title}\n(normalised: 1.0 = best per metric)')

    fig.suptitle('HVRT Benchmark — Radar Comparison  (all methods)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Box plots showing variance across datasets / conditions
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplots(results, task):
    """
    Box plots for each primary metric, one box per method.
    Shows variance across datasets and param conditions.
    """
    import matplotlib.pyplot as plt

    cols   = REDUCE_COLS[:-1] if task == 'reduce' else EXPAND_COLS[:-1]  # skip time
    cols   = [c for c in cols if any(
        c in r['metrics'] for r in results if r['task'] == task
    )]
    if not cols:
        return None

    agg_raw = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r['task'] != task:
            continue
        for c in cols:
            v = r['metrics'].get(c)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                agg_raw[r['method']][c].append(float(v))

    methods = sorted(agg_raw.keys())
    ncols   = min(3, len(cols))
    nrows   = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes      = np.array(axes).flatten()

    for ci, col in enumerate(cols):
        ax     = axes[ci]
        data   = [agg_raw[m].get(col, []) for m in methods]
        data   = [d if d else [np.nan] for d in data]
        colours = [_method_colour(m) for m in methods]
        bp     = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops={'color': 'black', 'linewidth': 1.5},
                            whiskerprops={'linewidth': 0.8},
                            capprops={'linewidth': 0.8},
                            flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5})
        for patch, col_c in zip(bp['boxes'], colours):
            patch.set_facecolor(col_c)
            patch.set_alpha(0.75)

        info = METRIC_INFO.get(col, {})
        target_map = {
            'tail_preservation':      1.0,
            'discriminator_accuracy': 0.5,
            'privacy_dcr':            1.0,
        }
        if col in target_map:
            ax.axhline(target_map[col], color=TARGET_COLOUR,
                       linestyle='--', linewidth=1.0, alpha=0.7,
                       label=f'target={target_map[col]}')
            ax.legend(fontsize=6)

        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=6.5)
        ax.set_ylabel(info.get('display', col), fontsize=7)
        ax.set_title(info.get('display', col), fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.grid(True, axis='y', alpha=0.25)

    for j in range(ci + 1, len(axes)):
        axes[j].set_visible(False)

    task_label = task.capitalize()
    fig.suptitle(f'HVRT Benchmark — {task_label} Distribution across Datasets & Conditions\n'
                 f'(boxes show IQR;  whiskers = 1.5×IQR;  dots = outliers)',
                 fontsize=10, fontweight='bold')
    _make_legend(fig)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(fig, path, dpi=150):
    if fig is None:
        return
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f'  Saved: {path}')


def main():
    parser = argparse.ArgumentParser(
        description='HVRT Benchmark Report Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--input', '-i',
        default='benchmarks/results/benchmark_results.json',
        help='Path to benchmark_results.json  (default: benchmarks/results/benchmark_results.json)',
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='benchmarks/results/report',
        help='Directory for saved figures and text report  (default: benchmarks/results/report)',
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip plot generation (text report only)',
    )
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='Figure DPI  (default: 150)',
    )
    args = parser.parse_args()

    # ── Stdout UTF-8 ────────────────────────────────────────────────────────
    # Windows terminals default to cp1252; box-drawing chars need UTF-8.
    import io as _io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = _io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
        )

    # ── Load ────────────────────────────────────────────────────────────────
    if not os.path.exists(args.input):
        print(f'ERROR: Results file not found: {args.input}', file=sys.stderr)
        print('Run  python benchmarks/run_benchmarks.py  first.', file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        results = json.load(f)

    if not results:
        print('No results found in file.', file=sys.stderr)
        sys.exit(1)

    n_total  = len(results)
    n_reduce = sum(1 for r in results if r['task'] == 'reduce')
    n_expand = sum(1 for r in results if r['task'] == 'expand')
    print(f'\nLoaded {n_total} results  ({n_reduce} reduction,  {n_expand} expansion)')

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Text report ─────────────────────────────────────────────────────────
    report_path = os.path.join(args.output_dir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as fh:
        generate_text_report(results, out=fh)
    generate_text_report(results, out=sys.stdout)
    print(f'\nText report saved to: {report_path}')

    # ── Plots ────────────────────────────────────────────────────────────────
    if args.no_plots:
        print('Skipping plots (--no-plots).')
        return

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            'font.family':  'DejaVu Sans',
            'axes.spines.top':   False,
            'axes.spines.right': False,
        })
    except ImportError:
        print('\nmatplotlib not available — skipping plots.  pip install matplotlib')
        return

    print('\nGenerating figures...')

    def _try(fn, name):
        try:
            fig = fn(results)
            if fig is not None:
                save_fig(fig, os.path.join(args.output_dir, name), dpi=args.dpi)
                plt.close(fig)
        except Exception as exc:
            print(f'  WARNING: {name} failed — {exc}')

    if n_reduce:
        _try(plot_reduction_overview,  'reduction_overview.png')
        _try(plot_heatmap_reduction,   'heatmap_reduction.png')
        _try(lambda r: plot_boxplots(r, 'reduce'), 'boxplots_reduction.png')

    if n_expand:
        _try(plot_expansion_overview,  'expansion_overview.png')
        _try(plot_heatmap_expansion,   'heatmap_expansion.png')
        _try(lambda r: plot_boxplots(r, 'expand'), 'boxplots_expansion.png')

    if n_reduce or n_expand:
        _try(plot_scatter_efficiency, 'scatter_efficiency.png')
        _try(plot_radar_comparison,   'radar_comparison.png')

    print(f'\nAll figures saved to: {args.output_dir}/')


if __name__ == '__main__':
    main()
