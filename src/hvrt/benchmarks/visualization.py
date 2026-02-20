"""
Visualization utilities for HVRT benchmark results.

Provides both ASCII tables (no dependencies beyond stdlib/numpy) and
matplotlib figures (optional dependency).
"""

import io
import sys
import numpy as np


# ---------------------------------------------------------------------------
# ASCII table
# ---------------------------------------------------------------------------

def print_results_table(results, task='reduce', metric=None, dataset=None,
                         param_value=None, file=None):
    """
    Print a formatted ASCII comparison table.

    Parameters
    ----------
    results    : list of result dicts (from run_full_benchmark)
    task       : 'reduce' or 'expand'
    metric     : str or None  primary metric column (auto-selected if None)
    dataset    : str or None  filter to one dataset (all if None)
    param_value: float or None  filter by ratio / expansion_ratio
    file       : file-like or None  defaults to stdout
    """
    # Ensure stdout can render Unicode box-drawing characters on Windows
    if file is None:
        if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower().replace('-', '') not in ('utf8', 'utf-8'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    out = file or sys.stdout

    # Filter
    rows = [r for r in results if r['task'] == task]
    if dataset:
        rows = [r for r in rows if r['dataset'] == dataset]
    if param_value is not None:
        param_key = 'ratio' if task == 'reduce' else 'expansion_ratio'
        rows = [r for r in rows if r['params'].get(param_key) == param_value]

    if not rows:
        print("No results match the filter criteria.", file=out)
        return

    # Choose columns
    if task == 'reduce':
        if metric is None:
            cols = ['marginal_fidelity', 'correlation_fidelity',
                    'tail_preservation', 'ml_utility_trtr',
                    'ml_utility_retention', 'ml_delta',
                    'operation_time_seconds']
        else:
            cols = [metric, 'operation_time_seconds']
        param_key = 'ratio'
    else:
        if metric is None:
            cols = ['marginal_fidelity', 'discriminator_accuracy',
                    'tail_preservation', 'privacy_dcr',
                    'ml_utility_trtr', 'ml_utility_tstr', 'ml_delta',
                    'operation_time_seconds']
        else:
            cols = [metric, 'operation_time_seconds']
        param_key = 'expansion_ratio'

    # Group by dataset + param_value → pivot on method
    grouped = {}
    for r in rows:
        key = (r['dataset'], r['params'].get(param_key))
        grouped.setdefault(key, {})[r['method']] = r['metrics']

    col_width = 12
    task_label = task.capitalize()
    header_sep = '═' * (20 + col_width * len(cols) + 4)

    print(f"\n╔{header_sep}╗", file=out)
    print(f"║  HVRT BENCHMARK — {task_label.upper():<{len(header_sep) - 22}}║", file=out)
    print(f"╠{header_sep}╣", file=out)

    for (ds, pv), method_metrics in sorted(grouped.items()):
        label = f"Dataset: {ds}  |  {param_key}={pv}"
        print(f"║  {label:<{len(header_sep) - 3}}║", file=out)

        # Header row
        header = f"  {'Method':<18}" + ''.join(f"{c[:col_width-1]:>{col_width}}" for c in cols)
        print(f"╠{'═' * (len(header_sep))}╣", file=out)
        print(f"║{header}║", file=out)
        print(f"╠{'─' * (len(header_sep))}╣", file=out)

        for method, metrics in sorted(method_metrics.items()):
            row_str = f"  {method:<18}"
            for c in cols:
                val = metrics.get(c, float('nan'))
                if isinstance(val, float):
                    if 'time' in c:
                        row_str += f"{val:>{col_width}.3f}s"[:-1].rjust(col_width)
                    else:
                        row_str += f"{val:>{col_width}.4f}"
                else:
                    row_str += f"{'N/A':>{col_width}}"
            print(f"║{row_str}║", file=out)

        print(f"╠{'═' * (len(header_sep))}╣", file=out)

    print(f"╚{header_sep}╝", file=out)


def results_to_string(results, **kwargs):
    """Return print_results_table output as a string."""
    buf = io.StringIO()
    print_results_table(results, file=buf, **kwargs)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Matplotlib plots
# ---------------------------------------------------------------------------

def plot_comparison(results, task='reduce', metric='marginal_fidelity',
                    dataset=None, figsize=(10, 6)):
    """
    Bar chart comparing methods on a single metric.

    Parameters
    ----------
    results  : list of result dicts
    task     : 'reduce' or 'expand'
    metric   : str
    dataset  : str or None  (average across datasets if None)
    figsize  : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError("matplotlib is required for plotting. pip install matplotlib")

    rows = [r for r in results if r['task'] == task]
    if dataset:
        rows = [r for r in rows if r['dataset'] == dataset]

    # Aggregate: mean metric per method across all conditions
    from collections import defaultdict
    method_vals = defaultdict(list)
    for r in rows:
        v = r['metrics'].get(metric)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            method_vals[r['method']].append(v)

    methods = sorted(method_vals.keys())
    means = [np.mean(method_vals[m]) for m in methods]
    stds = [np.std(method_vals[m]) for m in methods]

    # Colour: HVRT family highlighted
    colours = []
    for m in methods:
        if m.startswith('HVRT') or m.startswith('FastHVRT'):
            colours.append('#2196F3')
        else:
            colours.append('#9E9E9E')

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(methods, means, yerr=stds, capsize=4,
                  color=colours, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Method')
    ax.set_ylabel(metric.replace('_', ' ').title())
    title_ds = f' ({dataset})' if dataset else ' (all datasets)'
    ax.set_title(f'{task.capitalize()} — {metric.replace("_", " ").title()}{title_ds}')
    ax.tick_params(axis='x', rotation=35)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='HVRT family'),
        Patch(facecolor='#9E9E9E', label='Competitors'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    fig.tight_layout()
    return fig


def plot_unified_summary(results, figsize=(16, 10)):
    """
    Two-panel figure: left = reduction metrics, right = expansion metrics.
    Each panel shows marginal fidelity and the task-specific key metric.

    Parameters
    ----------
    results : list of result dicts
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        raise ImportError("matplotlib is required for plotting. pip install matplotlib")

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Top-left: reduction marginal fidelity
    ax1 = fig.add_subplot(gs[0, 0])
    _fill_ax(ax1, results, 'reduce', 'marginal_fidelity', 'Reduction — Marginal Fidelity')

    # Top-right: reduction ml_utility_retention
    ax2 = fig.add_subplot(gs[0, 1])
    _fill_ax(ax2, results, 'reduce', 'ml_utility_retention', 'Reduction — ML Utility Retention')

    # Bottom-left: expansion marginal fidelity
    ax3 = fig.add_subplot(gs[1, 0])
    _fill_ax(ax3, results, 'expand', 'marginal_fidelity', 'Expansion — Marginal Fidelity')

    # Bottom-right: expansion discriminator accuracy
    ax4 = fig.add_subplot(gs[1, 1])
    _fill_ax(ax4, results, 'expand', 'discriminator_accuracy',
             'Expansion — Discriminator Accuracy\n(target ≈ 50%)')

    fig.suptitle('HVRT v2 Benchmark Summary', fontsize=14, fontweight='bold', y=1.01)
    return fig


def _fill_ax(ax, results, task, metric, title):
    """Helper to fill a single subplot with a horizontal bar chart."""
    from collections import defaultdict

    rows = [r for r in results if r['task'] == task]
    method_vals = defaultdict(list)
    for r in rows:
        v = r['metrics'].get(metric)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            method_vals[r['method']].append(v)

    if not method_vals:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return

    methods = sorted(method_vals.keys())
    means = [np.mean(method_vals[m]) for m in methods]
    colours = [
        '#2196F3' if (m.startswith('HVRT') or m.startswith('FastHVRT')) else '#9E9E9E'
        for m in methods
    ]

    y_pos = np.arange(len(methods))
    ax.barh(y_pos, means, color=colours, edgecolor='white', linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel(metric.replace('_', ' '), fontsize=8)

    if metric == 'discriminator_accuracy':
        ax.axvline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.6, label='target')
        ax.legend(fontsize=7)
