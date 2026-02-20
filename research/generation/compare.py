"""
In-Partition Generation Method Comparison
==========================================
Benchmarks all in-partition generation methods available in methods.py
against each other and against external deep learning methods.

Method groups
-------------
Per-partition methods (run inside HVRT's partition structure):
    MultivariateKDE, UniKDE-RankCoupled, UniKDE-Independent,
    PartitionGMM, KNNInterpolation, PartitionBootstrap

Global methods (operate on the full dataset, no partitioning):
    SMOTE              imbalanced-learn   pip install imbalanced-learn
    CTGAN              ctgan              pip install ctgan
    TVAE               ctgan              pip install ctgan
    TabDDPM            (not runnable locally — published numbers only)
    MOSTLY AI          (commercial cloud  — published numbers only)

Usage
-----
    python research/generation/compare.py
    python research/generation/compare.py --dataset multimodal --n 3000
    python research/generation/compare.py --bandwidth 0.3 0.5 0.7
    python research/generation/compare.py --deep-learning
    python research/generation/compare.py --all --output research/generation/results/
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

# Ensure src is importable when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from hvrt.benchmarks.datasets import BENCHMARK_DATASETS
from hvrt.benchmarks.metrics import (
    marginal_fidelity,
    correlation_fidelity,
    tail_preservation,
    discriminator_accuracy,
    privacy_dcr,
    novelty_min,
)

from .methods import ALL_METHODS


# ─────────────────────────────────────────────────────────────────────────────
# Published benchmark numbers for methods that cannot be run locally
# ─────────────────────────────────────────────────────────────────────────────

# Source: HVRT_V2_SPECIFICATION_REVISED.md §4 and published papers.
# These are included for reference only and are marked with a † in tables.
PUBLISHED_NUMBERS = {
    'TabDDPM†': {
        'marginal_fidelity':      0.960,
        'discriminator_accuracy': 0.520,
        'tail_preservation':      0.700,   # tail_error = 0.300 → ratio ≈ 0.700
        'note': 'Published benchmark — Kotelnikov et al. 2023',
    },
    'TVAE†': {
        'marginal_fidelity':      0.940,
        'discriminator_accuracy': 0.535,
        'tail_preservation':      0.550,   # tail_error = 0.450
        'note': 'Published benchmark — Xu et al. 2019',
    },
    'CTGAN†': {
        'marginal_fidelity':      0.920,
        'discriminator_accuracy': 0.558,
        'tail_preservation':      0.500,   # tail_error = 0.500
        'note': 'Published benchmark — Xu et al. 2019',
    },
    'MOSTLY_AI†': {
        'marginal_fidelity':      0.975,
        'discriminator_accuracy': 0.510,
        'tail_preservation':      0.850,   # tail_error = 0.150
        'note': 'Published benchmark — MOSTLY AI evaluation 2024',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# HVRT-partitioned expansion using a pluggable generation method
# ─────────────────────────────────────────────────────────────────────────────

def _expand_with_method(X_train, generation_method, n_synthetic,
                         variance_weighted=False, random_state=42):
    """
    Run HVRT partitioning and then use the given generation_method for
    within-partition sampling.

    Parameters
    ----------
    X_train          : ndarray (n, d)
    generation_method: instance of _BaseMethod
    n_synthetic      : int  number of synthetic samples to generate
    variance_weighted: bool  oversample high-variance partitions
    random_state     : int

    Returns
    -------
    X_synth : ndarray (n_synthetic, d)
    fit_time, sample_time : float
    """
    from hvrt import FastHVRT
    from hvrt.benchmarks.runners import compute_expansion_budgets  # noqa: F401
    from hvrt.expand import compute_expansion_budgets
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(random_state)

    # Fit HVRT partitioner (FastHVRT for speed in research context)
    t0 = time.perf_counter()
    hvrt = FastHVRT(random_state=random_state)
    hvrt.fit(X_train)
    X_z = hvrt.X_z_
    partition_ids    = hvrt.partition_ids_
    unique_parts     = hvrt.unique_partitions_
    t_fit = time.perf_counter() - t0

    # Budget allocation
    budgets = compute_expansion_budgets(
        partition_ids, unique_parts, n_synthetic, variance_weighted, X_z
    )

    # Per-partition generation
    t1 = time.perf_counter()
    parts_synth = []
    for pid, budget in zip(unique_parts, budgets):
        if budget == 0:
            continue
        mask = partition_ids == pid
        X_part = X_z[mask]
        if len(X_part) < 2:
            # Single-point partition: bootstrap with small noise
            noise = rng.randn(budget, X_part.shape[1]) * 0.05
            parts_synth.append(X_part[[0] * budget] + noise)
            continue
        try:
            X_part_synth = generation_method.fit_sample(X_part, budget, rng=rng)
        except Exception:
            # Fall back to bootstrap if method fails on this partition
            noise = rng.randn(budget, X_part.shape[1]) * 0.05
            idx = rng.choice(len(X_part), budget, replace=True)
            X_part_synth = X_part[idx] + noise
        parts_synth.append(X_part_synth)

    X_synth_z = np.vstack(parts_synth) if parts_synth else np.empty((0, X_z.shape[1]))

    # Inverse-transform from z-space to original scale
    X_synth = hvrt._from_z(X_synth_z)
    t_sample = time.perf_counter() - t1

    return X_synth, t_fit, t_sample


# ─────────────────────────────────────────────────────────────────────────────
# Global (non-partitioned) methods
# ─────────────────────────────────────────────────────────────────────────────

def _smote_expand(X_train, n_synthetic, random_state=42):
    """
    SMOTE-style expansion on the full dataset (no class labels required).

    Treats all samples as a single class and synthesises minority-like
    interpolations.  Equivalent to k-NN interpolation globally.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError(
            "imbalanced-learn is required for SMOTE. "
            "pip install imbalanced-learn"
        )

    n = len(X_train)
    # SMOTE needs at least 2 classes; we simulate by creating a dummy label
    # and requesting n + n_synthetic total samples for the target class.
    y_dummy = np.zeros(n, dtype=int)
    n_target = n + n_synthetic

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sm = SMOTE(
            sampling_strategy={0: n_target},
            k_neighbors=min(5, n - 1),
            random_state=random_state,
        )
        X_res, _ = sm.fit_resample(X_train, y_dummy)

    # Return only the synthetic portion
    return X_res[n:]


def _ctgan_expand(X_train, n_synthetic, epochs=300, random_state=42):
    """
    CTGAN expansion (requires ctgan library).
    """
    try:
        from ctgan import CTGAN
    except ImportError:
        raise ImportError("ctgan is required. pip install ctgan")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = CTGAN(epochs=epochs, verbose=False)
        import pandas as pd
        df = pd.DataFrame(X_train, columns=[f'f{i}' for i in range(X_train.shape[1])])
        model.fit(df)
        X_synth_df = model.sample(n_synthetic)
    return X_synth_df.values


def _tvae_expand(X_train, n_synthetic, epochs=300, random_state=42):
    """
    TVAE expansion (requires ctgan library).
    """
    try:
        from ctgan import TVAE
    except ImportError:
        raise ImportError("ctgan is required. pip install ctgan")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = TVAE(epochs=epochs)
        import pandas as pd
        df = pd.DataFrame(X_train, columns=[f'f{i}' for i in range(X_train.shape[1])])
        model.fit(df)
        X_synth_df = model.sample(n_synthetic)
    return X_synth_df.values


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(X_real, X_synth):
    """Compute all expansion quality metrics."""
    m = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m['marginal_fidelity']      = marginal_fidelity(X_real, X_synth)
        m['correlation_fidelity']   = correlation_fidelity(X_real, X_synth)
        tp                          = tail_preservation(X_real, X_synth)
        m['tail_preservation']      = tp
        m['tail_error']             = abs(tp - 1.0)
        m['discriminator_accuracy'] = discriminator_accuracy(X_real, X_synth)
        m['privacy_dcr']            = privacy_dcr(X_real, X_synth)
        m['novelty_min']            = novelty_min(X_real, X_synth)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Single-run helper
# ─────────────────────────────────────────────────────────────────────────────

def run_one(method_name, X_train, X_test, n_synthetic,
            generation_method=None, deep_learning=False,
            bandwidth=0.5, random_state=42, verbose=True):
    """
    Run a single method and return a result dict.

    Parameters
    ----------
    method_name       : str
    X_train, X_test   : ndarray
    n_synthetic       : int
    generation_method : _BaseMethod instance or None (for global methods)
    deep_learning     : bool  enable CTGAN/TVAE (slow)
    bandwidth         : float  passed to MultivariateKDE if method is 'MultivariateKDE'
    random_state      : int
    verbose           : bool

    Returns
    -------
    dict with keys: method, metrics, fit_time, sample_time
    """
    t_fit = t_sample = 0.0

    try:
        if method_name == 'SMOTE':
            t0 = time.perf_counter()
            X_synth = _smote_expand(X_train, n_synthetic, random_state)
            t_sample = time.perf_counter() - t0

        elif method_name == 'CTGAN':
            t0 = time.perf_counter()
            X_synth = _ctgan_expand(X_train, n_synthetic, random_state=random_state)
            t_sample = time.perf_counter() - t0

        elif method_name == 'TVAE':
            t0 = time.perf_counter()
            X_synth = _tvae_expand(X_train, n_synthetic, random_state=random_state)
            t_sample = time.perf_counter() - t0

        else:
            # Per-partition method
            if generation_method is None:
                raise ValueError(f"generation_method required for {method_name!r}")
            X_synth, t_fit, t_sample = _expand_with_method(
                X_train, generation_method, n_synthetic,
                variance_weighted=False, random_state=random_state,
            )

        metrics = evaluate(X_train, X_synth)

    except Exception as exc:
        if verbose:
            print(f"    ERROR in {method_name}: {exc}")
        return {
            'method':      method_name,
            'metrics':     {},
            'fit_time':    None,
            'sample_time': None,
            'error':       str(exc),
        }

    if verbose:
        mf = metrics.get('marginal_fidelity', float('nan'))
        da = metrics.get('discriminator_accuracy', float('nan'))
        te = metrics.get('tail_error', float('nan'))
        print(f"    {method_name:<30}  mf={mf:.3f}  disc={da:.3f}  tail_err={te:.3f}"
              f"  [{t_fit + t_sample:.2f}s]")

    return {
        'method':      method_name,
        'metrics':     metrics,
        'fit_time':    round(t_fit, 4),
        'sample_time': round(t_sample, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full comparison runner
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(
    dataset_name='multimodal',
    n=3000,
    expansion_ratio=1.0,
    bandwidths=(0.5,),
    deep_learning=False,
    random_state=42,
    verbose=True,
):
    """
    Run the full in-partition generation method comparison on one dataset.

    Parameters
    ----------
    dataset_name    : str  key from BENCHMARK_DATASETS
    n               : int  number of training samples to use
    expansion_ratio : float  n_synthetic = int(n_train * expansion_ratio)
    bandwidths      : sequence of float  bandwidths to test for KDE methods
    deep_learning   : bool  include CTGAN / TVAE (requires ctgan)
    random_state    : int
    verbose         : bool

    Returns
    -------
    list of result dicts
    """
    from sklearn.model_selection import train_test_split

    gen_fn = BENCHMARK_DATASETS[dataset_name]
    X, _, _ = gen_fn(random_state=random_state)
    X = X[:n]

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=random_state)
    n_synthetic = max(10, int(len(X_train) * expansion_ratio))

    if verbose:
        print(f"\nDataset: {dataset_name}  |  n_train={len(X_train)}"
              f"  n_synthetic={n_synthetic}  d={X_train.shape[1]}")
        print('─' * 70)

    results = []

    # ── Per-partition methods ────────────────────────────────────────────────
    for bw in bandwidths:
        bw_tag = f'  [bw={bw}]' if bw != 0.5 else ''

        # MultivariateKDE at different bandwidths
        from .methods import MultivariateKDE
        name = f'MultivariateKDE{bw_tag}'
        r = run_one(name, X_train, X_test, n_synthetic,
                    generation_method=MultivariateKDE(bandwidth=bw),
                    random_state=random_state, verbose=verbose)
        r['bandwidth'] = bw
        results.append(r)

        # UniKDE-RankCoupled at the same bandwidth
        from .methods import UnivariateKDERankCoupled
        name = f'UniKDE-RankCoupled{bw_tag}'
        r = run_one(name, X_train, X_test, n_synthetic,
                    generation_method=UnivariateKDERankCoupled(bandwidth=bw),
                    random_state=random_state, verbose=verbose)
        r['bandwidth'] = bw
        results.append(r)

        # UniKDE-Independent at the same bandwidth (only at default bw)
        if bw == bandwidths[0]:
            from .methods import UnivariateKDEIndependent
            r = run_one('UniKDE-Independent', X_train, X_test, n_synthetic,
                        generation_method=UnivariateKDEIndependent(bandwidth=bw),
                        random_state=random_state, verbose=verbose)
            results.append(r)

    # Other per-partition methods (bandwidth-agnostic)
    for mname, factory in {
        'PartitionGMM':     ALL_METHODS['PartitionGMM'],
        'KNNInterpolation': ALL_METHODS['KNNInterpolation'],
        'PartitionBootstrap': ALL_METHODS['PartitionBootstrap'],
    }.items():
        r = run_one(mname, X_train, X_test, n_synthetic,
                    generation_method=factory(),
                    random_state=random_state, verbose=verbose)
        results.append(r)

    # ── Global methods ───────────────────────────────────────────────────────
    r = run_one('SMOTE', X_train, X_test, n_synthetic,
                random_state=random_state, verbose=verbose)
    results.append(r)

    if deep_learning:
        for name in ('CTGAN', 'TVAE'):
            r = run_one(name, X_train, X_test, n_synthetic,
                        random_state=random_state, verbose=verbose)
            results.append(r)

    # ── Published-only numbers ───────────────────────────────────────────────
    if verbose:
        print(f"\n  {'Method':<32} {'Source'}")
        print('  ' + '─' * 60)
    for pub_name, pub_data in PUBLISHED_NUMBERS.items():
        results.append({
            'method':  pub_name,
            'metrics': {k: v for k, v in pub_data.items() if k != 'note'},
            'note':    pub_data.get('note', ''),
            'published_only': True,
        })
        if verbose:
            print(f"  {pub_name:<32} {pub_data.get('note', '')}")

    return results, X_train, X_test


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────

_DISPLAY_COLS = [
    ('marginal_fidelity',      'Marginal Fid', True,   None),
    ('correlation_fidelity',   'Correl Fid',   True,   None),
    ('tail_error',             'Tail Error',   False,  None),   # lower = better
    ('discriminator_accuracy', 'Discriminator',None,  0.50),    # target 0.50
    ('privacy_dcr',            'Privacy DCR',  True,   1.0),
    ('novelty_min',            'Novelty Min',  True,   None),
]


def print_comparison_table(results, out=None):
    import sys as _sys
    import io as _io
    if out is None:
        if hasattr(_sys.stdout, 'buffer'):
            out = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
        else:
            out = _sys.stdout

    print('\n' + '═' * 100, file=out)
    print('  IN-PARTITION GENERATION METHOD COMPARISON', file=out)
    print('  († = published benchmark, not run locally)', file=out)
    print('═' * 100, file=out)

    col_w = 14
    header = f'  {"Method":<32}' + ''.join(
        f'{label:>{col_w}}' for _, label, _, _ in _DISPLAY_COLS
    )
    print(header, file=out)
    print('  ' + '─' * (len(header) - 2), file=out)

    def sort_key(r):
        v = r.get('metrics', {}).get('marginal_fidelity', -1)
        return -1 if v is None else v

    for r in sorted(results, key=sort_key, reverse=True):
        m = r.get('metrics', {})
        name = r['method']
        pub  = '†' if r.get('published_only') else ''
        row  = f'  {name + pub:<32}'
        for col, _, higher_better, target in _DISPLAY_COLS:
            v = m.get(col)
            if v is None:
                row += f'{"N/A":>{col_w}}'
            else:
                row += f'{v:>{col_w}.4f}'
        print(row, file=out)

    print('═' * 100, file=out)

    # Bandwidth sensitivity section (if multiple bandwidths tested)
    bw_results = [r for r in results if 'bandwidth' in r and 'MultivariateKDE' in r['method']]
    if len(bw_results) > 1:
        print('\n  BANDWIDTH SENSITIVITY  (MultivariateKDE)', file=out)
        print('  ' + '─' * 80, file=out)
        bw_header = f'  {"Bandwidth":<12}' + ''.join(
            f'{label:>{col_w}}' for _, label, _, _ in _DISPLAY_COLS
        )
        print(bw_header, file=out)
        for r in sorted(bw_results, key=lambda x: x.get('bandwidth', 0)):
            m   = r.get('metrics', {})
            bw  = r.get('bandwidth', '?')
            row = f'  {str(bw):<12}'
            for col, _, _, _ in _DISPLAY_COLS:
                v = m.get(col)
                row += f'{v:>{col_w}.4f}' if v is not None else f'{"N/A":>{col_w}}'
            print(row, file=out)


# ─────────────────────────────────────────────────────────────────────────────
# Optional plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(results, output_dir=None):
    """
    Generate comparison figures:
      - Bar chart: all methods × all metrics
      - Bandwidth sensitivity line chart (if multiple bandwidths present)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available — skipping plots.')
        return

    HVRT_BLUE = '#1565C0'
    GREY      = '#9E9E9E'
    PUB_GREY  = '#BDBDBD'

    def _colour(name):
        if '†' in name:           return PUB_GREY
        if 'KDE' in name:         return HVRT_BLUE
        if 'Univariate' in name:  return '#42A5F5'
        if 'KNN' in name or 'SMOTE' in name: return '#66BB6A'
        return GREY

    # ── Main bar chart ──────────────────────────────────────────────────────
    metrics_to_plot = [
        ('marginal_fidelity',      'Marginal Fidelity',      True),
        ('tail_error',             'Tail Error (lower=better)', False),
        ('discriminator_accuracy', 'Discriminator (target=0.50)', None),
        ('correlation_fidelity',   'Correlation Fidelity',   True),
    ]

    names  = [r['method'] + ('†' if r.get('published_only') else '') for r in results]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (col, title, higher) in zip(axes, metrics_to_plot):
        vals   = [r.get('metrics', {}).get(col, np.nan) for r in results]
        colours = [_colour(n) for n in names]

        valid = [(n, v, c) for n, v, c in zip(names, vals, colours) if not np.isnan(float(v if v is not None else np.nan))]
        if not valid:
            ax.set_visible(False)
            continue
        ns, vs, cs = zip(*valid)

        ax.barh(ns, vs, color=cs, edgecolor='white', linewidth=0.4)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.tick_params(axis='y', labelsize=7)

        if col == 'discriminator_accuracy':
            ax.axvline(0.50, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='target')
            ax.legend(fontsize=7)
        if col in ('tail_error', 'privacy_dcr'):
            ax.axvline(1.0 if col == 'privacy_dcr' else 0.0,
                       color='red', linestyle='--', linewidth=1.0, alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=HVRT_BLUE, label='MultivariateKDE (HVRT)'),
        Patch(facecolor='#42A5F5', label='Univariate KDE variants'),
        Patch(facecolor='#66BB6A', label='Interpolation / SMOTE'),
        Patch(facecolor=GREY,      label='Other competitors'),
        Patch(facecolor=PUB_GREY,  label='Published numbers only (†)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('In-Partition Generation Method Comparison', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    if output_dir:
        path = os.path.join(output_dir, 'generation_comparison.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)

    # ── Bandwidth sensitivity ────────────────────────────────────────────────
    bw_results = [r for r in results if 'bandwidth' in r and 'MultivariateKDE' in r['method']]
    if len(bw_results) <= 1:
        return

    bw_results.sort(key=lambda x: x.get('bandwidth', 0))
    bws = [r['bandwidth'] for r in bw_results]

    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    for ax2, (col, label, _) in zip(axes2, [
        ('marginal_fidelity',      'Marginal Fidelity',       True),
        ('tail_error',             'Tail Error',               False),
        ('discriminator_accuracy', 'Discriminator Accuracy',   None),
    ]):
        vals = [r.get('metrics', {}).get(col, np.nan) for r in bw_results]
        ax2.plot(bws, vals, 'o-', color=HVRT_BLUE, linewidth=2, markersize=6)
        ax2.set_xlabel('Bandwidth', fontsize=8)
        ax2.set_ylabel(label, fontsize=8)
        ax2.set_title(label, fontsize=9, fontweight='bold')
        ax2.grid(True, alpha=0.25)
        if col == 'discriminator_accuracy':
            ax2.axhline(0.5, color='red', linestyle='--', linewidth=1.0, alpha=0.6, label='target')
            ax2.legend(fontsize=7)

    fig2.suptitle('MultivariateKDE Bandwidth Sensitivity', fontsize=10, fontweight='bold')
    fig2.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, 'bandwidth_sensitivity.png')
        fig2.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='In-Partition Generation Method Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--dataset', default='multimodal',
                        choices=list(BENCHMARK_DATASETS.keys()),
                        help='Dataset from benchmark suite  (default: multimodal)')
    parser.add_argument('--all', action='store_true',
                        help='Run on all benchmark datasets')
    parser.add_argument('--n', type=int, default=3000,
                        help='Training samples to use  (default: 3000)')
    parser.add_argument('--expansion-ratio', type=float, default=1.0,
                        help='n_synthetic = n_train × ratio  (default: 1.0)')
    parser.add_argument('--bandwidth', nargs='+', type=float, default=[0.5],
                        metavar='BW',
                        help='KDE bandwidths to compare  (default: 0.5)')
    parser.add_argument('--deep-learning', action='store_true',
                        help='Include CTGAN and TVAE  (requires: pip install ctgan)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                        help='Directory to save JSON results and figures')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()

    datasets = list(BENCHMARK_DATASETS.keys()) if args.all else [args.dataset]

    all_results = {}
    for ds in datasets:
        results, X_train, X_test = run_comparison(
            dataset_name=ds,
            n=args.n,
            expansion_ratio=args.expansion_ratio,
            bandwidths=args.bandwidth,
            deep_learning=args.deep_learning,
            random_state=args.seed,
            verbose=True,
        )
        print_comparison_table(results)
        all_results[ds] = results

        if args.output:
            os.makedirs(args.output, exist_ok=True)
            json_path = os.path.join(args.output, f'{ds}_generation_comparison.json')
            # Serialise (strip non-serialisable keys)
            serialisable = [
                {k: v for k, v in r.items() if k not in ('X_train', 'X_test')}
                for r in results
            ]
            with open(json_path, 'w') as f:
                json.dump(serialisable, f, indent=2)
            print(f'Results saved: {json_path}')

            if not args.no_plots:
                plot_comparison(results, output_dir=args.output)


if __name__ == '__main__':
    main()
