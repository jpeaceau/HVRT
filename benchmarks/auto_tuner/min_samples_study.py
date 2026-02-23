#!/usr/bin/env python3
"""
Auto-Tuner Min-Samples-Leaf Study for KDE Generation
=====================================================

Systematically investigates the appropriate ``min_samples_leaf`` formula
for KDE-based expansion / augmentation in HVRT.

Background
----------
v2.1.1 changed the expansion auto-tuner from the old 40:1 sample-to-feature
ratio (shared with reduction) to a dataset-size-driven formula:

    min_samples_leaf = max(5, int(0.75 * n_samples ** (2 / 3)))

This formula is *feature-agnostic*.  Multivariate KDE using
``scipy.stats.gaussian_kde`` on an (n_part × d) partition matrix requires
a non-singular covariance, which is guaranteed only when n_part > d.
When min_samples_leaf < d + 1 the tree can produce partitions where KDE
silently degrades to bootstrap-noise fallback, losing all structural fidelity.

This benchmark sweeps small-scale datasets (n = 50–500, d = 2–20) across
seven candidate formulas and measures:

  kde_failure_rate  Fraction of partitions where KDE collapsed to None
                    (< 2 samples OR singular covariance → LinAlgError).
  risk_rate         Fraction of partitions with n_part < d + 1
                    (numerically at-risk even if scipy didn't raise).
  n_partitions      Number of leaf partitions actually created by the tree.
  wasserstein_mean  Mean 1-D Wasserstein distance (generated vs real),
                    averaged over all features.  Lower = more faithful.

Candidate formulas
------------------
  current    max(5, 0.75 * n^(2/3))                  v2.1.1 default
  feat_floor max(d+2, 0.75 * n^(2/3))                feature-aware floor
  feat_2x    max(5, 2*d)                              2× feature ratio
  feat_3x    max(5, 3*d)                              3× feature ratio
  sqrt_n     max(d+2, sqrt(n))                        sqrt(n) + feature floor
  hybrid_a   max(2*d, 0.5 * n^(2/3))                 balanced: size + features
  hybrid_b   max(d+2, 0.6 * n^(2/3))                 slight compromise

Usage
-----
    python benchmarks/auto_tuner/min_samples_study.py
    python benchmarks/auto_tuner/min_samples_study.py --seeds 10
    python benchmarks/auto_tuner/min_samples_study.py --plot
    python benchmarks/auto_tuner/min_samples_study.py --plot --seeds 10
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import warnings

import numpy as np
from scipy.stats import wasserstein_distance

# Allow running from repo root or directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'src'))

# Windows UTF-8 fix for box-drawing chars.
if hasattr(sys.stdout, 'buffer') and \
        getattr(sys.stdout, 'encoding', '').lower().replace('-', '') not in ('utf8', 'utf-8'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )

from hvrt import HVRT

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Candidate formulas
# ---------------------------------------------------------------------------

FORMULAS = {
    'current':    lambda n, d: max(5, int(0.75 * n ** (2 / 3))),
    'feat_floor': lambda n, d: max(d + 2, int(0.75 * n ** (2 / 3))),
    'feat_2x':    lambda n, d: max(5, 2 * d),
    'feat_3x':    lambda n, d: max(5, 3 * d),
    'sqrt_n':     lambda n, d: max(d + 2, int(n ** 0.5)),
    'hybrid_a':   lambda n, d: max(2 * d, int(0.5 * n ** (2 / 3))),
    'hybrid_b':   lambda n, d: max(d + 2, int(0.6 * n ** (2 / 3))),
}

# Test grid — (n_samples, n_features).
# Deliberately includes high-d / small-n stress cases where the current
# feature-agnostic formula is expected to produce at-risk partitions.
N_VALUES = [50, 100, 200, 500]
D_VALUES = [2, 5, 10, 15, 20]

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def make_dataset(n: int, d: int, seed: int) -> np.ndarray:
    """
    Gaussian mixture with ``d`` features and 3 modes.

    Includes mild off-diagonal covariance (features 0 and 1 correlated at 0.3)
    to make the partitioning and KDE non-trivial.
    """
    rng = np.random.RandomState(seed)
    n_modes = 3
    base_sz = n // n_modes
    extra = n - base_sz * n_modes

    centers = rng.randn(n_modes, d) * 2.5

    # Mildly correlated covariance
    cov = np.eye(d) * 0.5
    if d >= 2:
        cov[0, 1] = cov[1, 0] = 0.3

    parts = []
    for i, c in enumerate(centers):
        sz = base_sz + (1 if i < extra else 0)
        parts.append(rng.multivariate_normal(c, cov, sz))
    return np.vstack(parts)


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

def evaluate(formula_name: str, formula_fn, n: int, d: int, seed: int) -> dict:
    """
    Fit HVRT with ``min_samples_leaf`` from ``formula_fn``, expand 5×, and
    return a dict of diagnostic metrics.
    """
    X = make_dataset(n, d, seed)
    n_synth = min(n * 5, 2000)

    min_sl = formula_fn(n, d)

    try:
        model = HVRT(min_samples_leaf=min_sl, random_state=seed)
        model.fit(X)
        X_synth = model.expand(n=n_synth)

        # KDEs are populated lazily during expand(); inspect them now.
        kdes = model._kdes_                                      # dict[pid -> kde | None]
        n_parts = len(model.unique_partitions_)
        n_failed = sum(1 for v in kdes.values() if v is None)   # None = KDE collapsed

        # Count "at-risk" partitions (n_part < d+1 → near-singular covariance).
        part_sizes = np.array([
            int(np.sum(model.partition_ids_ == pid))
            for pid in model.unique_partitions_
        ])
        n_risky = int(np.sum(part_sizes < d + 1))

        # 1-D Wasserstein per feature (original vs generated).
        w_dists = [wasserstein_distance(X[:, j], X_synth[:, j]) for j in range(d)]

        return {
            'formula':    formula_name,
            'n':          n,
            'd':          d,
            'seed':       seed,
            'min_sl':     min_sl,
            'n_parts':    n_parts,
            'fail_rate':  n_failed / max(1, n_parts),
            'risk_rate':  n_risky  / max(1, n_parts),
            'wass_mean':  float(np.mean(w_dists)),
            'wass_std':   float(np.std(w_dists)),
            'ok':         True,
        }

    except Exception as exc:
        return {
            'formula':    formula_name,
            'n':          n,
            'd':          d,
            'seed':       seed,
            'min_sl':     min_sl,
            'n_parts':    0,
            'fail_rate':  1.0,
            'risk_rate':  1.0,
            'wass_mean':  float('nan'),
            'wass_std':   float('nan'),
            'ok':         False,
            'error':      str(exc),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agg(rows: list[dict], key: str) -> tuple[float, float]:
    """Return (mean, std) of ``key`` from a list of result dicts."""
    vals = [r[key] for r in rows if r['ok'] and not np.isnan(r[key])]
    if not vals:
        return float('nan'), float('nan')
    return float(np.mean(vals)), float(np.std(vals))


def _fmt(mean: float, std: float, pct: bool = False) -> str:
    if np.isnan(mean):
        return '      N/A'
    if pct:
        return f'{mean * 100:5.1f}% ±{std * 100:.1f}%'
    return f'{mean:.4f} ±{std:.4f}'


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(all_results: list[dict], formula_names: list[str]) -> None:
    """Print a per-formula summary table averaged across all (n, d, seed) runs."""
    W = 96
    print()
    print(f'╔{"═" * W}╗')
    title = 'MIN-SAMPLES-LEAF FORMULA STUDY  —  Summary (all n × d × seed)'
    print(f'║  {title:<{W - 3}}║')
    print(f'╠{"═" * W}╣')
    hdr = (
        f'  {"Formula":<14}'
        f'{"Avg min_sl":>11}'
        f'{"KDE fail %":>14}'
        f'{"At-risk %":>13}'
        f'{"N partitions":>14}'
        f'{"Wasserstein":>16}'
        f'{"Crashes":>9}'
    )
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    for name in formula_names:
        rows = [r for r in all_results if r['formula'] == name]
        crashes = sum(1 for r in rows if not r['ok'])

        sl_m, sl_s   = _agg(rows, 'min_sl')
        fr_m, fr_s   = _agg(rows, 'fail_rate')
        rr_m, rr_s   = _agg(rows, 'risk_rate')
        np_m, np_s   = _agg(rows, 'n_parts')
        wm_m, wm_s   = _agg(rows, 'wass_mean')

        row = (
            f'  {name:<14}'
            f'{sl_m:>9.1f}  '
            f'{fr_m * 100:>9.1f}%    '
            f'{rr_m * 100:>8.1f}%    '
            f'{np_m:>9.1f}    '
            f'{wm_m:>11.4f}    '
            f'{crashes:>5}'
        )
        print(f'║{row}║')

    print(f'╚{"═" * W}╝')


def print_failure_grid(all_results: list[dict], formula_names: list[str]) -> None:
    """
    Print per-formula heatmap grids (n × d) of KDE failure rate.
    Values are averaged over seeds.
    """
    print()
    for name in formula_names:
        rows = [r for r in all_results if r['formula'] == name]
        print(f'  KDE failure rate %  [{name}]')
        # Header
        d_header = '      ' + ''.join(f'  d={d:<4}' for d in D_VALUES)
        print(d_header)
        for n in N_VALUES:
            row_parts = [f'  n={n:<4}']
            for d in D_VALUES:
                cell_rows = [r for r in rows if r['n'] == n and r['d'] == d]
                if cell_rows:
                    fr = np.mean([r['fail_rate'] for r in cell_rows]) * 100
                    cell = f'  {fr:5.1f}%'
                else:
                    cell = '    N/A'
                row_parts.append(cell)
            print(''.join(row_parts))
        print()


def print_risk_grid(all_results: list[dict], formula_names: list[str]) -> None:
    """Per-formula heatmap of at-risk (n_part < d+1) partition rate."""
    print()
    for name in formula_names:
        rows = [r for r in all_results if r['formula'] == name]
        print(f'  At-risk partition %  [{name}]  (n_part < d+1 → near-singular KDE)')
        d_header = '      ' + ''.join(f'  d={d:<4}' for d in D_VALUES)
        print(d_header)
        for n in N_VALUES:
            row_parts = [f'  n={n:<4}']
            for d in D_VALUES:
                cell_rows = [r for r in rows if r['n'] == n and r['d'] == d]
                if cell_rows:
                    rr = np.mean([r['risk_rate'] for r in cell_rows]) * 100
                    marker = ' !' if rr > 0 else '  '
                    cell = f'  {rr:5.1f}%{marker}'
                else:
                    cell = '    N/A  '
                row_parts.append(cell)
            print(''.join(row_parts))
        print()


def print_wasserstein_grid(all_results: list[dict], formula_names: list[str]) -> None:
    """Per-formula heatmap of mean Wasserstein distance."""
    print()
    for name in formula_names:
        rows = [r for r in all_results if r['formula'] == name]
        print(f'  Mean Wasserstein  [{name}]  (lower = more faithful generation)')
        d_header = '      ' + ''.join(f'  d={d:<5}' for d in D_VALUES)
        print(d_header)
        for n in N_VALUES:
            row_parts = [f'  n={n:<4}']
            for d in D_VALUES:
                cell_rows = [r for r in rows if r['n'] == n and r['d'] == d and r['ok']]
                if cell_rows:
                    wm = np.nanmean([r['wass_mean'] for r in cell_rows])
                    cell = f'  {wm:6.3f} '
                else:
                    cell = '    N/A  '
                row_parts.append(cell)
            print(''.join(row_parts))
        print()


def print_formula_values(formula_names: list[str]) -> None:
    """Print the computed min_sl value for each (n, d, formula) combination."""
    print()
    print('  Computed min_samples_leaf values per (n, d, formula)')
    print()
    W_col = 10
    header = f'  {"n":>5}  {"d":>3}  ' + ''.join(f'{fn:>{W_col}}' for fn in formula_names)
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for n in N_VALUES:
        for d in D_VALUES:
            vals = '  '.join(
                f'{FORMULAS[fn](n, d):>{W_col - 2}}'
                for fn in formula_names
            )
            flag = '  *' if n < d * 2 else '   '
            print(f'  {n:>5}  {d:>3}  {vals}{flag}')
    print()
    print('  * = n < 2d  (small-n / high-d stress zone)')


def print_recommendation(all_results: list[dict], formula_names: list[str]) -> None:
    """Score formulas and print a ranked recommendation."""
    print()
    W = 80
    print(f'╔{"═" * W}╗')
    print(f'║  {"FORMULA RANKING & RECOMMENDATION":<{W - 3}}║')
    print(f'╠{"═" * W}╣')

    scores = {}
    for name in formula_names:
        rows  = [r for r in all_results if r['formula'] == name and r['ok']]
        fr_m  = np.mean([r['fail_rate'] for r in rows]) if rows else 1.0
        rr_m  = np.mean([r['risk_rate'] for r in rows]) if rows else 1.0
        wm_m  = np.nanmean([r['wass_mean'] for r in rows]) if rows else float('nan')
        np_m  = np.mean([r['n_parts'] for r in rows]) if rows else 0.0

        # Composite score: penalise KDE failures heavily, then risk, then Wasserstein.
        # Lower score = better.
        penalty = (fr_m * 100.0) + (rr_m * 20.0) + (wm_m if not np.isnan(wm_m) else 10.0)
        scores[name] = (penalty, fr_m, rr_m, wm_m, np_m)

    ranked = sorted(scores.items(), key=lambda x: x[1][0])

    hdr = (
        f'  {"Rank":<5}{"Formula":<14}'
        f'{"KDE fail":>10}'
        f'{"At-risk":>10}'
        f'{"Wasserstein":>14}'
        f'{"Avg parts":>11}'
        f'{"Score":>9}'
    )
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    for rank, (name, (score, fr, rr, wm, np_)) in enumerate(ranked, 1):
        row = (
            f'  {rank:<5}{name:<14}'
            f'{fr * 100:>9.1f}%'
            f'{rr * 100:>9.1f}%'
            f'{wm:>13.4f}'
            f'{np_:>10.1f}'
            f'{score:>9.2f}'
        )
        print(f'║{row}║')

    print(f'╠{"─" * W}╣')

    winner = ranked[0][0]
    fn     = FORMULAS[winner]
    example_values = ', '.join(
        f'n={n}/d={d}→{fn(n, d)}'
        for n, d in [(50, 10), (100, 15), (200, 20), (500, 10)]
    )
    rec_lines = [
        f'  Recommended formula:  {winner}',
        f'  Expression:           max(d+2, int(0.75 * n^(2/3)))' if winner == 'feat_floor' else
        f'  Expression:           see FORMULAS dict for definition',
        f'  Example values:       {example_values}',
        '',
        f'  Current formula (v2.1.1):  max(5, int(0.75 * n^(2/3)))',
        f'  Change needed in:     src/hvrt/_partitioning.py :: auto_tune_tree_params()',
        f'    is_reduction=False branch → replace floor constant 5 with (d + 2)',
    ]
    for line in rec_lines:
        print(f'║  {line:<{W - 3}}║')

    print(f'╚{"═" * W}╝')


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_results: list[dict], formula_names: list[str], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print('\nmatplotlib not available — skipping plot.  pip install matplotlib')
        return

    os.makedirs(out_dir, exist_ok=True)

    # --- Fig 1: KDE failure rate heatmaps (one subplot per formula) ----------
    n_formulas = len(formula_names)
    n_rows = (n_formulas + 2) // 3
    n_cols = min(3, n_formulas)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    fig.suptitle('KDE Failure Rate by (n, d)  per Formula\n'
                 '(fraction of partitions where KDE collapsed)',
                 fontsize=13, fontweight='bold')

    vmin, vmax = 0.0, 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = 'RdYlGn_r'

    for idx, name in enumerate(formula_names):
        ax = axes[idx // n_cols][idx % n_cols]
        rows = [r for r in all_results if r['formula'] == name]

        mat = np.full((len(N_VALUES), len(D_VALUES)), np.nan)
        for i, n in enumerate(N_VALUES):
            for j, d in enumerate(D_VALUES):
                cell = [r['fail_rate'] for r in rows if r['n'] == n and r['d'] == d and r['ok']]
                if cell:
                    mat[i, j] = np.mean(cell)

        im = ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm,
                       interpolation='nearest')
        ax.set_xticks(range(len(D_VALUES)))
        ax.set_xticklabels([f'd={d}' for d in D_VALUES], fontsize=8)
        ax.set_yticks(range(len(N_VALUES)))
        ax.set_yticklabels([f'n={n}' for n in N_VALUES], fontsize=8)
        ax.set_title(name, fontsize=10, fontweight='bold')

        for i in range(len(N_VALUES)):
            for j in range(len(D_VALUES)):
                v = mat[i, j]
                if not np.isnan(v):
                    txt = f'{v * 100:.0f}%'
                    color = 'white' if v > 0.5 else 'black'
                    ax.text(j, i, txt, ha='center', va='center',
                            fontsize=7, color=color, fontweight='bold')

        plt.colorbar(im, ax=ax, format='%.0%%', fraction=0.046, pad=0.04)

    # Hide unused subplots
    for idx in range(len(formula_names), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.tight_layout()
    path1 = os.path.join(out_dir, 'kde_failure_heatmaps.png')
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {path1}')

    # --- Fig 2: Wasserstein comparison bar chart (per formula, per d) --------
    fig2, axes2 = plt.subplots(1, len(D_VALUES), figsize=(4 * len(D_VALUES), 5),
                               sharey=False)
    fig2.suptitle('Mean Wasserstein Distance by d-value\n'
                  '(lower = more faithful 1-D marginals)',
                  fontsize=13, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 0.7, len(formula_names)))

    for j, d in enumerate(D_VALUES):
        ax2 = axes2[j]
        means, stds = [], []
        for name in formula_names:
            rows = [r for r in all_results if r['formula'] == name and r['d'] == d and r['ok']]
            vals = [r['wass_mean'] for r in rows if not np.isnan(r['wass_mean'])]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0.0)

        x = np.arange(len(formula_names))
        ax2.bar(x, means, yerr=stds, color=colors, edgecolor='white',
                linewidth=0.5, width=0.7, capsize=4)
        ax2.set_title(f'd = {d}', fontsize=10, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(formula_names, rotation=40, ha='right', fontsize=7)
        ax2.set_ylabel('Wasserstein', fontsize=8)
        ax2.tick_params(labelsize=7)

    plt.tight_layout()
    path2 = os.path.join(out_dir, 'wasserstein_by_d.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'  Saved → {path2}')

    # --- Fig 3: min_sl values (reference table as figure) --------------------
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.axis('off')
    col_labels = [f'd={d}' for d in D_VALUES]
    row_labels  = [f'n={n}' for n in N_VALUES]

    for fi, name in enumerate(formula_names):
        fn = FORMULAS[name]
        table_data = [
            [str(fn(n, d)) for d in D_VALUES]
            for n in N_VALUES
        ]
        offset_x = fi * 0.14
        for ri, row in enumerate(table_data):
            for ci, val in enumerate(row):
                ax3.text(
                    offset_x + ci * 0.025,
                    1.0 - ri * 0.15 - fi * 0.01,
                    val, fontsize=5, transform=ax3.transAxes,
                )

    fig3.suptitle('Computed min_samples_leaf per Formula × (n, d)', fontsize=11)
    path3 = os.path.join(out_dir, 'formula_values.png')
    fig3.savefig(path3, dpi=120, bbox_inches='tight')
    plt.close(fig3)
    print(f'  Saved → {path3}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='HVRT auto-tuner min_samples_leaf study for KDE generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--seeds', type=int, default=5,
                        help='Random seeds per (n, d, formula) cell (default: 5)')
    parser.add_argument('--plot', action='store_true',
                        help='Save diagnostic plots to benchmarks/auto_tuner/results/')
    args = parser.parse_args()

    formula_names = list(FORMULAS.keys())
    n_total = len(formula_names) * len(N_VALUES) * len(D_VALUES) * args.seeds

    print()
    print('HVRT Auto-Tuner — min_samples_leaf Study for KDE Generation')
    print(f'  Formulas  : {len(formula_names)}  ({", ".join(formula_names)})')
    print(f'  n values  : {N_VALUES}')
    print(f'  d values  : {D_VALUES}')
    print(f'  Seeds     : {args.seeds}')
    print(f'  Total runs: {n_total}')
    print()

    all_results: list[dict] = []

    for name in formula_names:
        fn = FORMULAS[name]
        print(f'  [{name}] ', end='', flush=True)
        for n in N_VALUES:
            for d in D_VALUES:
                for seed in range(args.seeds):
                    result = evaluate(name, fn, n, d, seed)
                    all_results.append(result)
                print('.', end='', flush=True)
        print(' done')

    # ── Print formula values reference ────────────────────────────────────
    print_formula_values(formula_names)

    # ── Summary table ──────────────────────────────────────────────────────
    print_summary_table(all_results, formula_names)

    # ── Per-formula grids ──────────────────────────────────────────────────
    print('\n' + '─' * 60)
    print('KDE FAILURE RATE GRIDS  (averaged over seeds)')
    print('─' * 60)
    print_failure_grid(all_results, formula_names)

    print('─' * 60)
    print('AT-RISK PARTITION GRIDS  (n_part < d+1, averaged over seeds)')
    print('─' * 60)
    print_risk_grid(all_results, formula_names)

    print('─' * 60)
    print('WASSERSTEIN GRIDS  (lower = more faithful)')
    print('─' * 60)
    print_wasserstein_grid(all_results, formula_names)

    # ── Recommendation ─────────────────────────────────────────────────────
    print_recommendation(all_results, formula_names)

    if args.plot:
        out_dir = os.path.join(_HERE, 'results')
        print(f'\nSaving plots to {out_dir}')
        plot_results(all_results, formula_names, out_dir)

    print()


if __name__ == '__main__':
    main()
