#!/usr/bin/env python3
"""
Targeted Benchmark: Bootstrap-Noise Failure Modes vs HVRT
==========================================================

Bootstrap-Noise (resample + Gaussian noise) is a strong baseline on the main
benchmark because the test regime happens to suit it: small low-dimensional
datasets, moderate expansion ratios, and metrics that reward staying near the
real data distribution.  This file probes five specific conditions where that
approach fundamentally degenerates, and checks whether HVRT is a sustained
strong performer in those regimes.

Failure modes tested
--------------------
1. Privacy (DCR)            Synthetic samples sit on top of real data.
                            Bootstrap-Noise DCR ≈ noise_level × σ — tiny.
                            High DCR = harder for an adversary to recover
                            individual real records from the synthetic set.

2. Correlation Decay        Independent per-feature noise destroys joint
                            structure.  A multivariate KDE (HVRT) draws from
                            the joint density and therefore preserves it.

3. Boundary Escape          Bootstrap-Noise is bounded by the observed data
                            range + noise.  When training data has been
                            clipped (e.g. sensor saturation, capping), the
                            generator must be able to recover unseen regions.
                            HVRT's KDE naturally extends beyond the boundary.

4. Diversity at Scale       At high expansion ratios (20×), Bootstrap-Noise
                            generates ~20 near-identical copies of each
                            original point.  Effective sample size ≈ n_orig.
                            HVRT's KDE draws genuinely distinct samples.

5. Manifold Gap Coverage    When real data lies on a low-dimensional manifold
                            (e.g. ring, seasonal cycle) with sparse coverage,
                            Bootstrap-Noise leaves gaps between real points.
                            HVRT's within-partition KDE smoothly fills them.

Usage
-----
    python benchmarks/bootstrap_failure_benchmark.py
    python benchmarks/bootstrap_failure_benchmark.py --n-seeds 10
    python benchmarks/bootstrap_failure_benchmark.py --plot
"""

import argparse
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Reconfigure stdout to UTF-8 on Windows (cp1252 cannot encode box-drawing
# characters or Greek letters used in the output tables).
if hasattr(sys.stdout, 'buffer') and \
        getattr(sys.stdout, 'encoding', '').lower().replace('-', '') not in ('utf8', 'utf8'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )
from hvrt import HVRT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_NOISE_LEVEL = 0.1          # Bootstrap-Noise per-feature noise fraction


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _bootstrap(X, n, rs):
    rng = np.random.RandomState(rs)
    stds = np.where(X.std(0) > 1e-10, X.std(0), 1.0)
    idx = rng.choice(len(X), n, replace=True)
    S = X[idx].copy()
    S += rng.normal(0, _NOISE_LEVEL, S.shape) * stds
    return S


def _hvrt(X, n, rs):
    model = HVRT(random_state=rs)
    model.fit(X)
    return model.expand(n=n)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _dcr(X_synth, X_real, chunk=200):
    """
    Mean Distance to Closest Record (z-score space).  Higher = better privacy:
    synthetic samples are harder to link back to individual real records.
    """
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_real)
    Zr = sc.transform(X_real)
    Zs = sc.transform(X_synth)
    mins = []
    for i in range(0, len(Zs), chunk):
        D = np.linalg.norm(Zs[i:i + chunk, None] - Zr[None, :], axis=2)
        mins.append(D.min(axis=1))
    return float(np.concatenate(mins).mean())


def _corr_frobenius(X_synth, C_target):
    """
    Frobenius norm between synthetic correlation matrix and target.
    Lower = better preservation of inter-feature correlation structure.
    """
    C = np.corrcoef(X_synth.T)
    return float(np.linalg.norm(C - C_target, 'fro'))


def _boundary_escape(X_synth, lo, hi):
    """
    Fraction of synthetic samples that fall outside [lo, hi] in any dimension.
    Higher = better: the generator can recover content beyond training bounds.
    The true N(0,1) baseline for ±1.5σ clipping is ~13.4 %.
    """
    return float(np.any((X_synth < lo) | (X_synth > hi), axis=1).mean())


def _mean_nn(X_synth, rs, n_sub=500):
    """
    Mean nearest-neighbour distance within the synthetic set (z-score space).
    Higher = more diverse; lower = repetitive noisy copies of real points.
    """
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(rs)
    Z = StandardScaler().fit_transform(X_synth)
    idx = rng.choice(len(Z), min(n_sub, len(Z)), replace=False)
    Z = Z[idx]
    D = np.linalg.norm(Z[:, None] - Z[None, :], axis=2)
    np.fill_diagonal(D, np.inf)
    return float(D.min(axis=1).mean())


def _arc_gap(X_synth, n_bins=36):
    """
    Maximum consecutive empty angular bin in a 2-D synthetic dataset.
    Lower = better manifold coverage; 0 = ring fully covered.
    Each bin spans 360 / n_bins degrees.
    """
    angles = np.arctan2(X_synth[:, 1], X_synth[:, 0])
    counts, _ = np.histogram(angles, bins=np.linspace(-np.pi, np.pi, n_bins + 1))
    # Circular scan — double the sequence to handle wraparound
    max_gap = cur = 0
    for c in list(counts) * 2:
        cur = cur + 1 if c == 0 else 0
        max_gap = max(max_gap, cur)
    return min(max_gap, n_bins)   # cap at full circle


# ---------------------------------------------------------------------------
# Per-seed test functions
# Each returns (bootstrap_value, hvrt_value).
# ---------------------------------------------------------------------------

def _test_privacy(seed):
    """n=200, d=8, expand 5× → measure mean DCR."""
    rng = np.random.RandomState(seed)
    X = rng.randn(200, 8)
    n = 1000
    return (
        _dcr(_bootstrap(X, n, seed), X),
        _dcr(_hvrt(X, n, seed), X),
    )


def _test_nonlinear_structure(seed):
    """
    Non-linear functional relationship: X1 ~ Uniform(-pi, pi), X2 = sin(X1) + noise.
    n=150, expand 5×.  Metric: mean |X2_synth - sin(X1_synth)| — the residual
    deviation from the true function.  Lower = better structural fidelity.

    Bootstrap-Noise adds INDEPENDENT noise to X1 and X2, so the synthetic
    (X1, X2) pairs no longer follow sin(X1) = X2.  HVRT's joint KDE fits the
    local joint density within each partition (a short arc of the sine curve)
    and therefore keeps X1 and X2 structurally coupled.
    """
    rng = np.random.RandomState(seed)
    n_train = 150
    X1 = rng.uniform(-np.pi, np.pi, n_train)
    X2 = np.sin(X1) + rng.randn(n_train) * 0.05
    X = np.column_stack([X1, X2])
    n = 750
    Xb = _bootstrap(X, n, seed)
    Xh = _hvrt(X, n, seed)
    resid_bn   = float(np.abs(Xb[:, 1] - np.sin(Xb[:, 0])).mean())
    resid_hvrt = float(np.abs(Xh[:, 1] - np.sin(Xh[:, 0])).mean())
    return resid_bn, resid_hvrt


def _test_boundary(seed):
    """
    n=200, d=5 drawn from N(0,1) then clipped to ±1.5σ.
    Measure fraction of synthetic samples that escape the training boundary.
    True N(0,1) baseline: ~13.4 % of mass is beyond ±1.5σ.
    """
    rng = np.random.RandomState(seed)
    X = np.clip(rng.randn(200, 5), -1.5, 1.5)
    lo, hi = np.full(5, -1.5), np.full(5, 1.5)
    n = 1000
    return (
        _boundary_escape(_bootstrap(X, n, seed), lo, hi),
        _boundary_escape(_hvrt(X, n, seed), lo, hi),
    )


def _test_diversity(seed):
    """
    n=100, d=5, expand 20× (2 000 synthetic).
    Metric: mean nearest-neighbour distance in synthetic set (z-score space).
    Bootstrap-Noise produces ~20 noisy copies per real point → dense clusters.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(100, 5)
    n = 2000
    return (
        _mean_nn(_bootstrap(X, n, seed), seed),
        _mean_nn(_hvrt(X, n, seed), seed),
    )


def _test_manifold(seed):
    """
    n=20 points drawn uniformly on a unit circle in 2-D, radius noise σ=0.05.
    Expand 10× (200 synthetic).  Metric: max empty angular bin (36 bins × 10°).
    Bootstrap-Noise noise ≈ ±4° tangential → 18° average spacing leaves gaps.
    HVRT KDE smooths within each partition and fills the gaps.
    """
    rng = np.random.RandomState(seed)
    n_train = 20
    theta = rng.uniform(0, 2 * np.pi, n_train)
    X = np.column_stack([
        np.cos(theta) + rng.randn(n_train) * 0.05,
        np.sin(theta) + rng.randn(n_train) * 0.05,
    ])
    n_synth = 200
    return (
        _arc_gap(_bootstrap(X, n_synth, seed)),
        _arc_gap(_hvrt(X, n_synth, seed)),
    )


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

TESTS = [
    #  (display_name,                    direction_note,                 fn,                      higher_is_better)
    ('1. Privacy (DCR)',                  '↑ higher = safer',             _test_privacy,            True),
    ('2. Non-linear Structure Fidelity', '↓ lower residual = better',    _test_nonlinear_structure, False),
    ('3. Boundary Escape',               '↑ higher = better coverage',   _test_boundary,            True),
    ('4. Diversity at 20× Expansion',    '↑ higher = more diverse',      _test_diversity,           True),
    ('5. Manifold Gap Coverage',         '↓ fewer empty 10° bins',       _test_manifold,            False),
]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(rows):
    """
    rows: list of (name, direction, bn_vals, hv_vals, higher_is_better)
    """
    W = 88
    print()
    print(f'╔{"═" * W}╗')
    title = 'BOOTSTRAP-NOISE FAILURE MODE BENCHMARK   (Bootstrap-Noise vs HVRT)'
    print(f'║  {title:<{W - 3}}║')
    print(f'╠{"═" * W}╣')
    hdr = (f'  {"Test / direction":<36}'
           f'{"Bootstrap-Noise":>20}'
           f'{"HVRT":>16}'
           f'{"Δ (HVRT − BN)":>12}'
           f'{"Winner":>8}')
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    hvrt_wins = 0
    for name, direction, bn_vals, hv_vals, higher_better in rows:
        bn_m, bn_s = np.mean(bn_vals), np.std(bn_vals)
        hv_m, hv_s = np.mean(hv_vals), np.std(hv_vals)
        delta = hv_m - bn_m
        if higher_better:
            winner = 'HVRT' if hv_m > bn_m else 'BN'
        else:
            winner = 'HVRT' if hv_m < bn_m else 'BN'
        if winner == 'HVRT':
            hvrt_wins += 1

        bn_str = f'{bn_m:.4f} ±{bn_s:.3f}'
        hv_str = f'{hv_m:.4f} ±{hv_s:.3f}'
        d_str  = f'{delta:+.4f}'
        row = (f'  {name:<36}'
               f'{bn_str:>20}'
               f'{hv_str:>16}'
               f'{d_str:>12}'
               f'{winner:>8}')
        print(f'║{row}║')
        dir_row = f'  {direction:<36}'
        print(f'║{dir_row}║')
        print(f'╠{"─" * W}╣')

    score_line = f'  HVRT wins: {hvrt_wins} / {len(rows)}'
    print(f'║{score_line:<{W}}║')
    print(f'╚{"═" * W}╝')


def _plot(rows):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print('\nmatplotlib not available — skipping plot.  pip install matplotlib')
        return

    n = len(rows)
    fig = plt.figure(figsize=(3.5 * n, 5))
    gs  = gridspec.GridSpec(1, n, figure=fig, wspace=0.4)

    for col, (name, direction, bn_vals, hv_vals, higher_better) in enumerate(rows):
        ax = fig.add_subplot(gs[0, col])
        bn_m, bn_s = np.mean(bn_vals), np.std(bn_vals)
        hv_m, hv_s = np.mean(hv_vals), np.std(hv_vals)

        bars = ax.bar(
            ['Bootstrap\nNoise', 'HVRT'],
            [bn_m, hv_m],
            yerr=[bn_s, hv_s],
            capsize=6,
            color=['#9E9E9E', '#2196F3'],
            edgecolor='white',
            linewidth=0.5,
            width=0.55,
        )
        label = name.split('.', 1)[1].strip()
        ax.set_title(label, fontsize=9, fontweight='bold', pad=6)
        ax.set_ylabel(direction, fontsize=7)
        ax.tick_params(labelsize=8)

        # Annotate winner
        winner_val = hv_m if (higher_better and hv_m > bn_m) or \
                             (not higher_better and hv_m < bn_m) else bn_m
        ax.annotate('★ better', xy=(1, hv_m), xytext=(1, winner_val * 1.08),
                    ha='center', fontsize=7, color='#1565C0')

    fig.suptitle(
        'Bootstrap-Noise Failure Modes vs HVRT',
        fontsize=13, fontweight='bold', y=1.02,
    )

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'bootstrap_failure_benchmark.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved → {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Targeted Bootstrap-Noise failure-mode benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='Random seeds per test (default: 5)')
    parser.add_argument('--plot', action='store_true',
                        help='Save a bar-chart summary to benchmarks/results/')
    args = parser.parse_args()

    print()
    print('Bootstrap-Noise Failure Mode Benchmark')
    print(f'  Seeds per test : {args.n_seeds}')
    print(f'  Noise level    : {_NOISE_LEVEL}  (per-feature fraction of std)')
    print()

    collected = []
    for name, direction, fn, higher_better in TESTS:
        print(f'  Running {name} ', end='', flush=True)
        bn_vals, hv_vals = [], []
        for seed in range(args.n_seeds):
            bn, hv = fn(seed)
            bn_vals.append(bn)
            hv_vals.append(hv)
            print('.', end='', flush=True)
        print(' done')
        collected.append((name, direction, bn_vals, hv_vals, higher_better))

    _print_table(collected)

    if args.plot:
        _plot(collected)


if __name__ == '__main__':
    main()
