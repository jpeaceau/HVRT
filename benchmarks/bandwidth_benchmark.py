#!/usr/bin/env python3
"""
KDE Bandwidth Method Benchmark for HVRT Expansion
==================================================

Compares KDE bandwidth selection methods for HVRT's per-partition multivariate
Gaussian KDE expansion, and tests a structurally motivated alternative kernel.

Four metrics
------------
  disc_err      Logistic-regression indistinguishability: |balanced_acc − 0.5|.
                Lower → synthetic harder to tell from real → better fidelity.
                5-fold stratified CV; capped at 2 000 samples per class.

  mw1           Mean per-feature Wasserstein-1 distance (marginal fidelity).
                Lower → better.

  corr_mae      Mean absolute error of pairwise Pearson correlation matrices.
                Lower → better.

  tstr_delta    ML utility: TSTR metric − TRTR metric (GradientBoosting;
                AUC for classification, R² for regression).  Higher → better.

Bandwidth candidates (Gaussian kernel)
---------------------------------------
  'scott'      Scott's rule n^(−1/(d+4)).  Theoretically AMISE-optimal for
               multivariate Gaussian targets.
  'silverman'  Silverman's rule.  Marginally wider than Scott's.
  h=0.10 …     Absolute covariance scale factors passed as scipy bw_method.
  h=2.00       Note: these are NOT multipliers of Scott's rule.  The kernel
               covariance becomes h² × data_cov.
  adaptive     Per-partition Scott factor scaled by (budget/n_part)^(1/d).

Alternative kernel
------------------
  epanechnikov  Product Epanechnikov kernel with Scott's bandwidth factor.
                Theoretically AMISE-optimal among all kernel families.
                Bounded support means no sample can fall outside the bandwidth
                window — unlike Gaussian whose tails extend to infinity.

Structural hypothesis (Epanechnikov rationale)
----------------------------------------------
HVRT's pairwise-interaction partitioning target produces splits aligned with
the data's principal variance hyperplane, yielding compact, locally homogeneous
partitions.  Because the data within each partition lies near a common
hyperplane, the within-partition distribution is expected to be:

  (a) approximately Gaussian (low skewness, low excess kurtosis), which
      supports Scott's rule as near-optimal for the Gaussian kernel, AND

  (b) compact in feature space, which makes the Epanechnikov kernel's finite
      support attractive — it cannot extrapolate beyond the local bandwidth
      window, staying faithful to the partition's compact support.

The normality section measures (a) empirically.  The metric results for
'epanechnikov' test whether (b) confers a fidelity advantage in practice.

Usage
-----
    python benchmarks/bandwidth_benchmark.py
    python benchmarks/bandwidth_benchmark.py --ratios 2 5 10 --max-n 500
    python benchmarks/bandwidth_benchmark.py --quick   # fast sanity check
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# UTF-8 stdout for Windows terminals that default to cp1252 or similar
if hasattr(sys.stdout, 'buffer') and \
        getattr(sys.stdout, 'encoding', '').lower().replace('-', '') not in \
        ('utf8', 'utf-8'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )

from hvrt import HVRT
from hvrt.benchmarks.datasets import BENCHMARK_DATASETS
from hvrt.generation_strategies import (
    StatefulGenerationStrategy, PartitionContext,
    _build_base_context, _resample_base_points,
)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark-local Epanechnikov variant
# (no per-feature std scaling — isolates bounded-support effect independently
# of within-partition variance)
# ─────────────────────────────────────────────────────────────────────────────

class _BenchmarkEpanechnikov:
    """
    Product Epanechnikov KDE without per-feature std scaling.

    This variant differs from the built-in ``epanechnikov`` strategy, which
    scales noise by within-partition std (``base + h * std * noise``).  This
    benchmark variant uses ``base + h * noise`` to isolate the effect of
    bounded support from the effect of per-partition variance adaptation.

    Implements StatefulGenerationStrategy so it integrates with context
    caching and avoids HVRTDeprecationWarning.
    """

    def prepare(self, X_z, partition_ids, unique_partitions):
        n_parts = len(unique_partitions)
        d = X_z.shape[1]
        pos, sort_idx, part_starts, part_sizes = _build_base_context(
            X_z, partition_ids, unique_partitions
        )
        part_h = np.maximum(part_sizes, 1).astype(float) ** (-1.0 / (d + 4))
        # Store part_h in a thin wrapper around PartitionContext
        ctx = _BenchmarkEpanechnikovContext(
            X_z=X_z, pos=pos, sort_idx=sort_idx,
            part_starts=part_starts, part_sizes=part_sizes,
            n_parts=n_parts, n_features=d,
            part_h=part_h,
        )
        return ctx

    def generate(self, context, budgets, random_state):
        ctx = context
        rng = np.random.RandomState(random_state)
        d = ctx.n_features
        total_budget = int(budgets.sum())
        if total_budget == 0:
            return np.empty((0, d))
        labels = np.repeat(np.arange(ctx.n_parts), budgets)
        base = _resample_base_points(ctx, labels, total_budget, rng)
        U1 = rng.uniform(-1.0, 1.0, (total_budget, d))
        U2 = rng.uniform(-1.0, 1.0, (total_budget, d))
        U3 = rng.uniform(-1.0, 1.0, (total_budget, d))
        use_U2 = (np.abs(U3) >= np.abs(U2)) & (np.abs(U3) >= np.abs(U1))
        noise_unit = np.where(use_U2, U2, U3)
        h_arr = ctx.part_h[labels][:, None]
        return base + h_arr * noise_unit


@dataclass(frozen=True)
class _BenchmarkEpanechnikovContext(PartitionContext):
    part_h: np.ndarray


epanechnikov_kde = _BenchmarkEpanechnikov()


# ─────────────────────────────────────────────────────────────────────────────
# Bandwidth candidate registry
# ─────────────────────────────────────────────────────────────────────────────

# Each entry is a dict of kwargs forwarded verbatim to model.expand().
# - Gaussian kernel candidates: use 'bandwidth' and 'adaptive_bandwidth' keys.
# - Epanechnikov: uses 'generation_strategy' key (bypasses gaussian_kde entirely).
BANDWIDTH_CANDIDATES: dict = {
    'scott':        {'bandwidth': 'scott',         'adaptive_bandwidth': False},
    'silverman':    {'bandwidth': 'silverman',      'adaptive_bandwidth': False},
    'h=0.10':       {'bandwidth': 0.10,             'adaptive_bandwidth': False},
    'h=0.30':       {'bandwidth': 0.30,             'adaptive_bandwidth': False},
    'h=0.50':       {'bandwidth': 0.50,             'adaptive_bandwidth': False},  # HVRT default
    'h=0.75':       {'bandwidth': 0.75,             'adaptive_bandwidth': False},
    'h=1.00':       {'bandwidth': 1.00,             'adaptive_bandwidth': False},
    'h=1.50':       {'bandwidth': 1.50,             'adaptive_bandwidth': False},
    'h=2.00':       {'bandwidth': 2.00,             'adaptive_bandwidth': False},
    'epanechnikov': {'generation_strategy': epanechnikov_kde},
    'adaptive':     {'bandwidth': None,             'adaptive_bandwidth': True},
}

QUICK_CANDIDATES: dict = {
    k: BANDWIDTH_CANDIDATES[k]
    for k in ('scott', 'silverman', 'h=0.50', 'epanechnikov', 'adaptive')
}


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def discriminator_error(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    random_state: int = 42,
) -> float:
    """
    Train a logistic-regression classifier to separate real (label=0) from
    synthetic (label=1) samples.  Returns |balanced_accuracy − 0.5|.

      0.00  balanced_acc ≈ 0.5  → indistinguishable → best
      0.50  balanced_acc = 1.0  → perfectly distinct → worst

    Lower is better.  5-fold stratified CV; each class capped at 2 000 samples.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score

    cap = 2_000
    rng = np.random.RandomState(random_state)
    n = min(len(X_real), len(X_synth), cap)
    idx_r = rng.choice(len(X_real),  size=n, replace=False)
    idx_s = rng.choice(len(X_synth), size=n, replace=False)

    X = np.vstack([X_real[idx_r], X_synth[idx_s]])
    y = np.array([0] * n + [1] * n)

    X_sc = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    scores = []
    for tr, te in cv.split(X_sc, y):
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=random_state)
        clf.fit(X_sc[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], clf.predict(X_sc[te])))

    return float(abs(np.mean(scores) - 0.5))


def marginal_w1(X_real: np.ndarray, X_synth: np.ndarray) -> float:
    """Mean per-feature Wasserstein-1 distance.  Lower is better."""
    from scipy.stats import wasserstein_distance
    return float(np.mean([
        wasserstein_distance(X_real[:, j], X_synth[:, j])
        for j in range(X_real.shape[1])
    ]))


def correlation_mae(X_real: np.ndarray, X_synth: np.ndarray) -> float:
    """
    Mean absolute error of pairwise Pearson correlation matrices (upper
    triangle only).  Lower is better.  Returns 0.0 for single-feature data.
    """
    d = X_real.shape[1]
    if d < 2:
        return 0.0
    idx = np.triu_indices(d, k=1)
    return float(np.mean(np.abs(
        np.corrcoef(X_real.T)[idx] - np.corrcoef(X_synth.T)[idx]
    )))


def tstr_scores(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    X_synth: np.ndarray, y_synth: np.ndarray,
    is_cls: bool,
    random_state: int = 42,
) -> tuple:
    """Return (trtr_score, tstr_score) using GradientBoosting."""
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.metrics import roc_auc_score, r2_score

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        def _make():
            if is_cls:
                return GradientBoostingClassifier(
                    n_estimators=100, random_state=random_state)
            return GradientBoostingRegressor(
                n_estimators=100, random_state=random_state)

        def _score(model, X, y):
            if is_cls:
                p = model.predict_proba(X)
                if p.shape[1] == 2:
                    return float(roc_auc_score(y, p[:, 1]))
                return float(roc_auc_score(
                    y, p, multi_class='ovr', average='weighted'))
            return float(r2_score(y, model.predict(X)))

        trtr_m = _make()
        trtr_m.fit(X_tr, y_tr)
        trtr = _score(trtr_m, X_te, y_te)

        tstr_m = _make()
        tstr_m.fit(X_synth, y_synth)
        tstr = _score(tstr_m, X_te, y_te)

    return trtr, tstr


# ─────────────────────────────────────────────────────────────────────────────
# Normality analysis
# ─────────────────────────────────────────────────────────────────────────────

def partition_normality_stats(X: np.ndarray, random_state: int = 42) -> dict:
    """
    Fit HVRT on X, then compute per-partition per-feature |skewness| and
    |excess kurtosis| (scipy convention: Gaussian → kurtosis = 0) across
    all partitions with ≥ 4 samples.

    Low values confirm the structural hypothesis: that HVRT's variance-
    retaining tree partitions produce locally Gaussian distributions —
    the condition under which Scott's rule is theoretically AMISE-optimal,
    and under which the Epanechnikov finite-support property is most useful.
    """
    model = HVRT(random_state=random_state)
    model.fit(X)

    abs_skews: list = []
    abs_kurts: list = []
    part_sizes: list = []

    for pid in model.unique_partitions_:
        X_p = model.X_z_[model.partition_ids_ == pid]
        part_sizes.append(len(X_p))
        if len(X_p) < 4:
            continue
        for j in range(X_p.shape[1]):
            col = X_p[:, j]
            if col.std() < 1e-8:
                # Near-constant column: degenerate partition dimension.
                # Skewness and kurtosis are undefined (and scipy warns);
                # treat as perfectly uniform/Gaussian (values of 0).
                abs_skews.append(0.0)
                abs_kurts.append(0.0)
                continue
            abs_skews.append(abs(float(scipy_stats.skew(col))))
            abs_kurts.append(abs(float(scipy_stats.kurtosis(col))))  # excess

    return {
        'n_partitions':    int(len(model.unique_partitions_)),
        'mean_part_size':  float(np.mean(part_sizes)),
        'mean_abs_skew':   float(np.mean(abs_skews))  if abs_skews else float('nan'),
        'std_abs_skew':    float(np.std(abs_skews))   if abs_skews else float('nan'),
        'mean_abs_exkurt': float(np.mean(abs_kurts))  if abs_kurts else float('nan'),
        'std_abs_exkurt':  float(np.std(abs_kurts))   if abs_kurts else float('nan'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _snap_labels(y_raw: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """Snap synthetic continuous y values to the nearest observed class label."""
    classes = np.unique(y_ref)
    return classes[np.argmin(np.abs(y_raw[:, None] - classes[None, :]), axis=1)]


def evaluate_one(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    n_synth: int,
    bw_kwargs: dict,
    is_cls: bool,
    seed: int,
) -> dict:
    """
    Fit HVRT on the training split (y stacked as last column for joint
    distribution modelling), expand using the given bandwidth kwargs, then
    compute all four metrics.

    bw_kwargs keys understood:
      bandwidth           → passed to expand()
      adaptive_bandwidth  → passed to expand()
      generation_strategy → passed to expand()  (Epanechnikov path)

    Returns a metrics dict; all values are NaN on failure.
    """
    _nan = {k: float('nan')
            for k in ('disc_err', 'mw1', 'corr_mae', 'trtr', 'tstr_delta')}

    try:
        # Stack y as last feature so HVRT captures the joint (X, y) distribution
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
        )

        X_s = XY_s[:, :-1]
        y_s = _snap_labels(XY_s[:, -1], y_tr) if is_cls else XY_s[:, -1]

        trtr, tstr = tstr_scores(
            X_tr, y_tr, X_te, y_te, X_s, y_s,
            is_cls=is_cls, random_state=seed,
        )

        return {
            'disc_err':   discriminator_error(X_tr, X_s, random_state=seed),
            'mw1':        marginal_w1(X_tr, X_s),
            'corr_mae':   correlation_mae(X_tr, X_s),
            'trtr':       trtr,
            'tstr_delta': tstr - trtr,
        }

    except Exception as exc:
        warnings.warn(f'evaluate_one failed: {exc}', stacklevel=2)
        return _nan


def run_condition(
    X_full: np.ndarray,
    y_full: np.ndarray,
    exp_ratio: float,
    bw_kwargs: dict,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    is_cls: bool,
    max_n: int,
) -> dict:
    """Run repeated k-fold CV for one (dataset, ratio, bandwidth) condition."""
    from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

    if is_cls:
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        splits = list(cv.split(X_full, y_full))
    else:
        cv = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        splits = list(cv.split(X_full))

    accum: dict = defaultdict(list)

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = X_full[tr_idx], X_full[te_idx]
        y_tr, y_te = y_full[tr_idx], y_full[te_idx]

        if len(X_tr) > max_n:
            rng = np.random.RandomState(random_state + fold_i)
            sel = rng.choice(len(X_tr), size=max_n, replace=False)
            X_tr, y_tr = X_tr[sel], y_tr[sel]

        n_synth = max(4, int(len(X_tr) * exp_ratio))
        seed = random_state + fold_i * 13

        res = evaluate_one(
            X_tr, y_tr, X_te, y_te, n_synth, bw_kwargs,
            is_cls=is_cls, seed=seed,
        )
        for k in ('disc_err', 'mw1', 'corr_mae', 'trtr', 'tstr_delta'):
            accum[k].append(res[k])

    def _ms(k):
        a = np.array(accum[k], dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a))

    out: dict = {'n_folds': len(splits)}
    for k in ('disc_err', 'mw1', 'corr_mae', 'trtr', 'tstr_delta'):
        m, s = _ms(k)
        out[f'{k}_mean'] = m
        out[f'{k}_std']  = s
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

# (metric_key, display_label, lower_is_better)
_METRIC_DEFS = [
    ('disc_err',   'Disc.Err ↓',  True),
    ('mw1',        'Marg.W1 ↓',   True),
    ('corr_mae',   'Corr.MAE ↓',  True),
    ('tstr_delta', 'TSTR Δ ↑',    False),
]


def _extremes(results_by_bw: dict, key: str, lower_better: bool):
    """Return (best_bw_name, worst_bw_name) for a given metric."""
    vals  = {bw: results_by_bw[bw][f'{key}_mean'] for bw in results_by_bw}
    valid = {bw: v for bw, v in vals.items() if not np.isnan(v)}
    if not valid:
        return None, None
    best  = min(valid, key=valid.__getitem__) if lower_better \
        else max(valid, key=valid.__getitem__)
    worst = max(valid, key=valid.__getitem__) if lower_better \
        else min(valid, key=valid.__getitem__)
    return best, worst


def print_condition_table(
    ds_name: str,
    ratio: float,
    results_by_bw: dict,
    n_splits: int,
    n_repeats: int,
):
    """
    One table per (dataset, ratio): rows = bandwidth candidates, cols = metrics.
    ◆ marks the best value per metric; ▽ marks the worst.
    """
    bw_names = list(results_by_bw.keys())
    extremes = {
        key: _extremes(results_by_bw, key, lb)
        for key, _, lb in _METRIC_DEFS
    }
    # Inner width: 2 + 14 (method) + 4 × 16 (metrics) = 80
    W = 80
    print(f'╔{"═" * W}╗')
    title = (f'  {ds_name}  |  ratio={ratio:.0f}x  |  '
             f'{n_splits}-fold × {n_repeats}-repeat')
    print(f'║{title:<{W}}║')
    print(f'╠{"═" * W}╣')
    hdr = (f'  {"Method":<14}'
           f'{"Disc.Err ↓":>16}'
           f'{"Marg.W1 ↓":>16}'
           f'{"Corr.MAE ↓":>16}'
           f'{"TSTR Δ ↑":>16}')
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    for bw in bw_names:
        row = f'  {bw:<14}'
        for key, _, lb in _METRIC_DEFS:
            m = results_by_bw[bw][f'{key}_mean']
            s = results_by_bw[bw][f'{key}_std']
            best_bw, worst_bw = extremes[key]
            if np.isnan(m):
                cell = '—'
            elif key == 'tstr_delta':
                cell = f'{m:+.4f}±{s:.3f}'
            else:
                cell = f'{m:.4f}±{s:.3f}'
            marker = '◆' if bw == best_bw else ('▽' if bw == worst_bw else ' ')
            row += f'{cell + marker:>16}'
        print(f'║{row}║')

    print(f'╚{"═" * W}╝')
    print()


def print_normality_table(normality_results: dict):
    """Per-dataset partition normality statistics."""
    W = 80
    print(f'╔{"═" * W}╗')
    print(f'║  {"WITHIN-PARTITION NORMALITY  (z-scored continuous features)":<{W - 3}}║')
    print(f'║  {"Hypothesis: HVRT partitions → locally Gaussian → Scott & Epan optimal":<{W - 3}}║')
    print(f'╠{"═" * W}╣')
    hdr = (f'  {"Dataset":<24}'
           f'{"Partitions":>12}'
           f'{"Mean size":>10}'
           f'{"Mean |skew|":>16}'
           f'{"Mean |ex.kurt|":>16}')
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    for ds, n in normality_results.items():
        skew_s = f'{n["mean_abs_skew"]:.3f}±{n["std_abs_skew"]:.3f}'
        kurt_s = f'{n["mean_abs_exkurt"]:.3f}±{n["std_abs_exkurt"]:.3f}'
        row = (f'  {ds:<24}'
               f'{n["n_partitions"]:>12}'
               f'{n["mean_part_size"]:>10.1f}'
               f'{skew_s:>16}'
               f'{kurt_s:>16}')
        print(f'║{row}║')

    print(f'╚{"═" * W}╝')


def print_normality_verdict(normality_results: dict):
    """Short interpretive verdict linking normality stats to the hypothesis."""
    print()
    print('  Normality interpretation:')
    for ds, n in normality_results.items():
        skew = n['mean_abs_skew']
        kurt = n['mean_abs_exkurt']
        if np.isnan(skew):
            verdict = 'insufficient data'
        elif skew < 0.3 and kurt < 0.5:
            verdict = 'strongly Gaussian  → Scott & Epanechnikov both well-matched'
        elif skew < 0.6 and kurt < 1.0:
            verdict = 'mildly non-Gaussian → rule-based bandwidths still reasonable'
        else:
            verdict = 'moderately skewed   → rule-based methods may underfit tails'
        print(f'    {ds:<28}  {verdict}')
    print()


def print_summary_table(all_results: dict, bw_names: list):
    """
    Win-count table across all (dataset × ratio) conditions.
    A method 'wins' a condition when it achieves the best value for that metric.
    """
    wins: dict = {bw: {m: 0 for m, *_ in _METRIC_DEFS} for bw in bw_names}
    n_cond = 0

    for _cond, ds_results in all_results.items():
        n_cond += 1
        for key, _, lb in _METRIC_DEFS:
            best, _ = _extremes(ds_results, key, lb)
            if best is not None:
                wins[best][key] += 1

    col_w = max(12, max(len(b) for b in bw_names) + 2)
    W = 20 + len(bw_names) * col_w

    print(f'╔{"═" * W}╗')
    print(f'║  {"WIN-COUNT SUMMARY  (" + str(n_cond) + " conditions: dataset × ratio)":<{W - 3}}║')
    print(f'╠{"═" * W}╣')
    hdr = f'  {"Metric":<18}' + ''.join(f'{b:>{col_w}}' for b in bw_names)
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    for key, label, _ in _METRIC_DEFS:
        row = f'  {label:<18}' + ''.join(
            f'{wins[b][key]:>{col_w}}' for b in bw_names
        )
        print(f'║{row}║')

    print(f'╠{"─" * W}╣')
    totals = {b: sum(wins[b].values()) for b in bw_names}
    max_t  = max(totals.values())
    total_row = f'  {"Total wins":<18}' + ''.join(
        f'{str(totals[b]) + ("◆" if totals[b] == max_t else ""):{col_w}}'
        for b in bw_names
    )
    print(f'║{total_row}║')
    print(f'╚{"═" * W}╝')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--ratios',    nargs='+', type=float, default=[2.0, 5.0, 10.0],
                        help='Expansion ratios to test (default: 2 5 10)')
    parser.add_argument('--max-n',     type=int, default=500,
                        help='Training-set cap per fold (default: 500)')
    parser.add_argument('--n-splits',  type=int, default=5)
    parser.add_argument('--n-repeats', type=int, default=3)
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--quick',     action='store_true',
                        help='5 candidates, 3 datasets, 3-fold × 1-repeat')
    args = parser.parse_args()

    if args.quick:
        candidates  = QUICK_CANDIDATES
        ds_subset   = {k: BENCHMARK_DATASETS[k]
                       for k in ('multimodal', 'adult_like', 'housing_like')
                       if k in BENCHMARK_DATASETS}
        n_splits, n_repeats = 3, 1
    else:
        candidates  = BANDWIDTH_CANDIDATES
        ds_subset   = BENCHMARK_DATASETS
        n_splits, n_repeats = args.n_splits, args.n_repeats

    bw_names = list(candidates.keys())
    n_total  = n_splits * n_repeats

    print()
    print('═' * 72)
    print('  HVRT — KDE Bandwidth & Kernel Benchmark')
    print('═' * 72)
    print(f'  Model         : HVRT (pairwise interaction partitioning target)')
    print(f'  Datasets      : {", ".join(ds_subset)}')
    print(f'  Ratios        : {args.ratios}')
    print(f'  Candidates    : {", ".join(bw_names)}')
    print(f'  max_n (train) : {args.max_n}')
    print(f'  CV            : {n_splits}-fold × {n_repeats}-repeat'
          f' = {n_total} evals per condition')
    print()

    all_results: dict      = {}  # (ds_name, ratio) → {bw_name: metrics_dict}
    normality_results: dict = {}  # ds_name → normality stats dict

    for ds_name, gen_fn in ds_subset.items():
        X_full, y_full, _ = gen_fn(random_state=args.seed)
        is_cls = len(np.unique(y_full)) <= 20
        task_label = 'classification (AUC)' if is_cls else 'regression (R²)'

        # ── Normality analysis (once per dataset) ────────────────────────────
        X_norm = X_full[:args.max_n]
        print(f'  [{ds_name}]  {task_label}')
        print(f'    Normality analysis ...', end=' ', flush=True)
        nstats = partition_normality_stats(X_norm, random_state=args.seed)
        normality_results[ds_name] = nstats
        print(f'{nstats["n_partitions"]} partitions | '
              f'mean size {nstats["mean_part_size"]:.1f} | '
              f'|skew| = {nstats["mean_abs_skew"]:.3f} | '
              f'|ex.kurt| = {nstats["mean_abs_exkurt"]:.3f}')

        # ── Per-ratio, per-bandwidth evaluation ──────────────────────────────
        for ratio in args.ratios:
            cond_key = (ds_name, ratio)
            all_results[cond_key] = {}

            for bw_name, bw_kwargs in candidates.items():
                print(f'    [{ratio:.0f}x  {bw_name:<14}] ...', end=' ', flush=True)
                res = run_condition(
                    X_full, y_full,
                    exp_ratio=ratio,
                    bw_kwargs=bw_kwargs,
                    n_splits=n_splits, n_repeats=n_repeats,
                    random_state=args.seed, is_cls=is_cls, max_n=args.max_n,
                )
                all_results[cond_key][bw_name] = res
                print(f'disc_err={res["disc_err_mean"]:.4f}  '
                      f'W1={res["mw1_mean"]:.4f}  '
                      f'Corr={res["corr_mae_mean"]:.4f}  '
                      f'TSTR Δ={res["tstr_delta_mean"]:+.4f}')

            print()
            print_condition_table(ds_name, ratio, all_results[cond_key],
                                  n_splits, n_repeats)

    print_normality_table(normality_results)
    print_normality_verdict(normality_results)
    print_summary_table(all_results, bw_names)
    print()


if __name__ == '__main__':
    main()
