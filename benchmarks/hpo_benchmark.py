#!/usr/bin/env python3
"""
HPO Benchmark: HVRTOptimizer vs. Default HVRT
==============================================

Compares HVRTOptimizer (Optuna-backed TPE) against HVRT with all defaults
across the six standard benchmark datasets, using a nested cross-validation
protocol: for each outer fold the optimiser tunes on the training split via
inner CV, then the best model is evaluated on the held-out test split.

This separates HPO fitting from evaluation and avoids information leakage.

Compares
--------
  default   HVRT(bandwidth='auto') — all hyperparameters auto-tuned from
            dataset size.  Represents the "no-tuning" starting point.

  hpo       HVRTOptimizer(n_trials, cv=inner_cv) — Bayesian search over
            n_partitions, min_samples_leaf, y_weight, kernel, variance_weighted.

Metrics (same as bandwidth_benchmark.py)
-----------------------------------------
  disc_err      |balanced_accuracy − 0.5| from LR discriminator  (lower = better)
  mw1           Mean per-feature Wasserstein-1 distance           (lower = better)
  corr_mae      Pairwise Pearson correlation MAE                  (lower = better)
  tstr_delta    TSTR − TRTR, GradientBoosting AUC/R²             (higher = better)

HPO objective
-------------
HVRTOptimizer minimises the SAME tstr_delta objective internally, so this
benchmark is measuring how much better the metric can get when the optimiser
is given access to the training-fold targets.

Trial 0 is always the HVRT defaults (warm start via enqueue_trial).  This
guarantees HPO can only match or beat defaults when the inner-CV signal is
reliable.  With n_trials < 50 and inner_cv < 3 the TPE sampler is still noisy
and can pick sub-optimal params — increase both for production use.

Usage
-----
    python benchmarks/hpo_benchmark.py
    python benchmarks/hpo_benchmark.py --quick
    python benchmarks/hpo_benchmark.py --n-trials 40 --outer-splits 5 --inner-cv 3
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# UTF-8 stdout for Windows terminals (cp1252 cannot render box-drawing chars)
if hasattr(sys.stdout, 'buffer') and \
        getattr(sys.stdout, 'encoding', '').lower().replace('-', '') not in \
        ('utf8', 'utf-8'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )

from hvrt import HVRT, HVRTOptimizer
from hvrt.benchmarks.datasets import BENCHMARK_DATASETS


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers (identical semantics to bandwidth_benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────

def discriminator_error(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    random_state: int = 42,
) -> float:
    """
    |balanced_accuracy − 0.5| from a logistic-regression discriminator.
    0.0 = indistinguishable; 0.5 = trivially separable.  Lower is better.
    5-fold stratified CV; each class capped at 2 000 samples.
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
    MAE of pairwise Pearson correlation matrices (upper triangle).
    Lower is better.  Returns 0.0 for single-feature data.
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


def _snap_labels(y_raw: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """Snap continuous synthetic y to the nearest observed class label."""
    classes = np.unique(y_ref)
    return classes[np.argmin(np.abs(y_raw[:, None] - classes[None, :]), axis=1)]


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold evaluators
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_default(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_synth: int,
    is_cls: bool,
    seed: int,
) -> dict:
    """
    Fit HVRT with all defaults on the training split (y stacked as last column
    for joint distribution modelling), expand, and compute the four metrics.

    Returns a metrics dict; all NaN on failure.
    """
    _nan = {k: float('nan')
            for k in ('disc_err', 'mw1', 'corr_mae', 'trtr', 'tstr_delta')}
    try:
        XY_tr = np.column_stack([X_tr, y_tr.reshape(-1, 1).astype(float)])
        model = HVRT(random_state=seed).fit(XY_tr)
        XY_s = model.expand(n=n_synth)
        X_s = XY_s[:, :-1]
        y_s = _snap_labels(XY_s[:, -1], y_tr) if is_cls else XY_s[:, -1]

        trtr, tstr = tstr_scores(
            X_tr, y_tr, X_te, y_te, X_s, y_s, is_cls=is_cls, random_state=seed,
        )
        return {
            'disc_err':   discriminator_error(X_tr, X_s, random_state=seed),
            'mw1':        marginal_w1(X_tr, X_s),
            'corr_mae':   correlation_mae(X_tr, X_s),
            'trtr':       trtr,
            'tstr_delta': tstr - trtr,
        }
    except Exception as exc:
        warnings.warn(f'evaluate_default failed: {exc}', stacklevel=2)
        return _nan


def evaluate_hpo(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_synth: int,
    is_cls: bool,
    seed: int,
    n_trials: int,
    inner_cv: int,
    expansion_ratio: float,
) -> tuple:
    """
    Fit HVRTOptimizer on the training split (inner CV entirely on X_tr, y_tr),
    then expand using the best model and compute the four metrics.

    Returns (metrics_dict, best_params_dict, best_expand_params_dict, best_score).
    best_score is the HPO objective (mean inner-CV TSTR Δ).
    """
    _nan = {k: float('nan')
            for k in ('disc_err', 'mw1', 'corr_mae', 'trtr', 'tstr_delta')}

    try:
        task = 'classification' if is_cls else 'regression'
        opt = HVRTOptimizer(
            n_trials=n_trials,
            cv=inner_cv,
            expansion_ratio=expansion_ratio,
            task=task,
            random_state=seed,
            verbose=0,
        )
        opt.fit(X_tr, y_tr)

        # Access the internal model directly to obtain synthetic y alongside X
        # (HVRTOptimizer.expand() strips the y column; we need it for TSTR)
        XY_s = opt.best_model_.expand(n=n_synth, **opt.best_expand_params_)
        X_s = XY_s[:, :X_tr.shape[1]]   # strip synthetic y column
        y_s_raw = XY_s[:, -1]
        y_s = _snap_labels(y_s_raw, y_tr) if is_cls else y_s_raw

        trtr, tstr = tstr_scores(
            X_tr, y_tr, X_te, y_te, X_s, y_s, is_cls=is_cls, random_state=seed,
        )
        metrics = {
            'disc_err':   discriminator_error(X_tr, X_s, random_state=seed),
            'mw1':        marginal_w1(X_tr, X_s),
            'corr_mae':   correlation_mae(X_tr, X_s),
            'trtr':       trtr,
            'tstr_delta': tstr - trtr,
        }
        return metrics, opt.best_params_, opt.best_expand_params_, float(opt.best_score_)

    except Exception as exc:
        warnings.warn(f'evaluate_hpo failed: {exc}', stacklevel=2)
        return _nan, {}, {}, float('nan')


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level runner
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(
    X_full: np.ndarray,
    y_full: np.ndarray,
    is_cls: bool,
    outer_splits: int,
    outer_repeats: int,
    n_trials: int,
    inner_cv: int,
    expansion_ratio: float,
    max_n: int,
    seed: int,
) -> dict:
    """
    Nested CV for one dataset.

    Outer loop: RepeatedKFold / RepeatedStratifiedKFold.
    Per fold: evaluate_default + evaluate_hpo on the same split.

    Returns a dict with keys:
      'default'    → aggregated metric stats
      'hpo'        → aggregated metric stats
      'hpo_params' → list of (best_params, best_expand_params, best_score) per fold
      'n_folds'    → total number of folds run
    """
    from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

    if is_cls:
        cv = RepeatedStratifiedKFold(
            n_splits=outer_splits, n_repeats=outer_repeats, random_state=seed)
        splits = list(cv.split(X_full, y_full))
    else:
        cv = RepeatedKFold(
            n_splits=outer_splits, n_repeats=outer_repeats, random_state=seed)
        splits = list(cv.split(X_full))

    default_accum: dict = defaultdict(list)
    hpo_accum:     dict = defaultdict(list)
    hpo_params_log: list = []

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = X_full[tr_idx], X_full[te_idx]
        y_tr, y_te = y_full[tr_idx], y_full[te_idx]

        if len(X_tr) > max_n:
            rng = np.random.RandomState(seed + fold_i)
            sel = rng.choice(len(X_tr), size=max_n, replace=False)
            X_tr, y_tr = X_tr[sel], y_tr[sel]

        n_synth = max(4, int(len(X_tr) * expansion_ratio))
        fold_seed = seed + fold_i * 17

        d_res = evaluate_default(X_tr, y_tr, X_te, y_te, n_synth, is_cls, fold_seed)
        for k in ('disc_err', 'mw1', 'corr_mae', 'trtr', 'tstr_delta'):
            default_accum[k].append(d_res[k])

        h_res, h_params, h_expand, h_score = evaluate_hpo(
            X_tr, y_tr, X_te, y_te, n_synth, is_cls, fold_seed,
            n_trials=n_trials, inner_cv=inner_cv, expansion_ratio=expansion_ratio,
        )
        for k in ('disc_err', 'mw1', 'corr_mae', 'trtr', 'tstr_delta'):
            hpo_accum[k].append(h_res[k])
        hpo_params_log.append((h_params, h_expand, h_score))

    def _agg(accum: dict) -> dict:
        out = {}
        for k, vals in accum.items():
            arr = np.array(vals, dtype=float)
            out[f'{k}_mean'] = float(np.nanmean(arr))
            out[f'{k}_std']  = float(np.nanstd(arr))
        return out

    return {
        'default':    _agg(default_accum),
        'hpo':        _agg(hpo_accum),
        'hpo_params': hpo_params_log,
        'n_folds':    len(splits),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Best-params analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mode(values: list):
    """Return most common element; None for empty list."""
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _summarise_params(hpo_params_log: list) -> dict:
    """
    Summarise best_params and best_expand_params across folds.

    Returns a dict with:
      n_partitions_mode    most common n_partitions value (or 'auto')
      bandwidth_mode       most common bandwidth value
      y_weight_mean        mean y_weight across folds
      variance_weighted_mode  most common variance_weighted (True/False)
      generation_strategy_mode  most common generation_strategy (or None)
      hpo_score_mean       mean inner-CV TSTR Δ across folds
      hpo_score_std        std  inner-CV TSTR Δ across folds
    """
    n_parts_vals  = []
    bw_vals       = []
    yw_vals       = []
    vw_vals       = []
    gs_vals       = []
    score_vals    = []

    for params, expand_params, score in hpo_params_log:
        if params:
            n_parts_vals.append(
                'auto' if params.get('n_partitions') is None
                else str(params['n_partitions'])
            )
            bw_vals.append(str(params.get('bandwidth', '?')))
            yw_vals.append(float(params.get('y_weight', 0.0)))
        if expand_params:
            vw_vals.append(bool(expand_params.get('variance_weighted', False)))
            gs_vals.append(expand_params.get('generation_strategy', None))
        if not np.isnan(score) and np.isfinite(score):
            score_vals.append(score)

    scores_arr = np.array(score_vals, dtype=float) if score_vals else np.array([float('nan')])
    return {
        'n_partitions_mode':      _mode(n_parts_vals) if n_parts_vals else '?',
        'bandwidth_mode':         _mode(bw_vals)      if bw_vals      else '?',
        'y_weight_mean':          float(np.mean(yw_vals)) if yw_vals   else float('nan'),
        'variance_weighted_mode': _mode(vw_vals)      if vw_vals      else False,
        'generation_strategy_mode': _mode(gs_vals)   if gs_vals      else None,
        'hpo_score_mean':         float(np.nanmean(scores_arr)),
        'hpo_score_std':          float(np.nanstd(scores_arr)),
        '_n_parts_all':           n_parts_vals,
        '_bw_all':                bw_vals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_DEFS = [
    ('disc_err',   'Disc.Err ↓', True),
    ('mw1',        'Marg.W1 ↓',  True),
    ('corr_mae',   'Corr.MAE ↓', True),
    ('tstr_delta', 'TSTR Δ ↑',   False),
]


def print_dataset_comparison_table(
    ds_name: str,
    n_folds: int,
    default_stats: dict,
    hpo_stats: dict,
    task_label: str,
):
    """
    Two-row table (default vs hpo) with four metrics.
    Δ row shows HPO improvement: positive = HPO better for all metrics.
    """
    W = 80
    print(f'╔{"═" * W}╗')
    title = f'  {ds_name}  |  {task_label}  |  {n_folds} outer folds'
    print(f'║{title:<{W}}║')
    print(f'╠{"═" * W}╣')
    hdr = (f'  {"Method":<10}'
           f'{"Disc.Err ↓":>16}'
           f'{"Marg.W1 ↓":>16}'
           f'{"Corr.MAE ↓":>16}'
           f'{"TSTR Δ ↑":>16}')
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    rows = [('default', default_stats), ('hpo', hpo_stats)]
    for label, stats in rows:
        row = f'  {label:<10}'
        for key, _, lb in _METRIC_DEFS:
            m = stats.get(f'{key}_mean', float('nan'))
            s = stats.get(f'{key}_std',  float('nan'))
            if np.isnan(m):
                cell = '—'
            elif key == 'tstr_delta':
                cell = f'{m:+.4f}±{s:.3f}'
            else:
                cell = f'{m:.4f}±{s:.3f}'
            row += f'{cell:>16}'
        print(f'║{row}║')

    # Improvement row  (positive = HPO better regardless of metric direction)
    print(f'╠{"─" * W}╣')
    imp_row = f'  {"Δ (hpo−def)":<10}'
    for key, _, lb in _METRIC_DEFS:
        dm = default_stats.get(f'{key}_mean', float('nan'))
        hm = hpo_stats.get(f'{key}_mean',    float('nan'))
        if np.isnan(dm) or np.isnan(hm):
            cell = '—'
        else:
            # Positive = HPO is better in both directions
            diff = (dm - hm) if lb else (hm - dm)
            cell = f'{diff:+.4f}'
        imp_row += f'{cell:>16}'
    print(f'║{imp_row}║')
    print(f'╚{"═" * W}╝')
    print()


def print_best_params_table(ds_name: str, summary: dict):
    """
    Show what HVRTOptimizer consistently found for this dataset.
    """
    n_parts_str = summary['n_partitions_mode']
    bw_str      = summary['bandwidth_mode']
    yw_str      = f"{summary['y_weight_mean']:.3f}"
    vw_str      = str(summary['variance_weighted_mode'])
    gs          = summary['generation_strategy_mode']
    gs_str      = str(gs.__name__) if callable(gs) else str(gs) if gs else 'None'
    score_str   = (f"{summary['hpo_score_mean']:+.4f}"
                   f"±{summary['hpo_score_std']:.4f}")

    all_np = summary.get('_n_parts_all', [])
    all_bw = summary.get('_bw_all', [])
    np_counts = Counter(all_np).most_common(3)
    bw_counts = Counter(all_bw).most_common(3)

    W = 64
    print(f'╔{"═" * W}╗')
    print(f'║  Best params found by HPO — {ds_name:<{W - 32}}║')
    print(f'╠{"═" * W}╣')

    rows = [
        ('n_partitions (mode)',  n_parts_str),
        ('bandwidth (mode)',     bw_str),
        ('y_weight (mean)',      yw_str),
        ('variance_weighted',    vw_str),
        ('generation_strategy',  gs_str),
        ('HPO obj (inner-CV Δ)', score_str),
    ]
    for label, val in rows:
        print(f'║  {label:<28}  {val:<{W - 32}}║')

    print(f'╠{"─" * W}╣')
    # Frequency breakdowns
    np_line = ', '.join(f'{k}:{c}' for k, c in np_counts)
    bw_line = ', '.join(f'{k}:{c}' for k, c in bw_counts)
    print(f'║  n_partitions dist: {np_line:<{W - 22}}║')
    print(f'║  bandwidth dist:    {bw_line:<{W - 22}}║')
    print(f'╚{"═" * W}╝')
    print()


def print_summary_table(
    ds_names: list,
    all_results: dict,
    expansion_ratio: float,
    n_trials: int,
    inner_cv: int,
):
    """
    Cross-dataset summary: one row per dataset, shows default and HPO TSTR Δ
    and the improvement, plus a simple win/tie/loss verdict.
    """
    W = 88
    print(f'╔{"═" * W}╗')
    print(f'║  HPO BENCHMARK SUMMARY — {n_trials} trials, inner-cv={inner_cv},'
          f' expansion={expansion_ratio:.0f}x{"":<{W - 58}}║')
    print(f'╠{"═" * W}╣')
    hdr = (f'  {"Dataset":<26}'
           f'{"Default TSTR Δ":>16}'
           f'{"HPO TSTR Δ":>16}'
           f'{"Improvement":>14}'
           f'{"Verdict":>14}')
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    hpo_wins = 0
    for ds in ds_names:
        r = all_results[ds]
        dm = r['default']['tstr_delta_mean']
        ds_d = r['default']['tstr_delta_std']
        hm = r['hpo']['tstr_delta_mean']
        hs_d = r['hpo']['tstr_delta_std']
        imp = hm - dm
        if np.isnan(dm) or np.isnan(hm):
            verdict = '?'
        elif imp > 0.005:
            verdict = 'HPO wins  ◆'
            hpo_wins += 1
        elif imp < -0.005:
            verdict = 'Default ▽'
        else:
            verdict = 'Tie  ≈'

        def_cell = f'{dm:+.4f}±{ds_d:.3f}' if not np.isnan(dm) else '—'
        hpo_cell  = f'{hm:+.4f}±{hs_d:.3f}' if not np.isnan(hm) else '—'
        imp_cell  = f'{imp:+.4f}' if not (np.isnan(dm) or np.isnan(hm)) else '—'
        row = (f'  {ds:<26}{def_cell:>16}{hpo_cell:>16}{imp_cell:>14}{verdict:>14}')
        print(f'║{row}║')

    print(f'╠{"─" * W}╣')
    n_ds = len(ds_names)
    footer = f'  HPO wins {hpo_wins}/{n_ds} datasets on TSTR Δ (threshold ±0.005)'
    print(f'║{footer:<{W}}║')
    print(f'╚{"═" * W}╝')
    print()


def print_all_metrics_summary(ds_names: list, all_results: dict):
    """
    Win-count table across all datasets, all four metrics: default vs hpo.
    """
    wins = {'default': 0, 'hpo': 0, 'tie': 0}
    metric_wins = {m: {'default': 0, 'hpo': 0, 'tie': 0}
                   for m, *_ in _METRIC_DEFS}

    for ds in ds_names:
        r = all_results[ds]
        for key, _, lb in _METRIC_DEFS:
            dm = r['default'][f'{key}_mean']
            hm = r['hpo'][f'{key}_mean']
            if np.isnan(dm) or np.isnan(hm):
                metric_wins[key]['tie'] += 1
                continue
            better_is_lower = lb
            if better_is_lower:
                diff = dm - hm   # positive = hpo better
            else:
                diff = hm - dm   # positive = hpo better
            if diff > 0.001:
                metric_wins[key]['hpo'] += 1
            elif diff < -0.001:
                metric_wins[key]['default'] += 1
            else:
                metric_wins[key]['tie'] += 1

    W = 56
    print(f'╔{"═" * W}╗')
    print(f'║  WIN COUNTS ACROSS ALL DATASETS{"":<{W - 33}}║')
    print(f'╠{"═" * W}╣')
    hdr = f'  {"Metric":<20}{"Default":>10}{"HPO":>10}{"Tie":>10}'
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')
    for key, label, _ in _METRIC_DEFS:
        mw = metric_wins[key]
        row = f'  {label:<20}{mw["default"]:>10}{mw["hpo"]:>10}{mw["tie"]:>10}'
        print(f'║{row}║')
    print(f'╚{"═" * W}╝')
    print()


def print_findings_narrative(
    ds_names: list,
    all_results: dict,
    all_param_summaries: dict,
):
    """
    Auto-generated interpretive narrative based on the observed results.
    """
    W = 80
    print('═' * W)
    print('  FINDINGS')
    print('═' * W)
    print()

    # Rank datasets by HPO improvement on TSTR
    improvements = {}
    for ds in ds_names:
        r = all_results[ds]
        dm = r['default']['tstr_delta_mean']
        hm = r['hpo']['tstr_delta_mean']
        improvements[ds] = hm - dm if not (np.isnan(dm) or np.isnan(hm)) else 0.0

    ranked = sorted(ds_names, key=lambda d: improvements[d], reverse=True)
    gainers  = [d for d in ranked if improvements[d] > 0.005]
    neutral  = [d for d in ranked if -0.005 <= improvements[d] <= 0.005]
    degraded = [d for d in ranked if improvements[d] < -0.005]

    # 1. Which datasets benefit most
    print('  1. Datasets that benefit most from HPO')
    print('  ─────────────────────────────────────')
    if gainers:
        for ds in gainers:
            r = all_results[ds]
            hm = r['hpo']['tstr_delta_mean']
            dm = r['default']['tstr_delta_mean']
            ps = all_param_summaries[ds]
            gs = ps['generation_strategy_mode']
            gs_str = (gs.__name__ if callable(gs) else str(gs)) if gs else 'None'
            print(f'    {ds:<28}  +{improvements[ds]:.4f} TSTR Δ improvement')
            print(f'                                    default={dm:+.4f}  →  hpo={hm:+.4f}')
            print(f'                                    n_parts={ps["n_partitions_mode"]}  '
                  f'bw={ps["bandwidth_mode"]}  '
                  f'y_weight={ps["y_weight_mean"]:.2f}  '
                  f'gen_strat={gs_str}')
    else:
        print('    No dataset showed a clear TSTR improvement from HPO (> 0.005).')
    print()

    # 2. Datasets where defaults are sufficient
    print('  2. Datasets where defaults are sufficient (HPO improvement < 0.005)')
    print('  ────────────────────────────────────────────────────────────────────')
    if neutral:
        for ds in neutral:
            dm = all_results[ds]['default']['tstr_delta_mean']
            print(f'    {ds:<28}  default TSTR Δ={dm:+.4f}  improvement={improvements[ds]:+.4f}')
    else:
        print('    (none)')
    print()

    # 3. Datasets where HPO underperforms
    if degraded:
        print('  3. Datasets where defaults outperform HPO (HPO Δ < −0.005)')
        print('  ─────────────────────────────────────────────────────────')
        for ds in degraded:
            dm = all_results[ds]['default']['tstr_delta_mean']
            hm = all_results[ds]['hpo']['tstr_delta_mean']
            print(f'    {ds:<28}  default={dm:+.4f}  hpo={hm:+.4f}  '
                  f'diff={improvements[ds]:+.4f}')
        print()
        print('    This can indicate overfitting within HPO inner folds on small')
        print('    training sets, or that the 5-expansion-ratio objective does not')
        print('    align with the evaluation ratio used in this benchmark.')
        print()

    # 4. Consistent parameter patterns
    print('  4. Parameter patterns found by HPO')
    print('  ───────────────────────────────────')

    # Aggregate n_partitions across all datasets
    all_np = []
    all_bw = []
    for ps in all_param_summaries.values():
        all_np.extend(ps.get('_n_parts_all', []))
        all_bw.extend(ps.get('_bw_all', []))

    np_counter = Counter(all_np)
    bw_counter = Counter(all_bw)
    total_folds = sum(np_counter.values())

    print(f'    n_partitions distribution across all folds / datasets:')
    for val, cnt in np_counter.most_common():
        pct = 100 * cnt / total_folds if total_folds > 0 else 0
        print(f'      {val:<10}  {cnt:3d} folds  ({pct:.0f}%)')
    print()
    print(f'    bandwidth distribution across all folds / datasets:')
    for val, cnt in bw_counter.most_common():
        pct = 100 * cnt / total_folds if total_folds > 0 else 0
        print(f'      {val:<12}  {cnt:3d} folds  ({pct:.0f}%)')
    print()

    # 5. HPO objective alignment
    print('  5. HPO objective alignment with outer evaluation')
    print('  ────────────────────────────────────────────────')
    aligned = sum(1 for ds in ds_names if improvements[ds] >= -0.002)
    print(f'    {aligned}/{len(ds_names)} datasets: outer TSTR Δ (hpo) >= outer TSTR Δ (default) − 0.002')
    print('    When alignment is low, the inner-CV objective may not generalise:')
    print('    consider increasing n_trials or the outer fold count.')
    print()

    # 6. Y-weight finding
    yw_vals = [ps['y_weight_mean'] for ps in all_param_summaries.values()
               if not np.isnan(ps['y_weight_mean'])]
    if yw_vals:
        mean_yw = float(np.mean(yw_vals))
        print(f'  6. y_weight: mean across all datasets = {mean_yw:.3f}')
        if mean_yw > 0.15:
            print('     HPO consistently selected non-zero y_weight, suggesting that')
            print('     partitioning along y-extremeness improves synthetic utility.')
        else:
            print('     HPO generally selected low y_weight (≤ 0.1), consistent with')
            print('     unsupervised partitioning being sufficient for these datasets.')
        print()

    print('═' * W)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--n-trials', type=int, default=20,
        help='Optuna trials per outer fold (default: 20)',
    )
    parser.add_argument(
        '--inner-cv', type=int, default=3,
        help='Inner CV folds inside HVRTOptimizer (default: 3)',
    )
    parser.add_argument(
        '--outer-splits', type=int, default=3,
        help='Outer CV splits (default: 3)',
    )
    parser.add_argument(
        '--outer-repeats', type=int, default=2,
        help='Outer CV repeats (default: 2)',
    )
    parser.add_argument(
        '--expansion-ratio', type=float, default=5.0,
        help='Synthetic:real ratio for expand() (default: 5.0)',
    )
    parser.add_argument(
        '--max-n', type=int, default=500,
        help='Training-set cap per outer fold (default: 500)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Subset: 3 datasets, 3 outer folds (1 repeat), 10 trials',
    )
    parser.add_argument(
        '--datasets', nargs='+', default=None,
        help='Subset of datasets to run (default: all 6)',
    )
    args = parser.parse_args()

    if args.quick:
        n_trials     = 10
        inner_cv     = 2
        outer_splits = 3
        outer_repeats = 1
        ds_subset = {k: BENCHMARK_DATASETS[k]
                     for k in ('housing', 'multimodal', 'emergence_divergence')
                     if k in BENCHMARK_DATASETS}
    else:
        n_trials     = args.n_trials
        inner_cv     = args.inner_cv
        outer_splits = args.outer_splits
        outer_repeats = args.outer_repeats
        if args.datasets:
            ds_subset = {k: BENCHMARK_DATASETS[k]
                         for k in args.datasets if k in BENCHMARK_DATASETS}
        else:
            ds_subset = BENCHMARK_DATASETS

    n_outer_folds = outer_splits * outer_repeats

    print()
    print('═' * 80)
    print('  HVRT — HPO Benchmark  (HVRTOptimizer vs. default HVRT)')
    print('═' * 80)
    print(f'  Datasets       : {", ".join(ds_subset)}')
    print(f'  Outer CV       : {outer_splits}-fold × {outer_repeats}-repeat'
          f' = {n_outer_folds} folds per dataset')
    print(f'  HPO trials     : {n_trials} per outer fold  '
          f'(inner-CV = {inner_cv})')
    print(f'  Expansion ratio: {args.expansion_ratio:.0f}×')
    print(f'  max_n (train)  : {args.max_n}')
    print(f'  Seed           : {args.seed}')
    print(f'  Total HPO fits : ~{len(ds_subset) * n_outer_folds * n_trials * inner_cv:,}')
    print('═' * 80)
    print()

    all_results:        dict = {}
    all_param_summaries: dict = {}
    ds_names = list(ds_subset.keys())

    for ds_name, gen_fn in ds_subset.items():
        X_full, y_full, _ = gen_fn(random_state=args.seed)
        is_cls = len(np.unique(y_full)) <= 20
        task_label = 'classification (AUC)' if is_cls else 'regression (R²)'

        print(f'  ─── {ds_name}  [{task_label}]  n={len(X_full)}, d={X_full.shape[1]} ───')

        results = run_dataset(
            X_full, y_full,
            is_cls=is_cls,
            outer_splits=outer_splits,
            outer_repeats=outer_repeats,
            n_trials=n_trials,
            inner_cv=inner_cv,
            expansion_ratio=args.expansion_ratio,
            max_n=args.max_n,
            seed=args.seed,
        )
        all_results[ds_name] = results

        param_summary = _summarise_params(results['hpo_params'])
        all_param_summaries[ds_name] = param_summary

        # Per-fold progress indicator
        n_folds = results['n_folds']
        dm = results['default']['tstr_delta_mean']
        hm = results['hpo']['tstr_delta_mean']
        print(f'    done ({n_folds} folds)  '
              f'default TSTR Δ={dm:+.4f}  hpo TSTR Δ={hm:+.4f}  '
              f'improvement={hm - dm:+.4f}')
        print()

    # ── Detailed per-dataset tables ───────────────────────────────────────────
    print()
    print('═' * 80)
    print('  PER-DATASET RESULTS')
    print('═' * 80)
    print()

    for ds_name in ds_names:
        X_full, y_full, _ = ds_subset[ds_name](random_state=args.seed)
        is_cls = len(np.unique(y_full)) <= 20
        task_label = 'classification (AUC)' if is_cls else 'regression (R²)'
        r = all_results[ds_name]

        print_dataset_comparison_table(
            ds_name, r['n_folds'],
            r['default'], r['hpo'],
            task_label,
        )
        print_best_params_table(ds_name, all_param_summaries[ds_name])

    # ── Cross-dataset summary tables ─────────────────────────────────────────
    print_summary_table(
        ds_names, all_results,
        expansion_ratio=args.expansion_ratio,
        n_trials=n_trials, inner_cv=inner_cv,
    )
    print_all_metrics_summary(ds_names, all_results)

    # ── Interpretive findings ─────────────────────────────────────────────────
    print_findings_narrative(ds_names, all_results, all_param_summaries)


if __name__ == '__main__':
    main()
