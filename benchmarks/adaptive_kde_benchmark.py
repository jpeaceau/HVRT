#!/usr/bin/env python3
"""
Adaptive KDE Bandwidth Benchmark
==================================

Compares two HVRT expansion modes on the Heart Disease dataset:

  HVRT-standard  — existing Scott's rule bandwidth (fixed, independent of
                   expansion ratio)
  HVRT-adaptive  — per-partition bandwidth that scales proportionally with
                   both partition size and the local expansion factor:

                       bw_p = scott_p × max(1, budget_p/n_p)^(1/d)

                   At 1× both modes are identical.  At higher ratios the
                   adaptive mode explores proportionally further from the
                   observed data to avoid clumping.

Metrics (all via 5-fold × 3-repeat = 15 evaluations)
------------------------------------------------------
  AUC (TSTR)         ROC-AUC of a GBM trained on synthetic data, tested on
                     real held-out data.  Replaces F1 for interpretability.
  TRTR AUC           Baseline: same GBM trained on real data.
  delta AUC          TSTR − TRTR  (positive = synthetic augmentation helps)
  Coverage rate      Of the test cases the real model mis-classifies, fraction
                     that the synthetic model predicts correctly.
  Error-set AUC      ROC-AUC of the synthetic model specifically on the hard
                     cases (samples the real model fails on).

Expansion ratios tested: 1, 2, 5, 10, 100

Usage
-----
    python benchmarks/adaptive_kde_benchmark.py
    python benchmarks/adaptive_kde_benchmark.py --n-splits 5 --n-repeats 3
"""

import argparse
import io
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

if hasattr(sys.stdout, 'buffer') and \
        getattr(sys.stdout, 'encoding', '').lower().replace('-', '') not in ('utf8', 'utf-8'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )

from hvrt import HVRT
from hvrt.benchmarks.metrics import ml_utility_auc, error_explanation_rate

_CSV_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv'
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_heart_disease():
    import pandas as pd
    df = pd.read_csv(os.path.abspath(_CSV_PATH))
    target_col = 'Heart Disease'
    feat_names = [c for c in df.columns if c != target_col]
    X = df[feat_names].values.astype(float)
    y = (df[target_col].str.strip() == 'Presence').astype(int).values
    return X, y


# ---------------------------------------------------------------------------
# Generator wrappers
# ---------------------------------------------------------------------------

def _expand_standard(X_tr, y_tr, n_synth, seed):
    m = HVRT(random_state=seed)
    XY = np.column_stack([X_tr, y_tr.reshape(-1, 1).astype(float)])
    m.fit(XY)
    XY_s = m.expand(n=n_synth, adaptive_bandwidth=False)
    X_s = XY_s[:, :-1]
    y_raw = XY_s[:, -1]
    classes = np.unique(y_tr)
    y_s = classes[np.argmin(np.abs(y_raw[:, None] - classes[None, :]), axis=1)]
    return X_s, y_s


def _expand_adaptive(X_tr, y_tr, n_synth, seed):
    m = HVRT(random_state=seed)
    XY = np.column_stack([X_tr, y_tr.reshape(-1, 1).astype(float)])
    m.fit(XY)
    XY_s = m.expand(n=n_synth, adaptive_bandwidth=True)
    X_s = XY_s[:, :-1]
    y_raw = XY_s[:, -1]
    classes = np.unique(y_tr)
    y_s = classes[np.argmin(np.abs(y_raw[:, None] - classes[None, :]), axis=1)]
    return X_s, y_s


# ---------------------------------------------------------------------------
# K-fold evaluation
# ---------------------------------------------------------------------------

def kfold_evaluate(X, y, generator_fn, exp_ratio, n_splits, n_repeats, random_state):
    """
    5-fold × n_repeats repeated k-fold.

    For each fold:
      - TRTR AUC   : GBM on real train fold → AUC on real test fold
      - TSTR AUC   : GBM on synthetic (from train fold) → AUC on real test fold
      - Error explanation: coverage_rate and error_set_auc on TRTR error set

    Returns aggregated dict of mean/std for each metric.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    trtr_aucs, tstr_aucs = [], []
    coverage_rates, error_set_aucs = [], []

    for fold_i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        seed = random_state + fold_i * 7

        # TRTR AUC
        trtr_aucs.append(ml_utility_auc(X_tr, y_tr, X_te, y_te))

        # Generate synthetic from this fold's training data
        n_synth = max(4, int(len(X_tr) * exp_ratio))
        try:
            X_s, y_s = generator_fn(X_tr, y_tr, n_synth, seed)
        except Exception:
            tstr_aucs.append(float('nan'))
            coverage_rates.append(float('nan'))
            error_set_aucs.append(float('nan'))
            continue

        # TSTR AUC
        tstr_aucs.append(ml_utility_auc(X_s, y_s, X_te, y_te))

        # Error explanation
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            er = error_explanation_rate(X_tr, y_tr, X_s, y_s, X_te, y_te)
        coverage_rates.append(er['coverage_rate'])
        error_set_aucs.append(er['error_set_auc'])

    def _ms(arr):
        a = np.array(arr, dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a))

    trtr_m, trtr_s   = _ms(trtr_aucs)
    tstr_m, tstr_s   = _ms(tstr_aucs)
    cov_m,  cov_s    = _ms(coverage_rates)
    esa_m,  esa_s    = _ms(error_set_aucs)

    return {
        'trtr_auc_mean':      trtr_m,
        'trtr_auc_std':       trtr_s,
        'tstr_auc_mean':      tstr_m,
        'tstr_auc_std':       tstr_s,
        'delta_auc':          tstr_m - trtr_m,
        'coverage_mean':      cov_m,
        'coverage_std':       cov_s,
        'error_set_auc_mean': esa_m,
        'error_set_auc_std':  esa_s,
        'n_folds':            int(np.sum(~np.isnan(tstr_aucs))),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(mean, std):
    return f'{mean:.4f} ±{std:.4f}'


def _print_comparison_table(results, n_splits, n_repeats):
    """
    results: list of (mode_name, exp_ratio, cv_dict)
    """
    n_total = n_splits * n_repeats
    W = 116
    title = (f'ADAPTIVE KDE BENCHMARK  (Heart Disease, '
             f'{n_splits}-fold × {n_repeats}-repeat = {n_total} evals)')
    hdr = (f'  {"Mode":<18}'
           f'{"ratio":>6}'
           f'{"TRTR AUC":>22}'
           f'{"TSTR AUC":>22}'
           f'{"delta":>8}'
           f'{"coverage":>20}'
           f'{"error-set AUC":>20}')

    print()
    print(f'╔{"═" * W}╗')
    print(f'║  {title:<{W - 3}}║')
    print(f'╠{"═" * W}╣')
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    last_ratio = None
    for mode, exp_ratio, cv in results:
        if last_ratio is not None and exp_ratio != last_ratio:
            print(f'╠{"─" * W}╣')
        last_ratio = exp_ratio

        row = (f'  {mode:<18}'
               f'{exp_ratio:>5.0f}x'
               f'{_fmt(cv["trtr_auc_mean"], cv["trtr_auc_std"]):>22}'
               f'{_fmt(cv["tstr_auc_mean"], cv["tstr_auc_std"]):>22}'
               f'{cv["delta_auc"]:>+8.4f}'
               f'{_fmt(cv["coverage_mean"], cv["coverage_std"]):>20}'
               f'{_fmt(cv["error_set_auc_mean"], cv["error_set_auc_std"]):>20}')
        print(f'║{row}║')

    print(f'╚{"═" * W}╝')

    # Legend
    print()
    print('  Columns')
    print('  ───────')
    print('  TRTR AUC      : AUC of GBM trained on REAL data  (same for both modes at same ratio)')
    print('  TSTR AUC      : AUC of GBM trained on SYNTHETIC data, tested on real held-out data')
    print('  delta         : TSTR AUC − TRTR AUC  (positive = synthetic augmentation helps)')
    print('  coverage      : of cases TRTR mis-classifies, fraction TSTR gets right')
    print('  error-set AUC : ROC-AUC of TSTR model specifically on TRTR\'s error set')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Adaptive KDE bandwidth comparison on Heart Disease',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--n-splits',  type=int, default=5)
    parser.add_argument('--n-repeats', type=int, default=3)
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    print()
    print('Adaptive KDE Bandwidth Benchmark')
    print(f'  CV design : {args.n_splits}-fold × {args.n_repeats}-repeat = '
          f'{args.n_splits * args.n_repeats} evaluations per condition')
    print('  Loading Heart_Disease_Prediction.csv ...')
    X, y = load_heart_disease()
    print(f'  Dataset   : {X.shape[0]} samples, {X.shape[1]} features, '
          f'binary target ({y.mean():.1%} Presence)')
    print()

    MODES = [
        ('HVRT-standard', _expand_standard),
        ('HVRT-adaptive', _expand_adaptive),
    ]
    EXP_RATIOS = [1, 2, 5, 10, 100]

    results = []

    for exp_ratio in EXP_RATIOS:
        for mode_name, gen_fn in MODES:
            print(f'  [{exp_ratio:>3}x] {mode_name} ...', end=' ', flush=True)
            cv = kfold_evaluate(
                X, y, gen_fn, exp_ratio,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                random_state=args.seed,
            )
            results.append((mode_name, exp_ratio, cv))
            print(
                f'TSTR={cv["tstr_auc_mean"]:.4f}±{cv["tstr_auc_std"]:.4f}'
                f'  delta={cv["delta_auc"]:+.4f}'
                f'  cov={cv["coverage_mean"]:.3f}'
                f'  err_auc={cv["error_set_auc_mean"]:.4f}'
            )

    _print_comparison_table(results, args.n_splits, args.n_repeats)


if __name__ == '__main__':
    main()
