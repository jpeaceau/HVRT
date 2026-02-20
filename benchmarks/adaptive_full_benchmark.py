#!/usr/bin/env python3
"""
Adaptive vs Standard HVRT — Full Synthetic Benchmark Suite
============================================================

Compares HVRT-standard (Scott's rule) vs HVRT-adaptive (proportional
bandwidth) across all six synthetic benchmark datasets at expansion
ratios [1, 2, 5, 10, 100].

Each condition uses 5-fold × 3-repeat = 15 evaluations.
Within each fold, TRTR is computed once and shared across both modes
to eliminate evaluation variance between them.

Metrics
-------
  AUC / R²        Classification: ROC-AUC.  Regression: R².
  delta_vs_trtr   TSTR metric − TRTR metric  (positive = augmentation helps)
  coverage_rate   Fraction of TRTR errors that TSTR corrects  (cls only)
  error_set_auc   AUC of TSTR on TRTR's error set  (cls only)

Usage
-----
    python benchmarks/adaptive_full_benchmark.py
    python benchmarks/adaptive_full_benchmark.py --ratios 1 2 5 10 100
    python benchmarks/adaptive_full_benchmark.py --max-n 500 --n-splits 5
"""

import argparse
import io
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

if hasattr(sys.stdout, 'buffer') and \
        getattr(sys.stdout, 'encoding', '').lower().replace('-', '') not in ('utf8', 'utf-8'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )

from hvrt import HVRT
from hvrt.benchmarks.datasets import BENCHMARK_DATASETS
from hvrt.benchmarks.metrics import ml_utility_auc


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _hvrt_expand(X_tr, y_tr, n_synth, seed, adaptive):
    m = HVRT(random_state=seed)
    XY = np.column_stack([X_tr, y_tr.reshape(-1, 1).astype(float)])
    m.fit(XY)
    XY_s = m.expand(n=n_synth, adaptive_bandwidth=adaptive)
    X_s = XY_s[:, :-1]
    y_raw = XY_s[:, -1]

    is_cls = len(np.unique(y_tr)) <= 20
    if is_cls:
        classes = np.unique(y_tr)
        y_s = classes[np.argmin(np.abs(y_raw[:, None] - classes[None, :]), axis=1)]
    else:
        y_s = y_raw
    return X_s, y_s


# ---------------------------------------------------------------------------
# Per-fold evaluation (TRTR computed once, shared across modes)
# ---------------------------------------------------------------------------

def _fold_results(X_tr, y_tr, X_te, y_te, seed, exp_ratio, is_cls):
    """
    Returns dict keyed by mode ('standard', 'adaptive') with per-mode
    TSTR metric and error-explanation stats, plus the shared TRTR metric.
    """
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.metrics import roc_auc_score, r2_score

    n_synth = max(4, int(len(X_tr) * exp_ratio))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # ── TRTR model (shared) ──────────────────────────────────────────
        if is_cls:
            trtr_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            trtr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        trtr_model.fit(X_tr, y_tr)

        if is_cls:
            proba_trtr = trtr_model.predict_proba(X_te)
            y_pred_trtr = trtr_model.predict(X_te)
            if len(trtr_model.classes_) == 2:
                trtr_metric = float(roc_auc_score(y_te, proba_trtr[:, 1]))
            else:
                trtr_metric = float(roc_auc_score(y_te, proba_trtr,
                                                   multi_class='ovr',
                                                   average='weighted'))
            error_mask = (y_pred_trtr != y_te)
        else:
            trtr_metric = float(r2_score(y_te, trtr_model.predict(X_te)))
            error_mask = None

    results = {'trtr': trtr_metric}

    for mode_name, adaptive in (('standard', False), ('adaptive', True)):
        try:
            X_s, y_s = _hvrt_expand(X_tr, y_tr, n_synth, seed, adaptive)
        except Exception:
            results[mode_name] = {
                'tstr': float('nan'),
                'coverage': float('nan'),
                'error_set_auc': float('nan'),
            }
            continue

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            if is_cls:
                tstr_model = GradientBoostingClassifier(
                    n_estimators=100, random_state=42)
            else:
                tstr_model = GradientBoostingRegressor(
                    n_estimators=100, random_state=42)
            tstr_model.fit(X_s, y_s)

            if is_cls:
                proba_tstr = tstr_model.predict_proba(X_te)
                y_pred_tstr = tstr_model.predict(X_te)

                if len(tstr_model.classes_) == 2:
                    tstr_metric = float(roc_auc_score(y_te, proba_tstr[:, 1]))
                else:
                    tstr_metric = float(roc_auc_score(y_te, proba_tstr,
                                                       multi_class='ovr',
                                                       average='weighted'))

                n_errors = int(error_mask.sum())
                if n_errors == 0:
                    coverage = 1.0
                    esa = 1.0
                else:
                    coverage = float(
                        (y_pred_tstr[error_mask] == y_te[error_mask]).mean()
                    )
                    y_err = y_te[error_mask]
                    p_err = proba_tstr[error_mask]
                    if len(np.unique(y_err)) < 2:
                        esa = float('nan')
                    elif len(tstr_model.classes_) == 2:
                        esa = float(roc_auc_score(y_err, p_err[:, 1]))
                    else:
                        esa = float(roc_auc_score(y_err, p_err,
                                                   multi_class='ovr',
                                                   average='weighted'))
            else:
                tstr_metric = float(r2_score(y_te, tstr_model.predict(X_te)))
                coverage = float('nan')
                esa = float('nan')

        results[mode_name] = {
            'tstr': tstr_metric,
            'coverage': coverage,
            'error_set_auc': esa,
        }

    return results


# ---------------------------------------------------------------------------
# Full k-fold run for one (dataset, ratio) condition
# ---------------------------------------------------------------------------

def run_condition(X_full, y_full, exp_ratio, n_splits, n_repeats,
                  random_state, is_cls):
    from sklearn.model_selection import (
        RepeatedStratifiedKFold, RepeatedKFold
    )

    if is_cls:
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        splits = list(cv.split(X_full, y_full))
    else:
        cv = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        splits = list(cv.split(X_full))

    accum = defaultdict(list)

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X_full[train_idx], X_full[test_idx]
        y_tr, y_te = y_full[train_idx], y_full[test_idx]
        seed = random_state + fold_i * 7

        fold_res = _fold_results(X_tr, y_tr, X_te, y_te, seed, exp_ratio, is_cls)
        accum['trtr'].append(fold_res['trtr'])
        for mode in ('standard', 'adaptive'):
            for key in ('tstr', 'coverage', 'error_set_auc'):
                accum[f'{mode}_{key}'].append(fold_res[mode][key])

    def ms(key):
        arr = np.array(accum[key], dtype=float)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    trtr_m, trtr_s = ms('trtr')

    out = {
        'trtr_mean': trtr_m,
        'trtr_std':  trtr_s,
        'n_folds':   len(splits),
    }
    for mode in ('standard', 'adaptive'):
        tm, ts = ms(f'{mode}_tstr')
        cm, cs = ms(f'{mode}_coverage')
        em, es = ms(f'{mode}_error_set_auc')
        out[mode] = {
            'tstr_mean':         tm,
            'tstr_std':          ts,
            'delta':             tm - trtr_m,
            'coverage_mean':     cm,
            'coverage_std':      cs,
            'error_set_auc_mean': em,
            'error_set_auc_std': es,
        }
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(mean, std):
    return f'{mean:+.4f} ±{std:.4f}'


def _print_detailed_table(all_results, n_splits, n_repeats, metric_label):
    """
    all_results: dict {(ds_name, ratio): condition_dict}
    """
    n_total = n_splits * n_repeats
    W = 122

    print()
    print(f'╔{"═" * W}╗')
    title = (f'ADAPTIVE vs STANDARD HVRT — FULL SUITE  '
             f'({n_splits}-fold × {n_repeats}-repeat = {n_total} evals, '
             f'metric: {metric_label})')
    print(f'║  {title:<{W - 3}}║')
    print(f'╠{"═" * W}╣')
    hdr = (f'  {"Dataset":<22}'
           f'{"ratio":>6}'
           f'{"TRTR":>14}'
           f'{"Std delta":>16}'
           f'{"Ada delta":>16}'
           f'{"Δ(Ada−Std)":>12}'
           f'{"Std cover":>12}'
           f'{"Ada cover":>12}'
           f'{"Std err-AUC":>12}'
           f'{"Ada err-AUC":>12}')
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    last_ds = None
    ada_wins_auc = 0
    std_wins_auc = 0
    ada_wins_cov = 0
    std_wins_cov = 0
    n_valid = 0

    for (ds_name, ratio), res in sorted(all_results.items(),
                                         key=lambda x: (x[0][0], x[0][1])):
        if last_ds is not None and ds_name != last_ds:
            print(f'╠{"─" * W}╣')
        last_ds = ds_name

        trtr_s = f'{res["trtr_mean"]:+.4f}'
        std = res['standard']
        ada = res['adaptive']
        diff_delta = ada['delta'] - std['delta']

        # Track wins (ignoring datasets where coverage is NaN = regression)
        n_valid += 1
        if ada['delta'] > std['delta']:
            ada_wins_auc += 1
        elif std['delta'] > ada['delta']:
            std_wins_auc += 1

        cov_valid = not (np.isnan(std['coverage_mean']) or np.isnan(ada['coverage_mean']))
        if cov_valid:
            if ada['coverage_mean'] > std['coverage_mean']:
                ada_wins_cov += 1
            elif std['coverage_mean'] > ada['coverage_mean']:
                std_wins_cov += 1

        def _fcov(m, s):
            if np.isnan(m):
                return f'{"—":>12}'
            return f'{m:>+12.4f}'

        def _fesa(m, s):
            if np.isnan(m):
                return f'{"—":>12}'
            return f'{m:>12.4f}'

        row = (f'  {ds_name:<22}'
               f'{ratio:>5.0f}x'
               f'{trtr_s:>14}'
               f'{std["delta"]:>+16.4f}'
               f'{ada["delta"]:>+16.4f}'
               f'{diff_delta:>+12.4f}'
               f'{_fcov(std["coverage_mean"], std["coverage_std"])}'
               f'{_fcov(ada["coverage_mean"], ada["coverage_std"])}'
               f'{_fesa(std["error_set_auc_mean"], std["error_set_auc_std"])}'
               f'{_fesa(ada["error_set_auc_mean"], ada["error_set_auc_std"])}')
        print(f'║{row}║')

    print(f'╠{"═" * W}╣')
    summary = (f'  Adaptive wins on delta: {ada_wins_auc}/{n_valid}  '
               f'| Standard wins: {std_wins_auc}/{n_valid}  '
               f'| Tied: {n_valid - ada_wins_auc - std_wins_auc}/{n_valid}  '
               f'| Adaptive wins coverage: {ada_wins_cov}  '
               f'| Standard wins coverage: {std_wins_cov}')
    print(f'║{summary:<{W}}║')
    print(f'╚{"═" * W}╝')


def _print_ratio_summary(all_results, ratios):
    """Average delta across all datasets per ratio."""
    W = 88
    print()
    print(f'╔{"═" * W}╗')
    print(f'║  {"SUMMARY — average delta (TSTR − TRTR) across all datasets":<{W - 3}}║')
    print(f'╠{"═" * W}╣')
    hdr = (f'  {"ratio":>6}'
           f'{"Std delta":>16}'
           f'{"Ada delta":>16}'
           f'{"Δ(Ada−Std)":>14}'
           f'{"Ada wins":>10}'
           f'{"Std wins":>10}'
           f'{"Tied":>8}')
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    for ratio in ratios:
        cond_keys = [(ds, ratio) for ds, r in all_results.keys() if r == ratio]
        std_deltas = [all_results[k]['standard']['delta'] for k in cond_keys]
        ada_deltas = [all_results[k]['adaptive']['delta'] for k in cond_keys]
        std_m = np.mean(std_deltas)
        ada_m = np.mean(ada_deltas)
        diff = ada_m - std_m
        n = len(cond_keys)
        ada_wins = sum(a > s for a, s in zip(ada_deltas, std_deltas))
        std_wins = sum(s > a for a, s in zip(ada_deltas, std_deltas))
        row = (f'  {ratio:>5.0f}x'
               f'{std_m:>+16.4f}'
               f'{ada_m:>+16.4f}'
               f'{diff:>+14.4f}'
               f'{ada_wins:>10}/{n}'
               f'{std_wins:>10}/{n}'
               f'{n - ada_wins - std_wins:>8}/{n}')
        print(f'║{row}║')

    print(f'╚{"═" * W}╝')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--ratios',    nargs='+', type=float,
                        default=[1.0, 2.0, 5.0, 10.0, 100.0])
    parser.add_argument('--max-n',     type=int, default=500,
                        help='Training-set cap per dataset (default: 500)')
    parser.add_argument('--n-splits',  type=int, default=5)
    parser.add_argument('--n-repeats', type=int, default=3)
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    n_total = args.n_splits * args.n_repeats
    datasets = list(BENCHMARK_DATASETS.keys())

    print()
    print('Adaptive vs Standard HVRT — Full Synthetic Suite')
    print(f'  Datasets      : {", ".join(datasets)}')
    print(f'  Ratios        : {args.ratios}')
    print(f'  max_n (train) : {args.max_n}')
    print(f'  CV design     : {args.n_splits}-fold × {args.n_repeats}-repeat = {n_total} evals')
    print()

    all_results = {}
    cls_datasets = []
    reg_datasets = []

    for ds_name, gen_fn in BENCHMARK_DATASETS.items():
        X, y, _ = gen_fn(random_state=args.seed)
        if args.max_n and len(X) > args.max_n:
            X, y = X[:args.max_n], y[:args.max_n]

        is_cls = len(np.unique(y)) <= 20
        (cls_datasets if is_cls else reg_datasets).append(ds_name)
        task_label = 'cls (AUC)' if is_cls else 'reg (R²)'

        for ratio in args.ratios:
            print(f'  [{ds_name:<22}] {ratio:>5.0f}x  {task_label} ...',
                  end=' ', flush=True)
            res = run_condition(
                X, y, ratio,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                random_state=args.seed,
                is_cls=is_cls,
            )
            all_results[(ds_name, ratio)] = res
            std = res['standard']
            ada = res['adaptive']
            print(f'TRTR={res["trtr_mean"]:+.4f}  '
                  f'Std Δ={std["delta"]:+.4f}  '
                  f'Ada Δ={ada["delta"]:+.4f}  '
                  f'Δ(A-S)={ada["delta"] - std["delta"]:+.4f}')

    # Separate tables for classification and regression
    metric_label = 'ROC-AUC (cls) / R² (reg)'

    print()
    print('── Classification datasets (' + ', '.join(cls_datasets) + ')')
    cls_results = {k: v for k, v in all_results.items() if k[0] in cls_datasets}
    _print_detailed_table(cls_results, args.n_splits, args.n_repeats, 'ROC-AUC')

    print()
    print('── Regression datasets (' + ', '.join(reg_datasets) + ')')
    reg_results = {k: v for k, v in all_results.items() if k[0] in reg_datasets}
    _print_detailed_table(reg_results, args.n_splits, args.n_repeats, 'R²')

    print()
    print('── Summary across ALL datasets')
    _print_ratio_summary(all_results, args.ratios)


if __name__ == '__main__':
    main()
