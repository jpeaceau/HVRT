#!/usr/bin/env python3
"""
Heart Disease Dataset Benchmark
================================

Full method comparison (reduction + expansion) on the real-world
Heart Disease Prediction dataset (n=270, d=13, binary target).

ML utility is evaluated with a proper repeated k-fold design:
  - 5-fold cross-validation, repeated 3 times (15 evaluations per method)
  - TRTR (Train on Real, Test on Real) baseline computed on the same folds
  - TSTR (Train on Synthetic/Reduced, Test on Real) computed per method
  - Results reported as mean ± std and delta = TSTR − TRTR

All distribution-fidelity metrics (marginal_f, correlation_f, tail,
discriminator, DCR) are computed on a fixed 80/20 holdout for speed.

Usage
-----
    python benchmarks/heart_disease_benchmark.py
    python benchmarks/heart_disease_benchmark.py --tasks expand
    python benchmarks/heart_disease_benchmark.py --tasks reduce
    python benchmarks/heart_disease_benchmark.py --no-deep-learning
    python benchmarks/heart_disease_benchmark.py --n-splits 5 --n-repeats 3
"""

import argparse
import io
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# UTF-8 stdout for Windows (cp1252 cannot encode box-drawing characters)
if hasattr(sys.stdout, 'buffer') and \
        getattr(sys.stdout, 'encoding', '').lower().replace('-', '') not in ('utf8', 'utf-8'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )

from hvrt import HVRT, FastHVRT
from hvrt.benchmarks.metrics import (
    evaluate_reduction, evaluate_expansion, ml_utility_tstr,
)
from hvrt.benchmarks.runners import (
    _gmm_expand,
    _gaussian_copula_expand,
    _bootstrap_noise_expand,
    _smote_expand,
    _ctgan_expand,
    _tvae_expand,
    _kennard_stone,
    _qr_pivot,
    _stratified_reduce,
)

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_CSV_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv'
)


def load_heart_disease():
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required.  pip install pandas")

    csv_path = os.path.abspath(_CSV_PATH)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    target_col = 'Heart Disease'
    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].values.astype(float)
    y = (df[target_col].str.strip() == 'Presence').astype(int).values
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Reduction helper — returns the reduced subset (not metrics)
# ---------------------------------------------------------------------------

def _get_reduced(X_tr, y_tr, method, ratio, seed):
    n_target = max(2, int(len(X_tr) * ratio))

    if method in ('HVRT-size', 'HVRT-var', 'FastHVRT-size', 'FastHVRT-var'):
        Cls = HVRT if method.startswith('HVRT-') else FastHVRT
        var_w = method.endswith('-var')
        m = Cls(random_state=seed)
        m.fit(X_tr, y_tr)
        X_red, idx = m.reduce(n=n_target, variance_weighted=var_w, return_indices=True)
        return X_red, y_tr[idx]

    if method == 'Kennard-Stone':
        _KS_MAX = 5_000
        if len(X_tr) > _KS_MAX:
            rng = np.random.RandomState(seed)
            sub = rng.choice(len(X_tr), _KS_MAX, replace=False)
            n_sub = max(2, int(_KS_MAX * ratio))
            ks_idx = _kennard_stone(X_tr[sub], n_sub)
            idx = sub[ks_idx]
        else:
            idx = _kennard_stone(X_tr, n_target)
        return X_tr[idx], y_tr[idx]

    if method == 'QR-Pivot':
        idx = _qr_pivot(X_tr, n_target)
        return X_tr[idx], y_tr[idx]

    if method == 'Stratified':
        idx = _stratified_reduce(X_tr, y_tr, n_target, seed)
        return X_tr[idx], y_tr[idx]

    if method == 'Random':
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_tr), n_target, replace=False)
        return X_tr[idx], y_tr[idx]

    raise ValueError(f"Unknown reduction method: {method!r}")


# ---------------------------------------------------------------------------
# Expansion helper — generates (X_synth, y_synth) from a training fold
# dl_epochs controls CTGAN/TVAE training length; use fewer epochs for k-fold
# ---------------------------------------------------------------------------

def _get_expanded(X_tr, y_tr, method, n_synth, seed, dl_epochs=100):
    n_synth = max(4, n_synth)

    if method in ('HVRT-size', 'HVRT-var', 'FastHVRT-size', 'FastHVRT-var'):
        Cls = HVRT if method.startswith('HVRT-') else FastHVRT
        var_w = method.endswith('-var')
        m = Cls(random_state=seed)
        XY = np.column_stack([X_tr, y_tr.reshape(-1, 1).astype(float)])
        m.fit(XY)
        XY_synth = m.expand(n=n_synth, variance_weighted=var_w)
        X_synth = XY_synth[:, :-1]
        y_raw = XY_synth[:, -1]
        classes = np.unique(y_tr)
        y_synth = classes[np.argmin(np.abs(y_raw[:, None] - classes[None, :]), axis=1)]
        return X_synth, y_synth

    if method == 'GMM':
        X_synth = _gmm_expand(X_tr, n_synth, seed)
    elif method == 'Gaussian-Copula':
        X_synth = _gaussian_copula_expand(X_tr, n_synth, seed)
    elif method == 'Bootstrap-Noise':
        X_synth = _bootstrap_noise_expand(X_tr, n_synth, random_state=seed)
    elif method == 'SMOTE':
        X_synth = _smote_expand(X_tr, n_synth, random_state=seed)
    elif method == 'CTGAN':
        X_synth = _ctgan_expand(X_tr, n_synth, epochs=dl_epochs, random_state=seed)
    elif method == 'TVAE':
        X_synth = _tvae_expand(X_tr, n_synth, epochs=dl_epochs, random_state=seed)
    else:
        raise ValueError(f"Unknown expansion method: {method!r}")

    from sklearn.ensemble import GradientBoostingClassifier
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        proxy = GradientBoostingClassifier(n_estimators=50, random_state=42)
        proxy.fit(X_tr, y_tr)
    y_synth = proxy.predict(X_synth)
    return X_synth, y_synth


# ---------------------------------------------------------------------------
# K-fold ML utility: 5-fold × 3-repeat for TRTR and TSTR
# ---------------------------------------------------------------------------

def kfold_expansion_ml(X, y, method, exp_ratio, n_splits, n_repeats, random_state,
                        dl_epochs=100):
    """
    Proper repeated k-fold TSTR vs TRTR comparison for expansion.

    For each fold:
      - TRTR: train GBM on real training fold, test on real test fold
      - TSTR: generate synthetic from real training fold (to avoid leakage),
              train GBM on synthetic, test on real test fold

    Returns dict with trtr_mean, trtr_std, tstr_mean, tstr_std, delta_mean.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    trtr_list, tstr_list = [], []

    for fold_i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        seed = random_state + fold_i * 7

        # TRTR
        trtr_list.append(ml_utility_tstr(X_tr, y_tr, X_te, y_te))

        # TSTR
        n_synth = max(4, int(len(X_tr) * exp_ratio))
        try:
            X_synth, y_synth = _get_expanded(
                X_tr, y_tr, method, n_synth, seed, dl_epochs=dl_epochs
            )
            tstr_list.append(ml_utility_tstr(X_synth, y_synth, X_te, y_te))
        except Exception:
            tstr_list.append(float('nan'))

    trtr_arr = np.array(trtr_list)
    tstr_arr = np.array(tstr_list)
    return {
        'trtr_mean': float(np.nanmean(trtr_arr)),
        'trtr_std':  float(np.nanstd(trtr_arr)),
        'tstr_mean': float(np.nanmean(tstr_arr)),
        'tstr_std':  float(np.nanstd(tstr_arr)),
        'delta_mean': float(np.nanmean(tstr_arr) - np.nanmean(trtr_arr)),
        'n_folds': int(np.sum(~np.isnan(tstr_arr))),
    }


def kfold_reduction_ml(X, y, method, ratio, n_splits, n_repeats, random_state):
    """
    Proper repeated k-fold TSTR vs TRTR comparison for reduction.

    For each fold the training fold is reduced by `ratio`, then a GBM is
    trained on the reduced subset and tested on the real test fold.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    trtr_list, tstr_list = [], []

    for fold_i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        seed = random_state + fold_i * 7

        # TRTR
        trtr_list.append(ml_utility_tstr(X_tr, y_tr, X_te, y_te))

        # TSTR (reduction)
        try:
            X_red, y_red = _get_reduced(X_tr, y_tr, method, ratio, seed)
            tstr_list.append(ml_utility_tstr(X_red, y_red, X_te, y_te))
        except Exception:
            tstr_list.append(float('nan'))

    trtr_arr = np.array(trtr_list)
    tstr_arr = np.array(tstr_list)
    return {
        'trtr_mean': float(np.nanmean(trtr_arr)),
        'trtr_std':  float(np.nanstd(trtr_arr)),
        'tstr_mean': float(np.nanmean(tstr_arr)),
        'tstr_std':  float(np.nanstd(tstr_arr)),
        'delta_mean': float(np.nanmean(tstr_arr) - np.nanmean(trtr_arr)),
        'n_folds': int(np.sum(~np.isnan(tstr_arr))),
    }


# ---------------------------------------------------------------------------
# Single-split distribution fidelity run (for mf, corr_f, tail, discrim, dcr)
# ---------------------------------------------------------------------------

def _run_expansion_fixed(X_train, y_train, X_test, y_test,
                          method, exp_ratio, seed, dl_epochs=300):
    n_synth = int(len(X_train) * exp_ratio)
    t0 = time.perf_counter()
    try:
        X_synth, y_synth = _get_expanded(
            X_train, y_train, method, n_synth, seed, dl_epochs=dl_epochs
        )
    except Exception as exc:
        return None, str(exc)
    elapsed = time.perf_counter() - t0
    metrics = evaluate_expansion(X_train, y_train, X_synth, y_synth, X_test, y_test)
    metrics['time_s'] = round(elapsed, 4)
    return metrics, None


def _run_reduction_fixed(X_train, y_train, X_test, y_test,
                          method, ratio, seed):
    t0 = time.perf_counter()
    try:
        X_red, y_red = _get_reduced(X_train, y_train, method, ratio, seed)
    except Exception as exc:
        return None, str(exc)
    elapsed = time.perf_counter() - t0
    metrics = evaluate_reduction(X_train, y_train, X_red, y_red, X_test, y_test)
    metrics['time_s'] = round(elapsed, 4)
    return metrics, None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_ms(mean, std):
    """Format mean ± std to a fixed-width string."""
    return f'{mean:.4f} ±{std:.4f}'


def _print_fidelity_table(rows, task):
    """
    Distribution fidelity table (single fixed split).
    rows: list of (method, param, metrics_dict or None, error_str)
    """
    if task == 'expand':
        W = 112
        hdr = (f'  {"Method":<22}'
               f'{"ratio":>7}'
               f'{"n_synth":>8}'
               f'{"marginal_f":>12}'
               f'{"corr_f":>8}'
               f'{"tail":>8}'
               f'{"discrim":>9}'
               f'{"dcr":>8}'
               f'{"time_s":>8}')
        title = 'HEART DISEASE — EXPANSION   Distribution Fidelity  (fixed 80/20 split)'
    else:
        W = 100
        hdr = (f'  {"Method":<22}'
               f'{"ratio":>6}'
               f'{"n_red":>7}'
               f'{"marginal_f":>12}'
               f'{"corr_f":>8}'
               f'{"tail":>8}'
               f'{"time_s":>8}')
        title = 'HEART DISEASE — REDUCTION   Distribution Fidelity  (fixed 80/20 split)'

    print()
    print(f'╔{"═" * W}╗')
    print(f'║  {title:<{W - 3}}║')
    print(f'╠{"═" * W}╣')
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    last_param = None
    for method, param, n_train, metrics, err in rows:
        if last_param is not None and param != last_param:
            print(f'╠{"─" * W}╣')
        last_param = param

        if err or metrics is None:
            tag = f'  {method:<22} {"ERROR: " + (err or "")}'
            print(f'║{tag:<{W}}║')
            continue

        if task == 'expand':
            n_s = int(n_train * param)
            row = (f'  {method:<22}'
                   f'{param:>6.0f}x'
                   f'{n_s:>8}'
                   f'{metrics.get("marginal_fidelity", float("nan")):>12.4f}'
                   f'{metrics.get("correlation_fidelity", float("nan")):>8.4f}'
                   f'{metrics.get("tail_preservation", float("nan")):>8.4f}'
                   f'{metrics.get("discriminator_accuracy", float("nan")):>9.4f}'
                   f'{metrics.get("privacy_dcr", float("nan")):>8.4f}'
                   f'{metrics.get("time_s", float("nan")):>8.3f}')
        else:
            n_r = max(2, int(n_train * param))
            row = (f'  {method:<22}'
                   f'{param:>6.0%}'
                   f'{n_r:>7}'
                   f'{metrics.get("marginal_fidelity", float("nan")):>12.4f}'
                   f'{metrics.get("correlation_fidelity", float("nan")):>8.4f}'
                   f'{metrics.get("tail_preservation", float("nan")):>8.4f}'
                   f'{metrics.get("time_s", float("nan")):>8.3f}')
        print(f'║{row}║')

    print(f'╚{"═" * W}╝')


def _print_ml_utility_table(rows, task, n_splits, n_repeats):
    """
    ML utility table with k-fold mean ± std.
    rows: list of (method, param, n_train, cv_dict or None, error_str)
    """
    W = 102
    n_folds_total = n_splits * n_repeats
    if task == 'expand':
        title = (f'HEART DISEASE — EXPANSION   ML Utility  '
                 f'(TRTR vs TSTR, {n_splits}-fold × {n_repeats}-repeat = {n_folds_total} evals)')
        param_hdr = f'{"ratio":>7}'
    else:
        title = (f'HEART DISEASE — REDUCTION   ML Utility  '
                 f'(TRTR vs TSTR, {n_splits}-fold × {n_repeats}-repeat = {n_folds_total} evals)')
        param_hdr = f'{"ratio":>6}'

    hdr = (f'  {"Method":<22}'
           f'{param_hdr}'
           f'{"TRTR mean±std":>22}'
           f'{"TSTR mean±std":>22}'
           f'{"delta":>9}'
           f'{"folds":>6}')

    print()
    print(f'╔{"═" * W}╗')
    print(f'║  {title:<{W - 3}}║')
    print(f'╠{"═" * W}╣')
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    last_param = None
    for method, param, n_train, cv, err in rows:
        if last_param is not None and param != last_param:
            print(f'╠{"─" * W}╣')
        last_param = param

        if err or cv is None:
            tag = f'  {method:<22} {"ERROR: " + (err or "")}'
            print(f'║{tag:<{W}}║')
            continue

        trtr_s = _fmt_ms(cv['trtr_mean'], cv['trtr_std'])
        tstr_s = _fmt_ms(cv['tstr_mean'], cv['tstr_std'])
        delta  = cv['delta_mean']
        nf     = cv['n_folds']

        if task == 'expand':
            param_s = f'{param:>6.0f}x'
        else:
            param_s = f'{param:>6.0%}'

        row = (f'  {method:<22}'
               f'{param_s}'
               f'{trtr_s:>22}'
               f'{tstr_s:>22}'
               f'{delta:>+9.4f}'
               f'{nf:>6}')
        print(f'║{row}║')

    print(f'╚{"═" * W}╝')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Heart Disease dataset benchmark — all methods with k-fold ML utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--tasks', nargs='+', choices=['reduce', 'expand'], default=['reduce', 'expand'],
    )
    parser.add_argument(
        '--no-deep-learning', action='store_true',
        help='Skip CTGAN and TVAE',
    )
    parser.add_argument('--n-splits',  type=int, default=5,
                        help='k in k-fold CV (default: 5)')
    parser.add_argument('--n-repeats', type=int, default=3,
                        help='Repetitions of k-fold (default: 3)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    n_folds_total = args.n_splits * args.n_repeats

    print()
    print('Heart Disease Benchmark')
    print(f'  CV design      : {args.n_splits}-fold × {args.n_repeats}-repeat = {n_folds_total} evaluations')
    print('  Loading data/Heart_Disease_Prediction.csv ...')
    X, y, feat_names = load_heart_disease()
    print(f'  Dataset shape  : {X.shape}  (features: {len(feat_names)})')
    print(f'  Target balance : {y.mean():.1%} Presence  /  {1 - y.mean():.1%} Absence')

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.seed, stratify=y,
    )
    print(f'  Train / Test   : {len(X_train)} / {len(X_test)} samples  (fixed split for fidelity metrics)')
    print()

    # ---- REDUCTION ----------------------------------------------------------
    if 'reduce' in args.tasks:
        REDUCE_METHODS = [
            'HVRT-size', 'HVRT-var', 'FastHVRT-size', 'FastHVRT-var',
            'Kennard-Stone', 'QR-Pivot', 'Stratified', 'Random',
        ]
        RATIOS = [0.7, 0.5, 0.3]

        fidelity_rows = []
        ml_rows = []

        print('── REDUCTION ─────────────────────────────────────────────────')
        for ratio in RATIOS:
            for method in REDUCE_METHODS:
                # Fixed-split fidelity metrics
                metrics, err = _run_reduction_fixed(
                    X_train, y_train, X_test, y_test, method, ratio, args.seed
                )
                fidelity_rows.append((method, ratio, len(X_train), metrics, err))

                # K-fold ML utility
                print(f'  [{ratio:.0%}] {method}  k-fold ...', end=' ', flush=True)
                try:
                    cv = kfold_reduction_ml(
                        X, y, method, ratio,
                        n_splits=args.n_splits,
                        n_repeats=args.n_repeats,
                        random_state=args.seed,
                    )
                    ml_rows.append((method, ratio, len(X_train), cv, None))
                    print(f'TRTR={cv["trtr_mean"]:.4f}±{cv["trtr_std"]:.4f}'
                          f'  TSTR={cv["tstr_mean"]:.4f}±{cv["tstr_std"]:.4f}'
                          f'  delta={cv["delta_mean"]:+.4f}')
                except Exception as exc:
                    ml_rows.append((method, ratio, len(X_train), None, str(exc)))
                    print(f'ERROR: {exc}')

        _print_fidelity_table(fidelity_rows, task='reduce')
        _print_ml_utility_table(ml_rows, task='reduce',
                                 n_splits=args.n_splits, n_repeats=args.n_repeats)

    # ---- EXPANSION ----------------------------------------------------------
    if 'expand' in args.tasks:
        EXPAND_METHODS = [
            'HVRT-size', 'HVRT-var', 'FastHVRT-size', 'FastHVRT-var',
            'GMM', 'Gaussian-Copula', 'Bootstrap-Noise', 'SMOTE',
        ]
        if not args.no_deep_learning:
            EXPAND_METHODS += ['CTGAN', 'TVAE']

        EXP_RATIOS = [1.0, 2.0, 5.0]
        DL_EPOCHS_FIXED = 300   # single-eval (best quality)
        DL_EPOCHS_CV    = 100   # k-fold (speed / quality trade-off)

        fidelity_rows = []
        ml_rows = []

        print()
        print('── EXPANSION ─────────────────────────────────────────────────')
        for exp_ratio in EXP_RATIOS:
            for method in EXPAND_METHODS:
                dl_ep = DL_EPOCHS_FIXED if method in ('CTGAN', 'TVAE') else 300

                # Fixed-split fidelity metrics
                metrics, err = _run_expansion_fixed(
                    X_train, y_train, X_test, y_test,
                    method, exp_ratio, args.seed, dl_epochs=dl_ep,
                )
                if err:
                    print(f'  [{exp_ratio:.0f}x] {method}  fidelity ERROR: {err}')
                fidelity_rows.append((method, exp_ratio, len(X_train), metrics, err))

                # K-fold ML utility
                print(f'  [{exp_ratio:.0f}x] {method}  k-fold ...', end=' ', flush=True)
                try:
                    cv = kfold_expansion_ml(
                        X, y, method, exp_ratio,
                        n_splits=args.n_splits,
                        n_repeats=args.n_repeats,
                        random_state=args.seed,
                        dl_epochs=DL_EPOCHS_CV,
                    )
                    ml_rows.append((method, exp_ratio, len(X_train), cv, None))
                    print(f'TRTR={cv["trtr_mean"]:.4f}±{cv["trtr_std"]:.4f}'
                          f'  TSTR={cv["tstr_mean"]:.4f}±{cv["tstr_std"]:.4f}'
                          f'  delta={cv["delta_mean"]:+.4f}')
                except Exception as exc:
                    ml_rows.append((method, exp_ratio, len(X_train), None, str(exc)))
                    print(f'ERROR: {exc}')

        _print_fidelity_table(fidelity_rows, task='expand')
        _print_ml_utility_table(ml_rows, task='expand',
                                 n_splits=args.n_splits, n_repeats=args.n_repeats)


if __name__ == '__main__':
    main()
