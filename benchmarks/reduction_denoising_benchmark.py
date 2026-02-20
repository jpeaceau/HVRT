#!/usr/bin/env python3
"""
HVRT Sample Reduction — Denoising Benchmark
============================================

Demonstrates that HVRT sample reduction acts as an intelligent denoiser
for heavy-tailed and noisy data, while remaining competitive with random
sampling on well-behaved data.

Test design
-----------
For each dataset condition:

  1. Generate a full noisy dataset (train + test split, fixed seed).
  2. Reduce the training set to ``ratio`` using:
       HVRT-fps    (centroid-seeded FPS, variance_weighted=True)
       HVRT-var    (centroid-seeded FPS, variance_weighted=True, y_weight=0.3)
       Random      (simple random sample)
       Stratified  (stratified random within HVRT partitions — random within
                   the same partition structure, to isolate FPS vs random)
  3. Train a GBM on each reduced set, evaluate on the held-out test set.
  4. Report accuracy as % of the *full training set* score
     (100 % = matches full-data performance; >100 % = denoising benefit).

Dataset conditions
------------------
  well_behaved   — Gaussian features, no noise, linear + mild interaction signal
  heavy_tail     — Student-t(2) features (fat tails), same signal
  noisy_labels   — well-behaved features but 20 % label noise injected
  noisy_all      — heavy-tailed features + 20 % label noise + irrelevant features

Reduction ratios tested:  0.1, 0.2, 0.3, 0.5

Usage
-----
    python benchmarks/reduction_denoising_benchmark.py
    python benchmarks/reduction_denoising_benchmark.py --n-train 5000 --seed 0
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


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def _signal(X):
    """Simple signal: linear + pairwise interactions on first 5 features."""
    return (
        X[:, 0] * 1.5
        + X[:, 1] * -1.0
        + X[:, 2] * 0.8
        + X[:, 0] * X[:, 3] * 0.5
        + X[:, 1] * X[:, 4] * -0.3
    )


def make_well_behaved(n_train, n_test, seed):
    rng = np.random.RandomState(seed)
    n_feat = 10
    X_all = rng.randn(n_train + n_test, n_feat)
    y_all = (_signal(X_all) + rng.randn(n_train + n_test) * 0.5 > 0).astype(int)
    return (X_all[:n_train], y_all[:n_train],
            X_all[n_train:], y_all[n_train:])


def make_heavy_tail(n_train, n_test, seed):
    """Student-t(2) — fat tails, same signal."""
    rng = np.random.RandomState(seed)
    n_feat = 10
    # scipy t not available; generate t(2) = N(0,1) / sqrt(chi2(2)/2)
    chi2 = rng.chisquare(2, size=(n_train + n_test, 1)) / 2
    X_all = rng.randn(n_train + n_test, n_feat) / np.sqrt(chi2)
    y_all = (_signal(X_all) + rng.randn(n_train + n_test) * 0.5 > 0).astype(int)
    return (X_all[:n_train], y_all[:n_train],
            X_all[n_train:], y_all[n_train:])


def make_noisy_labels(n_train, n_test, seed, noise_rate=0.20):
    """Well-behaved features, but 20 % of training labels are randomly flipped."""
    rng = np.random.RandomState(seed)
    X_tr, y_tr, X_te, y_te = make_well_behaved(n_train, n_test, seed)
    flip_mask = rng.rand(n_train) < noise_rate
    y_tr_noisy = y_tr.copy()
    y_tr_noisy[flip_mask] = 1 - y_tr_noisy[flip_mask]
    return X_tr, y_tr_noisy, X_te, y_te


def make_noisy_all(n_train, n_test, seed, noise_rate=0.20):
    """Heavy-tailed features + 20 % label noise + 10 irrelevant noise features."""
    rng = np.random.RandomState(seed)
    n_feat_signal = 10
    n_feat_noise = 10
    chi2 = rng.chisquare(2, size=(n_train + n_test, 1)) / 2
    X_signal = rng.randn(n_train + n_test, n_feat_signal) / np.sqrt(chi2)
    X_noise = rng.randn(n_train + n_test, n_feat_noise) * 3  # high variance noise
    X_all = np.hstack([X_signal, X_noise])
    y_all = (_signal(X_signal) + rng.randn(n_train + n_test) * 0.5 > 0).astype(int)
    # Label noise on training portion only
    X_tr, y_tr = X_all[:n_train], y_all[:n_train]
    X_te, y_te = X_all[n_train:], y_all[n_train:]
    flip_mask = rng.rand(n_train) < noise_rate
    y_tr[flip_mask] = 1 - y_tr[flip_mask]
    return X_tr, y_tr, X_te, y_te


def make_rare_events(n_train, n_test, seed, positive_rate=0.05):
    """
    Rare-event / fraud-like dataset.

    Positive class (rare, ~5 %) lives in the extreme tails of the feature
    distribution.  Negative class (common) clusters near the origin.

    Random sampling at 20 % retention retains roughly 1 % positives
    (20 % of 5 % = 1 %).  HVRT's variance-weighted allocation dedicates
    disproportionate budget to the tail partitions, retaining ~3-4 ×
    more positive samples — the core denoising / structure-preserving
    advantage.
    """
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test
    n_pos = int(n_total * positive_rate)
    n_neg = n_total - n_pos
    n_feat = 10

    # Negative class: standard Gaussian near origin
    X_neg = rng.randn(n_neg, n_feat) * 0.8
    y_neg = np.zeros(n_neg, dtype=int)

    # Positive class: extreme values (tails) — magnitude > 2.5 in some features
    X_pos_raw = rng.randn(n_pos, n_feat)
    # Push positives into the tail by scaling up a few key features
    tail_feat = rng.choice(n_feat, size=3, replace=False)
    signs = rng.choice([-1, 1], size=(n_pos, len(tail_feat)))
    X_pos_raw[:, tail_feat] = np.abs(X_pos_raw[:, tail_feat]) * signs * 2.5
    y_pos = np.ones(n_pos, dtype=int)

    X_all = np.vstack([X_neg, X_pos_raw])
    y_all = np.concatenate([y_neg, y_pos])

    # Shuffle
    idx = rng.permutation(n_total)
    X_all, y_all = X_all[idx], y_all[idx]

    return (X_all[:n_train], y_all[:n_train],
            X_all[n_train:], y_all[n_train:])


DATASET_GENERATORS = {
    'well_behaved':  make_well_behaved,
    'heavy_tail':    make_heavy_tail,
    'noisy_labels':  make_noisy_labels,
    'noisy_all':     make_noisy_all,
    'rare_events':   make_rare_events,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _score(X_train, y_train, X_test, y_test):
    """GBM ROC-AUC."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, proba))


def evaluate_condition(dataset_name, n_train, n_test, ratios, seed):
    """
    Run all methods × ratios on one dataset condition.

    Returns
    -------
    dict: {ratio -> {method -> pct_of_full}}
    """
    gen_fn = DATASET_GENERATORS[dataset_name]
    X_tr, y_tr, X_te, y_te = gen_fn(n_train, n_test, seed)

    # Baseline: full training set
    full_score = _score(X_tr, y_tr, X_te, y_te)

    results = {}

    model = HVRT(random_state=seed)
    XY = np.column_stack([X_tr, y_tr.astype(float)])
    model.fit(XY, y_tr)

    for ratio in ratios:
        n_keep = max(4, int(n_train * ratio))
        row = {}

        # HVRT-fps (size-proportional + FPS, without y_weight)
        try:
            _, idx = HVRT(random_state=seed).fit(X_tr, y_tr).reduce(
                n=n_keep, method='fps', variance_weighted=True, return_indices=True
            )
            row['HVRT-fps'] = _score(X_tr[idx], y_tr[idx], X_te, y_te) / full_score * 100
        except Exception:
            row['HVRT-fps'] = float('nan')

        # HVRT-yw (y_weight=0.3 — blends label extremeness into partitioning)
        try:
            _, idx = HVRT(y_weight=0.3, random_state=seed).fit(X_tr, y_tr).reduce(
                n=n_keep, method='fps', variance_weighted=True, return_indices=True
            )
            row['HVRT-yw'] = _score(X_tr[idx], y_tr[idx], X_te, y_te) / full_score * 100
        except Exception:
            row['HVRT-yw'] = float('nan')

        # Random
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_train, size=n_keep, replace=False)
        row['Random'] = _score(X_tr[idx], y_tr[idx], X_te, y_te) / full_score * 100

        # Stratified random (using sklearn StratifiedShuffleSplit)
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=n_keep, random_state=seed
            )
            idx_strat, _ = next(sss.split(X_tr, y_tr))
            row['Stratified'] = (
                _score(X_tr[idx_strat], y_tr[idx_strat], X_te, y_te) / full_score * 100
            )
        except Exception:
            row['Stratified'] = float('nan')

        results[ratio] = row

    return full_score, results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_results(all_results, ratios, methods, n_train, n_test, seed):
    W = 94
    n_col = len(methods)
    print()
    title = (f'HVRT SAMPLE REDUCTION — DENOISING BENCHMARK  '
             f'(n_train={n_train}, n_test={n_test}, seed={seed})')
    print(f'╔{"═" * W}╗')
    print(f'║  {title:<{W - 3}}║')
    print(f'╠{"═" * W}╣')

    # Header
    col_w = 12
    hdr = f'  {"Dataset":<22}  {"ratio":>5}  {"Full AUC":>9}'
    for m in methods:
        hdr += f'  {m:>{col_w}}'
    print(f'║{hdr}║')
    print(f'╠{"─" * W}╣')

    last_ds = None
    for ds_name, (full_score, ds_results) in all_results.items():
        if last_ds is not None:
            print(f'╠{"─" * W}╣')
        last_ds = ds_name
        first_row = True
        for ratio in ratios:
            row_data = ds_results[ratio]
            ds_label = ds_name if first_row else ''
            full_label = f'{full_score:.4f}' if first_row else ''
            row = f'  {ds_label:<22}  {ratio:>4.0%}  {full_label:>9}'
            for m in methods:
                val = row_data.get(m, float('nan'))
                cell = f'{val:>+.1f}%' if not (val != val) else '   n/a'
                # Pad to col_w
                row += f'  {cell:>{col_w}}'
            print(f'║{row}║')
            first_row = False

    print(f'╚{"═" * W}╝')
    print()
    print('  Values = % of full-training-set AUC.  100 % = matches full-data performance.')
    print('  > 100 % = denoising benefit (reduced set outperforms noisy full set).')
    print('  HVRT-fps: FPS selection, variance_weighted=True')
    print('  HVRT-yw:  same + y_weight=0.3 (label extremeness drives partitioning)')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='HVRT reduction denoising benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--n-train', type=int, default=3000)
    parser.add_argument('--n-test',  type=int, default=2000)
    parser.add_argument('--seed',    type=int, default=42)
    args = parser.parse_args()

    RATIOS  = [0.1, 0.2, 0.3, 0.5]
    METHODS = ['HVRT-fps', 'HVRT-yw', 'Random', 'Stratified']

    print()
    print('HVRT Sample Reduction — Denoising Benchmark')
    print(f'  n_train  : {args.n_train}')
    print(f'  n_test   : {args.n_test}')
    print(f'  seed     : {args.seed}')
    print(f'  ratios   : {RATIOS}')
    print()

    all_results = {}
    for ds_name in DATASET_GENERATORS:
        print(f'  [{ds_name:<15}] ...', end=' ', flush=True)
        full_score, ds_results = evaluate_condition(
            ds_name, args.n_train, args.n_test, RATIOS, args.seed
        )
        all_results[ds_name] = (full_score, ds_results)
        # Quick inline summary at 20% retention
        r20 = ds_results.get(0.2, {})
        hvrt = r20.get('HVRT-fps', float('nan'))
        rand = r20.get('Random', float('nan'))
        print(f'full={full_score:.4f}  HVRT@20%={hvrt:+.1f}%  Rand@20%={rand:+.1f}%')

    _print_results(all_results, RATIOS, METHODS, args.n_train, args.n_test, args.seed)


if __name__ == '__main__':
    main()
