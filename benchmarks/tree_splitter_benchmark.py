"""
tree_splitter='best' vs tree_splitter='random' quality benchmark.

Runs HVRT-var across all 6 reduction datasets (4 ratios) and all 4
expansion datasets (3 ratios) for both splitters, reporting delta on
every quality metric.  FastHVRT-var is included for expansion to cover
the joint-(X,y) generation path.

Metrics
-------
Reduction:
  mf        marginal_fidelity (higher = better)
  ml        ml_utility_retention (higher = better)
  ml_delta  TSTR - TRTR (higher = better)
  corr_mae  correlation MAE (lower = better)

Expansion:
  disc      discriminator_accuracy (lower = better; 0.5 = indistinguishable)
  mf        marginal_fidelity (higher = better)
  ml_delta  TSTR - TRTR (higher = better)
  dcr       privacy_dcr (higher = better)

Usage
-----
    python benchmarks/tree_splitter_benchmark.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hvrt import HVRT, FastHVRT
from hvrt.benchmarks.datasets import BENCHMARK_DATASETS, make_emergence_divergence, make_emergence_bifurcation
from hvrt.benchmarks.metrics import evaluate_reduction, evaluate_expansion, ml_utility_tstr

RANDOM_STATE = 42

REDUCE_DATASETS = [
    'adult', 'fraud', 'housing', 'multimodal',
    'emergence_divergence', 'emergence_bifurcation',
]
EXPAND_DATASETS = ['adult', 'fraud', 'housing', 'multimodal']
EMERGENCE_DATASETS = {'emergence_divergence', 'emergence_bifurcation'}

REDUCE_RATIOS = [0.5, 0.3, 0.2, 0.1]
EXPAND_RATIOS = [1.0, 2.0, 5.0]
MAX_N_EXPAND  = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(ds_name, max_n=None):
    X, y, _ = BENCHMARK_DATASETS[ds_name](random_state=RANDOM_STATE)
    if max_n is not None:
        X, y = X[:max_n], y[:max_n]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def _run_reduce(X_tr, y_tr, X_te, y_te, splitter, ratio, is_emergence, trtr):
    n_target = max(2, int(len(X_tr) * ratio))
    model = HVRT(random_state=RANDOM_STATE, tree_splitter=splitter)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t0
    X_red, idx = model.reduce(n=n_target, variance_weighted=True, return_indices=True)
    y_red = y_tr[idx]
    m = evaluate_reduction(X_tr, y_tr, X_red, y_red, X_te, y_te, is_emergence=is_emergence)
    m['ml_delta'] = round(m.get('ml_utility_retention', 0.0) - trtr, 4)
    m['fit_ms'] = round(t_fit * 1000, 1)
    return m


def _run_expand(X_tr, y_tr, X_te, y_te, splitter, exp_ratio, trtr):
    n_syn = int(len(X_tr) * exp_ratio)
    is_cls = len(np.unique(y_tr)) <= 20
    results = {}
    for ModelCls, tag in [(HVRT, 'HVRT'), (FastHVRT, 'FastHVRT')]:
        model = ModelCls(random_state=RANDOM_STATE, tree_splitter=splitter)
        y_col = y_tr.reshape(-1, 1).astype(float)
        XY_tr = np.column_stack([X_tr, y_col])
        t0 = time.perf_counter()
        model.fit(XY_tr)
        t_fit = time.perf_counter() - t0
        XY_syn = model.expand(n=n_syn, variance_weighted=True)
        X_syn = XY_syn[:, :-1]
        y_syn_raw = XY_syn[:, -1]
        if is_cls:
            classes = np.unique(y_tr)
            y_syn = classes[np.argmin(np.abs(y_syn_raw[:, None] - classes[None, :]), axis=1)]
        else:
            y_syn = y_syn_raw
        m = evaluate_expansion(X_tr, y_tr, X_syn, y_syn, X_te, y_te)
        m['ml_delta'] = round(m.get('ml_utility_tstr', 0.0) - trtr, 4)
        m['fit_ms'] = round(t_fit * 1000, 1)
        results[tag] = m
    return results


def _fmt(v, decimals=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '  n/a  '
    return f'{v:+.{decimals}f}' if isinstance(v, float) else str(v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    reduce_rows  = []   # one row per (dataset, ratio)
    expand_rows  = []   # one row per (dataset, exp_ratio, model)

    # -----------------------------------------------------------------------
    # Reduction
    # -----------------------------------------------------------------------
    print('=' * 72)
    print('REDUCTION: HVRT-var  best vs random  (6 datasets × 4 ratios)')
    print('=' * 72)
    print(f'{"dataset":<26} {"ratio":>5}  {"mf_best":>7} {"mf_rnd":>7} {"dmf":>7}  '
          f'{"ml_best":>7} {"ml_rnd":>7} {"dml":>7}  '
          f'{"fit_best":>8} {"fit_rnd":>7} {"speedup":>7}')
    print('-' * 72)

    for ds_name in REDUCE_DATASETS:
        is_em = ds_name in EMERGENCE_DATASETS
        X_tr, X_te, y_tr, y_te = _load(ds_name)
        trtr = round(ml_utility_tstr(X_tr, y_tr, X_te, y_te), 4)

        for ratio in REDUCE_RATIOS:
            try:
                mb = _run_reduce(X_tr, y_tr, X_te, y_te, 'best',   ratio, is_em, trtr)
                mr = _run_reduce(X_tr, y_tr, X_te, y_te, 'random', ratio, is_em, trtr)
            except Exception as e:
                print(f'  ERROR {ds_name} ratio={ratio}: {e}')
                continue

            mf_b = mb.get('marginal_fidelity', float('nan'))
            mf_r = mr.get('marginal_fidelity', float('nan'))
            ml_b = mb.get('ml_utility_retention', float('nan'))
            ml_r = mr.get('ml_utility_retention', float('nan'))
            fit_b = mb['fit_ms']
            fit_r = mr['fit_ms']
            speedup = fit_b / fit_r if fit_r > 0 else float('nan')

            label = f'{ds_name}@{ratio:.0%}'
            print(f'{label:<26} {ratio:>5.0%}  '
                  f'{mf_b:>7.3f} {mf_r:>7.3f} {mf_r-mf_b:>+7.3f}  '
                  f'{ml_b:>7.3f} {ml_r:>7.3f} {ml_r-ml_b:>+7.3f}  '
                  f'{fit_b:>7.0f}ms {fit_r:>6.0f}ms {speedup:>6.1f}x')

            reduce_rows.append({
                'dataset': ds_name, 'ratio': ratio,
                'best': mb, 'random': mr,
            })

    # -----------------------------------------------------------------------
    # Expansion
    # -----------------------------------------------------------------------
    print()
    print('=' * 72)
    print('EXPANSION: HVRT-var + FastHVRT-var  best vs random  (4 datasets × 3 ratios)')
    print('=' * 72)
    print(f'{"dataset/model":<30} {"exp":>4}  '
          f'{"disc_b":>6} {"disc_r":>6} {"ddisc":>6}  '
          f'{"mf_b":>6} {"mf_r":>6} {"dmf":>6}  '
          f'{"ml_b":>6} {"ml_r":>6} {"dml":>6}')
    print('-' * 72)

    for ds_name in EXPAND_DATASETS:
        X_tr, X_te, y_tr, y_te = _load(ds_name, max_n=MAX_N_EXPAND)
        trtr = round(ml_utility_tstr(X_tr, y_tr, X_te, y_te), 4)

        for exp_ratio in EXPAND_RATIOS:
            try:
                res_b = _run_expand(X_tr, y_tr, X_te, y_te, 'best',   exp_ratio, trtr)
                res_r = _run_expand(X_tr, y_tr, X_te, y_te, 'random', exp_ratio, trtr)
            except Exception as e:
                print(f'  ERROR {ds_name} exp={exp_ratio}: {e}')
                continue

            for tag in ('HVRT', 'FastHVRT'):
                mb = res_b[tag]
                mr = res_r[tag]
                disc_b = mb.get('discriminator_accuracy', float('nan'))
                disc_r = mr.get('discriminator_accuracy', float('nan'))
                mf_b   = mb.get('marginal_fidelity', float('nan'))
                mf_r   = mr.get('marginal_fidelity', float('nan'))
                ml_b   = mb.get('ml_delta', float('nan'))
                ml_r   = mr.get('ml_delta', float('nan'))

                label = f'{ds_name}/{tag}@{exp_ratio:.0f}x'
                print(f'{label:<30} {exp_ratio:>4.0f}x  '
                      f'{disc_b:>6.3f} {disc_r:>6.3f} {disc_r-disc_b:>+6.3f}  '
                      f'{mf_b:>6.3f} {mf_r:>6.3f} {mf_r-mf_b:>+6.3f}  '
                      f'{ml_b:>6.3f} {ml_r:>6.3f} {ml_r-ml_b:>+6.3f}')

                expand_rows.append({
                    'dataset': ds_name, 'model': tag, 'exp_ratio': exp_ratio,
                    'best': mb, 'random': mr,
                })

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print('=' * 72)
    print('SUMMARY')
    print('=' * 72)

    if reduce_rows:
        mf_deltas  = [r['random'].get('marginal_fidelity', float('nan')) -
                      r['best'].get('marginal_fidelity', float('nan'))
                      for r in reduce_rows]
        ml_deltas  = [r['random'].get('ml_utility_retention', float('nan')) -
                      r['best'].get('ml_utility_retention', float('nan'))
                      for r in reduce_rows]
        speedups   = [r['best']['fit_ms'] / r['random']['fit_ms']
                      for r in reduce_rows if r['random']['fit_ms'] > 0]
        mf_deltas  = [x for x in mf_deltas  if not np.isnan(x)]
        ml_deltas  = [x for x in ml_deltas  if not np.isnan(x)]

        print(f'\nReduction ({len(reduce_rows)} conditions):')
        print(f'  marginal_fidelity  delta: mean={np.mean(mf_deltas):+.4f}  '
              f'std={np.std(mf_deltas):.4f}  '
              f'min={np.min(mf_deltas):+.4f}  max={np.max(mf_deltas):+.4f}')
        print(f'  ml_utility         delta: mean={np.mean(ml_deltas):+.4f}  '
              f'std={np.std(ml_deltas):.4f}  '
              f'min={np.min(ml_deltas):+.4f}  max={np.max(ml_deltas):+.4f}')
        print(f'  fit() speedup:            mean={np.mean(speedups):.1f}x  '
              f'min={np.min(speedups):.1f}x  max={np.max(speedups):.1f}x')

    if expand_rows:
        disc_deltas = [r['random'].get('discriminator_accuracy', float('nan')) -
                       r['best'].get('discriminator_accuracy', float('nan'))
                       for r in expand_rows]
        mf_deltas_e = [r['random'].get('marginal_fidelity', float('nan')) -
                       r['best'].get('marginal_fidelity', float('nan'))
                       for r in expand_rows]
        ml_deltas_e = [r['random'].get('ml_delta', float('nan')) -
                       r['best'].get('ml_delta', float('nan'))
                       for r in expand_rows]
        disc_deltas = [x for x in disc_deltas if not np.isnan(x)]
        mf_deltas_e = [x for x in mf_deltas_e if not np.isnan(x)]
        ml_deltas_e = [x for x in ml_deltas_e if not np.isnan(x)]

        print(f'\nExpansion ({len(expand_rows)} conditions):')
        print(f'  discriminator_acc  delta: mean={np.mean(disc_deltas):+.4f}  '
              f'std={np.std(disc_deltas):.4f}  '
              f'min={np.min(disc_deltas):+.4f}  max={np.max(disc_deltas):+.4f}')
        print(f'  marginal_fidelity  delta: mean={np.mean(mf_deltas_e):+.4f}  '
              f'std={np.std(mf_deltas_e):.4f}  '
              f'min={np.min(mf_deltas_e):+.4f}  max={np.max(mf_deltas_e):.4f}')
        print(f'  ml_delta           delta: mean={np.mean(ml_deltas_e):+.4f}  '
              f'std={np.std(ml_deltas_e):.4f}  '
              f'min={np.min(ml_deltas_e):+.4f}  max={np.max(ml_deltas_e):+.4f}')

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out = {
        'meta': {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'random_state': RANDOM_STATE,
            'max_n_expand': MAX_N_EXPAND,
        },
        'reduce': reduce_rows,
        'expand': expand_rows,
    }
    out_path = os.path.join(os.path.dirname(__file__), 'results', 'tree_splitter_benchmark.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nResults saved → {out_path}')
