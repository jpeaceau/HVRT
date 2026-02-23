#!/usr/bin/env python3
"""
Two-Phase KDE Pipeline Benchmark
=================================

Hypothesis: generating a large intermediate synthetic dataset with narrow
Gaussian KDE (h=0.10), re-fitting HVRT on it so auto-tuning produces >=50
fine partitions, then running Epanechnikov on those stable, data-rich
partitions yields better TSTR than:

  (a) single-phase bandwidth='auto' on the original small dataset, or
  (b) single-phase epanechnikov with a forced n_partitions=50 override.

Pipeline
--------
  Phase 1 : HVRT.fit(X_original).expand(n=n_mid, bandwidth=0.10) -> X_mid
  Phase 2 : HVRT.fit(X_mid).expand(n=n_final, generation_strategy=epanechnikov)
             -> X_synth   [evaluated with TSTR]

The Phase 2 HVRT re-runs auto-tuning on the larger X_mid, naturally producing
finer partitions (each with many samples) without manual n_partitions overrides.

Expected Phase 2 partition counts (n_tr ~= 400 from max_n=500):
  phase1_mult  n_mid   min_leaf (d=6)  expected_n_parts
  ----------   -----   -------------   ----------------
  5x           2000    44              ~45
  10x          4000    63              ~63
  25x         10000   100              ~100

Candidates
----------
  baseline_auto      : single-phase, bandwidth='auto'
  baseline_h010      : single-phase, bandwidth=0.10 (explicit narrow Gaussian)
  baseline_epan_50p  : single-phase, epanechnikov + n_partitions=50 forced
  two_phase_5x       : Phase1 to 5x n_tr, Phase2 Epanechnikov (auto-tuned)
  two_phase_10x      : Phase1 to 10x n_tr, Phase2 Epanechnikov (auto-tuned)
  two_phase_25x      : Phase1 to 25x n_tr, Phase2 Epanechnikov (auto-tuned)

Metrics (same as prior benchmarks)
-----------------------------------
  tstr_delta  ML utility; primary metric. Higher -> better.
  corr_mae    Pairwise correlation structure preservation. Lower -> better.
  mw1         Mean per-feature Wasserstein-1. Lower -> better.
  n_parts     Actual partition count used (Phase 2 for two-phase candidates).

Usage
-----
    python benchmarks/two_phase_pipeline_benchmark.py
    python benchmarks/two_phase_pipeline_benchmark.py --quick
    python benchmarks/two_phase_pipeline_benchmark.py --ratios 5 10 --max-n 500
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

if hasattr(sys.stdout, 'buffer') and \
        getattr(sys.stdout, 'encoding', '').lower().replace('-', '') not in \
        ('utf8', 'utf-8'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )

from hvrt import HVRT, epanechnikov
from hvrt.benchmarks.datasets import BENCHMARK_DATASETS


# -----------------------------------------------------------------------------
# Candidate registry
# Each candidate is a dict with 'type' and relevant kwargs.
# 'single'    -> expand_kwargs, n_partitions_override
# 'two_phase' -> phase1_mult, phase1_expand_kwargs, phase2_expand_kwargs
# -----------------------------------------------------------------------------

CANDIDATES = {
    'baseline_auto': {
        'type': 'single',
        'expand_kwargs': {'bandwidth': 'auto'},
        'n_partitions_override': None,
    },
    'baseline_h010': {
        'type': 'single',
        'expand_kwargs': {'bandwidth': 0.10},
        'n_partitions_override': None,
    },
    'baseline_epan_50p': {
        'type': 'single',
        'expand_kwargs': {'generation_strategy': epanechnikov},
        'n_partitions_override': 50,
    },
    'two_phase_5x': {
        'type': 'two_phase',
        'phase1_mult': 5,
        'phase1_expand_kwargs': {'bandwidth': 0.10},
        'phase2_expand_kwargs': {'generation_strategy': epanechnikov},
    },
    'two_phase_10x': {
        'type': 'two_phase',
        'phase1_mult': 10,
        'phase1_expand_kwargs': {'bandwidth': 0.10},
        'phase2_expand_kwargs': {'generation_strategy': epanechnikov},
    },
    'two_phase_25x': {
        'type': 'two_phase',
        'phase1_mult': 25,
        'phase1_expand_kwargs': {'bandwidth': 0.10},
        'phase2_expand_kwargs': {'generation_strategy': epanechnikov},
    },
}


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------

def marginal_w1(X_real: np.ndarray, X_synth: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance
    return float(np.mean([
        wasserstein_distance(X_real[:, j], X_synth[:, j])
        for j in range(X_real.shape[1])
    ]))


def correlation_mae(X_real: np.ndarray, X_synth: np.ndarray) -> float:
    d = X_real.shape[1]
    if d < 2:
        return 0.0
    idx = np.triu_indices(d, k=1)
    return float(np.mean(np.abs(
        np.corrcoef(X_real.T)[idx] - np.corrcoef(X_synth.T)[idx]
    )))


def tstr_scores(X_tr, y_tr, X_te, y_te, X_synth, y_synth, random_state=42):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        trtr_m = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        trtr_m.fit(X_tr, y_tr)
        trtr = float(r2_score(y_te, trtr_m.predict(X_te)))

        tstr_m = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        tstr_m.fit(X_synth, y_synth)
        tstr = float(r2_score(y_te, tstr_m.predict(X_te)))

    return trtr, tstr


# -----------------------------------------------------------------------------
# Per-fold evaluation — single-phase
# -----------------------------------------------------------------------------

def evaluate_single_phase(
    X_tr, y_tr, X_te, y_te,
    n_synth: int,
    cand: dict,
    seed: int,
) -> dict:
    _nan = {k: float('nan') for k in ('mw1', 'corr_mae', 'trtr', 'tstr_delta', 'n_parts')}

    try:
        XY_tr = np.column_stack([X_tr, y_tr.reshape(-1, 1).astype(float)])

        model = HVRT(random_state=seed).fit(XY_tr)

        XY_s = model.expand(
            n=n_synth,
            n_partitions=cand['n_partitions_override'],
            **cand['expand_kwargs'],
        )

        n_parts = int(model.n_partitions_)
        X_s, y_s = XY_s[:, :-1], XY_s[:, -1]

        trtr, tstr = tstr_scores(X_tr, y_tr, X_te, y_te, X_s, y_s, random_state=seed)

        return {
            'mw1':        marginal_w1(X_tr, X_s),
            'corr_mae':   correlation_mae(X_tr, X_s),
            'trtr':       trtr,
            'tstr_delta': tstr - trtr,
            'n_parts':    n_parts,
        }

    except Exception as exc:
        warnings.warn(f'evaluate_single_phase failed: {exc}', stacklevel=2)
        return _nan


# -----------------------------------------------------------------------------
# Per-fold evaluation — two-phase
# -----------------------------------------------------------------------------

def evaluate_two_phase(
    X_tr, y_tr, X_te, y_te,
    n_synth: int,
    cand: dict,
    seed: int,
) -> dict:
    _nan = {k: float('nan') for k in ('mw1', 'corr_mae', 'trtr', 'tstr_delta', 'n_parts')}

    try:
        XY_tr = np.column_stack([X_tr, y_tr.reshape(-1, 1).astype(float)])

        # ── Phase 1: narrow Gaussian on original data ──────────────────────
        model_p1 = HVRT(random_state=seed).fit(XY_tr)
        n_mid = max(len(X_tr), int(len(X_tr) * cand['phase1_mult']))
        XY_mid = model_p1.expand(n=n_mid, **cand['phase1_expand_kwargs'])

        # ── Phase 2: auto-tuned HVRT on intermediate → Epanechnikov ───────
        model_p2 = HVRT(random_state=seed).fit(XY_mid)
        XY_s = model_p2.expand(n=n_synth, **cand['phase2_expand_kwargs'])

        n_parts_p2 = int(model_p2.n_partitions_)
        X_s, y_s = XY_s[:, :-1], XY_s[:, -1]

        # Fidelity metrics compare synthetic against REAL training data,
        # not the intermediate, so comparisons are fair across all candidates.
        trtr, tstr = tstr_scores(X_tr, y_tr, X_te, y_te, X_s, y_s, random_state=seed)

        return {
            'mw1':        marginal_w1(X_tr, X_s),
            'corr_mae':   correlation_mae(X_tr, X_s),
            'trtr':       trtr,
            'tstr_delta': tstr - trtr,
            'n_parts':    n_parts_p2,
        }

    except Exception as exc:
        warnings.warn(f'evaluate_two_phase failed: {exc}', stacklevel=2)
        return _nan


# -----------------------------------------------------------------------------
# CV wrapper — unified dispatcher
# -----------------------------------------------------------------------------

def run_condition(
    X_full, y_full, exp_ratio, cand,
    n_splits, n_repeats, random_state, max_n,
) -> dict:
    from sklearn.model_selection import RepeatedKFold

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    accum = defaultdict(list)

    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X_full)):
        X_tr, X_te = X_full[tr_idx], X_full[te_idx]
        y_tr, y_te = y_full[tr_idx], y_full[te_idx]

        if len(X_tr) > max_n:
            rng = np.random.RandomState(random_state + fold_i)
            sel = rng.choice(len(X_tr), size=max_n, replace=False)
            X_tr, y_tr = X_tr[sel], y_tr[sel]

        n_synth = max(4, int(len(X_tr) * exp_ratio))
        seed = random_state + fold_i * 13

        if cand['type'] == 'single':
            res = evaluate_single_phase(X_tr, y_tr, X_te, y_te, n_synth, cand, seed)
        else:
            res = evaluate_two_phase(X_tr, y_tr, X_te, y_te, n_synth, cand, seed)

        for k in res:
            accum[k].append(res[k])

    def _ms(k):
        a = np.array(accum[k], dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a))

    out = {}
    for k in ('mw1', 'corr_mae', 'trtr', 'tstr_delta', 'n_parts'):
        m, s = _ms(k)
        out[f'{k}_mean'] = m
        out[f'{k}_std']  = s
    return out


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def print_pipeline_table(
    ds_name: str,
    ratio: float,
    results: dict,     # {cand_name: metrics}
    n_splits: int,
    n_repeats: int,
    cand_names: list,
):
    """
    Rows = candidates.
    Columns = TSTR delta, Corr.MAE, n_parts.
    Best TSTR and best Corr are marked with a diamond.
    """
    col_label_w = 22
    col_tstr_w  = 12
    col_corr_w  = 12
    col_parts_w = 10
    W = col_label_w + col_tstr_w + col_corr_w + col_parts_w
    W = max(W, 72)

    # Find winners
    tstr_vals = {c: results[c]['tstr_delta_mean'] for c in cand_names
                 if not np.isnan(results[c]['tstr_delta_mean'])}
    corr_vals = {c: results[c]['corr_mae_mean']   for c in cand_names
                 if not np.isnan(results[c]['corr_mae_mean'])}
    best_tstr = max(tstr_vals, key=tstr_vals.__getitem__) if tstr_vals else None
    best_corr = min(corr_vals, key=corr_vals.__getitem__) if corr_vals else None

    print(f'\u2554{"\u2550" * W}\u2557')
    title = (f'  {ds_name}  |  ratio={ratio:.0f}x  |  two-phase pipeline  |  '
             f'{n_splits}-fold \u00d7 {n_repeats}-repeat')
    print(f'\u2551{title:<{W}}\u2551')
    print(f'\u2560{"\u2550" * W}\u2563')

    hdr = (f'  {"candidate":<{col_label_w - 2}}'
           f'{"TSTR \u0394":>{col_tstr_w}}'
           f'{"Corr.MAE":>{col_corr_w}}'
           f'{"n_parts":>{col_parts_w}}')
    print(f'\u2551{hdr:<{W}}\u2551')
    print(f'\u2560{"\u2500" * W}\u2563')

    for cand in cand_names:
        r = results[cand]
        tstr_m = r['tstr_delta_mean']
        corr_m = r['corr_mae_mean']
        npts_m = r['n_parts_mean']

        tstr_s = f'{tstr_m:+.4f}' if not np.isnan(tstr_m) else '\u2014'
        corr_s = f'{corr_m:.4f}'  if not np.isnan(corr_m) else '\u2014'
        npts_s = f'\u2248{int(round(npts_m))}' if not np.isnan(npts_m) else '\u2014'

        tstr_mk = '\u25c6' if cand == best_tstr else ' '
        corr_mk = '\u25c6' if cand == best_corr else ' '

        row = (f'  {cand:<{col_label_w - 2}}'
               f'{tstr_s + tstr_mk:>{col_tstr_w}}'
               f'{corr_s + corr_mk:>{col_corr_w}}'
               f'{npts_s:>{col_parts_w}}')
        print(f'\u2551{row:<{W}}\u2551')

    print(f'\u255a{"\u2550" * W}\u255d')
    print()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Two-phase KDE pipeline benchmark')
    parser.add_argument('--ratios',  nargs='+', type=float, default=[5.0, 10.0])
    parser.add_argument('--max-n',   type=int,  default=500)
    parser.add_argument('--quick',   action='store_true',
                        help='Faster: 3-fold x1-repeat, fewer two-phase variants')
    args = parser.parse_args()

    if args.quick:
        n_splits, n_repeats = 3, 1
        active_cands = [
            'baseline_auto', 'baseline_h010', 'baseline_epan_50p',
            'two_phase_10x',
        ]
    else:
        n_splits, n_repeats = 5, 3
        active_cands = list(CANDIDATES.keys())

    # Regression datasets only
    all_ds = {}
    for ds_name, gen_fn in BENCHMARK_DATASETS.items():
        X_full, y_full, _ = gen_fn(random_state=42)
        X_full = np.asarray(X_full, dtype=float)
        y_full = np.asarray(y_full, dtype=float)
        if len(np.unique(y_full)) > 20:            # regression
            all_ds[ds_name] = (X_full, y_full)

    print()
    print('=' * 72)
    print('  HVRT \u2014 Two-Phase KDE Pipeline Benchmark')
    print('=' * 72)
    print(f'  Hypothesis : Phase1 Gaussian expansion enables fine partitions;')
    print(f'               Phase2 Epanechnikov exploits them for better TSTR.')
    print(f'  Datasets   : {", ".join(all_ds)}')
    print(f'  Ratios     : {args.ratios}')
    print(f'  Candidates : {active_cands}')
    print(f'  max_n      : {args.max_n}')
    print(f'  CV         : {n_splits}-fold \u00d7 {n_repeats}-repeat = '
          f'{n_splits * n_repeats} evals per condition')
    print()

    for ds_name, (X_full, y_full) in all_ds.items():
        print(f'  [{ds_name}]  n={len(X_full)}, d={X_full.shape[1]}')

        for ratio in args.ratios:
            all_results = {}

            for cand_name in active_cands:
                cand = CANDIDATES[cand_name]
                phase_lbl = (f'Phase1={cand["phase1_mult"]}x'
                             if cand['type'] == 'two_phase'
                             else 'single')
                print(f'    [{ratio:.0f}x  {cand_name:<22}  ({phase_lbl})] ...',
                      end=' ', flush=True)

                res = run_condition(
                    X_full, y_full, ratio, cand,
                    n_splits, n_repeats,
                    random_state=42, max_n=args.max_n,
                )
                all_results[cand_name] = res
                print(
                    f'TSTR \u0394={res["tstr_delta_mean"]:+.4f}  '
                    f'Corr={res["corr_mae_mean"]:.4f}  '
                    f'(n_parts\u2248{int(round(res["n_parts_mean"]))})'
                )

            print()
            print_pipeline_table(
                ds_name, ratio, all_results,
                n_splits, n_repeats, active_cands,
            )


if __name__ == '__main__':
    main()
