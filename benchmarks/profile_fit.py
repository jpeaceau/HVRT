"""
Profile HVRT.fit() to establish the breakdown across sub-operations.

Patches the key callsites in hvrt._base so that each timed section is
captured at the right module-level reference.

Sections timed
--------------
  preprocess  — fit_preprocess_data (z-score, StandardScaler, LabelEncoder)
  target      — _compute_x_component (pairwise or z-sum synthetic target)
  tree_fit    — fit_hvrt_tree (DecisionTreeRegressor.fit)
  tree_apply  — tree_.apply + unique_partitions_ + partition_ids_ assignment
  other       — remainder (resolve_tree_params, cache invalidation, strategy
                            eager prepare, etc.)

Usage
-----
    python benchmarks/profile_fit.py
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Import target modules before patching
# ---------------------------------------------------------------------------
import hvrt._base as _base_mod
import hvrt._partitioning as _partitioning_mod
import hvrt._preprocessing as _preprocessing_mod
from hvrt.model.hvrt import HVRT as _HVRTClass
from hvrt import HVRT

_times = {}

# ---------------------------------------------------------------------------
# Patch fit_preprocess_data in _base (where it's imported and called)
# ---------------------------------------------------------------------------
_orig_preprocess = _base_mod.fit_preprocess_data

def _timed_preprocess(*args, **kwargs):
    t0 = time.perf_counter()
    result = _orig_preprocess(*args, **kwargs)
    _times.setdefault("preprocess", []).append(time.perf_counter() - t0)
    return result

_base_mod.fit_preprocess_data = _timed_preprocess

# ---------------------------------------------------------------------------
# Patch fit_hvrt_tree in _base (where it's imported and called)
# ---------------------------------------------------------------------------
_orig_fit_tree = _base_mod.fit_hvrt_tree

def _timed_fit_tree(*args, **kwargs):
    t0 = time.perf_counter()
    result = _orig_fit_tree(*args, **kwargs)
    _times.setdefault("tree_fit", []).append(time.perf_counter() - t0)
    return result

_base_mod.fit_hvrt_tree = _timed_fit_tree

# ---------------------------------------------------------------------------
# Patch HVRT._compute_x_component
# ---------------------------------------------------------------------------
_orig_compute = _HVRTClass._compute_x_component

def _timed_compute(self, X_z):
    t0 = time.perf_counter()
    result = _orig_compute(self, X_z)
    _times.setdefault("target", []).append(time.perf_counter() - t0)
    return result

_HVRTClass._compute_x_component = _timed_compute

# ---------------------------------------------------------------------------
# Wrap _fit_tree to capture tree_.apply separately
# ---------------------------------------------------------------------------
_orig_fit_tree_method = _base_mod._HVRTBase._fit_tree

def _timed_fit_tree_method(self, X_z, target, n_partitions_override=None, is_reduction=False):
    # _fit_tree contains: resolve_tree_params, fit_hvrt_tree (already patched),
    # tree_.apply, np.unique.  We measure the whole method, then subtract
    # the tree_fit time to get apply+unique time.
    _times.setdefault("_fit_tree_outer", [])
    t0 = time.perf_counter()
    result = _orig_fit_tree_method(self, X_z, target,
                                   n_partitions_override=n_partitions_override,
                                   is_reduction=is_reduction)
    _times["_fit_tree_outer"].append(time.perf_counter() - t0)
    return result

_base_mod._HVRTBase._fit_tree = _timed_fit_tree_method

# ---------------------------------------------------------------------------
# Wrap fit() to capture total and compute residual
# ---------------------------------------------------------------------------
_orig_fit = _base_mod._HVRTBase.fit

def _timed_fit(self, X, y=None, feature_types=None):
    # Clear per-call accumulators
    for k in ("preprocess", "target", "tree_fit", "_fit_tree_outer"):
        _times[k] = []

    t0 = time.perf_counter()
    result = _orig_fit(self, X, y=y, feature_types=feature_types)
    total = time.perf_counter() - t0

    preprocess_t  = sum(_times.get("preprocess", []))
    target_t      = sum(_times.get("target", []))
    tree_fit_t    = sum(_times.get("tree_fit", []))
    fit_tree_outer_t = sum(_times.get("_fit_tree_outer", []))

    # tree_apply = _fit_tree outer overhead minus fit_hvrt_tree itself
    tree_apply_t  = max(fit_tree_outer_t - tree_fit_t, 0.0)
    other_t       = max(total - preprocess_t - target_t - fit_tree_outer_t, 0.0)

    _times["_last"] = {
        "total":      total,
        "preprocess": preprocess_t,
        "target":     target_t,
        "tree_fit":   tree_fit_t,
        "tree_apply": tree_apply_t,
        "other":      other_t,
    }
    return result

_base_mod._HVRTBase.fit = _timed_fit

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def ms(v):
    return f"{v * 1000:6.1f}"

def pct(v, total):
    return f"{v / total * 100:3.0f}%"

def run(n, d, label, n_repeats=5):
    rng = np.random.RandomState(0)
    X = rng.randn(n, d)

    # Warm-up (JIT compile on first call)
    HVRT(random_state=0).fit(X)

    rows = {k: [] for k in ("total", "preprocess", "target", "tree_fit", "tree_apply", "other")}
    for _ in range(n_repeats):
        HVRT(random_state=0).fit(X)
        last = _times["_last"]
        for k in rows:
            rows[k].append(last[k])

    med = {k: np.median(v) for k, v in rows.items()}
    t = med["total"]
    print(
        f"{label:<18}"
        f"  total={ms(t)}ms"
        f"  preproc={ms(med['preprocess'])}ms({pct(med['preprocess'],t)})"
        f"  target={ms(med['target'])}ms({pct(med['target'],t)})"
        f"  tree={ms(med['tree_fit'])}ms({pct(med['tree_fit'],t)})"
        f"  apply={ms(med['tree_apply'])}ms({pct(med['tree_apply'],t)})"
        f"  other={ms(med['other'])}ms({pct(med['other'],t)})"
    )


if __name__ == "__main__":
    from hvrt._kernels import _NUMBA_AVAILABLE
    print(f"Numba available: {_NUMBA_AVAILABLE}")
    print()
    header = (
        f"{'Config':<18}"
        f"  {'total':>14}"
        f"  {'preprocess':>18}"
        f"  {'target':>16}"
        f"  {'tree.fit':>14}"
        f"  {'tree.apply':>16}"
        f"  {'other':>12}"
    )
    print(header)
    print("-" * len(header))

    configs = [
        (1_000,   10, "n=1k  d=10"),
        (1_000,   20, "n=1k  d=20"),
        (5_000,   10, "n=5k  d=10"),
        (5_000,   20, "n=5k  d=20"),
        (20_000,  10, "n=20k d=10"),
        (20_000,  20, "n=20k d=20"),
        (50_000,  10, "n=50k d=10"),
        (50_000,  20, "n=50k d=20"),
    ]

    for n, d, label in configs:
        run(n, d, label)
