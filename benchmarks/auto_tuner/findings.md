# Auto-Tuner Min-Samples-Leaf Study — Findings

**Date**: 2026-02-22
**Benchmark**: `benchmarks/auto_tuner/min_samples_study.py`
**Seeds**: 5 per cell | **Grid**: n ∈ {50, 100, 200, 500} × d ∈ {2, 5, 10, 15, 20}

---

## Problem Statement

The v2.1.1 hotfix changed `min_samples_leaf` for expansion/augmentation from the
old 40:1 reduction ratio to:

```python
min_samples_leaf = max(5, int(0.75 * n_samples ** (2 / 3)))
```

This formula is **feature-agnostic**. `scipy.stats.gaussian_kde` requires a
non-singular covariance matrix, which is guaranteed only when the number of
samples in a partition `n_part > d` (number of features). When
`min_samples_leaf < d + 1` the tree can produce partitions where KDE silently
collapses (returns `None`) and falls back to bootstrap-noise, destroying all
structural fidelity for those partitions.

---

## Key Findings

### Current formula failures

| n  | d  | KDE failure rate | At-risk rate |
|----|-----|-----------------|--------------|
| 50 | 10  | **20%**         | 25%          |
| 50 | 15  | **80%**         | 80%          |
| 50 | 20  | **100%**        | 100%         |
| 100 | 20 | **85%**         | 85%          |

For datasets with `n < 2d` (the small-n / high-d zone), the current formula
produces `min_samples_leaf` values below `d + 1`, creating partitions too small
for full-rank covariance. In the worst case (`n=50, d=20`) every single
partition has a failed KDE — all generation silently degrades to
bootstrap-noise.

### Formula ranking (all 700 runs, 5 seeds)

| Rank | Formula     | Expression                        | KDE fail | At-risk | Wasserstein | Avg parts |
|------|-------------|-----------------------------------|----------|---------|-------------|-----------|
| 1    | `sqrt_n`    | `max(d+2, int(n**0.5))`           | 0.0%     | 0.0%    | **0.1423**  | 9.4       |
| 2    | `hybrid_b`  | `max(d+2, int(0.6 * n^(2/3)))`   | 0.0%     | 0.0%    | 0.1507      | 6.7       |
| 3    | `feat_2x`   | `max(5, 2 * d)`                   | 0.0%     | 0.0%    | 0.1589      | 13.7      |
| 4    | `feat_floor`| `max(d+2, int(0.75 * n^(2/3)))`  | 0.0%     | 0.0%    | 0.1601      | 5.5       |
| 5    | `hybrid_a`  | `max(2*d, int(0.5 * n^(2/3)))`   | 0.0%     | 0.0%    | 0.1643      | 6.7       |
| 6    | `feat_3x`   | `max(5, 3 * d)`                   | 0.0%     | 0.0%    | 0.1804      | 10.2      |
| 7    | **current** | `max(5, int(0.75 * n^(2/3)))`    | **1.4%** | **1.7%**| 0.1491      | 6.0       |

All six alternatives achieve **zero KDE failures** and **zero at-risk
partitions** across the full grid.

---

## Recommendation

### Primary: `sqrt_n` — `max(d + 2, int(n ** 0.5))`

- **Best Wasserstein** (0.1423) of all safe formulas — most faithful generation.
- Feature-aware floor `d + 2` guarantees non-singular covariance for any (n, d).
- `sqrt(n)` grows slower than `n^(2/3)`, allowing finer partitioning (avg 9.4
  leaves) which better captures local density structure.
- Example values compared to current:

  | n   | d  | current | sqrt_n |
  |-----|----|---------|--------|
  | 50  | 10 | 10 ❌   | **12** |
  | 50  | 15 | 10 ❌   | **17** |
  | 50  | 20 | 10 ❌   | **22** |
  | 100 | 20 | 16 ❌   | **22** |
  | 200 | 20 | 25 ✔    | 22     |
  | 500 | 10 | 47 ✔    | 22     |

### Conservative alternative: `feat_floor` — `max(d + 2, int(0.75 * n^(2/3)))`

If minimal code divergence from the current formula is preferred, simply
replacing the floor constant `5` with `d + 2` eliminates all failures. This
formula is identical to `current` for low-d datasets (where `d+2 < 5` or
`n^(2/3)` dominates) and differs only in the dangerous small-n/high-d zone.

---

## Change Required

**File**: `src/hvrt/_partitioning.py`
**Function**: `auto_tune_tree_params()`
**Branch**: `is_reduction=False` (expansion / augmentation)

### Option A — Primary recommendation (`sqrt_n`)

```python
# Before (v2.1.1):
min_samples_leaf = max(5, int(0.75 * n_samples ** (2 / 3)))

# After (v2.2):
min_samples_leaf = max(n_features + 2, int(n_samples ** 0.5))
```

### Option B — Conservative minimal change (`feat_floor`)

```python
# Before (v2.1.1):
min_samples_leaf = max(5, int(0.75 * n_samples ** (2 / 3)))

# After:
min_samples_leaf = max(n_features + 2, int(0.75 * n_samples ** (2 / 3)))
```

---

## Notes

- Both options are strictly safer than the current formula.
- `sqrt_n` allows finer partitioning for large-n / low-d datasets (e.g.
  n=500, d=5: current=47, sqrt_n=22) which improves structural fidelity.
- `feat_floor` is more conservative for large-n / low-d (matches current
  behaviour exactly when `n^(2/3) * 0.75 > d + 2`).
- The `doc/` comment in `auto_tune_tree_params` referencing "1:1" ratio should
  be updated to describe the new formula.
