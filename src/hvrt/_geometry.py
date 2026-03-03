"""
Cooperative geometry statistics for whitened tabular data.

Definitions (Peace 2026, §2, Notation table):

    z ∈ ℝᵈ   whitened feature vector (mean/std or median/MAD normalised)

    S  =  Σᵢ zᵢ                  linear cooperation sum
    Q  =  ‖z‖²  =  Σᵢ zᵢ²       Mahalanobis squared norm
    T  =  S² − Q  =  2Σᵢ<ⱼ zᵢzⱼ  quadratic cooperation statistic
    A  =  |S| − ‖z‖₁              ℓ₁ cooperation statistic

Algebraic properties (Peace 2026, §3 Theorems 1–3; §2 Proposition 1)
-----------------------------------------------------------------------
  T / Q orthogonality    Cov(T, Q) = 0  for z ~ N(0, Iᵈ)          [Theorem 1]
  Cooperation = cov sum  E[T]  = Σᵢ≠ⱼ Σᵢⱼ                         [Theorem 2]
  Noise invariance       E[T̃]  = E[T] under isotropic additive noise [Theorem 3]

  A bounded range        A ∈ [−r√d, 0]  (triangle inequality)       [Prop 1.1]
  Minority-sign exact    −A/2 = min(Σᵢ:zᵢ>0 zᵢ, Σᵢ:zᵢ<0 |zᵢ|)     [Prop 1.2]
  Outlier cancellation   dominant |zₖ| cancels from A exactly        [Prop 1.3]
  Degree-1 homogeneity   A(λz) = λA(z)  for λ ≥ 0                   [Prop 1.4]
  (contrast)             T(λz) = λ²T(z)

Trade-offs
----------
  Cov(A, Q) ≠ 0 in general — A does not satisfy Theorem 1.
  E[A] is a nonlinear function of Σ — A does not satisfy Theorem 2.
  E[A] is not preserved under isotropic noise — A does not satisfy Theorem 3.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Scalar statistics (per-sample, vectorised over n points)
# ---------------------------------------------------------------------------

def compute_S(X_z: np.ndarray) -> np.ndarray:
    """S = Σᵢ zᵢ — linear cooperation sum.

    FastHVRT and FastHART use S as their (approximate) partitioning target.
    S = **1**ᵀz is the projection of z onto the cooperative all-ones axis.

    Parameters
    ----------
    X_z : ndarray (n, d)
        Pre-whitened feature matrix.

    Returns
    -------
    S : ndarray (n,)
    """
    return X_z.sum(axis=1)


def compute_Q(X_z: np.ndarray) -> np.ndarray:
    """Q = ‖z‖² = Σᵢ zᵢ² — Mahalanobis squared norm.

    Q is the quantity used by all distance-based ML models (k-NN, k-means,
    RBF-SVM, LOF).  The isotropic cooperative cone {T > 0} is exactly
    orthogonal to Q in the sense Cov(T, Q) = 0 [Theorem 1].

    Parameters
    ----------
    X_z : ndarray (n, d)

    Returns
    -------
    Q : ndarray (n,)
    """
    return (X_z ** 2).sum(axis=1)


def compute_T(X_z: np.ndarray) -> np.ndarray:
    """T = S² − Q = 2Σᵢ<ⱼ zᵢzⱼ — quadratic cooperation statistic.

    Partitions z-space by T-level hyperboloids.  HVRT and HART use T (via
    the pairwise target P ≈ T/2) as their partitioning signal.

    Geometry:
      T > 0 → inside the cooperative cone (features deviate together)
      T = 0 → double cone boundary, half-angle arccos(1/√d) from 1/√d axis
      T < 0 → outside the cone (anti-cooperative)

    T(λz) = λ²T(z): degree-2 homogeneous.

    Parameters
    ----------
    X_z : ndarray (n, d)

    Returns
    -------
    T : ndarray (n,)
    """
    S = X_z.sum(axis=1)
    Q = (X_z ** 2).sum(axis=1)
    return S * S - Q


def compute_A(X_z: np.ndarray) -> np.ndarray:
    """A = |S| − ‖z‖₁ — ℓ₁ cooperation statistic (PyramidHART target).

    A ≤ 0 always by the triangle inequality.
    A = 0  iff all components zᵢ share a sign (maximally cooperative).
    A is minimised when features are equally split between positive and
    negative with equal magnitudes.

    Level sets {A = c} are axis-aligned piecewise-linear surfaces
    representable exactly by a decision tree.  The zero boundary {A = 0}
    is the union of coordinate hyperplanes {zᵢ = 0}; in d = 3 the eight
    sign-consistent octants form triangular faces on the unit sphere —
    a *double pyramid* rather than the smooth T = 0 cone.

    Key properties (Proposition 1):
      A ∈ [−r√d, 0]          bounded range, linear in r = ‖z‖₂
      −A/2 = minority-sign total (see minority_sign_total)
      A(λz) = λA(z) for λ ≥ 0  (degree-1 homogeneous)
      Single dominant |zₖ| cancels from A exactly

    Parameters
    ----------
    X_z : ndarray (n, d)

    Returns
    -------
    A : ndarray (n,)   values ≤ 0
    """
    return np.abs(X_z.sum(axis=1)) - np.abs(X_z).sum(axis=1)


def minority_sign_total(X_z: np.ndarray) -> np.ndarray:
    """−A/2 = min(Σᵢ:zᵢ>0 zᵢ, Σᵢ:zᵢ<0 |zᵢ|) — minority-sign total.

    The exact total magnitude of the minority-sign group: how much the
    smaller-sign subset contributes to sign cancellation.

      = 0     when all features share a sign  (A = 0)
      = r√d/2 when features split ±1 with equal magnitude  (A = −r√d)

    This is the exact minority interpretation from Proposition 1, item 2.

    Parameters
    ----------
    X_z : ndarray (n, d)

    Returns
    -------
    mst : ndarray (n,)   values ≥ 0,  equals −A(X_z)/2
    """
    pos = np.maximum(X_z, 0.0).sum(axis=1)
    neg = np.maximum(-X_z, 0.0).sum(axis=1)
    return np.minimum(pos, neg)


# ---------------------------------------------------------------------------
# Comprehensive geometry statistics (global; no partition awareness)
# ---------------------------------------------------------------------------

def geometry_stats(X_z: np.ndarray) -> Dict[str, Any]:
    """Compute cooperative geometry statistics on pre-whitened data.

    Computes all four statistics (S, Q, T, A) and their cross-relationships
    as characterised in Peace (2026), Sections 2–3 and Proposition 1.

    Parameters
    ----------
    X_z : ndarray (n, d)
        Pre-whitened feature matrix (mean/std or median/MAD normalised).

    Returns
    -------
    dict
        See Notes for the full key listing.

    Notes
    -----
    **Shape**
        ``n``, ``d``

    **S = Σᵢ zᵢ** (linear cooperation sum)
        ``S_mean``, ``S_std``, ``S_min``, ``S_max``

    **Q = ‖z‖²** (Mahalanobis norm squared)
        ``Q_mean``, ``Q_std``, ``Q_min``, ``Q_max``

    **T = S² − Q** (quadratic cooperation statistic; HVRT / HART target)
        ``T_mean``, ``T_std``, ``T_min``, ``T_max``
        ``cone_fraction``             — fraction with T > 0 (cooperative cone)
        ``cone_critical_angle_deg``   — arccos(1/√d) in degrees (boundary half-angle)
        ``cooperation_ratio``         — E[T] / E[Q]; 0 for isotropic z

    **A = |S| − ‖z‖₁** (ℓ₁ cooperation statistic; PyramidHART target)
        ``A_mean``, ``A_std``, ``A_min``, ``A_max``
        ``A_theoretical_lower_bound`` — −E[r]·√d  (from Prop 1.1)
        ``sign_consistent_fraction``  — fraction with A ≈ 0 (all same sign)
        ``minority_sign_mean``        — mean of −A/2  (Prop 1.2)

    **Cross-statistics** (relationships between the four quantities)
        ``corr_TQ``  — Pearson r(T, Q); ≈ 0 for isotropic z [Theorem 1]
        ``corr_AQ``  — Pearson r(A, Q); nonzero in general [Prop 1 trade-off]
        ``corr_AT``  — Pearson r(A, T); similarity of the two cooperation signals
        ``corr_AS``  — Pearson r(A, |S|); A and |S| are structurally related

    **Homogeneity verification** (scaling z by λ = 2)
        ``A_homogeneity``  — mean A(2z)/A(z); should be ≈ 2.0 [Prop 1.4, degree-1]
        ``T_homogeneity``  — mean T(2z)/T(z); should be ≈ 4.0 [degree-2]
    """
    X_z = np.asarray(X_z, dtype=np.float64)
    n, d = X_z.shape

    S   = X_z.sum(axis=1)
    Q   = (X_z ** 2).sum(axis=1)
    T   = S * S - Q
    A   = np.abs(S) - np.abs(X_z).sum(axis=1)   # ≤ 0
    mst = minority_sign_total(X_z)
    r   = np.sqrt(Q)                              # ‖z‖₂ per point

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    # Degree-1 check: A(2z)/A(z) → 2
    # Degree-2 check: T(2z)/T(z) → 4
    X_z2 = 2.0 * X_z
    A2   = np.abs(X_z2.sum(axis=1)) - np.abs(X_z2).sum(axis=1)
    T2   = X_z2.sum(axis=1) ** 2 - (X_z2 ** 2).sum(axis=1)

    a_mask = A < -1e-10
    t_mask = np.abs(T) > 1e-10

    A_hom = float(np.mean(A2[a_mask] / A[a_mask])) if a_mask.any() else float('nan')
    T_hom = float(np.mean(T2[t_mask] / T[t_mask])) if t_mask.any() else float('nan')

    crit_angle = float(np.degrees(np.arccos(1.0 / np.sqrt(max(d, 1)))))
    coop_ratio = float(T.mean()) / (float(Q.mean()) + 1e-12)

    return {
        # shape
        'n':                          int(n),
        'd':                          int(d),
        # S
        'S_mean':                     float(S.mean()),
        'S_std':                      float(S.std()),
        'S_min':                      float(S.min()),
        'S_max':                      float(S.max()),
        # Q
        'Q_mean':                     float(Q.mean()),
        'Q_std':                      float(Q.std()),
        'Q_min':                      float(Q.min()),
        'Q_max':                      float(Q.max()),
        # T
        'T_mean':                     float(T.mean()),
        'T_std':                      float(T.std()),
        'T_min':                      float(T.min()),
        'T_max':                      float(T.max()),
        'cone_fraction':              float((T > 0).mean()),
        'cone_critical_angle_deg':    crit_angle,
        'cooperation_ratio':          coop_ratio,
        # A
        'A_mean':                     float(A.mean()),
        'A_std':                      float(A.std()),
        'A_min':                      float(A.min()),
        'A_max':                      float(A.max()),
        'A_theoretical_lower_bound':  float(-r.mean() * np.sqrt(d)),
        'sign_consistent_fraction':   float((A >= -1e-10).mean()),
        'minority_sign_mean':         float(mst.mean()),
        # cross-statistics
        'corr_TQ':                    _safe_corr(T, Q),
        'corr_AQ':                    _safe_corr(A, Q),
        'corr_AT':                    _safe_corr(A, T),
        'corr_AS':                    _safe_corr(A, np.abs(S)),
        # homogeneity verification
        'A_homogeneity':              A_hom,
        'T_homogeneity':              T_hom,
    }
