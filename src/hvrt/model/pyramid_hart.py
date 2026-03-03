"""
PyramidHART — ℓ₁ cooperation statistic partitioner.

Uses A = |S| − ‖z‖₁ as the partitioning target, paired with MAD
normalisation and an absolute-error tree criterion.

The ℓ₁ zero-boundary {A = 0} is the union of coordinate hyperplanes
{zᵢ = 0} — a piecewise-linear, polyhedral structure.  In d = 3 the
eight sign-consistent octants form triangular faces on the unit sphere,
a *double pyramid* rather than the smooth T = 0 cone — hence the name.

Proposition 1 (Peace 2026)
--------------------------
1. Bounded range:  A ∈ [−r√d, 0].  T ∈ [−r², (d−1)r²] grows quadratically.
2. Minority-sign:  −A/2 = min(Σᵢ:zᵢ>0 zᵢ, Σᵢ:zᵢ<0 |zᵢ|).
3. Outlier cancellation:  if |zₖ| ≫ Σᵢ≠ₖ |zᵢ|, zₖ cancels from A exactly.
4. Degree-1 homogeneity:  A(λz) = λA(z) for λ ≥ 0.

Trade-offs vs HART
------------------
  Cov(A, Q) ≠ 0 — the T ⊥ Q orthogonality guarantee (Theorem 1) does
  not carry over.  E[A] is nonlinear in Σ and not noise-invariant.
  PyramidHART is a *complement* to HART, preferred when single-feature
  outlier immunity and a bounded, linearly-scaled cooperation signal
  are more important than the T ⊥ Q theoretical guarantees.
"""

from __future__ import annotations

import numpy as np

from .._hart_base import _HARTBase
from .._geometry import (
    compute_A,
    compute_S,
    compute_Q,
    compute_T,
    minority_sign_total,
    geometry_stats as _geo_stats,
)


class PyramidHART(_HARTBase):
    """
    PyramidHART — ℓ₁ cooperation statistic partitioner.

    Uses A = |S| − ‖z‖₁ as its partitioning target, with MAD normalisation
    and an absolute-error tree criterion (both inherited from ``_HARTBase``).

    Level sets {A = c} are axis-aligned piecewise-linear surfaces exactly
    representable by a decision tree.  The zero boundary {A = 0} is the
    union of coordinate hyperplanes {zᵢ = 0}; in d = 3 the sign-consistent
    regions form triangular faces on the unit sphere — a *double pyramid*.

    Properties (Proposition 1 of Peace 2026)
    -----------------------------------------
    1. **Bounded range**.
       A ∈ [−r√d, 0], scaling linearly with ‖z‖₂.  By contrast T
       ∈ [−r², (d−1)r²] grows quadratically with r and linearly with d.

    2. **Exact minority-sign interpretation**.
       −A/2 = min(Σᵢ:zᵢ>0 zᵢ, Σᵢ:zᵢ<0 |zᵢ|): the total magnitude
       of the minority-sign feature group.  Zero when all features share
       a sign; maximised when features split evenly between ±.

    3. **Single-feature outlier cancellation**.
       If |zₖ| ≫ Σᵢ≠ₖ |zᵢ|, the dominant component contributes equally
       to |S| and ‖z‖₁ and cancels exactly; A is then determined solely
       by the remaining components.  Empirically: a 50σ spike leaves the
       mean of A within 0.01 of its unperturbed value, while the mean of
       T shifts by O(50√d).

    4. **Degree-1 homogeneity**.
       A(λz) = λA(z) for λ ≥ 0 — A scales linearly with the data.
       T(λz) = λ²T(z) scales quadratically.

    Trade-offs vs HART
    ------------------
    * ``Cov(A, Q) ≠ 0``: the T ⊥ Q orthogonality guarantee (Theorem 1)
      does not carry over to A.
    * ``E[A]`` is a nonlinear (square-root) function of the covariance Σ.
    * ``E[A]`` is **not** preserved under isotropic additive noise
      (Theorem 3 does not apply).

    Preferred when
    --------------
    * Datasets contain isolated single-feature spikes (sensor faults,
      lab artefacts) that would inflate T without genuine cooperation.
    * A bounded, interpretable minority-sign decomposition is desired.
    * Quadratic growth of T causes instability at high d or large ‖z‖.

    Parameters
    ----------
    n_partitions : int or None
        Maximum leaf nodes.  Auto-tuned if None.
    min_samples_leaf : int or None
        Minimum samples per leaf.  Auto-tuned if None.
    max_depth : int or None
        Maximum tree depth.  No limit if None.
    min_samples_per_partition : int, default 5
    y_weight : float, default 0.0
        0.0 = unsupervised; 1.0 = supervised (y-extremeness drives splits).
    auto_tune : bool, default True
    n_jobs : int, default 1
    bandwidth : float or 'auto', default 'auto'
    random_state : int, default 42
    tree_splitter : {'best', 'random'}, default 'best'

    Examples
    --------
    >>> from hvrt import PyramidHART
    >>> import numpy as np
    >>> X = np.random.randn(1000, 8)
    >>> model = PyramidHART().fit(X)
    >>> X_red  = model.reduce(ratio=0.3)
    >>> X_syn  = model.expand(n=5000)
    >>> stats  = model.geometry_stats()
    >>> print(f"A mean: {stats['A_mean']:.3f}  "
    ...       f"sign-consistent: {stats['sign_consistent_fraction']:.1%}")
    """

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _compute_x_component(self, X_z: np.ndarray) -> np.ndarray:
        """A = |S| − ‖z‖₁.  ℓ₁ cooperation statistic.

        A ∈ [−r√d, 0] always; A = 0 iff all features share a sign.
        Degree-1 homogeneous: A(λz) = λA(z) for λ ≥ 0.

        Parameters
        ----------
        X_z : ndarray (n, d)
            MAD-normalised feature matrix.

        Returns
        -------
        A : ndarray (n,)  values ≤ 0
        """
        return compute_A(X_z)

    # ------------------------------------------------------------------
    # geometry_stats — full S / Q / T / A analysis
    # ------------------------------------------------------------------

    def geometry_stats(self, X=None):
        """Compute full cooperative geometry statistics: S, Q, T, and A.

        Extends the base-class T/Q cone analysis with the ℓ₁ cooperation
        statistic A = |S| − ‖z‖₁ and the algebraic properties from
        Proposition 1 of Peace (2026).

        Parameters
        ----------
        X : array-like (n, n_features) or None
            Data to evaluate.  ``None`` uses the fitted training data
            (X_z_ is reused directly, avoiding re-normalisation).

        Returns
        -------
        dict
            Global statistics (see Notes) plus:

            ``partitions`` : list of dict, one per tree leaf
                Keys: ``id``, ``n``,
                ``A_mean``, ``A_std``, ``sign_consistent_fraction``,
                ``minority_sign_mean``,
                ``T_mean``, ``T_std``, ``Q_mean``, ``Q_std``,
                ``cone_fraction``

        Notes
        -----
        **Shape**: ``n``, ``d``

        **S = Σᵢ zᵢ** (linear cooperation sum)
            ``S_mean``, ``S_std``, ``S_min``, ``S_max``

        **Q = ‖z‖²** (Mahalanobis norm squared)
            ``Q_mean``, ``Q_std``, ``Q_min``, ``Q_max``

        **T = S² − Q** (quadratic cooperation; HVRT / HART target)
            ``T_mean``, ``T_std``, ``T_min``, ``T_max``
            ``cone_fraction``           — fraction with T > 0
            ``cone_critical_angle_deg`` — arccos(1/√d) in degrees
            ``cooperation_ratio``       — E[T] / E[Q]

        **A = |S| − ‖z‖₁** (ℓ₁ cooperation; PyramidHART target)
            ``A_mean``, ``A_std``, ``A_min``, ``A_max``
            ``A_theoretical_lower_bound`` — −E[r]·√d  (Prop 1.1)
            ``sign_consistent_fraction``  — fraction with A ≈ 0
            ``minority_sign_mean``        — mean of −A/2  (Prop 1.2)

        **Cross-statistics**
            ``corr_TQ`` — ≈ 0 for isotropic z  [Theorem 1]
            ``corr_AQ`` — nonzero in general   [Prop 1 trade-off]
            ``corr_AT`` — A vs T agreement
            ``corr_AS`` — A vs |S|

        **Homogeneity verification** (scaling by λ = 2)
            ``A_homogeneity`` — A(2z)/A(z) ≈ 2.0  [Prop 1.4, degree-1]
            ``T_homogeneity`` — T(2z)/T(z) ≈ 4.0  [degree-2]
        """
        self._check_fitted()

        if X is None:
            X_z             = self.X_z_
            partition_ids   = self.partition_ids_
            unique_parts    = self.unique_partitions_
        else:
            X_z           = self._to_z(self._coerce_external_X(X))
            partition_ids = self.tree_.apply(X_z)
            unique_parts  = np.unique(partition_ids)

        # Global statistics (from _geometry module)
        stats = _geo_stats(X_z)

        # Per-sample arrays (recompute; _geo_stats already did this but
        # we need them for the per-partition loop without returning them)
        S   = compute_S(X_z)
        Q   = compute_Q(X_z)
        T   = S * S - Q
        A   = np.abs(S) - np.abs(X_z).sum(axis=1)
        mst = minority_sign_total(X_z)

        # Per-partition breakdown
        partitions = []
        for pid in unique_parts:
            mask  = partition_ids == pid
            n_p   = int(mask.sum())
            A_p   = A[mask]
            T_p   = T[mask]
            Q_p   = Q[mask]
            mst_p = mst[mask]
            partitions.append({
                'id':                      int(pid),
                'n':                       n_p,
                'A_mean':                  float(A_p.mean())           if n_p > 0 else float('nan'),
                'A_std':                   float(A_p.std())            if n_p > 1 else float('nan'),
                'sign_consistent_fraction':float((A_p >= -1e-10).mean()) if n_p > 0 else float('nan'),
                'minority_sign_mean':      float(mst_p.mean())         if n_p > 0 else float('nan'),
                'T_mean':                  float(T_p.mean())           if n_p > 0 else float('nan'),
                'T_std':                   float(T_p.std())            if n_p > 1 else float('nan'),
                'Q_mean':                  float(Q_p.mean())           if n_p > 0 else float('nan'),
                'Q_std':                   float(Q_p.std())            if n_p > 1 else float('nan'),
                'cone_fraction':           float((T_p > 0).mean())     if n_p > 0 else float('nan'),
            })

        stats['partitions'] = partitions
        return stats
