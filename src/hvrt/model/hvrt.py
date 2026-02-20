"""
HVRT: Hierarchical Variance-Retaining Transformer — pairwise interactions.

Uses the sum of z-score-normalised pairwise feature interactions as the
synthetic partitioning target.  Points with co-occurring extreme values
across feature pairs receive high |target|, naturally isolating structural
outliers into dedicated tree leaves.

Complexity: O(n · d²) for target computation (d = number of features).

Preferred for sample *reduction*; comparable to FastHVRT for expansion.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from .._base import _HVRTBase


class HVRT(_HVRTBase):
    """
    Hierarchical Variance-Retaining Transformer — pairwise interactions.

    Uses the sum of z-score-normalised pairwise feature interactions as the
    synthetic partitioning target.  Points with co-occurring extreme values
    across feature pairs receive high |target|, naturally isolating structural
    outliers into dedicated tree leaves.

    Complexity: O(n · d²) for target computation (d = number of features).

    Preferred for sample *reduction*; comparable to FastHVRT for expansion.

    Parameters
    ----------
    n_partitions : int or None
        Maximum leaf nodes.  Auto-tuned if None.
    min_samples_leaf : int or None
        Minimum samples per leaf.  Auto-tuned if None.
    max_depth : int or None
        Maximum tree depth.
    min_samples_per_partition : int, default=5
    y_weight : float, default=0.0
        0.0 = unsupervised; 1.0 = supervised (y-extremeness drives splits).
    auto_tune : bool, default=True
    mode : str, default='reduce'
        Default operation for fit_transform().
    bandwidth : float, default=0.5
        KDE bandwidth for expand().  0.5 achieves near-perfect tail
        preservation (error 0.004) while maintaining strong marginal
        fidelity (0.974).  Can be overridden per expand() call.
    random_state : int, default=42

    Examples
    --------
    >>> from hvrt import HVRT
    >>> model = HVRT().fit(X)
    >>> X_reduced = model.reduce(ratio=0.3)
    >>> X_synth   = model.expand(n=50000, min_novelty=0.2)
    >>> X_aug     = model.augment(n=30000)

    >>> # Sklearn pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> pipe = Pipeline([('hvrt', HVRT(mode='reduce', ratio=0.3))])
    >>> X_reduced = pipe.fit_transform(X)
    """

    def _compute_x_component(self, X_z):
        """
        Pairwise feature interaction synthetic target (O(d²)).

        For all feature pairs (i, j), i < j:
          1. Compute element-wise product X_z[:,i] ⊙ X_z[:,j]
          2. Z-score normalise each interaction column
          3. Sum across all interaction columns

        High |score| indicates samples where many feature pairs are jointly
        extreme — the defining characteristic of structural outliers.
        """
        n_samples, n_features = X_z.shape

        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        interactions_all = poly.fit_transform(X_z)
        interactions = interactions_all[:, n_features:]  # drop original columns

        means = interactions.mean(axis=0)
        stds = interactions.std(axis=0)
        stds_safe = np.where(stds > 1e-10, stds, 1.0)

        interactions_z = (interactions - means) / stds_safe
        # Zero out constant interactions (no variance information)
        interactions_z = np.where(stds[None, :] > 1e-10, interactions_z, 0.0)

        return interactions_z.sum(axis=1)
