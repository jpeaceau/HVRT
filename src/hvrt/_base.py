"""
_HVRTBase: Shared logic for HVRT and FastHVRT.

Subclasses implement _compute_x_component(X_z) to define the synthetic
partitioning target.  Everything else — fit, reduce, expand, augment,
sklearn pipeline compatibility, presets, and utility methods — lives here.

Design principle
----------------
Operation parameters (n, ratio, method, variance_weighted, min_novelty,
bandwidth) belong in reduce() / expand() / augment() call signatures, not
in the constructor.  The constructor holds only model hyperparameters.

reduce() and expand() both accept an optional n_partitions argument that
causes the tree to be re-fitted with the requested granularity before
selecting / generating samples.  This "trim or expand" capability allows
callers to tune tree granularity per operation without a separate fit() call.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .utils import auto_tune_tree_params, fit_hvrt_tree
from .reduce import compute_partition_budgets, select_from_partitions
from .expand import (
    fit_partition_kdes,
    compute_expansion_budgets,
    sample_from_kdes,
    sample_with_strategy,
    sample_categorical_from_freqs,
    compute_novelty_distances,
)


def _build_cat_partition_freqs(X_cat, partition_ids, unique_partitions):
    """
    Build per-partition empirical frequency distributions for categorical columns.

    Parameters
    ----------
    X_cat : ndarray (n_samples, n_cat_cols)
        Original categorical column values (as stored in self.X_).
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray

    Returns
    -------
    dict {int pid -> list of (unique_values, probs) per categorical column}
    """
    freqs = {}
    for pid in unique_partitions:
        mask = partition_ids == pid
        X_part = X_cat[mask]
        col_freqs = []
        for j in range(X_part.shape[1]):
            unique_vals, counts = np.unique(X_part[:, j], return_counts=True)
            probs = counts.astype(float) / counts.sum()
            col_freqs.append((unique_vals, probs))
        freqs[int(pid)] = col_freqs
    return freqs


class _HVRTBase(BaseEstimator, TransformerMixin):
    """
    Base class shared by HVRT (pairwise interactions) and FastHVRT (z-score sum).

    Parameters
    ----------
    n_partitions : int or None
        Maximum leaf nodes in the default partitioning tree.  Auto-tuned if None.
        Individual reduce() / expand() calls can override this per-operation
        via their own n_partitions argument.
    min_samples_leaf : int or None
        Minimum samples per leaf.  Auto-tuned if None.
    max_depth : int or None
        Maximum tree depth.  No limit if None.
    min_samples_per_partition : int, default=5
        Minimum samples selected from any partition during reduce().
    y_weight : float, default=0.0
        Blend weight for y-extremeness in the synthetic target.
        0.0 = fully unsupervised; 1.0 = supervised (y drives splits).
    auto_tune : bool, default=True
        Auto-tune tree hyperparameters from dataset size when explicit
        values are not provided.
    mode : str, default='reduce'
        Default operation dispatched by fit_transform().
        One of 'reduce', 'expand', 'augment'.
    bandwidth : float, default=0.5
        Default KDE bandwidth used by expand().  0.5 achieves near-perfect
        tail preservation (error 0.004) while maintaining strong marginal
        fidelity (0.974).  Individual expand() calls may override this.
    random_state : int, default=42
    """

    def __init__(
        self,
        n_partitions=None,
        min_samples_leaf=None,
        max_depth=None,
        min_samples_per_partition=5,
        y_weight=0.0,
        auto_tune=True,
        mode='reduce',
        bandwidth=0.5,
        random_state=42,
    ):
        self.n_partitions = n_partitions
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_samples_per_partition = min_samples_per_partition
        self.y_weight = y_weight
        self.auto_tune = auto_tune
        self.mode = mode
        self.bandwidth = bandwidth
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _compute_x_component(self, X_z):
        """Return the X-based partitioning signal.  Overridden by subclasses."""
        raise NotImplementedError(
            "_compute_x_component must be implemented by HVRT or FastHVRT."
        )

    # ------------------------------------------------------------------
    # Synthetic target
    # ------------------------------------------------------------------

    def _compute_synthetic_target(self, X_z, y=None):
        """
        Compute the full synthetic target, blending X-component with y if needed.

        Parameters
        ----------
        X_z : ndarray (n_samples, n_features)
        y : ndarray (n_samples,) or None

        Returns
        -------
        target : ndarray (n_samples,)
        """
        x_component = self._compute_x_component(X_z)

        if y is None or self.y_weight == 0.0:
            return x_component

        y_norm = (y - y.mean()) / (y.std() + 1e-10)
        y_extremeness = np.abs(y_norm - np.median(y_norm))
        y_component = (y_extremeness - y_extremeness.mean()) / (
            y_extremeness.std() + 1e-10
        )

        return (1.0 - self.y_weight) * x_component + self.y_weight * y_component

    # ------------------------------------------------------------------
    # Categorical encoding
    # ------------------------------------------------------------------

    def _encode_categorical(self, X_cat, fit=True):
        if not hasattr(self, 'label_encoders_'):
            self.label_encoders_ = {}

        X_encoded = np.zeros(X_cat.shape, dtype=np.float64)
        for col in range(X_cat.shape[1]):
            if fit:
                le = LabelEncoder()
                X_encoded[:, col] = le.fit_transform(X_cat[:, col].astype(str))
                self.label_encoders_[col] = le
            else:
                le = self.label_encoders_[col]
                X_encoded[:, col] = le.transform(X_cat[:, col].astype(str))
        return X_encoded

    # ------------------------------------------------------------------
    # Z-score transform / inverse transform
    # ------------------------------------------------------------------

    def _to_z(self, X):
        """Transform X to z-score space using fitted scalers."""
        parts = []
        if self.continuous_mask_.any():
            parts.append(self.scaler_.transform(X[:, self.continuous_mask_]))
        if self.categorical_mask_.any():
            X_enc = self._encode_categorical(X[:, self.categorical_mask_], fit=False)
            parts.append(self.cat_scaler_.transform(X_enc))
        return np.hstack(parts) if len(parts) > 1 else parts[0]

    def _from_z(self, X_z):
        """
        Inverse-transform from z-score space to original feature scale.

        Z-space column layout: [continuous... | categorical...]
        Output column layout:  original order (per continuous_mask_).

        Note: expand() no longer calls this method for the categorical path —
        it samples categoricals directly from per-partition empirical
        distributions.  This method is retained for completeness and decodes
        categorical columns by applying le.inverse_transform() to recover the
        original class labels.  The output dtype is object when categorical
        features are present (to accommodate string labels).
        """
        n_samples = len(X_z)
        n_orig = len(self.continuous_mask_)
        has_cat = self.categorical_mask_.any()

        X_out = np.empty((n_samples, n_orig), dtype=object if has_cat else float)

        z_offset = 0

        if self.continuous_mask_.any():
            n_cont = int(self.continuous_mask_.sum())
            X_cont = self.scaler_.inverse_transform(X_z[:, z_offset:z_offset + n_cont])
            X_out[:, self.continuous_mask_] = X_cont
            z_offset += n_cont

        if has_cat:
            n_cat = int(self.categorical_mask_.sum())
            X_cat_raw = self.cat_scaler_.inverse_transform(
                X_z[:, z_offset:z_offset + n_cat]
            )
            X_cat_int = np.round(X_cat_raw).astype(int)
            cat_col_positions = np.where(self.categorical_mask_)[0]
            for local_idx in range(n_cat):
                le = self.label_encoders_.get(local_idx)
                if le is not None:
                    codes = np.clip(
                        X_cat_int[:, local_idx], 0, len(le.classes_) - 1
                    )
                    X_out[:, cat_col_positions[local_idx]] = le.inverse_transform(codes)
                else:
                    X_out[:, cat_col_positions[local_idx]] = X_cat_int[:, local_idx]

        return X_out

    # ------------------------------------------------------------------
    # Internal tree helpers
    # ------------------------------------------------------------------

    def _resolve_tree_params(self, n_samples, n_features, n_partitions_override=None):
        """
        Resolve max_leaf_nodes and min_samples_leaf for a tree fit.

        n_partitions_override takes precedence over self.n_partitions and
        auto-tuning, allowing per-operation tree granularity control.

        When n_partitions_override is given, min_samples_leaf is set to allow
        that many leaves rather than the usual feature-ratio formula, so the
        tree can actually reach the requested leaf count.
        """
        max_leaf_nodes = n_partitions_override if n_partitions_override is not None \
            else self.n_partitions
        min_samples_leaf = self.min_samples_leaf

        if n_partitions_override is not None:
            # Honour the explicit leaf budget: set min_samples_leaf small enough
            # to allow n_partitions_override leaves, unless the user pinned it.
            if min_samples_leaf is None:
                min_samples_leaf = max(2, n_samples // (n_partitions_override * 3))
        elif self.auto_tune:
            max_leaf_nodes, min_samples_leaf = auto_tune_tree_params(
                n_samples, n_features, max_leaf_nodes, min_samples_leaf
            )
        else:
            if max_leaf_nodes is None:
                max_leaf_nodes = max(2, n_samples // 100)
            if min_samples_leaf is None:
                min_samples_leaf = max(5, n_samples // 1000)

        return max_leaf_nodes, min_samples_leaf

    def _fit_tree(self, X_z, target, n_partitions_override=None):
        """
        (Re-)fit the partitioning tree and update partition state.

        If n_partitions_override is provided the tree is always refitted with
        that leaf budget; otherwise a previously fitted tree is reused when the
        requested granularity matches (no-op for same parameters).
        """
        n_samples, n_features = X_z.shape
        max_leaf, min_leaf = self._resolve_tree_params(
            n_samples, n_features, n_partitions_override
        )

        # Reuse existing tree if parameters match
        if (
            hasattr(self, 'tree_')
            and getattr(self, '_tree_max_leaf_', None) == max_leaf
            and getattr(self, '_tree_min_leaf_', None) == min_leaf
        ):
            return

        self.tree_ = fit_hvrt_tree(
            X_z, self._last_target_,
            max_leaf, min_leaf, self.max_depth, self.random_state,
        )
        self._tree_max_leaf_ = max_leaf
        self._tree_min_leaf_ = min_leaf
        self.partition_ids_ = self.tree_.apply(X_z)
        self.unique_partitions_ = np.unique(self.partition_ids_)
        self.n_partitions_ = len(self.unique_partitions_)
        self._kdes_ = None  # invalidate cached KDEs when tree changes
        self._cat_partition_freqs_ = None  # invalidate categorical freqs when tree changes

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y=None, feature_types=None):
        """
        Fit the HVRT partitioner.

        Normalises X, computes the synthetic target, and fits the default
        partitioning tree.  Calling reduce() or expand() with an explicit
        n_partitions argument will re-fit the tree to that granularity without
        requiring a new fit() call.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,), optional
        feature_types : list of str or None
            'continuous' or 'categorical' per column.  All continuous if None.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Feature type masks
        if feature_types is None:
            feature_types = ['continuous'] * n_features
        continuous_mask = np.array([ft == 'continuous' for ft in feature_types])
        categorical_mask = ~continuous_mask
        self.continuous_mask_ = continuous_mask
        self.categorical_mask_ = categorical_mask

        # Normalize
        parts = []
        if continuous_mask.any():
            self.scaler_ = StandardScaler()
            parts.append(self.scaler_.fit_transform(X[:, continuous_mask]))
        if categorical_mask.any():
            X_enc = self._encode_categorical(X[:, categorical_mask], fit=True)
            self.cat_scaler_ = StandardScaler()
            parts.append(self.cat_scaler_.fit_transform(X_enc))
        X_z = np.hstack(parts) if len(parts) > 1 else parts[0]

        # Synthetic target
        y_arr = np.asarray(y, dtype=np.float64).ravel() if y is not None else None
        target = self._compute_synthetic_target(X_z, y_arr)

        # Store normalized data and target so _fit_tree can reuse them
        self.X_ = X
        self.X_z_ = X_z
        self._last_target_ = target
        self._kdes_ = None
        self._cat_partition_freqs_ = None
        # Clear cached tree params so first _fit_tree always runs
        self._tree_max_leaf_ = None
        self._tree_min_leaf_ = None

        # Fit the default tree
        self._fit_tree(X_z, target)

        return self

    # ------------------------------------------------------------------
    # reduce
    # ------------------------------------------------------------------

    def reduce(
        self,
        n=None,
        ratio=None,
        method='fps',
        variance_weighted=True,
        return_indices=False,
        n_partitions=None,
    ):
        """
        Select a representative subset of the fitted data.

        Parameters
        ----------
        n : int or None
            Absolute target count.
        ratio : float or None
            Proportion to keep (e.g. 0.3 = 30%).  Provide n or ratio, not both.
        method : str, default='fps'
            Within-partition selection strategy.
            'fps' / 'centroid_fps' : centroid-seeded Furthest Point Sampling
            'medoid_fps'           : medoid-seeded FPS
            'variance_ordered'     : highest local-variance samples
            'stratified'           : random within partition
        variance_weighted : bool, default=True
            Oversample high-variance (tail) partitions.
        return_indices : bool, default=False
            When True, also return indices into the original X.
        n_partitions : int or None
            Override the number of tree leaves for this operation.
            Triggers a tree re-fit when it differs from the current tree.
            Useful for aggressive reductions (more partitions) or for
            rapid high-ratio reductions (fewer partitions).

        Returns
        -------
        X_reduced : ndarray (n_target, n_features)  original scale
        indices   : ndarray (n_target,)  [only when return_indices=True]
        """
        self._check_fitted()
        n_target = self._resolve_n(n, ratio, len(self.X_), 'reduce')

        # Re-fit tree if a specific partition count is requested
        if n_partitions is not None:
            self._fit_tree(self.X_z_, self._last_target_, n_partitions)

        budgets = compute_partition_budgets(
            self.partition_ids_, self.unique_partitions_,
            n_target, self.min_samples_per_partition,
            variance_weighted, self.X_z_,
        )
        indices = select_from_partitions(
            self.X_z_, self.partition_ids_, self.unique_partitions_,
            budgets, method, self.random_state,
        )

        X_reduced = self.X_[indices]
        if return_indices:
            return X_reduced, indices
        return X_reduced

    # ------------------------------------------------------------------
    # expand
    # ------------------------------------------------------------------

    def expand(
        self,
        n,
        min_novelty=0.0,
        variance_weighted=False,
        bandwidth=None,
        adaptive_bandwidth=False,
        generation_strategy=None,
        return_novelty_stats=False,
        n_partitions=None,
    ):
        """
        Generate synthetic samples via per-partition multivariate KDE.

        KDEs are fitted lazily on the first call (or when bandwidth / tree
        granularity changes).

        Parameters
        ----------
        n : int
            Number of synthetic samples to generate.
        min_novelty : float, default=0.0
            Minimum Euclidean distance (z-score space) from any original
            sample.  0.0 disables the novelty constraint.
        variance_weighted : bool, default=False
            True  → oversample high-variance partitions (tails).
            False → proportional to partition size (preserves distribution).
        bandwidth : float or None
            KDE bandwidth scalar.  None → Scott's rule.
        adaptive_bandwidth : bool, default=False
            When True, each partition's KDE bandwidth scales with both the
            local expansion factor and partition size::

                bw_p = scott_p × max(1, budget_p / n_p) ^ (1/d)

            where ``scott_p = n_p^(-1/(d+4))`` is Scott's rule for that
            partition, ``budget_p`` is the number of synthetic samples
            allocated to partition p, ``n_p`` is the partition's real sample
            count, and ``d`` is the number of continuous features.

            At expansion ratio 1 the bandwidth equals Scott's rule.  At
            higher ratios the KDE spreads proportionally so that synthetic
            samples explore further from the observed data without clustering.
            Ignored when ``generation_strategy`` is set.
        generation_strategy : str, callable, or None, default=None
            Override the per-partition generation strategy used to draw
            continuous synthetic samples.  When ``None`` (default) the built-in
            multivariate KDE path is used (respecting ``bandwidth`` and
            ``adaptive_bandwidth``).

            Pass a string to select a built-in strategy by name:

            ``'multivariate_kde'``
                Default multivariate Gaussian KDE (Scott's rule bandwidth).
            ``'univariate_kde_copula'``
                Per-feature 1-D KDE marginals + Gaussian copula for joint
                dependence.  More flexible marginals, similar correlation
                preservation.
            ``'bootstrap_noise'``
                Resample with replacement + per-feature Gaussian noise
                (10 % of partition std).  Fastest option.

            Any callable with signature
            ``(X_partition, budget, random_state) -> ndarray``
            is also accepted.
        return_novelty_stats : bool, default=False
            When True, also return a dict with distance statistics.
        n_partitions : int or None
            Override the number of tree leaves for this operation.
            Triggers a tree re-fit (and KDE re-fit) when it differs from the
            current tree.  Fewer partitions → smoother, coarser KDE.
            More partitions → finer local density estimation.

        Returns
        -------
        X_synthetic : ndarray (n, n_features)  original scale
        stats : dict  [only when return_novelty_stats=True]
            Keys: 'min', 'mean', 'p5'
        """
        self._check_fitted()

        # Re-fit tree if a specific partition count is requested
        if n_partitions is not None:
            self._fit_tree(self.X_z_, self._last_target_, n_partitions)

        # _ensure_kdes always runs: populates _cat_partition_freqs_ and the
        # default (Scott's rule) _kdes_ cache used by the non-adaptive path.
        self._ensure_kdes(bandwidth)

        budgets = compute_expansion_budgets(
            self.partition_ids_, self.unique_partitions_,
            n, variance_weighted, self.X_z_,
        )

        n_cont = int(self.continuous_mask_.sum())

        # Adaptive path: fit fresh per-partition KDEs whose bandwidth scales
        # with the local expansion factor.  These are NOT cached because they
        # depend on the budget, which varies with n and variance_weighted.
        if adaptive_bandwidth and n_cont > 0:
            bw_per_partition = self._adaptive_bandwidths(budgets, n_cont)
            kdes = fit_partition_kdes(
                self.X_z_[:, :n_cont], self.partition_ids_,
                self.unique_partitions_, bw_per_partition,
            )
        else:
            kdes = self._kdes_
        n_cat = int(self.categorical_mask_.sum())
        n_orig = len(self.continuous_mask_)

        # ── Resolve generation strategy (if provided) ────────────────────
        if generation_strategy is not None:
            if isinstance(generation_strategy, str):
                from .generation_strategies import get_generation_strategy
                strategy_fn = get_generation_strategy(generation_strategy)
            else:
                strategy_fn = generation_strategy
        else:
            strategy_fn = None

        # ── Continuous dimensions: sample, inverse-transform ─────────────
        if n_cont > 0:
            X_z_cont = self.X_z_[:, :n_cont]
            if strategy_fn is not None:
                # Strategy path: bypass KDE entirely
                X_synth_cont_z = sample_with_strategy(
                    strategy_fn, self.unique_partitions_, budgets,
                    X_z_cont, self.partition_ids_,
                    self.random_state, min_novelty=min_novelty,
                )
            else:
                # Default KDE path (supports adaptive_bandwidth, caching)
                X_synth_cont_z = sample_from_kdes(
                    kdes, self.unique_partitions_, budgets,
                    X_z_cont, self.partition_ids_,
                    self.random_state, min_novelty=min_novelty,
                )
            X_synth_cont = self.scaler_.inverse_transform(X_synth_cont_z)
        else:
            X_synth_cont_z = np.empty((n, 0))
            X_synth_cont = np.empty((n, 0))

        # ── Categorical dimensions: sample from per-partition empirical freqs ─
        # Categorical transformation (LabelEncoder + StandardScaler) was used
        # only for tree splitting and variance representation.  The generated
        # values come directly from the observed category distribution so that
        # output categories are always drawn from the original value set.
        if n_cat > 0:
            X_synth_cat = sample_categorical_from_freqs(
                self._cat_partition_freqs_, self.unique_partitions_, budgets,
                self.random_state,
            )
            X_synthetic = np.empty((n, n_orig))
            if n_cont > 0:
                X_synthetic[:, self.continuous_mask_] = X_synth_cont
            X_synthetic[:, self.categorical_mask_] = X_synth_cat
        else:
            X_synthetic = X_synth_cont

        if return_novelty_stats:
            # Novelty is measured in continuous z-score space only
            ref_z = self.X_z_[:, :n_cont] if n_cont > 0 else self.X_z_
            dists = compute_novelty_distances(X_synth_cont_z, ref_z)
            stats = {
                'min': float(dists.min()),
                'mean': float(dists.mean()),
                'p5': float(np.percentile(dists, 5)),
            }
            return X_synthetic, stats

        return X_synthetic

    # ------------------------------------------------------------------
    # augment
    # ------------------------------------------------------------------

    def augment(self, n, min_novelty=0.0, variance_weighted=False, n_partitions=None):
        """
        Return original X concatenated with (n - len(X)) synthetic samples.

        Parameters
        ----------
        n : int
            Total output size.  Must be strictly greater than len(X).
        min_novelty : float, default=0.0
        variance_weighted : bool, default=False
        n_partitions : int or None
            Override tree granularity for the expansion step.

        Returns
        -------
        X_augmented : ndarray (n, n_features)
        """
        self._check_fitted()
        n_orig = len(self.X_)
        if n <= n_orig:
            raise ValueError(
                f"augment() requires n ({n}) > original sample count ({n_orig})."
            )
        n_synthetic = n - n_orig
        X_synth = self.expand(
            n_synthetic,
            min_novelty=min_novelty,
            variance_weighted=variance_weighted,
            n_partitions=n_partitions,
        )
        return np.vstack([self.X_, X_synth])

    # ------------------------------------------------------------------
    # Sklearn fit_transform (mode-aware)
    # ------------------------------------------------------------------

    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit and transform in a single step.  Behaviour depends on self.mode.

        mode='reduce' : calls reduce(**kwargs)
        mode='expand' : calls expand(**kwargs)  — n must be supplied in kwargs
        mode='augment': calls augment(**kwargs) — n must be supplied in kwargs

        Examples
        --------
        >>> pipe = Pipeline([('hvrt', HVRT(mode='reduce'))])
        >>> X_red = pipe.fit_transform(X, ratio=0.3)

        >>> pipe = Pipeline([('hvrt', FastHVRT(mode='expand'))])
        >>> X_synth = pipe.fit_transform(X, n=50000, min_novelty=0.2)
        """
        self.fit(X, y)

        if self.mode == 'reduce':
            return self.reduce(**kwargs)
        elif self.mode == 'expand':
            return self.expand(**kwargs)
        elif self.mode == 'augment':
            return self.augment(**kwargs)
        else:
            raise ValueError(
                f"Unknown mode {self.mode!r}. Use 'reduce', 'expand', or 'augment'."
            )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_partitions(self):
        """
        Return metadata for every partition.

        Returns
        -------
        list of dict, one per partition:
            'id'         : int   leaf node ID
            'size'       : int   number of original samples
            'mean_abs_z' : float mean |z-score| across samples and features
            'variance'   : float mean per-feature variance within partition
        """
        self._check_fitted()
        result = []
        for pid in self.unique_partitions_:
            mask = self.partition_ids_ == pid
            X_part = self.X_z_[mask]
            result.append({
                'id': int(pid),
                'size': int(mask.sum()),
                'mean_abs_z': float(np.mean(np.abs(X_part))),
                'variance': float(X_part.var()),
            })
        return result

    def compute_novelty(self, X_new):
        """
        Compute minimum Euclidean distance from each sample in X_new to any
        fitted original sample (in z-score space).

        Parameters
        ----------
        X_new : array-like (n, n_features)

        Returns
        -------
        min_dists : ndarray (n,)
        """
        self._check_fitted()
        X_new = np.asarray(X_new, dtype=np.float64)
        X_new_z = self._to_z(X_new)
        return compute_novelty_distances(X_new_z, self.X_z_)

    @classmethod
    def recommend_params(cls, X):
        """
        Return auto-tuned parameter recommendations for a given dataset.

        Parameters
        ----------
        X : array-like (n_samples, n_features)

        Returns
        -------
        dict with keys 'n_partitions', 'min_samples_leaf', 'n_samples', 'n_features'
        """
        X = np.asarray(X)
        n, d = X.shape
        max_leaf, min_leaf = auto_tune_tree_params(n, d)
        return {
            'n_partitions': max_leaf,
            'min_samples_leaf': min_leaf,
            'n_samples': n,
            'n_features': d,
        }

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def for_ml_reduction(cls, **kwargs):
        """
        Preset for ML training-set compression.

        Recommended usage::

            model = HVRT.for_ml_reduction().fit(X, y)
            X_reduced = model.reduce(ratio=0.3)
            # variance_weighted=True, method='fps' are the reduce() defaults
        """
        defaults = dict(mode='reduce')
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_synthetic_data(cls, **kwargs):
        """
        Preset for privacy-safe synthetic data generation.

        Recommended usage::

            model = HVRT.for_synthetic_data().fit(X)
            X_synth = model.expand(n=50000, min_novelty=0.2)
            # variance_weighted=False preserves marginal distribution
        """
        defaults = dict(mode='expand')
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_anomaly_augmentation(cls, **kwargs):
        """
        Preset for tail-preserving anomaly / fraud augmentation.

        Recommended usage::

            model = HVRT.for_anomaly_augmentation().fit(X)
            X_aug = model.augment(n=30000, min_novelty=0.1, variance_weighted=True)
        """
        defaults = dict(mode='augment')
        defaults.update(kwargs)
        return cls(**defaults)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not hasattr(self, 'X_'):
            raise ValueError("Model must be fitted before calling this method.")

    def _resolve_n(self, n, ratio, n_orig, mode):
        if n is not None and ratio is not None:
            raise ValueError("Provide either n or ratio, not both.")
        if n is None and ratio is None:
            raise ValueError(f"{mode}() requires n or ratio to be specified.")
        if ratio is not None:
            n = max(1, int(n_orig * ratio))
        return n

    def _ensure_kdes(self, bandwidth=None):
        """
        Lazily fit per-partition KDEs (continuous dims only) and build
        per-partition categorical frequency tables.  Re-fits when bandwidth
        or tree changes.

        KDEs are fitted on z-scored CONTINUOUS features only.  Categorical
        columns are never passed through the KDE; their values are sampled
        from per-partition empirical distributions instead.
        """
        bw = bandwidth if bandwidth is not None else self.bandwidth
        n_cont = int(self.continuous_mask_.sum())

        if self._kdes_ is None or bw != getattr(self, '_kdes_bandwidth_', object()):
            if n_cont > 0:
                # Fit KDE on continuous z-score dims only (first n_cont columns)
                self._kdes_ = fit_partition_kdes(
                    self.X_z_[:, :n_cont], self.partition_ids_, self.unique_partitions_, bw
                )
            else:
                # All-categorical data: no KDE needed
                self._kdes_ = {}
            self._kdes_bandwidth_ = bw

        # Build categorical frequency tables if not already built
        if self.categorical_mask_.any() and self._cat_partition_freqs_ is None:
            self._cat_partition_freqs_ = _build_cat_partition_freqs(
                self.X_[:, self.categorical_mask_],
                self.partition_ids_,
                self.unique_partitions_,
            )

    def _adaptive_bandwidths(self, budgets, n_cont):
        """
        Compute per-partition adaptive KDE bandwidth::

            bw_p = scott_p × max(1, budget_p / n_p) ^ (1/d)

        Parameters
        ----------
        budgets   : ndarray of int  synthetic samples per partition
        n_cont    : int             continuous feature count (KDE dimensionality)

        Returns
        -------
        dict {int pid: float}
            Per-partition bandwidth factors, ready to pass as ``bw_method``
            to ``scipy.stats.gaussian_kde``.
        """
        d = max(1, n_cont)
        bw_dict = {}
        for pid, budget in zip(self.unique_partitions_, budgets):
            n_p = int(np.sum(self.partition_ids_ == pid))
            if n_p < 2:
                bw_dict[int(pid)] = 'scott'
                continue
            # Scott's rule bandwidth factor for this partition
            scott = n_p ** (-1.0 / (d + 4))
            # Local expansion ratio — scale up proportionally, never shrink
            local_ratio = max(1.0, float(budget) / n_p)
            bw_dict[int(pid)] = float(scott * local_ratio ** (1.0 / d))
        return bw_dict
