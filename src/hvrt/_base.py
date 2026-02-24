"""
_HVRTBase: shared logic for HVRT and FastHVRT.

Subclasses override ``_compute_x_component(X_z)`` to define the synthetic
partitioning target.  All other behaviour — fit, reduce, expand, augment,
sklearn pipeline compatibility, presets, and utility methods — is
implemented here.

Constructor parameters are model hyperparameters that affect partitioning.
Operation parameters (n, ratio, method, variance_weighted, etc.) belong in
reduce_params / expand_params / augment_params (for pipeline use) or directly
in the call signatures of reduce() / expand() / augment() (for direct use).

``reduce()`` and ``expand()`` both accept an optional ``n_partitions``
argument that re-fits the tree to the requested leaf count for that call
only, without a separate ``fit()`` call.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Callable, Literal, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ._warnings import HVRTDeprecationWarning, HVRTFeatureWarning
from ._params import ReduceParams, ExpandParams, AugmentParams
from ._preprocessing import (
    fit_preprocess_data,
    to_z,
    from_z,
    build_cat_partition_freqs,
)
from ._partitioning import auto_tune_tree_params, fit_hvrt_tree, resolve_tree_params
from .reduce import compute_partition_budgets, select_from_partitions
from .expand import (
    fit_partition_kdes,
    compute_expansion_budgets,
    sample_from_kdes,
    sample_categorical_from_freqs,
    compute_novelty_distances,
)


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
    n_jobs : int, default=1
        Number of parallel jobs used for within-partition work (KDE fitting,
        KDE sampling, selection strategies).  ``-1`` uses all available CPU
        cores.  Parallelism is provided by ``joblib`` (already a transitive
        dependency of scikit-learn).  Overhead dominates for small datasets
        (n < ~1000 or n_partitions < 8); use ``n_jobs=1`` in those cases.
    bandwidth : float or 'auto', default='auto'
        Default KDE bandwidth used by expand().  Options:

        * ``float`` — scalar passed directly as scipy's ``bw_method``;
          kernel covariance = ``bandwidth² × data_cov`` within each partition.
          0.1 (10 % of within-partition std) is recommended for most cases.
        * ``'scott'`` / ``'silverman'`` — scipy's built-in rules.
          Reliably suboptimal for HVRT partitions; avoid.
        * ``'auto'`` — chooses at expand-time based on mean partition size:
          narrow Gaussian (h=0.1) when partitions are large enough for stable
          covariance estimation (mean size ≥ max(15, 2·d)); Epanechnikov
          product kernel otherwise.  Robust default when partition count
          is unknown in advance.

        Individual expand() calls may override this per-call.
    random_state : int, default=42
    reduce_params : ReduceParams or None
        Operation parameters for fit_transform() pipeline use.
        When set, fit_transform() calls reduce(**vars(reduce_params)).
    expand_params : ExpandParams or None
        Operation parameters for fit_transform() pipeline use.
        When set, fit_transform() calls expand(**vars(expand_params)).
    augment_params : AugmentParams or None
        Operation parameters for fit_transform() pipeline use.
        When set, fit_transform() calls augment(**vars(augment_params)).
    mode : str or None
        Deprecated.  Use reduce_params / expand_params / augment_params
        instead.  Kept for backward compatibility; emits
        HVRTDeprecationWarning when fit_transform() is called.
    """

    def __init__(
        self,
        n_partitions=None,
        min_samples_leaf=None,
        max_depth=None,
        min_samples_per_partition=5,
        y_weight=0.0,
        auto_tune=True,
        n_jobs=1,
        bandwidth='auto',
        random_state=42,
        reduce_params=None,
        expand_params=None,
        augment_params=None,
        mode=None,
    ):
        self.n_partitions = n_partitions
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_samples_per_partition = min_samples_per_partition
        self.y_weight = y_weight
        self.auto_tune = auto_tune
        self.n_jobs = n_jobs
        self.bandwidth = bandwidth
        self.random_state = random_state
        self.reduce_params = reduce_params
        self.expand_params = expand_params
        self.augment_params = augment_params
        self.mode = mode

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
    # Preprocessing delegates
    # ------------------------------------------------------------------

    def _to_z(self, X):
        return to_z(
            X,
            self.continuous_mask_, self.categorical_mask_,
            self.scaler_, self.cat_scaler_, self.label_encoders_,
        )

    def _from_z(self, X_z):
        return from_z(
            X_z,
            self.continuous_mask_, self.categorical_mask_,
            self.scaler_, self.cat_scaler_, self.label_encoders_,
        )

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------

    def _fit_tree(self, X_z, target, n_partitions_override=None, is_reduction=False):
        """
        (Re-)fit the partitioning tree and update partition state.

        If n_partitions_override is provided the tree is always refitted
        with that leaf budget; otherwise a previously fitted tree is reused
        when the resolved parameters match (no-op).

        Parameters
        ----------
        is_reduction : bool, default False
            When True, the stricter 40:1 sample-to-feature auto-tune formula
            is used for min_samples_leaf.  When False, the permissive 1:1
            formula is used, allowing many more fine-grained partitions.
        """
        n_samples, n_features = X_z.shape
        max_leaf, min_leaf = resolve_tree_params(
            n_samples, n_features,
            n_partitions_override=n_partitions_override,
            n_partitions=self.n_partitions,
            min_samples_leaf=self.min_samples_leaf,
            auto_tune=self.auto_tune,
            is_reduction=is_reduction,
        )

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
        self._kdes_ = None
        self._cat_partition_freqs_ = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y=None, feature_types=None):
        """
        Fit the HVRT partitioner.

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
        (
            self.X_,
            self.X_z_,
            self.continuous_mask_,
            self.categorical_mask_,
            self.scaler_,
            self.cat_scaler_,
            self.label_encoders_,
            self.feature_names_in_,
        ) = fit_preprocess_data(X, feature_types)

        y_arr = np.asarray(y, dtype=np.float64).ravel() if y is not None else None
        self._last_target_ = self._compute_synthetic_target(self.X_z_, y_arr)

        self._kdes_ = None
        self._cat_partition_freqs_ = None
        self._tree_max_leaf_ = None
        self._tree_min_leaf_ = None

        # Use the stricter 40:1 leaf constraint only when reduction is the
        # declared pipeline operation; default to the permissive 1:1 formula
        # for expansion / augmentation (or when no params are declared).
        _is_reduction = (
            self.reduce_params is not None
            and self.expand_params is None
            and self.augment_params is None
        )
        self._fit_tree(self.X_z_, self._last_target_, is_reduction=_is_reduction)

        return self

    # ------------------------------------------------------------------
    # reduce
    # ------------------------------------------------------------------

    def reduce(
        self,
        n: Optional[int] = None,
        ratio: Optional[float] = None,
        method: Union[
            Literal['fps', 'centroid_fps', 'medoid_fps', 'variance_ordered', 'stratified'],
            Callable,
        ] = 'fps',
        variance_weighted: bool = True,
        return_indices: bool = False,
        n_partitions: Optional[int] = None,
        X: Optional[np.ndarray] = None,
    ):
        """
        Select a representative subset of a dataset.

        Parameters
        ----------
        n : int, optional
            Absolute target count.  Provide n or ratio, not both.
        ratio : float, optional
            Proportion to keep, e.g. 0.3 keeps 30 %.
        method : str or callable, default 'fps'
            Within-partition selection strategy.
            Built-in: 'fps'/'centroid_fps', 'medoid_fps', 'variance_ordered',
            'stratified'.
        variance_weighted : bool, default True
            Allocate proportionally to mean |z-score| per partition.
        return_indices : bool, default False
            When True, also return global indices into the source X.
        n_partitions : int, optional
            Override tree leaf count for this call only.
        X : ndarray (n_samples, n_features), optional
            Dataset to reduce.  When None, reduces the fitted training data.

        Returns
        -------
        X_reduced : ndarray
        indices : ndarray — only when return_indices=True
        """
        self._check_fitted()

        if n_partitions is not None:
            self._fit_tree(self.X_z_, self._last_target_, n_partitions, is_reduction=True)

        if X is None:
            X_src = self.X_
            X_z_src = self.X_z_
            partition_ids_src = self.partition_ids_
            unique_partitions_src = self.unique_partitions_
        else:
            X_src = self._coerce_external_X(X)
            X_z_src = self._to_z(X_src)
            partition_ids_src = self.tree_.apply(X_z_src)
            unique_partitions_src = np.unique(partition_ids_src)

        n_target = self._resolve_n(n, ratio, len(X_src), 'reduce')

        budgets = compute_partition_budgets(
            partition_ids_src, unique_partitions_src,
            n_target, self.min_samples_per_partition,
            variance_weighted, X_z_src,
        )
        indices = select_from_partitions(
            X_z_src, partition_ids_src, unique_partitions_src,
            budgets, method, self.random_state,
            n_jobs=self.n_jobs,
        )

        X_reduced = X_src[indices]
        if return_indices:
            return X_reduced, indices
        return X_reduced

    # ------------------------------------------------------------------
    # expand
    # ------------------------------------------------------------------

    def expand(
        self,
        n: int,
        min_novelty: float = 0.0,
        variance_weighted: bool = False,
        bandwidth: Union[float, str, None] = None,
        adaptive_bandwidth: bool = False,
        generation_strategy: Union[
            Literal['multivariate_kde', 'univariate_kde_copula', 'bootstrap_noise',
                    'epanechnikov'],
            Callable,
            None,
        ] = None,
        return_novelty_stats: bool = False,
        n_partitions: Optional[int] = None,
        X: Optional[np.ndarray] = None,
    ):
        """
        Generate synthetic samples via per-partition sampling.

        Parameters
        ----------
        n : int
            Number of synthetic samples to generate.
        min_novelty : float, default 0.0
            Deprecated.
        variance_weighted : bool, default False
        bandwidth : float or str, optional
            KDE bandwidth scalar or selector.  ``None`` uses ``self.bandwidth``.
            Accepts the same values as the constructor ``bandwidth`` parameter,
            including ``'auto'`` for data-driven kernel selection.
        adaptive_bandwidth : bool, default False
        generation_strategy : str or callable, optional
        return_novelty_stats : bool, default False
        n_partitions : int, optional
        X : ndarray (n_samples, n_features), optional

        Returns
        -------
        X_synthetic : ndarray (n, n_features)
        stats : dict — only when return_novelty_stats=True
        """
        self._check_fitted()

        if min_novelty > 0.0:
            warnings.warn(
                "min_novelty is deprecated and will be removed in a future release. "
                "Benchmarking shows KDE-generated samples are naturally novel; "
                "the filter is rarely effective.",
                HVRTDeprecationWarning,
                stacklevel=2,
            )

        if n_partitions is not None:
            self._fit_tree(self.X_z_, self._last_target_, n_partitions, is_reduction=False)

        n_cont = int(self.continuous_mask_.sum())
        n_cat = int(self.categorical_mask_.sum())
        n_orig = len(self.continuous_mask_)

        # Resolve 'auto' bandwidth before any KDE fitting.
        # 'auto' inspects mean partition size relative to n_cont and selects
        # narrow Gaussian (h=0.1) or Epanechnikov at call-time.
        _eff_bw = bandwidth if bandwidth is not None else self.bandwidth
        if _eff_bw == 'auto':
            if generation_strategy is None:
                bandwidth, generation_strategy = self._resolve_bandwidth_auto(n_cont)
            else:
                bandwidth = None  # explicit strategy overrides; clear sentinel

        if X is None:
            X_z_src = self.X_z_
            partition_ids_src = self.partition_ids_
            unique_partitions_src = self.unique_partitions_
            self._ensure_kdes(bandwidth)
            cat_partition_freqs = self._cat_partition_freqs_
            kdes_default = self._kdes_
        else:
            X_src = self._coerce_external_X(X)
            X_z_src = self._to_z(X_src)
            partition_ids_src = self.tree_.apply(X_z_src)
            unique_partitions_src = np.unique(partition_ids_src)
            bw = bandwidth if bandwidth is not None else self.bandwidth
            kdes_default = fit_partition_kdes(
                X_z_src[:, :n_cont], partition_ids_src, unique_partitions_src, bw,
                n_jobs=self.n_jobs,
            ) if n_cont > 0 else {}
            cat_partition_freqs = build_cat_partition_freqs(
                X_src[:, self.categorical_mask_], partition_ids_src, unique_partitions_src
            ) if n_cat > 0 else {}

        budgets = compute_expansion_budgets(
            partition_ids_src, unique_partitions_src,
            n, variance_weighted, X_z_src,
        )

        if generation_strategy is not None:
            if isinstance(generation_strategy, str):
                from .generation_strategies import get_generation_strategy
                strategy_fn = get_generation_strategy(generation_strategy)
            else:
                strategy_fn = generation_strategy
        else:
            strategy_fn = None

        if adaptive_bandwidth and n_cont > 0 and strategy_fn is None:
            bw_per_partition = self._adaptive_bandwidths(
                budgets, n_cont, partition_ids_src, unique_partitions_src
            )
            kdes = fit_partition_kdes(
                X_z_src[:, :n_cont], partition_ids_src,
                unique_partitions_src, bw_per_partition,
                n_jobs=self.n_jobs,
            )
        else:
            kdes = kdes_default

        if n_cont > 0:
            X_z_cont = X_z_src[:, :n_cont]
            if strategy_fn is not None:
                X_synth_cont_z = self._call_strategy(
                    strategy_fn,
                    X_z_cont, partition_ids_src, unique_partitions_src,
                    budgets, self.random_state,
                )
            else:
                X_synth_cont_z = sample_from_kdes(
                    kdes, unique_partitions_src, budgets,
                    X_z_cont, partition_ids_src,
                    self.random_state, min_novelty=min_novelty,
                    n_jobs=self.n_jobs,
                )
            X_synth_cont = self.scaler_.inverse_transform(X_synth_cont_z)
        else:
            X_synth_cont_z = np.empty((n, 0))
            X_synth_cont = np.empty((n, 0))

        if n_cat > 0:
            X_synth_cat = sample_categorical_from_freqs(
                cat_partition_freqs, unique_partitions_src, budgets,
                self.random_state,
            )
            X_synthetic = np.empty((n, n_orig))
            if n_cont > 0:
                X_synthetic[:, self.continuous_mask_] = X_synth_cont
            X_synthetic[:, self.categorical_mask_] = X_synth_cat
        else:
            X_synthetic = X_synth_cont

        if return_novelty_stats:
            ref_z = X_z_src[:, :n_cont] if n_cont > 0 else X_z_src
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
            Deprecated.
        variance_weighted : bool, default=False
        n_partitions : int or None

        Returns
        -------
        X_augmented : ndarray (n, n_features)
        """
        self._check_fitted()
        if min_novelty > 0.0:
            warnings.warn(
                "min_novelty is deprecated and will be removed in a future release. "
                "Benchmarking shows KDE-generated samples are naturally novel; "
                "the filter is rarely effective.",
                HVRTDeprecationWarning,
                stacklevel=2,
            )
        n_orig = len(self.X_)
        if n <= n_orig:
            raise ValueError(
                f"augment() requires n ({n}) > original sample count ({n_orig})."
            )
        n_synthetic = n - n_orig
        X_synth = self.expand(
            n_synthetic,
            variance_weighted=variance_weighted,
            n_partitions=n_partitions,
        )
        return np.vstack([self.X_, X_synth])

    # ------------------------------------------------------------------
    # fit_transform — sklearn pipeline support
    # ------------------------------------------------------------------

    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit and transform in a single step for sklearn pipeline use.

        Dispatch order:
        1. reduce_params / expand_params / augment_params set at construction
           (kwargs override individual fields).
        2. mode (deprecated) — emits HVRTDeprecationWarning.

        Examples
        --------
        >>> pipe = Pipeline([('hvrt', HVRT(reduce_params=ReduceParams(ratio=0.3)))])
        >>> X_red = pipe.fit_transform(X)

        >>> pipe = Pipeline([('hvrt', FastHVRT(expand_params=ExpandParams(n=50000)))])
        >>> X_synth = pipe.fit_transform(X)
        """
        self.fit(X, y)

        if self.reduce_params is not None:
            return self.reduce(**{**vars(self.reduce_params), **kwargs})
        if self.expand_params is not None:
            return self.expand(**{**vars(self.expand_params), **kwargs})
        if self.augment_params is not None:
            return self.augment(**{**vars(self.augment_params), **kwargs})

        # Deprecated mode-based dispatch
        if self.mode is not None:
            warnings.warn(
                "The 'mode' parameter is deprecated and will be removed in a future "
                "release. Use reduce_params, expand_params, or augment_params instead: "
                "HVRT(reduce_params=ReduceParams(ratio=0.3))",
                HVRTDeprecationWarning,
                stacklevel=2,
            )
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

        raise ValueError(
            "fit_transform() requires reduce_params, expand_params, or augment_params "
            "to be set at construction. For direct use, call fit() then "
            "reduce() / expand() / augment() separately."
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
            'id', 'size', 'mean_abs_z', 'variance'
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_strategy(self, strategy_fn, X_z, partition_ids, unique_partitions, budgets, random_state):
        """
        Call a generation strategy, passing ``n_jobs`` only when the strategy
        declares it.  Built-in strategies declare ``n_jobs``; custom strategies
        with the standard 5-arg signature are called without it.
        """
        try:
            sig = inspect.signature(strategy_fn)
            if 'n_jobs' in sig.parameters:
                return strategy_fn(
                    X_z, partition_ids, unique_partitions, budgets, random_state,
                    n_jobs=self.n_jobs,
                )
        except (ValueError, TypeError):
            pass
        return strategy_fn(X_z, partition_ids, unique_partitions, budgets, random_state)

    def _coerce_external_X(self, X):
        """
        Coerce an external array or DataFrame to match the training feature layout.

        DataFrame: columns matched by name (requires feature_names_in_).
        ndarray: positional match; trailing extra columns dropped with warning.
        """
        n_expected = self.X_.shape[1]

        if hasattr(X, 'columns') and self.feature_names_in_ is not None:
            X_cols = list(X.columns)
            missing = [c for c in self.feature_names_in_ if c not in X_cols]
            if missing:
                raise ValueError(
                    f"External X is missing column(s) seen during training: {missing}"
                )
            extra = [c for c in X_cols if c not in self.feature_names_in_]
            if extra:
                warnings.warn(
                    f"External X has {len(extra)} column(s) not present during "
                    f"training ({extra}); they will be ignored.",
                    HVRTFeatureWarning,
                    stacklevel=3,
                )
            return np.asarray(X[self.feature_names_in_], dtype=np.float64)

        X_arr = np.asarray(X, dtype=np.float64)
        n_given = X_arr.shape[1]
        if n_given == n_expected:
            return X_arr
        if n_given < n_expected:
            raise ValueError(
                f"External X has {n_given} feature(s) but the model was fitted "
                f"on {n_expected}. All training features must be present."
            )
        warnings.warn(
            f"External X has {n_given} column(s) but the model was fitted on "
            f"{n_expected}. The extra {n_given - n_expected} trailing column(s) "
            "will be ignored; the first columns are assumed to match the "
            "training feature layout. Pass a DataFrame for name-based matching.",
            HVRTFeatureWarning,
            stacklevel=3,
        )
        return X_arr[:, :n_expected]

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
        """Lazily fit per-partition KDEs and categorical frequency tables."""
        bw = bandwidth if bandwidth is not None else self.bandwidth
        n_cont = int(self.continuous_mask_.sum())

        if self._kdes_ is None or bw != getattr(self, '_kdes_bandwidth_', object()):
            if n_cont > 0:
                self._kdes_ = fit_partition_kdes(
                    self.X_z_[:, :n_cont], self.partition_ids_, self.unique_partitions_, bw,
                    n_jobs=self.n_jobs,
                )
            else:
                self._kdes_ = {}
            self._kdes_bandwidth_ = bw

        if self.categorical_mask_.any() and self._cat_partition_freqs_ is None:
            self._cat_partition_freqs_ = build_cat_partition_freqs(
                self.X_[:, self.categorical_mask_],
                self.partition_ids_,
                self.unique_partitions_,
            )

    def _resolve_bandwidth_auto(self, n_cont: int):
        """
        Resolve ``bandwidth='auto'`` at expand-time.

        Compares mean partition size against a feature-scaled threshold and
        returns either narrow Gaussian KDE or Epanechnikov:

        * mean partition size ≥ ``max(15, 2 × n_cont)``:
          narrow Gaussian ``bandwidth=0.1``.  Partitions are large enough for
          stable multivariate covariance estimation.

        * mean partition size < ``max(15, 2 × n_cont)``:
          Epanechnikov product kernel.  Covariance-free; robust to small
          partitions where Gaussian KDE degenerates.

        Returns
        -------
        (bandwidth, generation_strategy) : exactly one is non-None.
        """
        from .generation_strategies import epanechnikov as _epan
        from ._budgets import _partition_pos

        pos = _partition_pos(self.partition_ids_, self.unique_partitions_)
        sizes = np.bincount(pos, minlength=len(self.unique_partitions_))
        mean_part_size = float(sizes.mean())
        threshold = max(15, 2 * max(1, n_cont))

        if mean_part_size >= threshold:
            return 0.1, None   # narrow Gaussian: partitions are covariance-stable
        else:
            return None, _epan  # Epanechnikov: covariance-free product kernel

    def _adaptive_bandwidths(self, budgets, n_cont, partition_ids=None, unique_partitions=None):
        """Compute per-partition adaptive KDE bandwidth scaling with expansion ratio."""
        from ._budgets import _partition_pos

        if partition_ids is None:
            partition_ids = self.partition_ids_
        if unique_partitions is None:
            unique_partitions = self.unique_partitions_
        d = max(1, n_cont)

        pos = _partition_pos(partition_ids, unique_partitions)
        sizes = np.bincount(pos, minlength=len(unique_partitions))

        bw_dict = {}
        for i, (pid, budget) in enumerate(zip(unique_partitions, budgets)):
            n_p = int(sizes[i])
            if n_p < 2:
                bw_dict[int(pid)] = 'scott'
                continue
            scott = n_p ** (-1.0 / (d + 4))
            local_ratio = max(1.0, float(budget) / n_p)
            bw_dict[int(pid)] = float(scott * local_ratio ** (1.0 / d))
        return bw_dict
