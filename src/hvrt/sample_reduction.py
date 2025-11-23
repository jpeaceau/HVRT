"""
H-VRT Sample Reduction - Sklearn-Compatible

Intelligent sample reduction using H-VRT (Heterogeneous Variance Reduction Tree)
partitioning with diversity-based selection.

Key Features:
- Sklearn BaseEstimator interface (fit/transform)
- Partition-based stratified sampling
- Diversity preservation via furthest-point sampling
- Adaptive reduction ratio to maintain sample:feature ratio
- Handles continuous and categorical features
- Vectorized operations for efficiency

Usage:
    from src.sample_reduction import HVRTSampleReducer

    reducer = HVRTSampleReducer(reduction_ratio=0.2, auto_tune=True)
    X_reduced, y_reduced = reducer.fit_transform(X, y)

    # Or get indices
    reducer.fit(X, y)
    indices = reducer.selected_indices_
"""

import numpy as np
import warnings
from typing import Optional, List, Tuple, Union, Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .selection_strategies import (
    SelectionStrategy,
    get_strategy,
    BUILTIN_STRATEGIES
)


class HVRTSampleReducer(BaseEstimator, TransformerMixin):
    """
    H-VRT Sample Reducer

    Reduces samples while preserving data diversity using H-VRT partitioning
    and diversity-based selection within each partition.

    Parameters
    ----------
    reduction_ratio : float, default=0.2
        Proportion of samples to keep (e.g., 0.2 = keep 20%)

    max_leaf_nodes : int, optional
        Maximum leaf nodes in H-VRT tree. If None, auto-tuned.

    min_samples_leaf : int, optional
        Minimum samples per leaf. If None, uses 1% rule.

    min_samples_per_partition : int, default=5
        Minimum samples to select from each partition

    auto_tune : bool, default=True
        Automatically tune hyperparameters based on data size

    maintain_ratio : bool, default=True
        Maintain 40:1 sample:feature ratio after reduction

    y_weight : float, default=0.0
        Weight for y-extremeness in hybrid synthetic target (0.0-1.0).
        - 0.0: Pure X-interactions (original H-VRT, best for interaction-driven data)
        - 0.25-0.50: Balanced hybrid (recommended for CLT-failure scenarios)
        - 1.0: Pure y-extremeness (supervised, best for outlier detection)
        Higher values prioritize capturing rare extreme outcomes.

    selection_strategy : str or callable, default='centroid_fps'
        Strategy for selecting samples within each partition.

        Built-in strategies (str):
        - 'centroid_fps': Centroid-seeded FPS (default, filters noise, preserves diversity)
        - 'medoid_fps': Medoid-seeded FPS (robust to outliers)
        - 'variance_ordered': Select by local variance (boundary preservation)
        - 'stratified': Random sampling within partition (baseline comparison)

        Custom strategy (callable):
        Must follow SelectionStrategy protocol:
            def strategy(X_partition, n_select, random_state) -> selected_indices

        See selection_strategies.py for detailed documentation.

    random_state : int, default=42
        Random seed for reproducibility

    Attributes
    ----------
    selected_indices_ : ndarray of shape (n_selected,)
        Indices of selected samples after fitting

    tree_ : DecisionTreeRegressor
        Fitted H-VRT tree

    n_partitions_ : int
        Number of partitions created

    Examples
    --------
    >>> from src.sample_reduction import HVRTSampleReducer
    >>> import numpy as np
    >>> X = np.random.randn(1000, 20)
    >>> y = np.random.randn(1000)
    >>> reducer = HVRTSampleReducer(reduction_ratio=0.2)
    >>> X_reduced, y_reduced = reducer.fit_transform(X, y)
    >>> print(f"Reduced from {len(X)} to {len(X_reduced)} samples")
    """

    def __init__(
        self,
        reduction_ratio: float = 0.2,
        max_leaf_nodes: Optional[int] = None,
        min_samples_leaf: Optional[int] = None,
        max_depth: Optional[int] = None,
        min_samples_per_partition: int = 5,
        auto_tune: bool = True,
        maintain_ratio: bool = True,
        y_weight: float = 0.0,
        selection_strategy: Union[str, Callable[[np.ndarray, int, int], np.ndarray]] = 'centroid_fps',
        random_state: int = 42
    ):
        self.reduction_ratio = reduction_ratio
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_samples_per_partition = min_samples_per_partition
        self.auto_tune = auto_tune
        self.maintain_ratio = maintain_ratio
        self.y_weight = y_weight
        self.random_state = random_state

        # Validate and store selection strategy
        if isinstance(selection_strategy, str):
            self.selection_strategy = get_strategy(selection_strategy)
            self.selection_strategy_name_ = selection_strategy
        elif callable(selection_strategy):
            # Validate it follows the protocol (runtime check)
            if not isinstance(selection_strategy, SelectionStrategy):
                warnings.warn(
                    "Custom selection_strategy should follow SelectionStrategy protocol. "
                    "Expected signature: (X_partition, n_select, random_state) -> selected_indices"
                )
            self.selection_strategy = selection_strategy
            self.selection_strategy_name_ = getattr(selection_strategy, '__name__', 'custom')
        else:
            raise TypeError(
                f"selection_strategy must be str or callable, got {type(selection_strategy)}"
            )

    def _auto_tune_hyperparameters(self, n_samples: int, n_features: int):
        """Auto-tune tree hyperparameters based on dataset size.

        Only tunes parameters that are None (allows manual override).

        FINER PARTITIONING STRATEGY (3x):
        Creates more variance-based partitions to naturally isolate outliers
        into their own partitions, improving CLT-failure robustness.
        """
        if self.auto_tune:
            if self.min_samples_leaf is None:
                # Maintain 40:1 sample-to-feature ratio for statistical reliability
                min_required = n_features * 40
                # Allow smaller partitions (divide by 3) to support 3x more partitions
                self.min_samples_leaf = max(5, min_required * 2 // 3)

            if self.max_leaf_nodes is None:
                # Scale partitions with dataset size (min 30, max 1500)
                # 3x multiplier: Creates finer partitions that naturally isolate outliers
                target_partitions = max(30, min(1500, 3 * n_samples // (self.min_samples_leaf * 2)))
                self.max_leaf_nodes = target_partitions

    def _compute_synthetic_y(self, X_normalized: np.ndarray) -> np.ndarray:
        """
        Compute synthetic target using pairwise feature interactions (Paper Algorithm 1).

        VECTORIZED IMPLEMENTATION for production scalability.

        For each pair of features (i, j) where i < j:
        - Compute element-wise product: X[:, i] ⊙ X[:, j]
        - Z-score normalize the interaction
        - Sum all normalized interactions

        This emphasizes regions where multiple interactions are active,
        capturing structural heterogeneity in the data.

        Rationale:
        - Pairwise interactions reveal structural relationships
        - Z-score normalization ensures scale-invariance (all features contribute equally)
        - Sum aggregation maximizes predictive signal across all interaction patterns

        Complexity: O(d² · n) but vectorized with sklearn (5-10x faster than Python loops)
        Scales to 100k+ samples in seconds

        References: Paper Section 3.2.1, Algorithm 1 (lines 295-310)
        """
        n_samples, n_features = X_normalized.shape

        # VECTORIZED: Generate all pairwise interactions at once using sklearn
        # PolynomialFeatures uses optimized Cython code
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        interactions_all = poly.fit_transform(X_normalized)

        # Remove original features (keep only interaction terms)
        # First d columns are original features, rest are interactions
        interactions = interactions_all[:, n_features:]

        # Z-score normalize each interaction column (vectorized)
        interaction_means = interactions.mean(axis=0)
        interaction_stds = interactions.std(axis=0)

        # Avoid division by zero for constant interactions
        interaction_stds = np.where(interaction_stds > 1e-10, interaction_stds, 1.0)
        interactions_z = (interactions - interaction_means) / interaction_stds

        # Zero out constant interactions (they contribute no variance information)
        interactions_z = np.where(interaction_stds[None, :] > 1e-10, interactions_z, 0.0)

        # Sum across all interaction terms
        synthetic_y = interactions_z.sum(axis=1)

        return synthetic_y

    def _compute_hybrid_synthetic_y(
        self,
        X_normalized: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Compute hybrid synthetic target combining X-interactions and y-extremeness.

        This addresses a fundamental limitation of pure X-interaction partitioning:
        rare high-signal events defined by extreme y-values (e.g., heavy-tailed noise,
        outliers, rare events) may not be distinguished by their feature patterns alone.

        The hybrid approach combines two orthogonal variance signals:

        1. **X-Component (Feature Interactions)**:
           Captures correlation patterns in feature space. High values indicate
           regions with strong pairwise feature interactions.

        2. **Y-Component (Outcome Extremeness)**:
           Measures distance from typical outcomes. High values indicate rare
           extreme cases, outliers, or anomalous samples.

        Formula:
            y_synthetic = (1 - α) × X_component + α × Y_component

        where α is self.y_weight.

        **Interpretability**:
        - The weighting α has clear semantic meaning based on domain knowledge
        - α=0.0: Variance is interaction-driven (embeddings, image features)
        - α=0.25-0.50: Balanced (most real-world scenarios)
        - α=0.50-0.75: Variance is outcome-driven (anomaly detection, rare events)
        - α=1.0: Pure outlier detection (supervised)

        **Auditability**:
        - Can decompose each sample's score into X and Y contributions
        - Can explain WHY specific samples were prioritized
        - Maintains determinism and reproducibility

        Parameters
        ----------
        X_normalized : array of shape (n_samples, n_features)
            Normalized feature matrix
        y : array of shape (n_samples,)
            Target values

        Returns
        -------
        synthetic_y : array of shape (n_samples,)
            Hybrid synthetic target for partitioning

        References
        ----------
        Investigation findings: INVESTIGATION_FINDINGS.md, FINAL_RECOMMENDATIONS.md
        """
        # Component 1: X-space interactions (original H-VRT)
        x_component = self._compute_synthetic_y(X_normalized)

        # Component 2: Y-space extremeness
        y_normalized = (y - y.mean()) / (y.std() + 1e-10)
        y_median = np.median(y_normalized)
        y_extremeness = np.abs(y_normalized - y_median)

        # Normalize to same scale as x_component for fair weighting
        y_component = (y_extremeness - y_extremeness.mean()) / (y_extremeness.std() + 1e-10)

        # Combine with weighting
        synthetic_y = (1.0 - self.y_weight) * x_component + self.y_weight * y_component

        return synthetic_y

    def _encode_categorical(
        self,
        X_cat: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """Encode categorical features using LabelEncoder"""
        if not hasattr(self, 'label_encoders_'):
            self.label_encoders_ = {}

        n_features = X_cat.shape[1]
        X_encoded = np.zeros_like(X_cat, dtype=np.int32)

        for col_idx in range(n_features):
            if fit:
                le = LabelEncoder()
                X_encoded[:, col_idx] = le.fit_transform(X_cat[:, col_idx].astype(str))
                self.label_encoders_[col_idx] = le
            else:
                le = self.label_encoders_[col_idx]
                X_encoded[:, col_idx] = le.transform(X_cat[:, col_idx].astype(str))

        return X_encoded

    def _furthest_point_sampling(
        self,
        X: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Deterministic Furthest Point Sampling (FPS) for X-diversity.

        Selects samples that maximize diversity in feature space using greedy FPS.
        FULLY DETERMINISTIC for reproducibility:
        - Seed from centroid (closest point)
        - Iteratively select points farthest from selected set
        - Preserves X-space diversity within partitions

        This approach complements H-VRT's variance-based partitioning by
        maintaining feature diversity within each partition (finer partitions
        strategy creates 3x more partitions to naturally isolate outliers).

        Parameters
        ----------
        X : array of shape (n_points, n_features)
            Feature matrix for the partition
        n_samples : int
            Number of samples to select
        """
        n_points = len(X)
        if n_points <= n_samples:
            return np.arange(n_points)

        # Compute centroid for deterministic seed
        centroid = X.mean(axis=0)

        # DETERMINISTIC FPS: Single centroid seed for reproducibility
        # OPTIMIZED: Use squared distances (monotonic ordering preserved, no sqrt needed)
        diff = X - centroid
        squared_distances = np.sum(diff * diff, axis=1)

        # Single seed: Closest to centroid (deterministic starting point)
        seed_idx = np.argmin(squared_distances)
        selected = [seed_idx]
        min_squared_distances = np.full(n_points, np.inf)

        # Greedy FPS: Select points that maximize minimum distance to selected set
        for _ in range(n_samples - 1):
            last_idx = selected[-1]

            # Vectorized squared distance update (no sqrt - much faster)
            diff = X - X[last_idx]
            squared_distances = np.sum(diff * diff, axis=1)
            min_squared_distances = np.minimum(min_squared_distances, squared_distances)

            # Select farthest point (squared distance ordering = regular distance ordering)
            next_idx = np.argmax(min_squared_distances)
            selected.append(next_idx)

        return np.array(selected)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_types: Optional[List[str]] = None
    ) -> 'HVRTSampleReducer':
        """
        Fit H-VRT sample reducer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,), optional
            Target values. If provided, used for tree fitting.
            If None, synthetic target is computed.

        feature_types : list of str, optional
            Feature types: 'continuous' or 'categorical'.
            If None, assumes all continuous.

        Returns
        -------
        self : HVRTSampleReducer
            Fitted reducer
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # Auto-tune hyperparameters
        self._auto_tune_hyperparameters(n_samples, n_features)

        # If auto_tune is False, ensure max_leaf_nodes and min_samples_leaf have sensible defaults
        if not self.auto_tune:
            if self.max_leaf_nodes is None:
                self.max_leaf_nodes = max(2, n_samples // 100) # At least 2, or 1% of samples
            if self.min_samples_leaf is None:
                self.min_samples_leaf = max(5, n_samples // 1000) # At least 5, or 0.1% of samples

        # Check reduction ratio feasibility
        if self.maintain_ratio:
            min_samples_needed = n_features * 40
            safe_reduction_ratio = max(self.reduction_ratio, min_samples_needed / n_samples)

            if safe_reduction_ratio > 1.0:
                raise ValueError(
                    f"Dataset too small for sample reduction. "
                    f"Need >= {min_samples_needed} samples for {n_features} features (40:1 ratio). "
                    f"Current: {n_samples} samples (ratio {n_samples/n_features:.1f}:1)"
                )

            if safe_reduction_ratio > self.reduction_ratio:
                warnings.warn(
                    f"Adjusted reduction ratio from {self.reduction_ratio:.1%} to "
                    f"{safe_reduction_ratio:.1%} to maintain 40:1 sample:feature ratio."
                )
                self.effective_reduction_ratio_ = safe_reduction_ratio
            else:
                self.effective_reduction_ratio_ = self.reduction_ratio
        else:
            self.effective_reduction_ratio_ = self.reduction_ratio

        # Determine feature types
        if feature_types is None:
            feature_types = ['continuous'] * n_features

        continuous_mask = np.array([ft == 'continuous' for ft in feature_types])
        categorical_mask = ~continuous_mask

        self.continuous_mask_ = continuous_mask
        self.categorical_mask_ = categorical_mask

        # Process features
        processed_parts = []

        if continuous_mask.any():
            X_cont = X[:, continuous_mask]
            self.scaler_ = StandardScaler()
            X_cont_norm = self.scaler_.fit_transform(X_cont)
            processed_parts.append(X_cont_norm)

        if categorical_mask.any():
            X_cat = X[:, categorical_mask]
            X_cat_encoded = self._encode_categorical(X_cat, fit=True)
            self.cat_scaler_ = StandardScaler()
            X_cat_norm = self.cat_scaler_.fit_transform(X_cat_encoded)
            processed_parts.append(X_cat_norm)

        X_processed = np.hstack(processed_parts) if len(processed_parts) > 1 else processed_parts[0]

        # Compute target
        if y is not None:
            y_array = np.asarray(y).ravel()
            if self.y_weight > 0:
                # Hybrid: combine X-interactions and y-extremeness
                target = self._compute_hybrid_synthetic_y(X_processed, y_array)
            else:
                # Original: use y directly for partitioning
                target = y_array
        else:
            # Unsupervised: use pure X-interactions
            target = self._compute_synthetic_y(X_processed)

        # Fit H-VRT tree (DecisionTreeRegressor already discretizes efficiently)
        # No presampling needed - tree construction is O(n log n) and learns optimal splits
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            min_impurity_decrease=0.0,
            random_state=self.random_state
        )
        self.tree_.fit(X_processed, target)

        # Get partition assignments
        partition_ids = self.tree_.apply(X_processed)
        unique_partitions = np.unique(partition_ids)
        self.n_partitions_ = len(unique_partitions)

        # Compute target sample count
        n_target = int(n_samples * self.effective_reduction_ratio_)

        # Proportional sampling from each partition
        partition_sizes = np.array([np.sum(partition_ids == pid) for pid in unique_partitions])
        samples_per_partition = np.maximum(
            self.min_samples_per_partition,
            (partition_sizes / partition_sizes.sum() * n_target).astype(int)
        )

        # Adjust to exact target
        while samples_per_partition.sum() > n_target:
            idx_max = np.argmax(samples_per_partition)
            samples_per_partition[idx_max] -= 1
        while samples_per_partition.sum() < n_target:
            idx_min = np.argmin(samples_per_partition)
            samples_per_partition[idx_min] += 1

        # Select samples using diversity-based sampling
        selected_idx = []
        for pid, n_select in zip(unique_partitions, samples_per_partition):
            if n_select == 0:
                continue

            mask = partition_ids == pid
            partition_indices = np.where(mask)[0]

            if len(partition_indices) <= n_select:
                # Take all samples from small partitions
                selected_idx.extend(partition_indices)
            else:
                # Apply selection strategy for larger partitions
                selected = self.selection_strategy(
                    X_processed[mask],
                    n_select,
                    self.random_state
                )
                selected_idx.extend(partition_indices[selected])

        self.selected_indices_ = np.array(selected_idx[:n_target])

        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Reduce samples using fitted reducer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to reduce

        y : array-like of shape (n_samples,), optional
            Target values to reduce correspondingly

        Returns
        -------
        X_reduced : ndarray of shape (n_selected, n_features)
            Reduced data

        y_reduced : ndarray of shape (n_selected,), optional
            Reduced target (if y was provided)
        """
        if not hasattr(self, 'selected_indices_'):
            raise ValueError("Reducer must be fitted before transform. Call fit() first.")

        X = np.asarray(X)
        X_reduced = X[self.selected_indices_]

        if y is not None:
            y = np.asarray(y)
            y_reduced = y[self.selected_indices_]
            return X_reduced, y_reduced
        else:
            return X_reduced

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_types: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit reducer and reduce samples in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,), optional
            Target values

        feature_types : list of str, optional
            Feature types: 'continuous' or 'categorical'

        Returns
        -------
        X_reduced : ndarray of shape (n_selected, n_features)
            Reduced data

        y_reduced : ndarray of shape (n_selected,), optional
            Reduced target (if y was provided)
        """
        self.fit(X, y, feature_types)
        return self.transform(X, y)

    def get_reduction_info(self) -> dict:
        """
        Get information about the reduction.

        Returns
        -------
        info : dict
            Dictionary containing reduction statistics
        """
        if not hasattr(self, 'selected_indices_'):
            raise ValueError("Reducer must be fitted first.")

        return {
            'n_partitions': self.n_partitions_,
            'n_selected': len(self.selected_indices_),
            'reduction_ratio': self.effective_reduction_ratio_,
            'tree_depth': self.tree_.get_depth(),
            'max_leaf_nodes': self.max_leaf_nodes,
            'min_samples_leaf': self.min_samples_leaf
        }
