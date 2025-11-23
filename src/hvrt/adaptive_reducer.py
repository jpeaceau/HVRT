"""
Adaptive Pipeline Sample Reducer

Automatically finds the optimal reduction level by testing progressively
aggressive reductions until accuracy drops below threshold.

Key Features:
- Progressive reduction with accuracy tracking
- XGBoost validation by default (fast, accurate)
- Custom pipeline support for specific use cases
- Stores all reduction results for user decision-making
- Returns best reduction that meets accuracy threshold
"""

import numpy as np
from typing import Optional, Dict, List, Any, Union, Callable
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import (
    get_scorer,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
import warnings

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

from .sample_reduction import HVRTSampleReducer


class AdaptiveHVRTReducer:
    """
    Adaptive sample reducer that finds optimal reduction via accuracy testing.

    Use Cases:
    1. SVM training: Use XGBoost to validate, get reduced samples for SVM
    2. Faster inference: Reduce to minimum samples maintaining accuracy
    3. Unknown optimal reduction: Let the tool find it automatically

    Parameters
    ----------
    accuracy_threshold : float, default=0.95
        Minimum accuracy retention (0.95 = 95% of baseline accuracy).
        Reduction stops when accuracy drops below this.

    reduction_ratios : list of float, optional
        Reduction levels to test (e.g., [0.5, 0.3, 0.2, 0.1]).
        If None, defaults to [0.5, 0.3, 0.2, 0.15, 0.1, 0.05].

    validator : estimator, optional
        Model for validation (default: XGBClassifier/Regressor).
        Use XGBoost even if final model is SVM for fast validation.

    cv : int, default=3
        Cross-validation folds for accuracy estimation.

    scoring : str, callable, or dict, optional
        Scoring metric for evaluation. Options:

        **Built-in metrics (str):**
        - Classification: 'accuracy', 'f1', 'precision', 'recall', 'roc_auc'
        - Regression: 'r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'

        **Custom scorer (callable):**
        - Function with signature: score = scorer(y_true, y_pred)
        - Higher scores are better (use negative for loss metrics)

        **Multiple metrics (dict):**
        - {'primary': 'accuracy', 'f1': 'f1', 'recall': 'recall'}
        - First metric used for threshold comparison

        If None, uses estimator's default scorer.

    y_weight : float, default=0.0
        H-VRT hybrid mode weight (0.0-1.0).
        Use 0.25-0.5 for heavy-tailed data.

    random_state : int, default=42
        Random seed for reproducibility.

    verbose : bool, default=True
        Print progress during reduction testing.

    Attributes
    ----------
    reduction_results_ : list of dict
        Results for each tested reduction level:
        - 'reduction_ratio': Reduction level tested
        - 'n_samples': Number of samples after reduction
        - 'cv_score_mean': Mean CV accuracy
        - 'cv_score_std': Std CV accuracy
        - 'accuracy_retention': % of baseline accuracy retained
        - 'X_reduced': Reduced feature matrix
        - 'y_reduced': Reduced target vector
        - 'selected_indices': Indices of selected samples

    best_reduction_ : dict
        Best reduction meeting accuracy threshold.

    baseline_score_ : float
        Baseline accuracy on full dataset.

    Examples
    --------
    >>> from hvrt import AdaptiveHVRTReducer
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.svm import SVC
    >>>
    >>> # Generate data
    >>> X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
    >>>
    >>> # Find optimal reduction for SVM (using XGBoost for validation)
    >>> reducer = AdaptiveHVRTReducer(
    ...     accuracy_threshold=0.95,  # Keep 95% accuracy
    ...     verbose=True
    ... )
    >>> reducer.fit(X, y)
    >>>
    >>> # Get best reduced dataset
    >>> X_reduced, y_reduced = reducer.transform(X, y)
    >>>
    >>> # Train SVM on reduced data
    >>> svm = SVC()
    >>> svm.fit(X_reduced, y_reduced)
    >>>
    >>> # Review all reduction results
    >>> for result in reducer.reduction_results_:
    ...     print(f"Reduction {result['reduction_ratio']:.0%}: "
    ...           f"{result['accuracy_retention']:.1f}% accuracy, "
    ...           f"{result['n_samples']} samples")
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.95,
        reduction_ratios: Optional[List[float]] = None,
        validator: Optional[Any] = None,
        cv: int = 3,
        scoring: Optional[Union[str, Callable]] = None,
        y_weight: float = 0.0,
        random_state: int = 42,
        verbose: bool = True
    ):
        self.accuracy_threshold = accuracy_threshold
        self.reduction_ratios = reduction_ratios or [0.5, 0.3, 0.2, 0.15, 0.1, 0.05]
        self.validator = validator
        self.cv = cv
        self.scoring = scoring
        self.y_weight = y_weight
        self.random_state = random_state
        self.verbose = verbose

        # Results
        self.reduction_results_: List[Dict] = []
        self.best_reduction_: Optional[Dict] = None
        self.baseline_score_: Optional[float] = None
        self._X_fit = None
        self._y_fit = None

    def _get_validator(self, y: np.ndarray) -> Any:
        """Get or create validator model."""
        if self.validator is not None:
            return self.validator

        if not XGBOOST_AVAILABLE:
            raise ValueError(
                "No validator provided and XGBoost not available. "
                "Install XGBoost (pip install xgboost) or provide a validator model."
            )

        # Auto-detect task type
        is_classification = len(np.unique(y)) < 0.05 * len(y)  # <5% unique values

        if is_classification:
            return XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
        else:
            return XGBRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )

    def _prepare_scoring(self, scoring: Any) -> tuple:
        """
        Prepare scoring for cross-validation.

        Returns
        -------
        scoring_dict : dict or str or callable
            Scoring for cross_validate
        primary_metric : str
            Primary metric name for threshold comparison
        """
        if scoring is None:
            return None, 'score'

        # If callable, wrap it as a scorer
        if callable(scoring):
            return make_scorer(scoring), 'score'

        # If string, use as-is
        if isinstance(scoring, str):
            return scoring, scoring

        # If dict, use first key as primary
        if isinstance(scoring, dict):
            primary_metric = list(scoring.keys())[0]
            return scoring, primary_metric

        return scoring, 'score'

    def _evaluate_reduction(
        self,
        X: np.ndarray,
        y: np.ndarray,
        reduction_ratio: float,
        validator: Any
    ) -> Dict:
        """Evaluate a single reduction level."""

        # Apply H-VRT reduction
        reducer = HVRTSampleReducer(
            reduction_ratio=reduction_ratio,
            y_weight=self.y_weight,
            random_state=self.random_state
        )

        X_reduced, y_reduced = reducer.fit_transform(X, y)

        # Prepare scoring
        scoring_dict, primary_metric = self._prepare_scoring(self.scoring)

        # Evaluate via cross-validation
        if isinstance(scoring_dict, dict):
            # Multiple metrics
            cv_results = cross_validate(
                validator,
                X_reduced,
                y_reduced,
                cv=self.cv,
                scoring=scoring_dict,
                n_jobs=-1,
                return_train_score=False
            )

            # Extract primary metric
            test_key = f'test_{primary_metric}'
            cv_score_mean = cv_results[test_key].mean()
            cv_score_std = cv_results[test_key].std()

            # Store all metrics
            all_scores = {
                metric: cv_results[f'test_{metric}'].mean()
                for metric in scoring_dict.keys()
            }
        else:
            # Single metric
            scores = cross_val_score(
                validator,
                X_reduced,
                y_reduced,
                cv=self.cv,
                scoring=scoring_dict,
                n_jobs=-1
            )
            cv_score_mean = scores.mean()
            cv_score_std = scores.std()
            all_scores = {primary_metric: cv_score_mean}

        # Compute accuracy retention (using primary metric)
        accuracy_retention = (cv_score_mean / self.baseline_score_) if self.baseline_score_ else 0.0

        result = {
            'reduction_ratio': reduction_ratio,
            'n_samples': len(X_reduced),
            'cv_score_mean': cv_score_mean,
            'cv_score_std': cv_score_std,
            'accuracy_retention': accuracy_retention,
            'all_scores': all_scores,
            'primary_metric': primary_metric,
            'X_reduced': X_reduced,
            'y_reduced': y_reduced,
            'selected_indices': reducer.selected_indices_
        }

        return result

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptiveHVRTReducer':
        """
        Find optimal reduction level by testing progressively aggressive reductions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : AdaptiveHVRTReducer
            Fitted reducer with best_reduction_ set
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        self._X_fit = X
        self._y_fit = y

        # Get validator
        validator = self._get_validator(y)

        # Prepare scoring
        scoring_dict, primary_metric = self._prepare_scoring(self.scoring)

        # Compute baseline accuracy
        if self.verbose:
            print(f"Computing baseline score on {len(X)} samples...")

        # Evaluate baseline with same scoring
        if isinstance(scoring_dict, dict):
            cv_results = cross_validate(
                validator,
                X,
                y,
                cv=self.cv,
                scoring=scoring_dict,
                n_jobs=-1,
                return_train_score=False
            )
            baseline_scores = cv_results[f'test_{primary_metric}']
            self.baseline_score_ = baseline_scores.mean()
            baseline_std = baseline_scores.std()

            # Store all baseline metrics
            self.baseline_all_scores_ = {
                metric: cv_results[f'test_{metric}'].mean()
                for metric in scoring_dict.keys()
            }
        else:
            baseline_scores = cross_val_score(
                validator,
                X,
                y,
                cv=self.cv,
                scoring=scoring_dict,
                n_jobs=-1
            )
            self.baseline_score_ = baseline_scores.mean()
            baseline_std = baseline_scores.std()
            self.baseline_all_scores_ = {primary_metric: self.baseline_score_}

        if self.verbose:
            metric_name = primary_metric if primary_metric != 'score' else 'default'
            print(f"Baseline ({metric_name}): {self.baseline_score_:.4f} ± {baseline_std:.4f}")

            if isinstance(scoring_dict, dict) and len(scoring_dict) > 1:
                print("All baseline metrics:")
                for metric, score in self.baseline_all_scores_.items():
                    print(f"  {metric}: {score:.4f}")

            print(f"\nTesting reduction levels (threshold: {self.accuracy_threshold:.0%})...")
            print("=" * 70)

        # Test each reduction level
        self.reduction_results_ = []
        best_result = None

        for reduction_ratio in sorted(self.reduction_ratios, reverse=True):
            result = self._evaluate_reduction(X, y, reduction_ratio, validator)
            self.reduction_results_.append(result)

            if self.verbose:
                metric_str = f"{result['cv_score_mean']:.4f} ± {result['cv_score_std']:.4f}"

                # Show all metrics if multiple
                if len(result['all_scores']) > 1:
                    other_metrics = ", ".join(
                        f"{k}={v:.3f}"
                        for k, v in result['all_scores'].items()
                        if k != result['primary_metric']
                    )
                    metric_str += f" [{other_metrics}]"

                print(f"Reduction {reduction_ratio:5.0%}: "
                      f"{result['accuracy_retention']:6.1%} retention "
                      f"({metric_str}), "
                      f"{result['n_samples']:5d} samples")

            # Check if meets threshold
            if result['accuracy_retention'] >= self.accuracy_threshold:
                best_result = result
            else:
                # Accuracy dropped below threshold, stop
                if self.verbose:
                    print(f"\nAccuracy dropped below {self.accuracy_threshold:.0%} threshold. "
                          f"Using previous reduction.")
                break

        # Set best reduction
        if best_result is None:
            # Even least aggressive reduction failed, use it anyway
            best_result = self.reduction_results_[0]
            if self.verbose:
                print(f"\nWARNING: No reduction met {self.accuracy_threshold:.0%} threshold. "
                      f"Using least aggressive ({best_result['reduction_ratio']:.0%}).")

        self.best_reduction_ = best_result

        if self.verbose:
            print("=" * 70)
            print(f"\nBest reduction: {best_result['reduction_ratio']:.0%} "
                  f"({best_result['n_samples']} samples, "
                  f"{best_result['accuracy_retention']:.1%} accuracy)")

        return self

    def transform(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None
    ) -> tuple:
        """
        Return the best reduced dataset.

        Parameters
        ----------
        X : array-like, optional
            If provided, applies best reduction to new data.
            If None, returns stored best reduction from fit().

        y : array-like, optional
            Target values (required if X is provided).

        Returns
        -------
        X_reduced : array
            Reduced feature matrix

        y_reduced : array
            Reduced target vector
        """
        if self.best_reduction_ is None:
            raise ValueError("Reducer not fitted yet. Call fit() first.")

        if X is not None:
            # Apply best reduction to new data
            if y is None:
                raise ValueError("y must be provided when X is provided")

            reducer = HVRTSampleReducer(
                reduction_ratio=self.best_reduction_['reduction_ratio'],
                y_weight=self.y_weight,
                random_state=self.random_state
            )
            return reducer.fit_transform(X, y)
        else:
            # Return stored best reduction
            return self.best_reduction_['X_reduced'], self.best_reduction_['y_reduced']

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Fit and return best reduced dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        X_reduced : array
            Reduced feature matrix

        y_reduced : array
            Reduced target vector
        """
        self.fit(X, y)
        return self.transform()

    def get_reduction_summary(self) -> str:
        """
        Get human-readable summary of all tested reductions.

        Returns
        -------
        summary : str
            Formatted table of all reduction results
        """
        if not self.reduction_results_:
            return "No reductions tested yet. Call fit() first."

        lines = []
        lines.append("Reduction Summary")
        lines.append("=" * 80)
        lines.append(f"Baseline Accuracy: {self.baseline_score_:.4f}")
        lines.append(f"Accuracy Threshold: {self.accuracy_threshold:.0%}")
        lines.append("")
        lines.append(f"{'Reduction':<10} {'Samples':<8} {'CV Score':<20} {'Retention':<10} {'Status':<10}")
        lines.append("-" * 80)

        for result in self.reduction_results_:
            is_best = result == self.best_reduction_
            status = "[BEST]" if is_best else ("PASS" if result['accuracy_retention'] >= self.accuracy_threshold else "FAIL")

            lines.append(
                f"{result['reduction_ratio']:7.0%}    "
                f"{result['n_samples']:6d}   "
                f"{result['cv_score_mean']:.4f} ± {result['cv_score_std']:.4f}   "
                f"{result['accuracy_retention']:7.1%}    "
                f"{status}"
            )

        return "\n".join(lines)
