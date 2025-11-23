"""
Adaptive Reduction with Custom Scoring

Demonstrates all scoring options for AdaptiveHVRTReducer:
- Built-in metrics (accuracy, f1, precision, recall, MAE, MSE, R2)
- Custom callable scorers
- Multiple metrics simultaneously
"""

import numpy as np
from hvrt import AdaptiveHVRTReducer
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


def main():
    print("=" * 70)
    print("Adaptive H-VRT: Scoring Options Demo")
    print("=" * 70)

    # ========================================================================
    # Example 1: Built-in Classification Metrics
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 1: Built-in Classification Metrics")
    print("=" * 70)

    X_cls, y_cls = make_classification(
        n_samples=5000, n_features=20, n_informative=15, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42
    )

    # Option 1a: Accuracy (default for classification)
    print("\n1a. Using 'accuracy' scoring:")
    reducer_acc = AdaptiveHVRTReducer(
        accuracy_threshold=0.95,
        reduction_ratios=[0.5, 0.3, 0.2],
        scoring='accuracy',
        verbose=True
    )
    reducer_acc.fit(X_train, y_train)

    # Option 1b: F1 Score
    print("\n1b. Using 'f1' scoring:")
    reducer_f1 = AdaptiveHVRTReducer(
        accuracy_threshold=0.90,
        reduction_ratios=[0.5, 0.3, 0.2],
        scoring='f1',
        verbose=True
    )
    reducer_f1.fit(X_train, y_train)

    # Option 1c: Precision
    print("\n1c. Using 'precision' scoring:")
    reducer_prec = AdaptiveHVRTReducer(
        accuracy_threshold=0.90,
        reduction_ratios=[0.5, 0.3, 0.2],
        scoring='precision',
        verbose=True
    )
    reducer_prec.fit(X_train, y_train)

    # Option 1d: Recall
    print("\n1d. Using 'recall' scoring:")
    reducer_rec = AdaptiveHVRTReducer(
        accuracy_threshold=0.90,
        reduction_ratios=[0.5, 0.3, 0.2],
        scoring='recall',
        verbose=True
    )
    reducer_rec.fit(X_train, y_train)

    # ========================================================================
    # Example 2: Built-in Regression Metrics
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: Built-in Regression Metrics")
    print("=" * 70)

    X_reg, y_reg = make_regression(
        n_samples=5000, n_features=20, n_informative=15, noise=10, random_state=42
    )
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Option 2a: R² (default for regression)
    print("\n2a. Using 'r2' scoring:")
    reducer_r2 = AdaptiveHVRTReducer(
        accuracy_threshold=0.95,
        reduction_ratios=[0.5, 0.3, 0.2],
        scoring='r2',
        verbose=True
    )
    reducer_r2.fit(X_train_r, y_train_r)

    # Option 2b: Negative MAE (higher is better)
    print("\n2b. Using 'neg_mean_absolute_error' scoring:")
    reducer_mae = AdaptiveHVRTReducer(
        accuracy_threshold=0.95,
        reduction_ratios=[0.5, 0.3, 0.2],
        scoring='neg_mean_absolute_error',
        verbose=True
    )
    reducer_mae.fit(X_train_r, y_train_r)

    # Option 2c: Negative MSE (higher is better)
    print("\n2c. Using 'neg_mean_squared_error' scoring:")
    reducer_mse = AdaptiveHVRTReducer(
        accuracy_threshold=0.95,
        reduction_ratios=[0.5, 0.3, 0.2],
        scoring='neg_mean_squared_error',
        verbose=True
    )
    reducer_mse.fit(X_train_r, y_train_r)

    # ========================================================================
    # Example 3: Custom Callable Scorer
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Custom Callable Scorer")
    print("=" * 70)

    def balanced_accuracy_custom(y_true, y_pred):
        """Custom scorer: balanced accuracy for imbalanced datasets."""
        from sklearn.metrics import balanced_accuracy_score
        return balanced_accuracy_score(y_true, y_pred)

    print("\n3. Using custom callable scorer (balanced accuracy):")
    reducer_custom = AdaptiveHVRTReducer(
        accuracy_threshold=0.95,
        reduction_ratios=[0.5, 0.3, 0.2],
        scoring=balanced_accuracy_custom,
        verbose=True
    )
    reducer_custom.fit(X_train, y_train)

    # ========================================================================
    # Example 4: Multiple Metrics Simultaneously
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 4: Multiple Metrics Simultaneously")
    print("=" * 70)

    print("\n4. Using multiple metrics (accuracy, f1, precision, recall):")
    print("   Primary metric (accuracy) used for threshold comparison")
    print()

    reducer_multi = AdaptiveHVRTReducer(
        accuracy_threshold=0.95,
        reduction_ratios=[0.5, 0.3, 0.2],
        scoring={
            'accuracy': 'accuracy',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall'
        },
        verbose=True
    )
    reducer_multi.fit(X_train, y_train)

    # Review all metrics for each reduction
    print("\nDetailed multi-metric results:")
    print("-" * 70)
    for result in reducer_multi.reduction_results_:
        print(f"\nReduction {result['reduction_ratio']:.0%} ({result['n_samples']} samples):")
        for metric, score in result['all_scores'].items():
            print(f"  {metric:12s}: {score:.4f}")

    # ========================================================================
    # Example 5: Accessing Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 5: Accessing Results")
    print("=" * 70)

    # Get best reduction
    best = reducer_multi.best_reduction_
    print(f"\nBest reduction: {best['reduction_ratio']:.0%}")
    print(f"Samples: {best['n_samples']}")
    print(f"Primary metric retention: {best['accuracy_retention']:.1%}")
    print("\nAll metrics:")
    for metric, score in best['all_scores'].items():
        print(f"  {metric}: {score:.4f}")

    # Get reduced dataset
    X_reduced, y_reduced = reducer_multi.transform()
    print(f"\nReduced dataset shape: {X_reduced.shape}")

    # Review summary
    print("\n" + reducer_multi.get_reduction_summary())

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Scoring Options Summary")
    print("=" * 70)
    print("""
Classification Metrics (str):
  - 'accuracy': Overall accuracy
  - 'f1': F1 score (harmonic mean of precision/recall)
  - 'precision': Precision (TP / (TP + FP))
  - 'recall': Recall/Sensitivity (TP / (TP + FN))
  - 'roc_auc': ROC AUC score

Regression Metrics (str):
  - 'r2': R² coefficient of determination
  - 'neg_mean_absolute_error': Negative MAE (higher better)
  - 'neg_mean_squared_error': Negative MSE (higher better)

Custom Scorer (callable):
  - Function: score = scorer(y_true, y_pred)
  - Must return higher values for better performance
  - Use negative for loss metrics

Multiple Metrics (dict):
  - First key is primary metric for threshold
  - All metrics tracked and stored
  - Example: {'accuracy': 'accuracy', 'f1': 'f1'}

Default Behavior:
  - If scoring=None, uses validator's default scorer
  - XGBoost: accuracy (classification), R² (regression)
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
