"""
Adaptive Reduction Example

Demonstrates how to automatically find the optimal reduction level using
AdaptiveHVRTReducer with accuracy threshold testing.

Use Cases:
1. SVM Training: Use XGBoost for validation, get reduced samples for SVM
2. Faster Inference: Reduce to minimum samples while maintaining accuracy
3. Unknown Optimal Reduction: Let the tool find it automatically
"""

import numpy as np
from hvrt import AdaptiveHVRTReducer
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time


def main():
    print("=" * 70)
    print("Adaptive H-VRT: Automatic Reduction Level Selection")
    print("=" * 70)

    # Generate classification data
    print("\n1. Generating classification data (10k samples, 20 features)...")
    np.random.seed(42)
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )

    # Split train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")

    # Use Case 1: Find optimal reduction for SVM training
    print("\n2. Finding optimal reduction for SVM training...")
    print("   (Using XGBoost for fast validation, final model is SVM)")
    print()

    reducer = AdaptiveHVRTReducer(
        accuracy_threshold=0.95,  # Keep 95% of baseline accuracy
        reduction_ratios=[0.5, 0.3, 0.2, 0.15, 0.1, 0.05],
        cv=3,
        random_state=42,
        verbose=True
    )

    # Fit reducer (finds best reduction automatically)
    reducer.fit(X_train, y_train)

    # Get best reduced dataset
    X_train_reduced, y_train_reduced = reducer.transform()

    # Review all tested reductions
    print("\n3. Reduction results summary:")
    print(reducer.get_reduction_summary())

    # Train SVM on full vs reduced data
    print("\n4. Training SVM on full vs reduced data...")

    # Full dataset
    print("\n   Training SVM on FULL dataset (8000 samples)...")
    start = time.time()
    svm_full = SVC(kernel='rbf', random_state=42)
    svm_full.fit(X_train, y_train)
    time_full = time.time() - start
    y_pred_full = svm_full.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred_full)

    print(f"   Training time: {time_full:.2f}s")
    print(f"   Test accuracy: {acc_full:.4f}")

    # Reduced dataset
    n_reduced = len(X_train_reduced)
    print(f"\n   Training SVM on REDUCED dataset ({n_reduced} samples)...")
    start = time.time()
    svm_reduced = SVC(kernel='rbf', random_state=42)
    svm_reduced.fit(X_train_reduced, y_train_reduced)
    time_reduced = time.time() - start
    y_pred_reduced = svm_reduced.predict(X_test)
    acc_reduced = accuracy_score(y_test, y_pred_reduced)

    print(f"   Training time: {time_reduced:.2f}s")
    print(f"   Test accuracy: {acc_reduced:.4f}")

    # Compare
    print("\n5. Performance comparison:")
    print("-" * 70)
    speedup = time_full / time_reduced
    acc_retention = (acc_reduced / acc_full) * 100
    reduction_pct = (1 - n_reduced / len(X_train)) * 100

    print(f"   Reduction:        {reduction_pct:.0f}% fewer samples")
    print(f"   Training speedup: {speedup:.1f}x")
    print(f"   Accuracy:         {acc_full:.4f} -> {acc_reduced:.4f} ({acc_retention:.1f}% retention)")

    # Use Case 2: Review results and make manual decision
    print("\n6. Making manual decisions from results...")
    print("\n   All tested reductions:")
    for result in reducer.reduction_results_:
        print(f"      {result['reduction_ratio']:.0%}: "
              f"{result['n_samples']:4d} samples, "
              f"{result['accuracy_retention']:.1%} accuracy")

    print("\n   You can access any reduction for manual selection:")
    print("   - reducer.reduction_results_[0] for most aggressive")
    print("   - reducer.best_reduction_ for auto-selected best")

    # Use Case 3: Heavy-tailed data with hybrid mode
    print("\n7. Bonus: Heavy-tailed data example...")
    print("   (Using y_weight=0.25 for hybrid mode)")

    # Generate heavy-tailed data
    X_heavy = np.random.randn(5000, 20)
    weights = np.exp(-np.arange(20) / 5.0)
    y_heavy_signal = X_heavy @ weights

    # Add Cauchy noise (heavy tails)
    cauchy_noise = np.random.standard_cauchy(5000) * 2.0
    cauchy_noise = np.clip(cauchy_noise, -50, 50)
    y_heavy = y_heavy_signal + cauchy_noise

    # Binarize for classification
    y_heavy_class = (y_heavy > np.median(y_heavy)).astype(int)

    X_heavy_train, X_heavy_test, y_heavy_train, y_heavy_test = train_test_split(
        X_heavy, y_heavy_class, test_size=0.2, random_state=42
    )

    reducer_heavy = AdaptiveHVRTReducer(
        accuracy_threshold=0.90,  # Slightly lower threshold for heavy tails
        reduction_ratios=[0.5, 0.3, 0.2],
        y_weight=0.25,  # Hybrid mode for rare events
        cv=3,
        random_state=42,
        verbose=False
    )

    reducer_heavy.fit(X_heavy_train, y_heavy_train)

    print(f"\n   Best reduction for heavy-tailed data:")
    print(f"   - Reduction level: {reducer_heavy.best_reduction_['reduction_ratio']:.0%}")
    print(f"   - Samples: {reducer_heavy.best_reduction_['n_samples']}")
    print(f"   - Accuracy retention: {reducer_heavy.best_reduction_['accuracy_retention']:.1%}")

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("  1. AdaptiveHVRTReducer automatically finds optimal reduction")
    print("  2. Use XGBoost for validation even if final model is SVM")
    print("  3. All reduction results stored for manual review")
    print("  4. Hybrid mode (y_weight>0) for heavy-tailed data")
    print("  5. Significant speedup with minimal accuracy loss")
    print("=" * 70)


if __name__ == "__main__":
    main()
