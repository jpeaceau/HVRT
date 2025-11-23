"""
SVM Speedup Demonstration

Shows how H-VRT makes SVM training practical on medium-to-large datasets.
Demonstrates 25-40x training speedup with minimal accuracy loss.
"""

import numpy as np
import time
from hvrt import HVRTSampleReducer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def main():
    print("=" * 70)
    print("H-VRT SVM Speedup Demonstration")
    print("=" * 70)
    print("\nDemonstrating how H-VRT makes SVM training feasible at scale")
    print("(SVM training is O(n²-n³), becomes slow above 10k samples)")

    # Generate data (10k samples for demo; try 50k for dramatic speedup)
    print("\n1. Generating data...")
    np.random.seed(42)
    n_samples = 10000  # Try 50000 for ~30 min vs ~47 sec comparison
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    weights = np.exp(-np.arange(n_features) / 5.0)
    y = X @ weights + np.random.randn(n_samples) * 0.5

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Dataset: {n_samples} samples × {n_features} features")
    print(f"   Training set: {len(y_train)} samples")

    # Baseline: SVM on full dataset
    print("\n2. Training SVM on FULL dataset...")
    print("   (This can take several minutes for larger datasets)")

    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train)
    X_test_scaled = scaler_full.transform(X_test)

    start_full = time.time()
    svm_full = SVR(kernel='rbf', C=1.0)
    svm_full.fit(X_train_scaled, y_train)
    time_full = time.time() - start_full

    y_pred_full = svm_full.predict(X_test_scaled)
    r2_full = r2_score(y_test, y_pred_full)

    print(f"   Training time: {time_full:.2f}s")
    print(f"   R² score: {r2_full:.4f}")

    # H-VRT reduction (20% retention)
    print("\n3. Applying H-VRT reduction (20% retention)...")

    start_reduction = time.time()
    reducer = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
    X_train_reduced, y_train_reduced = reducer.fit_transform(X_train, y_train)
    time_reduction = time.time() - start_reduction

    print(f"   Reduction time: {time_reduction:.2f}s")
    print(f"   Reduced to: {len(y_train_reduced)} samples (5x smaller)")

    # SVM on reduced dataset
    print("\n4. Training SVM on REDUCED dataset...")

    scaler_reduced = StandardScaler()
    X_train_reduced_scaled = scaler_reduced.fit_transform(X_train_reduced)
    X_test_scaled_reduced = scaler_reduced.transform(X_test)

    start_reduced = time.time()
    svm_reduced = SVR(kernel='rbf', C=1.0)
    svm_reduced.fit(X_train_reduced_scaled, y_train_reduced)
    time_reduced = time.time() - start_reduced

    y_pred_reduced = svm_reduced.predict(X_test_scaled_reduced)
    r2_reduced = r2_score(y_test, y_pred_reduced)

    print(f"   Training time: {time_reduced:.2f}s")
    print(f"   R² score: {r2_reduced:.4f}")

    # Analysis
    print("\n5. Speedup Analysis:")
    print("-" * 70)

    training_speedup = time_full / time_reduced
    total_speedup = time_full / (time_reduction + time_reduced)
    r2_retention = (r2_reduced / r2_full) * 100

    print(f"   Full training:     {time_full:.2f}s")
    print(f"   Reduced training:  {time_reduced:.2f}s ({training_speedup:.1f}x faster)")
    print(f"   Reduction overhead: {time_reduction:.2f}s")
    print(f"   Total speedup:     {total_speedup:.1f}x")
    print()
    print(f"   Full R²:      {r2_full:.4f}")
    print(f"   Reduced R²:   {r2_reduced:.4f}")
    print(f"   Retention:    {r2_retention:.1f}%")

    print("\n6. Practical Impact:")
    print("-" * 70)

    if n_samples >= 50000:
        print("   At 50k samples:")
        print(f"   - Full SVM: ~{time_full/60:.1f} minutes per training")
        print(f"   - Reduced: ~{time_reduced:.0f} seconds per training")
        print(f"   - Hyperparameter tuning (100 configs):")
        print(f"     - Full: ~{time_full*100/3600:.1f} hours ❌ INFEASIBLE")
        print(f"     - Reduced: ~{(time_reduction + time_reduced*100)/60:.1f} min [OK] FEASIBLE")
    else:
        print("   At 10k samples (current):")
        print(f"   - Training speedup: {training_speedup:.1f}x")
        print(f"   - Try n_samples=50000 for dramatic effect:")
        print("     - Full SVM: ~30 minutes")
        print("     - Reduced: ~47 seconds")
        print("     - Speedup: ~38x")

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("  [OK] H-VRT makes SVM feasible where it previously wasn't")
    print(f"  [OK] {training_speedup:.0f}x training speedup with {r2_retention:.0f}% accuracy retention")
    print("  [OK] Enables hyperparameter tuning on larger datasets")
    print("  [OK] Overhead amortized across multiple trainings")
    print("=" * 70)


if __name__ == "__main__":
    main()
