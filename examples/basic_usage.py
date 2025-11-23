"""
Basic H-VRT Usage Example

Demonstrates the simplest way to use H-VRT for sample reduction.
This example reduces a 10k sample dataset to 20% while preserving accuracy.
"""

import numpy as np
from hvrt import HVRTSampleReducer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def main():
    print("=" * 70)
    print("H-VRT Basic Usage Example")
    print("=" * 70)

    # Generate synthetic data (10k samples, 20 features)
    print("\n1. Generating data (10k samples, 20 features)...")
    np.random.seed(42)
    n_samples = 10000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    # Target: combination of features with noise
    weights = np.exp(-np.arange(n_features) / 5.0)
    y = X @ weights + np.random.randn(n_samples) * 0.5

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training set: {len(y_train)} samples")
    print(f"   Test set: {len(y_test)} samples")

    # Baseline: Train on full dataset
    print("\n2. Training baseline model (full dataset)...")
    model_full = RandomForestRegressor(n_estimators=100, random_state=42)
    model_full.fit(X_train, y_train)
    y_pred_full = model_full.predict(X_test)
    r2_full = r2_score(y_test, y_pred_full)
    print(f"   Baseline R²: {r2_full:.4f}")

    # Apply H-VRT reduction (20% retention)
    print("\n3. Applying H-VRT sample reduction (20% retention)...")
    reducer = HVRTSampleReducer(
        reduction_ratio=0.2,  # Keep 20% of samples
        random_state=42       # Reproducible
    )
    X_train_reduced, y_train_reduced = reducer.fit_transform(X_train, y_train)

    print(f"   Original: {len(y_train)} samples")
    print(f"   Reduced:  {len(y_train_reduced)} samples")
    print(f"   Reduction: {(1 - len(y_train_reduced)/len(y_train))*100:.0f}%")

    # Train on reduced dataset
    print("\n4. Training on reduced dataset...")
    model_reduced = RandomForestRegressor(n_estimators=100, random_state=42)
    model_reduced.fit(X_train_reduced, y_train_reduced)
    y_pred_reduced = model_reduced.predict(X_test)
    r2_reduced = r2_score(y_test, y_pred_reduced)
    print(f"   Reduced R²: {r2_reduced:.4f}")

    # Compare
    print("\n5. Results:")
    print(f"   Baseline R²:  {r2_full:.4f} (using 100% of training data)")
    print(f"   Reduced R²:   {r2_reduced:.4f} (using 20% of training data)")
    print(f"   Retention:    {(r2_reduced/r2_full)*100:.1f}%")

    if r2_reduced/r2_full >= 0.95:
        print("\n   [OK] Excellent: >95% accuracy retained with 80% fewer samples")
    elif r2_reduced/r2_full >= 0.90:
        print("\n   [OK] Good: >90% accuracy retained with 80% fewer samples")
    else:
        print("\n   [WARN] Moderate: Consider using higher retention ratio")

    # Show determinism
    print("\n6. Demonstrating determinism...")
    reducer2 = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
    X_train_reduced2, _ = reducer2.fit_transform(X_train, y_train)

    if np.array_equal(X_train_reduced, X_train_reduced2):
        print("   [OK] Same random_state -> identical samples selected")

    print("\n" + "=" * 70)
    print("Example complete! Try modifying:")
    print("  - reduction_ratio (e.g., 0.1, 0.5)")
    print("  - n_samples (e.g., 50000 for larger scale)")
    print("  - different models (SVM, XGBoost, etc.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
