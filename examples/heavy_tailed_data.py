"""
Heavy-Tailed Data Example

Demonstrates H-VRT's hybrid mode for non-well-behaved data (heavy tails, rare events).
Shows how random sampling fails when CLT assumptions break down.
"""

import numpy as np
from hvrt import HVRTSampleReducer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def generate_heavy_tailed_data(n_samples, n_features, rare_fraction=0.05, random_state=42):
    """Generate data with heavy-tailed noise and rare extreme events."""
    rng = np.random.RandomState(random_state)

    # Features
    X = rng.randn(n_samples, n_features)
    weights = np.exp(-np.arange(n_features) / 5.0)

    # Base signal
    y = X @ weights

    # Heavy-tailed Cauchy noise (infinite variance!)
    cauchy_noise = rng.standard_cauchy(n_samples) * 2.0
    cauchy_noise = np.clip(cauchy_noise, -50, 50)  # Clip extremes

    # Rare events: 5% of samples get 10x signal boost
    n_rare = int(n_samples * rare_fraction)
    rare_indices = rng.choice(n_samples, size=n_rare, replace=False)
    rare_mask = np.zeros(n_samples, dtype=bool)
    rare_mask[rare_indices] = True

    y_with_rare = y.copy()
    y_with_rare[rare_mask] *= 10.0  # Boost rare events

    # Final target
    y_final = y_with_rare + cauchy_noise

    return X, y_final, rare_mask


def random_sample(X, y, rare_mask, reduction_ratio, random_state):
    """Simple random sampling."""
    rng = np.random.RandomState(random_state)
    n_select = int(len(X) * reduction_ratio)
    indices = rng.choice(len(X), size=n_select, replace=False)

    rare_fraction_original = np.sum(rare_mask) / len(rare_mask)
    rare_fraction_selected = np.sum(rare_mask[indices]) / len(indices)

    return X[indices], y[indices], indices, rare_fraction_selected / rare_fraction_original


def main():
    print("=" * 70)
    print("Heavy-Tailed Data: H-VRT Hybrid Mode vs Random Sampling")
    print("=" * 70)
    print("\nDemonstrating H-VRT's advantage on non-well-behaved data:")
    print("  - Heavy-tailed Cauchy noise (infinite variance)")
    print("  - Rare extreme events (5% with 10x signal)")
    print("  - Where random sampling fails (CLT doesn't apply)")

    # Generate heavy-tailed data
    print("\n1. Generating heavy-tailed data...")
    np.random.seed(42)
    n_samples = 10000
    n_features = 20

    X, y, rare_mask = generate_heavy_tailed_data(n_samples, n_features)

    X_train, X_test, y_train, y_test, rare_train, rare_test = train_test_split(
        X, y, rare_mask, test_size=0.2, random_state=42
    )

    rare_fraction = np.sum(rare_train) / len(rare_train)
    print(f"   Dataset: {n_samples} samples")
    print(f"   Rare events: {rare_fraction*100:.1f}% of samples")
    print(f"   Noise: Heavy-tailed Cauchy (infinite variance)")

    # Baseline: Full dataset
    print("\n2. Training on FULL dataset (baseline)...")
    model_full = RandomForestRegressor(n_estimators=100, random_state=42)
    model_full.fit(X_train, y_train)
    y_pred_full = model_full.predict(X_test)
    r2_full = r2_score(y_test, y_pred_full)
    print(f"   R²: {r2_full:.4f}")

    # Method 1: Random sampling
    print("\n3. Random sampling (20% retention)...")
    X_rand, y_rand, idx_rand, rare_capture_rand = random_sample(
        X_train, y_train, rare_train, 0.2, 42
    )

    model_rand = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rand.fit(X_rand, y_rand)
    y_pred_rand = model_rand.predict(X_test)
    r2_rand = r2_score(y_test, y_pred_rand)

    print(f"   R²: {r2_rand:.4f} ({(r2_rand/r2_full)*100:.1f}% of full)")
    print(f"   Rare event capture: {rare_capture_rand:.2f}x")

    # Method 2: H-VRT original (pure X-interactions)
    print("\n4. H-VRT original (y_weight=0.0, X-interactions only)...")
    reducer_original = HVRTSampleReducer(
        reduction_ratio=0.2,
        y_weight=0.0,  # Pure X-interactions
        random_state=42
    )
    X_hvrt0, y_hvrt0 = reducer_original.fit_transform(X_train, y_train)
    idx_hvrt0 = reducer_original.selected_indices_

    rare_fraction_hvrt0 = np.sum(rare_train[idx_hvrt0]) / len(idx_hvrt0)
    rare_capture_hvrt0 = rare_fraction_hvrt0 / rare_fraction

    model_hvrt0 = RandomForestRegressor(n_estimators=100, random_state=42)
    model_hvrt0.fit(X_hvrt0, y_hvrt0)
    y_pred_hvrt0 = model_hvrt0.predict(X_test)
    r2_hvrt0 = r2_score(y_test, y_pred_hvrt0)

    print(f"   R²: {r2_hvrt0:.4f} ({(r2_hvrt0/r2_full)*100:.1f}% of full)")
    print(f"   Rare event capture: {rare_capture_hvrt0:.2f}x")

    # Method 3: H-VRT hybrid (y-awareness)
    print("\n5. H-VRT hybrid (y_weight=0.25, includes y-extremeness)...")
    reducer_hybrid = HVRTSampleReducer(
        reduction_ratio=0.2,
        y_weight=0.25,  # Hybrid: 75% X + 25% y-extremeness
        random_state=42
    )
    X_hvrt25, y_hvrt25 = reducer_hybrid.fit_transform(X_train, y_train)
    idx_hvrt25 = reducer_hybrid.selected_indices_

    rare_fraction_hvrt25 = np.sum(rare_train[idx_hvrt25]) / len(idx_hvrt25)
    rare_capture_hvrt25 = rare_fraction_hvrt25 / rare_fraction

    model_hvrt25 = RandomForestRegressor(n_estimators=100, random_state=42)
    model_hvrt25.fit(X_hvrt25, y_hvrt25)
    y_pred_hvrt25 = model_hvrt25.predict(X_test)
    r2_hvrt25 = r2_score(y_test, y_pred_hvrt25)

    print(f"   R²: {r2_hvrt25:.4f} ({(r2_hvrt25/r2_full)*100:.1f}% of full)")
    print(f"   Rare event capture: {rare_capture_hvrt25:.2f}x")

    # Comparison
    print("\n6. Comparison (20% retention):")
    print("=" * 70)
    print(f"{'Method':<25} {'R²':<10} {'vs Full':<12} {'Rare Capture':<15}")
    print("-" * 70)
    print(f"{'Full dataset':<25} {r2_full:.4f}    {'100.0%':<12} {'1.00x':<15}")
    print(f"{'Random sampling':<25} {r2_rand:.4f}    {f'{(r2_rand/r2_full)*100:.1f}%':<12} {f'{rare_capture_rand:.2f}x':<15}")
    print(f"{'H-VRT (original)':<25} {r2_hvrt0:.4f}    {f'{(r2_hvrt0/r2_full)*100:.1f}%':<12} {f'{rare_capture_hvrt0:.2f}x':<15}")
    print(f"{'H-VRT (hybrid)':<25} {r2_hvrt25:.4f}    {f'{(r2_hvrt25/r2_full)*100:.1f}%':<12} {f'{rare_capture_hvrt25:.2f}x':<15}")
    print("=" * 70)

    # Analysis
    print("\n7. Key Insights:")
    print("-" * 70)

    hybrid_advantage = (r2_hvrt25 / r2_rand - 1) * 100
    print(f"   - Random sampling: {(r2_rand/r2_full)*100:.1f}% accuracy (CLT fails on heavy tails)")
    print(f"   - H-VRT hybrid: {(r2_hvrt25/r2_full)*100:.1f}% accuracy (+{hybrid_advantage:.1f}% vs random)")
    print(f"   - Rare event capture:")
    print(f"     - Random: {rare_capture_rand:.2f}x (undersamples)")
    print(f"     - H-VRT hybrid: {rare_capture_hvrt25:.2f}x (oversamples!)")

    print("\n   Why hybrid mode wins on heavy-tailed data:")
    print("     1. Random treats all samples equally -> misses rare events")
    print("     2. H-VRT (original) uses X-patterns -> doesn't see y-extremeness")
    print("     3. H-VRT (hybrid) includes |y - median(y)| -> prioritizes outliers")

    print("\n" + "=" * 70)
    print("Recommendations:")
    print("  - Well-behaved data (normal noise):  y_weight=0.0 (default)")
    print("  - Heavy-tailed / rare events:        y_weight=0.25-0.50")
    print("  - Extreme outlier detection:         y_weight=0.50-1.0")
    print("=" * 70)


if __name__ == "__main__":
    main()
