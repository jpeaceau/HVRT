"""
Demonstration of H-VRT Multi-Strategy Selection

Shows how to use built-in strategies and custom callables.
"""

import numpy as np
from hvrt import HVRTSampleReducer, centroid_fps, medoid_fps, variance_ordered, stratified
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def generate_synthetic_data(n_samples=5000, n_features=20, noise_level=0.5):
    """Generate synthetic regression data."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    # Y with interactions
    y = (
        2.0 * X[:, 0] +
        1.5 * X[:, 1] +
        3.0 * X[:, 0] * X[:, 2] +  # Interaction
        np.random.randn(n_samples) * noise_level
    )
    return X, y


def custom_hybrid_strategy(X_partition, n_select, random_state):
    """
    Custom strategy: Combine centroid FPS with variance ordering.

    This demonstrates how to create a custom selection strategy
    that follows the SelectionStrategy protocol.
    """
    n_points = len(X_partition)

    if n_points <= n_select:
        return np.arange(n_points)

    # Split selection: 70% from centroid FPS, 30% from variance-ordered
    n_fps = int(n_select * 0.7)
    n_var = n_select - n_fps

    # Use built-in strategies
    fps_indices = centroid_fps(X_partition, n_fps, random_state)
    var_indices = variance_ordered(X_partition, n_var, random_state)

    # Combine (ensure no duplicates)
    combined = np.unique(np.concatenate([fps_indices, var_indices]))

    # If we have too many, trim to n_select
    if len(combined) > n_select:
        combined = combined[:n_select]

    return combined.astype(np.int64)


def evaluate_strategy(X_train, y_train, X_test, y_test, strategy_name, reduction_ratio=0.2):
    """Evaluate a single selection strategy."""
    # Create reducer with specified strategy
    reducer = HVRTSampleReducer(
        reduction_ratio=reduction_ratio,
        selection_strategy=strategy_name,
        random_state=42
    )

    # Reduce training data
    reducer.fit(X_train, y_train)
    X_train_reduced = X_train[reducer.selected_indices_]
    y_train_reduced = y_train[reducer.selected_indices_]

    # Train model on full vs reduced
    model_full = RandomForestRegressor(n_estimators=100, random_state=42)
    model_reduced = RandomForestRegressor(n_estimators=100, random_state=42)

    model_full.fit(X_train, y_train)
    model_reduced.fit(X_train_reduced, y_train_reduced)

    # Evaluate
    r2_full = r2_score(y_test, model_full.predict(X_test))
    r2_reduced = r2_score(y_test, model_reduced.predict(X_test))
    retention = (r2_reduced / r2_full) * 100

    return {
        'strategy': strategy_name if isinstance(strategy_name, str) else strategy_name.__name__,
        'n_samples': len(X_train),
        'n_reduced': len(X_train_reduced),
        'r2_full': r2_full,
        'r2_reduced': r2_reduced,
        'retention_%': retention
    }


def main():
    print("=" * 70)
    print("H-VRT Multi-Strategy Selection Demo")
    print("=" * 70)

    # Generate data
    print("\n📊 Generating synthetic data...")
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")

    # Test all built-in strategies
    strategies = [
        'centroid_fps',
        'medoid_fps',
        'variance_ordered',
        'stratified',
        custom_hybrid_strategy  # Custom callable
    ]

    results = []
    print("\n🔬 Evaluating strategies (20% retention)...\n")

    for strategy in strategies:
        print(f"Testing: {strategy if isinstance(strategy, str) else strategy.__name__}...", end=" ")
        result = evaluate_strategy(X_train, y_train, X_test, y_test, strategy)
        results.append(result)
        print(f"[OK] {result['retention_%']:.2f}% retention")

    # Print results table
    print("\n" + "=" * 70)
    print("📈 Results Summary")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Reduced':<10} {'R² Full':<10} {'R² Reduced':<12} {'Retention':<10}")
    print("-" * 70)

    for result in results:
        print(
            f"{result['strategy']:<20} "
            f"{result['n_reduced']:<10} "
            f"{result['r2_full']:<10.4f} "
            f"{result['r2_reduced']:<12.4f} "
            f"{result['retention_%']:<10.2f}%"
        )

    # Find best strategy
    best = max(results, key=lambda x: x['retention_%'])
    print("\n" + "=" * 70)
    print(f"🏆 Best strategy: {best['strategy']} ({best['retention_%']:.2f}% retention)")
    print("=" * 70)

    # Demonstrate custom callable usage
    print("\n" + "=" * 70)
    print("💡 Custom Strategy Example")
    print("=" * 70)
    print("""
A custom strategy must follow the SelectionStrategy protocol:

    def my_strategy(X_partition, n_select, random_state):
        # Your selection logic here
        selected_indices = ...
        return selected_indices

Then use it:

    reducer = HVRTSampleReducer(
        selection_strategy=my_strategy,
        reduction_ratio=0.2
    )
    """)

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
