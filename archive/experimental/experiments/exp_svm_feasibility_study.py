"""
SVM Feasibility Study: H-VRT vs Random at Scale (50k+ samples)

Research Question:
Does H-VRT make SVM feasible where it previously couldn't be, while retaining
accuracy, especially on non-well-behaved data?

Key Hypotheses:
1. At 50k+ samples, SVM is infeasible without reduction (5-30 min training)
2. H-VRT reduces training to <1 min while preserving 75-95% accuracy
3. Random sampling fails on non-well-behaved data (CLT breakdown)
4. H-VRT (especially hybrid mode) handles heavy tails / rare events

Methodology:
- Scales: 50k, 100k samples (where SVM becomes infeasible)
- Data types: Well-behaved (normal) vs Non-well-behaved (heavy-tailed, rare events)
- Reduction levels: 10%, 20%, 50% (focus on 20% as optimal)
- Comparison: H-VRT vs Random sampling
- Model: SVM (SVR for regression)
- Metrics: Training time, accuracy retention, rare event capture

Expected Results:
- Well-behaved data (50k): Random ~95%, H-VRT ~96% (CLT works, both good)
- Heavy-tailed data (50k): Random ~60-70%, H-VRT ~85-90% (CLT fails, H-VRT wins)
- 100k scale: Both enable feasibility, but accuracy gap persists

Duration: ~2-3 hours (SVM is slow at this scale)
"""

import numpy as np
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from hvrt.sample_reduction import HVRTSampleReducer
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


def generate_wellbehaved_data(
    n_samples: int,
    n_features: int,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate well-behaved data where CLT assumptions hold.

    DGP: y = sum(Xi * wi) + normal_noise
    - Linear relationship
    - Normal noise
    - IID samples
    - No outliers or heavy tails
    """
    rng = np.random.RandomState(random_state)

    # Features: standard normal
    X = rng.randn(n_samples, n_features)

    # Weights: decreasing importance
    weights = np.exp(-np.arange(n_features) / 5.0)

    # Target: linear combination + normal noise
    y = X @ weights + rng.randn(n_samples) * 0.5

    return X, y


def generate_heavy_tailed_data(
    n_samples: int,
    n_features: int,
    rare_event_fraction: float = 0.05,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate non-well-behaved data where CLT fails.

    DGP: y = sum(Xi * wi) + Cauchy_noise + rare_extreme_events
    - Linear relationship
    - Heavy-tailed Cauchy noise (infinite variance)
    - Rare extreme events (5% of samples)
    - Non-IID structure

    This tests H-VRT's hybrid approach (y_weight > 0).
    """
    rng = np.random.RandomState(random_state)

    # Features: standard normal
    X = rng.randn(n_samples, n_features)

    # Weights
    weights = np.exp(-np.arange(n_features) / 5.0)

    # Base signal
    y_signal = X @ weights

    # Heavy-tailed Cauchy noise (scale=2.0, very heavy tails)
    cauchy_noise = rng.standard_cauchy(n_samples) * 2.0
    # Clip extreme outliers to prevent numerical issues
    cauchy_noise = np.clip(cauchy_noise, -50, 50)

    # Rare events: 5% of samples get 10x signal boost
    n_rare = int(n_samples * rare_event_fraction)
    rare_indices = rng.choice(n_samples, size=n_rare, replace=False)
    rare_mask = np.zeros(n_samples, dtype=bool)
    rare_mask[rare_indices] = True

    # Assemble target
    y = y_signal.copy()
    y[rare_mask] *= 10.0  # Boost rare events
    y += cauchy_noise

    metadata = {
        'rare_event_fraction': rare_event_fraction,
        'rare_indices': rare_indices.tolist(),
        'rare_mask': rare_mask
    }

    return X, y, metadata


def random_sample(X: np.ndarray, y: np.ndarray, reduction_ratio: float, random_state: int):
    """Simple random sampling baseline."""
    n_select = int(len(X) * reduction_ratio)
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X), size=n_select, replace=False)
    return X[indices], y[indices], indices


def run_single_trial(
    data_type: str,
    n_samples: int,
    n_features: int,
    reduction_ratio: float,
    reducer_name: str,
    y_weight: float,
    random_state: int
) -> Dict:
    """Run a single SVM feasibility trial."""

    # Generate data
    if data_type == 'wellbehaved':
        X, y = generate_wellbehaved_data(n_samples, n_features, random_state)
        metadata = {'rare_mask': None}
    else:  # heavy_tailed
        X, y, metadata = generate_heavy_tailed_data(n_samples, n_features, random_state=random_state)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Track rare events in train set
    if metadata['rare_mask'] is not None:
        train_indices = np.arange(len(y))
        train_mask = np.isin(train_indices, range(len(y_train)))
        rare_mask_train = metadata['rare_mask'][train_mask]
    else:
        rare_mask_train = None

    # === BASELINE: Full training set ===
    start_time = time.time()

    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train)
    X_test_scaled = scaler_full.transform(X_test)

    model_full = SVR(kernel='rbf', C=1.0, gamma='scale', cache_size=1000)
    model_full.fit(X_train_scaled, y_train)

    time_train_full = time.time() - start_time

    y_pred_full = model_full.predict(X_test_scaled)
    r2_full = r2_score(y_test, y_pred_full)
    mse_full = mean_squared_error(y_test, y_pred_full)

    # === REDUCED: Apply sample reduction ===
    start_reduction = time.time()

    if reducer_name == 'H-VRT':
        reducer = HVRTSampleReducer(
            reduction_ratio=reduction_ratio,
            y_weight=y_weight,
            auto_tune=True,
            random_state=random_state
        )
        X_train_reduced, y_train_reduced = reducer.fit_transform(X_train, y_train)
        selected_indices = reducer.selected_indices_
    else:  # Random
        X_train_reduced, y_train_reduced, selected_indices = random_sample(
            X_train, y_train, reduction_ratio, random_state
        )

    time_reduction = time.time() - start_reduction

    # Train on reduced data
    start_train_reduced = time.time()

    scaler_reduced = StandardScaler()
    X_train_reduced_scaled = scaler_reduced.fit_transform(X_train_reduced)
    X_test_scaled_reduced = scaler_reduced.transform(X_test)

    model_reduced = SVR(kernel='rbf', C=1.0, gamma='scale', cache_size=1000)
    model_reduced.fit(X_train_reduced_scaled, y_train_reduced)

    time_train_reduced = time.time() - start_train_reduced
    time_total_reduced = time_reduction + time_train_reduced

    y_pred_reduced = model_reduced.predict(X_test_scaled_reduced)
    r2_reduced = r2_score(y_test, y_pred_reduced)
    mse_reduced = mean_squared_error(y_test, y_pred_reduced)

    # Compute metrics
    r2_retention_pct = (r2_reduced / r2_full * 100) if r2_full > 0 else 0.0
    training_speedup = time_train_full / time_train_reduced if time_train_reduced > 0 else 0.0
    total_speedup = time_train_full / time_total_reduced if time_total_reduced > 0 else 0.0

    # Rare event capture (only for heavy-tailed data)
    if rare_mask_train is not None:
        n_rare_full = np.sum(rare_mask_train)
        n_rare_reduced = np.sum(rare_mask_train[selected_indices])
        rare_fraction_full = n_rare_full / len(y_train)
        rare_fraction_reduced = n_rare_reduced / len(y_train_reduced)
        rare_capture_ratio = (rare_fraction_reduced / rare_fraction_full) if rare_fraction_full > 0 else 0.0
    else:
        rare_fraction_full = 0.0
        rare_fraction_reduced = 0.0
        rare_capture_ratio = 1.0

    return {
        'data_type': data_type,
        'n_samples': n_samples,
        'n_features': n_features,
        'reduction_ratio': reduction_ratio,
        'reducer': reducer_name,
        'y_weight': y_weight,

        # Timing
        'time_train_full_sec': time_train_full,
        'time_reduction_sec': time_reduction,
        'time_train_reduced_sec': time_train_reduced,
        'time_total_reduced_sec': time_total_reduced,
        'training_speedup': training_speedup,
        'total_speedup': total_speedup,

        # Accuracy
        'r2_full': r2_full,
        'r2_reduced': r2_reduced,
        'r2_retention_pct': r2_retention_pct,
        'mse_full': mse_full,
        'mse_reduced': mse_reduced,

        # Rare events
        'rare_fraction_full': rare_fraction_full,
        'rare_fraction_reduced': rare_fraction_reduced,
        'rare_capture_ratio': rare_capture_ratio,

        # Sample counts
        'n_train': len(y_train),
        'n_reduced': len(y_train_reduced),
        'n_test': len(y_test)
    }


def main():
    """Run SVM feasibility study."""

    print("=" * 80)
    print("SVM FEASIBILITY STUDY: H-VRT vs Random at Scale (50k+)")
    print("=" * 80)
    print()

    # Configuration
    scales = [50000, 100000]  # Focus on infeasible scales
    n_features = 20
    reduction_ratios = [0.10, 0.20, 0.50]  # 10%, 20%, 50% retention
    data_types = ['wellbehaved', 'heavy_tailed']
    reducers = [
        ('H-VRT', 0.0),      # Original (pure X-interactions)
        ('H-VRT', 0.25),     # Hybrid (balanced)
        ('Random', 0.0)      # Random baseline
    ]
    n_replications = 5  # Fewer reps due to SVM slowness

    results = []

    total_configs = (
        len(scales) *
        len(data_types) *
        len(reduction_ratios) *
        len(reducers) *
        n_replications
    )

    current = 0
    start_time_total = time.time()

    for scale in scales:
        for data_type in data_types:
            for reduction_ratio in reduction_ratios:
                for reducer_name, y_weight in reducers:

                    # Skip H-VRT y_weight=0.25 for well-behaved data (not needed)
                    if data_type == 'wellbehaved' and y_weight > 0:
                        continue

                    # Skip 100k scale if running out of time (estimate)
                    # SVM at 100k takes ~2-4 hours per trial, may need to skip
                    if scale == 100000:
                        print(f"\n[WARN] Skipping 100k scale (SVM training ~2-4 hours per trial)")
                        print(f"       To run, remove this skip condition and allocate ~10+ hours")
                        continue

                    for rep in range(n_replications):
                        current += 1
                        random_state = 42 + rep

                        print(f"\n[{current}/{total_configs}] "
                              f"data={data_type}, n={scale}, ratio={reduction_ratio:.0%}, "
                              f"reducer={reducer_name}, y_weight={y_weight}, rep={rep+1}")

                        try:
                            result = run_single_trial(
                                data_type=data_type,
                                n_samples=scale,
                                n_features=n_features,
                                reduction_ratio=reduction_ratio,
                                reducer_name=reducer_name,
                                y_weight=y_weight,
                                random_state=random_state
                            )

                            results.append(result)

                            # Print key metrics
                            print(f"  Full training: {result['time_train_full_sec']:.1f}s")
                            print(f"  Reduced training: {result['time_train_reduced_sec']:.1f}s "
                                  f"({result['training_speedup']:.1f}x speedup)")
                            print(f"  Total time: {result['time_total_reduced_sec']:.1f}s "
                                  f"({result['total_speedup']:.1f}x speedup)")
                            print(f"  R² retention: {result['r2_retention_pct']:.1f}%")

                            if data_type == 'heavy_tailed':
                                print(f"  Rare event capture: {result['rare_capture_ratio']:.2f}x "
                                      f"({result['rare_fraction_reduced']:.1%} vs {result['rare_fraction_full']:.1%} expected)")

                        except Exception as e:
                            print(f"  ERROR: {e}")
                            import traceback
                            traceback.print_exc()

    # Save results
    output_dir = project_root / 'results' / 'svm_feasibility'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'raw_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print(f"Total time: {(time.time() - start_time_total) / 60:.1f} minutes")
    print(f"{'=' * 80}")

    # Generate summary
    generate_summary(results, output_dir)


def generate_summary(results: List[Dict], output_dir: Path):
    """Generate summary statistics."""

    if len(results) == 0:
        print("No results to summarize")
        return

    import pandas as pd

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Group by data type, reducer, reduction ratio
    grouped = df.groupby(['data_type', 'reducer', 'y_weight', 'reduction_ratio'])

    summary = grouped.agg({
        'r2_retention_pct': ['mean', 'std'],
        'training_speedup': ['mean', 'std'],
        'total_speedup': ['mean', 'std'],
        'rare_capture_ratio': ['mean', 'std']
    }).round(2)

    print("\nAccuracy Retention (R² %):")
    print(summary['r2_retention_pct'])

    print("\nTraining Speedup:")
    print(summary['training_speedup'])

    print("\nTotal Speedup (including overhead):")
    print(summary['total_speedup'])

    print("\nRare Event Capture (heavy-tailed only):")
    heavy_tail_summary = df[df['data_type'] == 'heavy_tailed'].groupby(
        ['reducer', 'y_weight', 'reduction_ratio']
    )['rare_capture_ratio'].agg(['mean', 'std']).round(2)
    print(heavy_tail_summary)

    # Save summary
    summary_file = output_dir / 'summary_statistics.csv'
    summary.to_csv(summary_file)
    print(f"\nSummary saved to: {summary_file}")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Compare H-VRT vs Random on well-behaved
    wb_hvrt = df[(df['data_type'] == 'wellbehaved') &
                  (df['reducer'] == 'H-VRT') &
                  (df['reduction_ratio'] == 0.2)]['r2_retention_pct'].mean()
    wb_random = df[(df['data_type'] == 'wellbehaved') &
                    (df['reducer'] == 'Random') &
                    (df['reduction_ratio'] == 0.2)]['r2_retention_pct'].mean()

    print(f"\n1. Well-behaved data (20% retention):")
    print(f"   H-VRT:  {wb_hvrt:.1f}%")
    print(f"   Random: {wb_random:.1f}%")
    print(f"   Difference: {wb_hvrt - wb_random:+.1f}pp")

    # Compare on heavy-tailed
    ht_hvrt_0 = df[(df['data_type'] == 'heavy_tailed') &
                    (df['reducer'] == 'H-VRT') &
                    (df['y_weight'] == 0.0) &
                    (df['reduction_ratio'] == 0.2)]['r2_retention_pct'].mean()
    ht_hvrt_25 = df[(df['data_type'] == 'heavy_tailed') &
                     (df['reducer'] == 'H-VRT') &
                     (df['y_weight'] == 0.25) &
                     (df['reduction_ratio'] == 0.2)]['r2_retention_pct'].mean()
    ht_random = df[(df['data_type'] == 'heavy_tailed') &
                    (df['reducer'] == 'Random') &
                    (df['reduction_ratio'] == 0.2)]['r2_retention_pct'].mean()

    print(f"\n2. Heavy-tailed data (20% retention):")
    print(f"   H-VRT (y_weight=0.00): {ht_hvrt_0:.1f}%")
    print(f"   H-VRT (y_weight=0.25): {ht_hvrt_25:.1f}%")
    print(f"   Random:                {ht_random:.1f}%")
    print(f"   H-VRT hybrid advantage: {ht_hvrt_25 - ht_random:+.1f}pp")

    # Training time feasibility
    full_time_50k = df[df['n_samples'] == 50000]['time_train_full_sec'].mean()
    reduced_time_50k_hvrt = df[(df['n_samples'] == 50000) &
                                (df['reducer'] == 'H-VRT') &
                                (df['reduction_ratio'] == 0.2)]['time_train_reduced_sec'].mean()
    reduced_time_50k_random = df[(df['n_samples'] == 50000) &
                                  (df['reducer'] == 'Random') &
                                  (df['reduction_ratio'] == 0.2)]['time_train_reduced_sec'].mean()

    print(f"\n3. Training time feasibility (50k samples):")
    print(f"   Full:          {full_time_50k:.1f}s ({full_time_50k/60:.1f} min)")
    print(f"   H-VRT (20%):   {reduced_time_50k_hvrt:.1f}s ({reduced_time_50k_hvrt/60:.1f} min)")
    print(f"   Random (20%):  {reduced_time_50k_random:.1f}s ({reduced_time_50k_random/60:.1f} min)")
    print(f"   Speedup: {full_time_50k / reduced_time_50k_hvrt:.1f}x")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if ht_hvrt_25 > ht_random + 5:
        print("\n✅ H-VRT MAKES SVM FEASIBLE with significant accuracy advantage")
        print(f"   - Training time: {full_time_50k/60:.0f} min → {reduced_time_50k_hvrt/60:.0f} min")
        print(f"   - Heavy-tailed accuracy: +{ht_hvrt_25 - ht_random:.1f}pp over random")
        print(f"   - Use hybrid mode (y_weight=0.25) for non-well-behaved data")
    elif wb_hvrt > wb_random + 2:
        print("\n⚠️ H-VRT enables SVM feasibility, modest accuracy gain")
        print(f"   - Training time: {full_time_50k/60:.0f} min → {reduced_time_50k_hvrt/60:.0f} min")
        print(f"   - Well-behaved advantage: +{wb_hvrt - wb_random:.1f}pp")
        print(f"   - Main value: Determinism + interpretability, not raw performance")
    else:
        print("\n⚠️ Random sampling sufficient for speed, H-VRT adds determinism")
        print(f"   - Both enable SVM feasibility (similar accuracy)")
        print(f"   - Choose H-VRT for: regulatory compliance, interpretability")
        print(f"   - Choose Random for: raw speed (no overhead)")


if __name__ == '__main__':
    main()
