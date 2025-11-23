"""
SVM Feasibility PILOT WITH SNR MEASUREMENTS

Re-runs the SVM pilot experiment with SNR (Signal-to-Noise Ratio) measurements
to quantify data quality for baseline, random sampling, and H-VRT.

SNR Definition:
    SNR = Var(signal) / Var(noise)

Where:
    - signal = true underlying pattern (X @ weights)
    - noise = y - signal

Higher SNR = cleaner data, easier to learn from
Lower SNR = noisier data, harder to learn from

This helps answer: Does H-VRT preserve SNR better than random sampling?
"""

import sys
from pathlib import Path
import numpy as np
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from hvrt.sample_reduction import HVRTSampleReducer
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


def compute_snr(X, y, weights):
    """
    Compute signal-to-noise ratio.

    SNR = Var(signal) / Var(noise)

    Parameters
    ----------
    X : array (n_samples, n_features)
        Feature matrix
    y : array (n_samples,)
        Target values
    weights : array (n_features,)
        True generative weights

    Returns
    -------
    snr : float
        Signal-to-noise ratio
    signal_var : float
        Variance of signal component
    noise_var : float
        Variance of noise component
    """
    signal = X @ weights
    noise = y - signal

    signal_var = np.var(signal)
    noise_var = np.var(noise)

    # Avoid division by zero
    if noise_var < 1e-10:
        snr = np.inf
    else:
        snr = signal_var / noise_var

    return snr, signal_var, noise_var


def generate_wellbehaved_data(n_samples, n_features, random_state=42):
    """Generate well-behaved data where CLT assumptions hold."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    weights = np.exp(-np.arange(n_features) / 5.0)
    y = X @ weights + rng.randn(n_samples) * 0.5

    return X, y, weights


def generate_heavy_tailed_data(n_samples, n_features, rare_event_fraction=0.05, random_state=42):
    """Generate non-well-behaved data where CLT fails."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    weights = np.exp(-np.arange(n_features) / 5.0)

    y_signal = X @ weights
    cauchy_noise = rng.standard_cauchy(n_samples) * 2.0
    cauchy_noise = np.clip(cauchy_noise, -50, 50)

    # Rare events
    n_rare = int(n_samples * rare_event_fraction)
    rare_indices = rng.choice(n_samples, size=n_rare, replace=False)
    rare_mask = np.zeros(n_samples, dtype=bool)
    rare_mask[rare_indices] = True

    y = y_signal.copy()
    y[rare_mask] *= 10.0
    y += cauchy_noise

    metadata = {
        'rare_event_fraction': rare_event_fraction,
        'rare_indices': rare_indices.tolist(),
        'rare_mask': rare_mask
    }

    return X, y, weights, metadata


def random_sample(X, y, reduction_ratio, random_state):
    """Simple random sampling baseline."""
    n_select = int(len(X) * reduction_ratio)
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X), size=n_select, replace=False)
    return X[indices], y[indices], indices


def run_single_trial(
    data_type,
    n_samples,
    n_features,
    reduction_ratio,
    reducer_name,
    y_weight,
    random_state
):
    """Run a single SVM feasibility trial WITH SNR measurements."""

    # Generate data
    if data_type == 'wellbehaved':
        X, y, weights = generate_wellbehaved_data(n_samples, n_features, random_state)
        metadata = None
    else:
        X, y, weights, metadata = generate_heavy_tailed_data(
            n_samples, n_features, random_state=random_state
        )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Compute SNR for FULL training set
    snr_full, signal_var_full, noise_var_full = compute_snr(X_train, y_train, weights)

    # Train SVM on full data
    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train)
    X_test_scaled = scaler_full.transform(X_test)

    start = time.time()
    svm_full = SVR(kernel='rbf', C=1.0)
    svm_full.fit(X_train_scaled, y_train)
    time_full = time.time() - start

    y_pred_full = svm_full.predict(X_test_scaled)
    r2_full = r2_score(y_test, y_pred_full)
    mse_full = mean_squared_error(y_test, y_pred_full)

    # Apply reduction
    start_reduction = time.time()
    if reducer_name == 'Random':
        X_train_reduced, y_train_reduced, selected_indices = random_sample(
            X_train, y_train, reduction_ratio, random_state
        )
    else:  # H-VRT
        reducer = HVRTSampleReducer(
            reduction_ratio=reduction_ratio,
            y_weight=y_weight,
            random_state=random_state
        )
        X_train_reduced, y_train_reduced = reducer.fit_transform(X_train, y_train)
        selected_indices = reducer.selected_indices_

    time_reduction = time.time() - start_reduction

    # Compute SNR for REDUCED training set
    snr_reduced, signal_var_reduced, noise_var_reduced = compute_snr(
        X_train_reduced, y_train_reduced, weights
    )

    # Train SVM on reduced data
    scaler_reduced = StandardScaler()
    X_train_reduced_scaled = scaler_reduced.fit_transform(X_train_reduced)
    X_test_scaled_reduced = scaler_reduced.transform(X_test)

    start = time.time()
    svm_reduced = SVR(kernel='rbf', C=1.0)
    svm_reduced.fit(X_train_reduced_scaled, y_train_reduced)
    time_reduced = time.time() - start

    y_pred_reduced = svm_reduced.predict(X_test_scaled_reduced)
    r2_reduced = r2_score(y_test, y_pred_reduced)
    mse_reduced = mean_squared_error(y_test, y_pred_reduced)

    # Rare event capture (if heavy-tailed)
    if metadata is not None:
        rare_mask_train = metadata['rare_mask'][:len(X_train)]
        rare_fraction_full = rare_mask_train.sum() / len(rare_mask_train)
        rare_fraction_reduced = rare_mask_train[selected_indices].sum() / len(selected_indices)
        rare_capture_ratio = rare_fraction_reduced / (rare_fraction_full + 1e-10)
    else:
        rare_fraction_full = 0.0
        rare_fraction_reduced = 0.0
        rare_capture_ratio = 1.0

    # Compile results
    result = {
        'data_type': data_type,
        'n_samples': n_samples,
        'n_features': n_features,
        'reduction_ratio': reduction_ratio,
        'reducer': reducer_name,
        'y_weight': y_weight,

        # Timing
        'time_train_full_sec': time_full,
        'time_reduction_sec': time_reduction,
        'time_train_reduced_sec': time_reduced,
        'time_total_reduced_sec': time_reduction + time_reduced,
        'training_speedup': time_full / time_reduced,
        'total_speedup': time_full / (time_reduction + time_reduced),

        # Accuracy
        'r2_full': r2_full,
        'r2_reduced': r2_reduced,
        'r2_retention_pct': (r2_reduced / r2_full) * 100,
        'mse_full': mse_full,
        'mse_reduced': mse_reduced,

        # SNR measurements (NEW!)
        'snr_full': snr_full,
        'snr_reduced': snr_reduced,
        'snr_retention_pct': (snr_reduced / snr_full) * 100,
        'signal_var_full': signal_var_full,
        'signal_var_reduced': signal_var_reduced,
        'noise_var_full': noise_var_full,
        'noise_var_reduced': noise_var_reduced,

        # Rare events
        'rare_fraction_full': rare_fraction_full,
        'rare_fraction_reduced': rare_fraction_reduced,
        'rare_capture_ratio': rare_capture_ratio,

        # Metadata
        'n_train': len(X_train),
        'n_reduced': len(X_train_reduced),
        'n_test': len(X_test)
    }

    return result


def main_pilot_with_snr():
    """Run pilot at 10k scale WITH SNR measurements."""

    print("=" * 80)
    print("SVM FEASIBILITY PILOT WITH SNR MEASUREMENTS")
    print("=" * 80)
    print("Purpose: Add SNR (Signal-to-Noise Ratio) measurements to pilot results")
    print("Metric: SNR = Var(signal) / Var(noise)")
    print("Question: Does H-VRT preserve SNR better than random sampling?")
    print()

    # Pilot configuration
    scales = [10000]
    n_features = 20
    reduction_ratios = [0.20]
    data_types = ['wellbehaved', 'heavy_tailed']
    reducers = [
        ('H-VRT', 0.0),
        ('H-VRT', 0.25),
        ('Random', 0.0)
    ]
    n_replications = 3

    results = []
    current = 0
    start_time = time.time()

    # Calculate total
    total = len(scales) * len(reduction_ratios) * n_replications * (
        2 * 2 +  # wellbehaved: 2 reducers (skip hybrid)
        3        # heavy_tailed: 3 reducers
    )

    for scale in scales:
        for data_type in data_types:
            for reduction_ratio in reduction_ratios:
                for reducer_name, y_weight in reducers:

                    # Skip hybrid for well-behaved
                    if data_type == 'wellbehaved' and y_weight > 0:
                        continue

                    for rep in range(n_replications):
                        current += 1
                        random_state = 42 + rep

                        print(f"\n[{current}/{total}] {data_type}, n={scale}, "
                              f"ratio={reduction_ratio:.0%}, {reducer_name}, "
                              f"y_weight={y_weight}, rep={rep+1}")

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

                            # Print metrics
                            print(f"  Speedup: {result['training_speedup']:.1f}x")
                            print(f"  R² retention: {result['r2_retention_pct']:.1f}%")
                            print(f"  SNR: {result['snr_full']:.2f} -> {result['snr_reduced']:.2f} "
                                  f"({result['snr_retention_pct']:.1f}%)")

                            if data_type == 'heavy_tailed':
                                print(f"  Rare capture: {result['rare_capture_ratio']:.2f}x")

                        except Exception as e:
                            print(f"  ERROR: {e}")
                            import traceback
                            traceback.print_exc()

    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'svm_pilot'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'pilot_results_with_snr.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("PILOT COMPLETE WITH SNR")
    print("=" * 80)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_file}")
    print()

    # Quick summary by data type and reducer
    print("SUMMARY BY DATA TYPE AND REDUCER:")
    print("-" * 80)

    for data_type in data_types:
        print(f"\n{data_type.upper()}:")
        for reducer_name, y_weight in reducers:
            if data_type == 'wellbehaved' and y_weight > 0:
                continue

            subset = [r for r in results if r['data_type'] == data_type
                     and r['reducer'] == reducer_name and r['y_weight'] == y_weight]

            if subset:
                avg_r2_retention = np.mean([r['r2_retention_pct'] for r in subset])
                avg_snr_retention = np.mean([r['snr_retention_pct'] for r in subset])
                avg_speedup = np.mean([r['training_speedup'] for r in subset])

                label = f"{reducer_name} (y_weight={y_weight})" if reducer_name == 'H-VRT' else reducer_name
                print(f"  {label:25s}: R² {avg_r2_retention:5.1f}%, "
                      f"SNR {avg_snr_retention:6.1f}%, Speedup {avg_speedup:4.1f}x")

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("-" * 80)

    # Compare SNR retention
    wellbehaved_hvrt = [r for r in results if r['data_type'] == 'wellbehaved'
                       and r['reducer'] == 'H-VRT' and r['y_weight'] == 0.0]
    wellbehaved_random = [r for r in results if r['data_type'] == 'wellbehaved'
                         and r['reducer'] == 'Random']

    heavy_hvrt = [r for r in results if r['data_type'] == 'heavy_tailed'
                 and r['reducer'] == 'H-VRT' and r['y_weight'] == 0.25]
    heavy_random = [r for r in results if r['data_type'] == 'heavy_tailed'
                   and r['reducer'] == 'Random']

    print("\nWell-behaved data:")
    if wellbehaved_hvrt and wellbehaved_random:
        print(f"  H-VRT:  SNR retention = {np.mean([r['snr_retention_pct'] for r in wellbehaved_hvrt]):.1f}%")
        print(f"  Random: SNR retention = {np.mean([r['snr_retention_pct'] for r in wellbehaved_random]):.1f}%")

    print("\nHeavy-tailed data:")
    if heavy_hvrt and heavy_random:
        print(f"  H-VRT (hybrid): SNR retention = {np.mean([r['snr_retention_pct'] for r in heavy_hvrt]):.1f}%")
        print(f"  Random:         SNR retention = {np.mean([r['snr_retention_pct'] for r in heavy_random]):.1f}%")

    print("=" * 80)


if __name__ == '__main__':
    main_pilot_with_snr()
