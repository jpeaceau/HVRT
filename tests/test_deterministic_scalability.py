"""
Quick scalability test for deterministic FPS.

Verifies that centroid-based seed selection doesn't degrade performance.
"""

import numpy as np
import sys
import time

from src.hvrt import HVRTSampleReducer

def test_scalability():
    """Test reduction time at different scales."""

    scales = [1000, 5000, 10000, 50000]

    print("=" * 70)
    print("DETERMINISTIC FPS SCALABILITY TEST")
    print("=" * 70)
    print()

    for n_samples in scales:
        # Generate data
        np.random.seed(42)
        X = np.random.randn(n_samples, 20)
        y = np.random.randn(n_samples)

        # Time reduction
        start = time.time()
        reducer = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
        X_reduced, y_reduced = reducer.fit_transform(X, y)
        elapsed = time.time() - start

        throughput = n_samples / elapsed

        print(f"{n_samples:6,} samples: {elapsed:6.3f}s ({throughput:8,.0f} samples/sec)")

    print()
    print("=" * 70)
    print("All scales completed successfully")
    print("=" * 70)

if __name__ == '__main__':
    test_scalability()
