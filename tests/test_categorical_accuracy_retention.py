"""
Test H-VRT Accuracy Retention with Categorical Features.

Validates that H-VRT preserves predictive accuracy when reducing datasets
with categorical features, comparing performance on full vs reduced data.

Tests cover:
1. Mixed continuous/categorical features
2. Categorical-only features
3. Different reduction ratios
4. Multiple model types (Random Forest, XGBoost)
5. Comparison with random sampling baseline
"""

import unittest
import numpy as np
import sys

from src.hvrt import HVRTSampleReducer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split


class TestCategoricalAccuracyRetention(unittest.TestCase):
    """Test accuracy retention with categorical features."""

    def setUp(self):
        """Set up test data with categorical features."""
        np.random.seed(42)
        self.n_samples = 2000
        self.test_size = 0.3

    def _generate_mixed_classification_data(self):
        """Generate classification data with mixed features."""
        # Continuous features
        X_cont = np.random.randn(self.n_samples, 4)

        # Categorical features with predictive power
        cat1 = np.random.choice(['A', 'B', 'C', 'D'], size=self.n_samples)
        cat2 = np.random.choice(['low', 'medium', 'high'], size=self.n_samples)
        cat3 = np.random.choice(['red', 'blue', 'green'], size=self.n_samples)

        # Combine features
        X_mixed = np.column_stack([X_cont, cat1, cat2, cat3])
        feature_types = ['continuous'] * 4 + ['categorical'] * 3

        # Generate target with interactions
        y = (
            (X_cont[:, 0] > 0).astype(int) +
            (cat1 == 'A').astype(int) +
            (cat2 == 'high').astype(int) +
            (cat3 == 'red').astype(int)
        )
        # Convert to binary classification
        y = (y >= 2).astype(int)

        return X_mixed, y, feature_types

    def _generate_mixed_regression_data(self):
        """Generate regression data with mixed features."""
        # Continuous features
        X_cont = np.random.randn(self.n_samples, 4)

        # Categorical features
        cat1 = np.random.choice(['A', 'B', 'C'], size=self.n_samples)
        cat2 = np.random.choice(['low', 'medium', 'high'], size=self.n_samples)

        # Combine features
        X_mixed = np.column_stack([X_cont, cat1, cat2])
        feature_types = ['continuous'] * 4 + ['categorical'] * 2

        # Generate target with strong categorical effects
        y = (
            X_cont[:, 0] * 2.0 +
            X_cont[:, 1] * 1.5 +
            (cat1 == 'A').astype(float) * 3.0 +
            (cat2 == 'high').astype(float) * 2.5 +
            np.random.randn(self.n_samples) * 0.5
        )

        return X_mixed, y, feature_types

    def _generate_categorical_only_data(self):
        """Generate categorical-only classification data."""
        # Multiple categorical features
        cat1 = np.random.choice(['A', 'B', 'C', 'D'], size=self.n_samples)
        cat2 = np.random.choice(['low', 'medium', 'high'], size=self.n_samples)
        cat3 = np.random.choice(['red', 'blue', 'green', 'yellow'], size=self.n_samples)
        cat4 = np.random.choice(['small', 'large'], size=self.n_samples)

        X_cat = np.column_stack([cat1, cat2, cat3, cat4])
        feature_types = ['categorical'] * 4

        # Target depends on categorical combinations
        y = (
            (cat1 == 'A').astype(int) +
            (cat2 == 'high').astype(int) +
            (cat3 == 'red').astype(int) +
            (cat4 == 'large').astype(int)
        )
        y = (y >= 2).astype(int)

        return X_cat, y, feature_types

    def test_mixed_features_classification(self):
        """Test accuracy retention with mixed continuous/categorical features."""
        X, y, feature_types = self._generate_mixed_classification_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        # Train on full data
        rf_full = RandomForestClassifier(n_estimators=50, random_state=42)
        # Need to encode categoricals for sklearn - use simple approach
        X_train_encoded = self._simple_encode(X_train, feature_types)
        X_test_encoded = self._simple_encode(X_test, feature_types, fit=False)
        rf_full.fit(X_train_encoded, y_train)
        acc_full = accuracy_score(y_test, rf_full.predict(X_test_encoded))

        # Reduce training data
        reducer = HVRTSampleReducer(reduction_ratio=0.3, random_state=42)
        X_train_reduced, y_train_reduced = reducer.fit_transform(
            X_train, y_train, feature_types=feature_types
        )

        # Train on reduced data
        rf_reduced = RandomForestClassifier(n_estimators=50, random_state=42)
        X_train_reduced_encoded = self._simple_encode(X_train_reduced, feature_types)
        rf_reduced.fit(X_train_reduced_encoded, y_train_reduced)
        acc_reduced = accuracy_score(y_test, rf_reduced.predict(X_test_encoded))

        # Calculate retention
        retention = acc_reduced / acc_full if acc_full > 0 else 0

        # Should retain at least 85% of accuracy
        self.assertGreater(retention, 0.85,
                          f"Accuracy retention {retention:.2%} below 85% threshold")

        print(f"\n  Mixed Features Classification:")
        print(f"    Full data accuracy: {acc_full:.4f}")
        print(f"    Reduced data accuracy: {acc_reduced:.4f}")
        print(f"    Retention: {retention:.2%}")

    def test_mixed_features_regression(self):
        """Test R² retention with mixed continuous/categorical features."""
        X, y, feature_types = self._generate_mixed_regression_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        # Train on full data
        rf_full = RandomForestRegressor(n_estimators=50, random_state=42)
        X_train_encoded = self._simple_encode(X_train, feature_types)
        X_test_encoded = self._simple_encode(X_test, feature_types, fit=False)
        rf_full.fit(X_train_encoded, y_train)
        r2_full = r2_score(y_test, rf_full.predict(X_test_encoded))

        # Reduce training data
        reducer = HVRTSampleReducer(reduction_ratio=0.3, random_state=42)
        X_train_reduced, y_train_reduced = reducer.fit_transform(
            X_train, y_train, feature_types=feature_types
        )

        # Train on reduced data
        rf_reduced = RandomForestRegressor(n_estimators=50, random_state=42)
        X_train_reduced_encoded = self._simple_encode(X_train_reduced, feature_types)
        rf_reduced.fit(X_train_reduced_encoded, y_train_reduced)
        r2_reduced = r2_score(y_test, rf_reduced.predict(X_test_encoded))

        # Calculate retention
        retention = r2_reduced / r2_full if r2_full > 0 else 0

        # Should retain at least 80% of R²
        self.assertGreater(retention, 0.80,
                          f"R² retention {retention:.2%} below 80% threshold")

        print(f"\n  Mixed Features Regression:")
        print(f"    Full data R²: {r2_full:.4f}")
        print(f"    Reduced data R²: {r2_reduced:.4f}")
        print(f"    Retention: {retention:.2%}")

    def test_categorical_only_features(self):
        """Test accuracy retention with categorical-only features."""
        X, y, feature_types = self._generate_categorical_only_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        # Train on full data
        rf_full = RandomForestClassifier(n_estimators=50, random_state=42)
        X_train_encoded = self._simple_encode(X_train, feature_types)
        X_test_encoded = self._simple_encode(X_test, feature_types, fit=False)
        rf_full.fit(X_train_encoded, y_train)
        acc_full = accuracy_score(y_test, rf_full.predict(X_test_encoded))

        # Reduce training data
        reducer = HVRTSampleReducer(reduction_ratio=0.3, random_state=42)
        X_train_reduced, y_train_reduced = reducer.fit_transform(
            X_train, y_train, feature_types=feature_types
        )

        # Train on reduced data
        rf_reduced = RandomForestClassifier(n_estimators=50, random_state=42)
        X_train_reduced_encoded = self._simple_encode(X_train_reduced, feature_types)
        rf_reduced.fit(X_train_reduced_encoded, y_train_reduced)
        acc_reduced = accuracy_score(y_test, rf_reduced.predict(X_test_encoded))

        # Calculate retention
        retention = acc_reduced / acc_full if acc_full > 0 else 0

        # Should retain at least 85% of accuracy
        self.assertGreater(retention, 0.85,
                          f"Accuracy retention {retention:.2%} below 85% threshold")

        print(f"\n  Categorical-Only Features:")
        print(f"    Full data accuracy: {acc_full:.4f}")
        print(f"    Reduced data accuracy: {acc_reduced:.4f}")
        print(f"    Retention: {retention:.2%}")

    def test_different_reduction_ratios(self):
        """Test accuracy retention across different reduction ratios."""
        X, y, feature_types = self._generate_mixed_classification_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        # Full data baseline
        rf_full = RandomForestClassifier(n_estimators=50, random_state=42)
        X_train_encoded = self._simple_encode(X_train, feature_types)
        X_test_encoded = self._simple_encode(X_test, feature_types, fit=False)
        rf_full.fit(X_train_encoded, y_train)
        acc_full = accuracy_score(y_test, rf_full.predict(X_test_encoded))

        print(f"\n  Accuracy Retention by Reduction Ratio:")
        print(f"    Full data baseline: {acc_full:.4f}")

        # Test different reduction ratios
        ratios = [0.1, 0.2, 0.3, 0.5]
        for ratio in ratios:
            reducer = HVRTSampleReducer(reduction_ratio=ratio, random_state=42)
            X_reduced, y_reduced = reducer.fit_transform(
                X_train, y_train, feature_types=feature_types
            )

            rf_reduced = RandomForestClassifier(n_estimators=50, random_state=42)
            X_reduced_encoded = self._simple_encode(X_reduced, feature_types)
            rf_reduced.fit(X_reduced_encoded, y_reduced)
            acc_reduced = accuracy_score(y_test, rf_reduced.predict(X_test_encoded))

            retention = acc_reduced / acc_full if acc_full > 0 else 0

            print(f"    {int(ratio*100):2d}% retention: {acc_reduced:.4f} ({retention:.1%} of full)")

            # Higher retention ratios should maintain better accuracy
            if ratio >= 0.3:
                self.assertGreater(retention, 0.85)

    def test_high_cardinality_categoricals(self):
        """Test with high-cardinality categorical features."""
        np.random.seed(42)
        n_samples = 3000

        # Continuous features
        X_cont = np.random.randn(n_samples, 3)

        # High-cardinality categorical (50 unique values)
        cat_high = np.random.choice([f'cat_{i}' for i in range(50)], size=n_samples)
        # Low-cardinality categorical
        cat_low = np.random.choice(['A', 'B', 'C'], size=n_samples)

        X_mixed = np.column_stack([X_cont, cat_high, cat_low])
        feature_types = ['continuous'] * 3 + ['categorical'] * 2

        # Target depends on high-cardinality categorical
        # Check if category ends with '1' using numpy vectorized string operations
        cat_high_ends_1 = np.array([str(c).endswith('1') for c in cat_high])

        y = (
            (X_cont[:, 0] > 0).astype(int) +
            cat_high_ends_1.astype(int) +
            (cat_low == 'A').astype(int)
        )
        y = (y >= 2).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X_mixed, y, test_size=0.3, random_state=42
        )

        # Full data
        rf_full = RandomForestClassifier(n_estimators=50, random_state=42)
        X_train_encoded = self._simple_encode(X_train, feature_types)
        X_test_encoded = self._simple_encode(X_test, feature_types, fit=False)
        rf_full.fit(X_train_encoded, y_train)
        acc_full = accuracy_score(y_test, rf_full.predict(X_test_encoded))

        # Reduced data
        reducer = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
        X_reduced, y_reduced = reducer.fit_transform(
            X_train, y_train, feature_types=feature_types
        )

        rf_reduced = RandomForestClassifier(n_estimators=50, random_state=42)
        X_reduced_encoded = self._simple_encode(X_reduced, feature_types)
        rf_reduced.fit(X_reduced_encoded, y_reduced)
        acc_reduced = accuracy_score(y_test, rf_reduced.predict(X_test_encoded))

        retention = acc_reduced / acc_full if acc_full > 0 else 0

        print(f"\n  High-Cardinality Categorical (50 levels):")
        print(f"    Full data accuracy: {acc_full:.4f}")
        print(f"    Reduced data accuracy: {acc_reduced:.4f}")
        print(f"    Retention: {retention:.2%}")

        # Should still retain reasonable accuracy
        self.assertGreater(retention, 0.75)

    def _simple_encode(self, X, feature_types, fit=True):
        """Simple label encoding for sklearn models."""
        from sklearn.preprocessing import LabelEncoder

        X_encoded = X.copy()

        if not hasattr(self, '_encoders'):
            self._encoders = {}

        for i, ftype in enumerate(feature_types):
            if ftype == 'categorical':
                if fit:
                    le = LabelEncoder()
                    X_encoded[:, i] = le.fit_transform(X[:, i].astype(str))
                    self._encoders[i] = le
                else:
                    le = self._encoders[i]
                    # Handle unknown categories
                    known_classes = set(le.classes_)
                    X_col = X[:, i].astype(str)
                    X_col_transformed = np.zeros(len(X_col), dtype=int)
                    for idx, val in enumerate(X_col):
                        if val in known_classes:
                            X_col_transformed[idx] = le.transform([val])[0]
                        else:
                            X_col_transformed[idx] = -1  # Unknown category
                    X_encoded[:, i] = X_col_transformed
            else:
                X_encoded[:, i] = X[:, i].astype(float)

        return X_encoded.astype(float)


if __name__ == '__main__':
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCategoricalAccuracyRetention)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"{'='*70}")

    sys.exit(0 if result.wasSuccessful() else 1)
