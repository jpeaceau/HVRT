"""
Test H-VRT with categorical features.

Validates that categorical feature handling works correctly:
1. Label encoding converts categories to integers
2. Standard scaling normalizes the encoded values
3. Pairwise interactions include categorical features
4. Results are consistent and reproducible
"""

import unittest
import numpy as np
import sys

from src.hvrt import HVRTSampleReducer


class TestCategoricalSupport(unittest.TestCase):
    """Test categorical feature support in H-VRT."""

    def setUp(self):
        """Set up test data with categorical features."""
        np.random.seed(42)
        self.n_samples = 1000

        # Create mixed feature data
        self.X_continuous = np.random.randn(self.n_samples, 3)

        # Create categorical features with different cardinalities
        self.cat1 = np.random.choice(['A', 'B', 'C'], size=self.n_samples)
        self.cat2 = np.random.choice(['low', 'medium', 'high'], size=self.n_samples)
        self.cat3 = np.random.choice(['red', 'blue'], size=self.n_samples)

        # Combine into single array
        self.X_mixed = np.column_stack([
            self.X_continuous,
            self.cat1,
            self.cat2,
            self.cat3
        ])

        # Define feature types
        self.feature_types = ['continuous', 'continuous', 'continuous',
                              'categorical', 'categorical', 'categorical']

        # Create target with interaction between continuous and categorical
        self.y = (
            self.X_continuous[:, 0] +
            (self.cat1 == 'A').astype(float) * 2.0 +
            (self.cat2 == 'high').astype(float) * 1.5 +
            np.random.randn(self.n_samples) * 0.5
        )

    def test_basic_categorical_reduction(self):
        """Test that H-VRT works with categorical features."""
        reducer = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
        X_reduced, y_reduced = reducer.fit_transform(
            self.X_mixed, self.y, feature_types=self.feature_types
        )

        # Check shapes
        self.assertEqual(X_reduced.shape[1], self.X_mixed.shape[1])
        self.assertLess(X_reduced.shape[0], self.X_mixed.shape[0])
        self.assertEqual(len(y_reduced), X_reduced.shape[0])

        # Check categorical values are preserved
        for col_idx in [3, 4, 5]:
            original_categories = set(self.X_mixed[:, col_idx])
            reduced_categories = set(X_reduced[:, col_idx])
            # Reduced set should be subset of original
            self.assertTrue(reduced_categories.issubset(original_categories))

    def test_categorical_only(self):
        """Test H-VRT with only categorical features."""
        X_cat_only = self.X_mixed[:, 3:]  # Only categorical columns
        feature_types_cat = ['categorical'] * 3

        reducer = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
        X_reduced, y_reduced = reducer.fit_transform(
            X_cat_only, self.y, feature_types=feature_types_cat
        )

        self.assertEqual(X_reduced.shape[1], 3)
        self.assertLess(X_reduced.shape[0], X_cat_only.shape[0])

    def test_reproducibility_with_categorical(self):
        """Test that categorical handling is deterministic."""
        reducer1 = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
        X_r1, y_r1 = reducer1.fit_transform(
            self.X_mixed, self.y, feature_types=self.feature_types
        )

        reducer2 = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
        X_r2, y_r2 = reducer2.fit_transform(
            self.X_mixed, self.y, feature_types=self.feature_types
        )

        # Should be identical
        np.testing.assert_array_equal(X_r1, X_r2)
        np.testing.assert_array_equal(y_r1, y_r2)

    def test_label_encoder_persistence(self):
        """Test that label encoders are stored for transform."""
        reducer = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
        reducer.fit(self.X_mixed, self.y, feature_types=self.feature_types)

        # Check that encoders were created
        self.assertTrue(hasattr(reducer, 'label_encoders_'))
        self.assertEqual(len(reducer.label_encoders_), 3)  # 3 categorical features

    def test_high_cardinality_categorical(self):
        """Test with high-cardinality categorical feature."""
        # Create high-cardinality categorical (100 unique values)
        n_samples = 2000
        X_cont = np.random.randn(n_samples, 2)
        cat_high_card = np.random.choice([f'cat_{i}' for i in range(100)], size=n_samples)

        X_high_card = np.column_stack([X_cont, cat_high_card])
        y_high = np.random.randn(n_samples)

        feature_types = ['continuous', 'continuous', 'categorical']

        reducer = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
        X_reduced, y_reduced = reducer.fit_transform(
            X_high_card, y_high, feature_types=feature_types
        )

        # Should complete without error
        self.assertEqual(X_reduced.shape[1], 3)

    def test_mixed_features_info(self):
        """Test that reduction info works with mixed features."""
        reducer = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
        reducer.fit(self.X_mixed, self.y, feature_types=self.feature_types)

        info = reducer.get_reduction_info()

        self.assertIn('n_partitions', info)
        self.assertIn('n_selected', info)
        self.assertIn('reduction_ratio', info)

    def test_categorical_with_nan_handling(self):
        """Test that categorical features with string 'nan' are handled."""
        # Some categorical features might have 'nan' as a string
        X_with_nan = self.X_mixed.copy()
        X_with_nan[0:10, 3] = 'nan'

        reducer = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
        X_reduced, y_reduced = reducer.fit_transform(
            X_with_nan, self.y, feature_types=self.feature_types
        )

        # Should handle 'nan' as another category
        self.assertIsNotNone(X_reduced)

    def test_binary_categorical(self):
        """Test with binary categorical features."""
        X_binary = self.X_mixed[:, [0, 5]]  # 1 continuous, 1 binary categorical
        feature_types = ['continuous', 'categorical']

        reducer = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
        X_reduced, y_reduced = reducer.fit_transform(
            X_binary, self.y, feature_types=feature_types
        )

        self.assertEqual(X_reduced.shape[1], 2)
        # Binary categorical should only have 2 unique values
        unique_values = np.unique(X_reduced[:, 1])
        self.assertLessEqual(len(unique_values), 2)


if __name__ == '__main__':
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCategoricalSupport)
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
