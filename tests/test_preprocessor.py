"""
Algions AI Labs - Recommendation Engine Tests
Test Suite: Preprocessing, Leakage Prevention, and Implicit Aggregation
"""

import unittest
import pandas as pd
import numpy as np
from src.data_generator import generate_catalog, generate_users, generate_interactions
from src.preprocessor import RecommenderDataPreprocessor


class TestPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.catalog_df = generate_catalog(num_items=60, random_state=42)
        cls.users_df = generate_users(num_users=120, random_state=42)
        cls.interactions_df = generate_interactions(
            cls.users_df, cls.catalog_df, duration_days=20, random_state=42
        )
        cls.preprocessor = RecommenderDataPreprocessor(time_split_quantile=0.80)

    def test_chronological_split_prevents_future_leakage(self):
        """CRITICAL: Assert that max(train_timestamp) < min(test_timestamp)."""
        train_df, test_df = self.preprocessor.chronological_split(self.interactions_df)

        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)

        max_train_time = train_df["timestamp"].max()
        min_test_time = test_df["timestamp"].min()

        # Strict chronological separation
        self.assertLess(max_train_time, min_test_time, "Train and test overlap! Temporal leakage detected.")
        self.assertEqual(len(train_df) + len(test_df), len(self.interactions_df))

    def test_implicit_feedback_aggregation(self):
        """Check aggregation of event weights with log-damping."""
        sample_interactions = pd.DataFrame([
            {"user_id": "U1", "item_id": "I1", "event_weight": 1.0, "timestamp": pd.Timestamp("2026-08-01 10:00:00")},
            {"user_id": "U1", "item_id": "I1", "event_weight": 2.5, "timestamp": pd.Timestamp("2026-08-01 10:05:00")},
            {"user_id": "U1", "item_id": "I1", "event_weight": 5.0, "timestamp": pd.Timestamp("2026-08-01 10:10:00")},
            {"user_id": "U2", "item_id": "I2", "event_weight": 1.0, "timestamp": pd.Timestamp("2026-08-01 11:00:00")}
        ])
        agg = self.preprocessor.aggregate_implicit_feedback(sample_interactions)
        self.assertEqual(len(agg), 2)
        
        u1_i1 = agg[(agg["user_id"] == "U1") & (agg["item_id"] == "I1")].iloc[0]
        self.assertEqual(u1_i1["total_weight"], 8.5)
        self.assertEqual(u1_i1["interaction_count"], 3)
        self.assertAlmostEqual(u1_i1["rating"], np.log1p(8.5), places=4)

    def test_sparsity_calculation(self):
        """Test matrix sparsity calculation."""
        sparsity = self.preprocessor.compute_sparsity(num_users=100, num_items=50, num_interactions=250)
        expected_sparsity = 1.0 - (250 / 5000.0)  # 0.95
        self.assertAlmostEqual(sparsity, expected_sparsity, places=4)


if __name__ == "__main__":
    unittest.main()

