"""
Algions AI Labs - Recommendation Engine Tests
Test Suite: Data Generation and Schema Validation
"""

import unittest
import pandas as pd
from src.data_generator import generate_catalog, generate_users, generate_interactions


class TestDataGenerator(unittest.TestCase):

    def setUp(self):
        self.catalog_df = generate_catalog(num_items=50, random_state=42)
        self.users_df = generate_users(num_users=100, random_state=42)
        self.interactions_df = generate_interactions(
            self.users_df, self.catalog_df, duration_days=10, random_state=42
        )

    def test_catalog_schema_and_integrity(self):
        """Verify catalog contains all required columns and valid values."""
        required_cols = {"item_id", "title", "category", "sub_category", "brand", "price", "rating", "review_count", "tags", "description"}
        self.assertTrue(required_cols.issubset(self.catalog_df.columns))
        self.assertEqual(len(self.catalog_df), 50)
        self.assertEqual(self.catalog_df["item_id"].nunique(), 50)
        self.assertFalse(self.catalog_df["price"].isnull().any())
        self.assertTrue((self.catalog_df["price"] > 0).all())

    def test_users_schema_and_diversity(self):
        """Verify user cohort is properly structured without nulls."""
        required_cols = {"user_id", "age_group", "gender", "primary_category", "secondary_category", "price_tier", "activity_tier"}
        self.assertTrue(required_cols.issubset(self.users_df.columns))
        self.assertEqual(len(self.users_df), 100)
        self.assertEqual(self.users_df["user_id"].nunique(), 100)

    def test_interactions_integrity_and_funnel(self):
        """Verify interaction records, temporal monotonic ordering, and event funnel."""
        self.assertGreater(len(self.interactions_df), 100)
        event_types = set(self.interactions_df["event_type"].unique())
        self.assertTrue({"view", "cart", "purchase"}.issubset(event_types))
        
        # Verify timestamps are sorted chronologically
        self.assertTrue(self.interactions_df["timestamp"].is_monotonic_increasing)
        
        # Foreign key integrity
        self.assertTrue(set(self.interactions_df["item_id"]).issubset(set(self.catalog_df["item_id"])))
        self.assertTrue(set(self.interactions_df["user_id"]).issubset(set(self.users_df["user_id"])))


if __name__ == "__main__":
    unittest.main()

