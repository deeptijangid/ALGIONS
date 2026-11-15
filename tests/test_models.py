"""
Algions AI Labs - Recommendation Engine Tests
Test Suite: Model Training, Personalization, Cold-Start Routing, and Ranking
"""

import unittest
import pandas as pd
from src.data_generator import generate_catalog, generate_users, generate_interactions
from src.models import (
    PopularityRecommender,
    ContentBasedRecommender,
    CollaborativeSVDRecommender,
    HybridRecommender
)


class TestRecommendationModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.catalog_df = generate_catalog(num_items=40, random_state=42)
        cls.users_df = generate_users(num_users=80, random_state=42)
        cls.interactions_df = generate_interactions(
            cls.users_df, cls.catalog_df, duration_days=15, random_state=42
        )
        # Train on first 80% interactions
        cutoff = int(len(cls.interactions_df) * 0.8)
        cls.train_df = cls.interactions_df.iloc[:cutoff].copy()

    def test_popularity_recommender(self):
        """Test popularity baseline ranks items properly and handles category conditioning."""
        model = PopularityRecommender()
        model.fit(self.train_df, self.catalog_df, self.users_df)

        recs = model.recommend(user_id="USER_0001", top_k=5)
        self.assertEqual(len(recs), 5)
        self.assertEqual(recs[0]["rank"], 1)
        self.assertIn("score", recs[0])

        # Test category conditioning
        cat_recs = model.recommend(user_id="USER_0001", top_k=5, category="Electronics")
        for r in cat_recs:
            self.assertEqual(r["category"], "Electronics")

    def test_content_based_recommender(self):
        """Test content-based model personalization and cold-start fallback."""
        model = ContentBasedRecommender()
        model.fit(self.train_df, self.catalog_df, self.users_df)

        active_user = self.train_df["user_id"].iloc[0]
        recs = model.recommend(user_id=active_user, top_k=5)
        self.assertEqual(len(recs), 5)
        self.assertIn("score", recs[0])
        self.assertTrue(recs[0]["explanation"])

        # Cold-start user test
        cold_recs = model.recommend(user_id="USER_UNSEEN_9999", top_k=5)
        self.assertEqual(len(cold_recs), 5)
        self.assertTrue(any("ColdStart" in r["model"] for r in cold_recs))

    def test_collaborative_svd_recommender(self):
        """Test collaborative filtering with matrix decomposition."""
        model = CollaborativeSVDRecommender(n_components=10, random_state=42)
        model.fit(self.train_df, self.catalog_df, self.users_df)

        active_user = self.train_df["user_id"].iloc[0]
        recs = model.recommend(user_id=active_user, top_k=5)
        self.assertEqual(len(recs), 5)
        self.assertIn("score", recs[0])

        # Cold-start user fallback
        cold_recs = model.recommend(user_id="USER_UNSEEN_9999", top_k=5)
        self.assertEqual(len(cold_recs), 5)

    def test_hybrid_recommender_and_dynamic_routing(self):
        """Test adaptive hybrid recommender blends models and handles cold-start gracefully."""
        model = HybridRecommender(cf_weight=0.65, svd_components=10, random_state=42)
        model.fit(self.train_df, self.catalog_df, self.users_df)

        # 1. Warm active user
        active_user = self.train_df["user_id"].iloc[0]
        recs = model.recommend(user_id=active_user, top_k=8)
        self.assertEqual(len(recs), 8)
        self.assertEqual(recs[0]["rank"], 1)
        self.assertIn("cf_score", recs[0])
        self.assertIn("content_score", recs[0])

        # 2. Cold-start user routing
        cold_recs = model.recommend(user_id="USER_NEW_9999", top_k=5)
        self.assertEqual(len(cold_recs), 5)
        self.assertEqual(cold_recs[0]["alpha_cf"], 0.0)
        self.assertIn("ColdStart", cold_recs[0]["model"])


if __name__ == "__main__":
    unittest.main()

