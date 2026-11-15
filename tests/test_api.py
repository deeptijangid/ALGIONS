"""
Algions AI Labs - Recommendation Engine Tests
Test Suite: FastAPI Inference Endpoints, Cold-Start Routing, and Contract Validation
"""

import unittest
from api.main import (
    load_artifacts,
    health_check,
    get_recommendations,
    get_similar_products,
    get_user_profile,
    get_catalog_sample,
    get_metrics_benchmark
)
from api.schemas import RecommendationRequest, SimilarItemsRequest


class TestRecommendationAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Trigger artifact loading
        load_artifacts()

    def test_health_check(self):
        """Test health check returns system status and catalog count."""
        response = health_check()
        self.assertEqual(response.status, "HEALTHY")
        self.assertIn("hybrid", response.models_loaded)
        self.assertTrue(response.models_loaded["hybrid"])
        self.assertGreater(response.catalog_item_count, 0)

    def test_recommend_endpoint_active_user(self):
        """Test real prediction for an active user input."""
        req = RecommendationRequest(
            user_id="USER_0012",
            top_k=5,
            model_type="hybrid",
            filter_history=True
        )
        response = get_recommendations(req)
        self.assertEqual(response.user_id, "USER_0012")
        self.assertEqual(len(response.recommendations), 5)
        self.assertGreater(response.latency_ms, 0.0)
        self.assertTrue(response.decision_support_advisory)

        top_rec = response.recommendations[0]
        self.assertEqual(top_rec.rank, 1)
        self.assertTrue(top_rec.explanation)
        self.assertIn(top_rec.confidence_level, ["High", "Medium", "Low (Cold-Start)"])

    def test_recommend_endpoint_cold_start_user(self):
        """Test recommendation for brand new user (cold start routing)."""
        req = RecommendationRequest(
            user_id="COLD_START_USER_9999",
            top_k=5,
            model_type="hybrid",
            filter_history=False
        )
        response = get_recommendations(req)
        self.assertIn("Cold-Start", response.user_segment)
        self.assertEqual(len(response.recommendations), 5)

    def test_recommend_endpoint_category_filtering(self):
        """Test recommendation strictly obeys category filter."""
        req = RecommendationRequest(
            user_id="USER_0012",
            top_k=5,
            category="Electronics",
            model_type="hybrid"
        )
        response = get_recommendations(req)
        self.assertEqual(len(response.recommendations), 5)
        for item in response.recommendations:
            self.assertEqual(item.category, "Electronics")

    def test_similar_products_endpoint(self):
        """Test content-based item-to-item similarity endpoint."""
        req = SimilarItemsRequest(item_id="PROD_0001", top_k=3)
        response = get_similar_products(req)
        self.assertEqual(response.anchor_item_id, "PROD_0001")
        self.assertEqual(len(response.similar_items), 3)
        self.assertGreaterEqual(response.similar_items[0].similarity_score, 0.0)

    def test_catalog_endpoint(self):
        """Test catalog browsing with category filter."""
        response = get_catalog_sample(category="Electronics", limit=5)
        self.assertLessEqual(len(response["items"]), 5)
        for item in response["items"]:
            self.assertEqual(item["category"], "Electronics")

    def test_user_profile_endpoint(self):
        """Test retrieving user demographic profile and history."""
        profile = get_user_profile("USER_0012")
        self.assertEqual(profile.user_id, "USER_0012")
        self.assertIn("primary_category", profile.demographics)

    def test_metrics_endpoint(self):
        """Test offline benchmark evaluation results retrieval."""
        metrics = get_metrics_benchmark()
        self.assertIn("Adaptive Hybrid", metrics)
        self.assertIn("NDCG@10", metrics["Adaptive Hybrid"])


if __name__ == "__main__":
    unittest.main()
