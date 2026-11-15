"""
Algions AI Labs - E-Commerce Recommendation Engine
Model: Adaptive Hybrid Recommender with Dynamic Cold-Start Routing,
Collaborative-Content Ensembling, and Transparent Decision-Support Explanations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Set
from .baseline_popularity import PopularityRecommender
from .content_based import ContentBasedRecommender
from .collaborative_svd import CollaborativeSVDRecommender


class HybridRecommender:
    """
    Adaptive Hybrid Recommender that dynamically blends Collaborative Filtering
    and Content-Based predictions while providing seamless fallback for cold-start
    users and new catalog items.
    """

    def __init__(
        self,
        cf_weight: float = 0.65,
        svd_components: int = 30,
        decay_half_life_days: float = 30.0,
        random_state: int = 42
    ):
        self.default_cf_weight = cf_weight
        self.random_state = random_state

        self.popularity_model = PopularityRecommender(decay_half_life_days=decay_half_life_days)
        self.content_model = ContentBasedRecommender()
        self.cf_model = CollaborativeSVDRecommender(n_components=svd_components, random_state=random_state)

        self.catalog_df: Optional[pd.DataFrame] = None
        self.item_metadata: Dict[str, Dict[str, Any]] = {}
        self.user_history: Dict[str, Set[str]] = {}
        self.user_interaction_counts: Dict[str, int] = {}
        self.user_demographics: Dict[str, Dict[str, Any]] = {}

    def fit(self, train_df: pd.DataFrame, catalog_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None):
        """Train all underlying sub-models on historical data."""
        self.catalog_df = catalog_df.copy()
        self.item_metadata = catalog_df.set_index("item_id").to_dict(orient="index")

        # Track user interaction volume
        self.user_history = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
        self.user_interaction_counts = train_df.groupby("user_id")["item_id"].count().to_dict()

        if users_df is not None:
            self.user_demographics = users_df.set_index("user_id").to_dict(orient="index")

        print("Fitting Popularity Baseline...")
        self.popularity_model.fit(train_df, catalog_df, users_df)

        print("Fitting Content-Based Model...")
        self.content_model.fit(train_df, catalog_df, users_df)

        print("Fitting Collaborative SVD Model...")
        self.cf_model.fit(train_df, catalog_df, users_df)

        return self

    def _normalize_scores(self, score_dict: Dict[str, float]) -> Dict[str, float]:
        """Min-Max normalize scores to [0, 1] range for harmonious blending."""
        if not score_dict:
            return {}
        vals = list(score_dict.values())
        min_val = min(vals)
        max_val = max(vals)
        if max_val == min_val:
            return {k: 0.5 for k in score_dict}
        return {k: (v - min_val) / (max_val - min_val) for k, v in score_dict.items()}

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        category: Optional[str] = None,
        filter_history: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized top-K recommendations using adaptive hybrid routing.
        """
        seen_items = self.user_history.get(user_id, set()) if filter_history else set()
        user_num_interactions = self.user_interaction_counts.get(user_id, 0)
        demographics = self.user_demographics.get(user_id, {})
        primary_cat = demographics.get("primary_category", None)

        # ----------------------------------------------------
        # Scenario 1: Pure Cold-Start User (0 interactions)
        # ----------------------------------------------------
        if user_num_interactions == 0:
            effective_cat = category or primary_cat
            cold_recs = self.popularity_model.recommend(
                user_id=user_id,
                top_k=top_k,
                category=effective_cat,
                filter_history=False
            )
            for r in cold_recs:
                r["model"] = "Hybrid (ColdStart Routing)"
                r["alpha_cf"] = 0.0
                r["explanation"] = (
                    f"Trending in {r['category']}: Featured bestseller recommended for new shoppers."
                    if effective_cat else
                    "Store-wide bestseller recommended to introduce diverse popular catalog choices."
                )
            return cold_recs

        # ----------------------------------------------------
        # Scenario 2: Active User (Dynamic Alpha Blending)
        # ----------------------------------------------------
        # Adaptive weight: users with few interactions rely more on content similarity
        if user_num_interactions >= 8:
            alpha = self.default_cf_weight  # e.g. 0.65 CF, 0.35 Content
        elif user_num_interactions >= 3:
            alpha = 0.45
        else:
            alpha = 0.20  # sparse user: 0.20 CF, 0.80 Content

        # Query sub-models for candidate pool (top 100 each)
        cf_candidates = self.cf_model.recommend(
            user_id=user_id,
            top_k=100,
            category=category,
            filter_history=filter_history
        )
        content_candidates = self.content_model.recommend(
            user_id=user_id,
            top_k=100,
            category=category,
            filter_history=filter_history
        )

        cf_scores = {r["item_id"]: float(r["score"]) for r in cf_candidates}
        content_scores = {r["item_id"]: float(r["score"]) for r in content_candidates}

        # Normalize both score sets to [0, 1]
        norm_cf = self._normalize_scores(cf_scores)
        norm_content = self._normalize_scores(content_scores)

        # Merge unique items across both sets
        all_candidate_ids = set(norm_cf.keys()) | set(norm_content.keys())

        blended_items = []
        for item_id in all_candidate_ids:
            if item_id in seen_items:
                continue

            meta = self.item_metadata.get(item_id)
            if not meta:
                continue

            if category and meta["category"] != category:
                continue

            s_cf = norm_cf.get(item_id, 0.0)
            s_content = norm_content.get(item_id, 0.0)

            final_score = (alpha * s_cf) + ((1.0 - alpha) * s_content)

            # Determine dominant explanation signal
            if s_cf > s_content and s_cf > 0.4:
                reason = "Shoppers with similar purchase trajectories frequently bought this."
                driver = "Collaborative"
            else:
                reason = f"Attributes closely match your browsing interest in {meta['sub_category']} ({meta['brand']})."
                driver = "Content"

            blended_items.append({
                "item_id": item_id,
                "title": meta["title"],
                "category": meta["category"],
                "sub_category": meta["sub_category"],
                "brand": meta["brand"],
                "price": meta["price"],
                "score": round(final_score, 4),
                "cf_score": round(s_cf, 4),
                "content_score": round(s_content, 4),
                "alpha_cf": round(alpha, 2),
                "primary_driver": driver,
                "explanation": reason,
                "model": "Hybrid (Collaborative + Content)"
            })

        # Sort descending by blended score
        blended_items = sorted(blended_items, key=lambda x: x["score"], reverse=True)

        # Assign final rank
        for idx, item in enumerate(blended_items[:top_k]):
            item["rank"] = idx + 1

        return blended_items[:top_k]

