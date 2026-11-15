"""
Algions AI Labs - E-Commerce Recommendation Engine
Model: Popularity Baseline Recommender with Time-Decay and Category Conditioning.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Set


class PopularityRecommender:
    """
    Non-personalized and category-conditioned baseline recommender.
    Ranks items based on aggregated interaction volume and recency-weighted scores.
    """

    def __init__(self, decay_half_life_days: float = 30.0):
        self.decay_half_life_days = decay_half_life_days
        self.global_ranked_items: List[Dict[str, Any]] = []
        self.category_ranked_items: Dict[str, List[Dict[str, Any]]] = {}
        self.catalog_df: Optional[pd.DataFrame] = None
        self.item_metadata: Dict[str, Dict[str, Any]] = {}
        self.user_history: Dict[str, Set[str]] = {}

    def fit(self, train_df: pd.DataFrame, catalog_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None):
        """Fit the popularity baseline using historical training interactions."""
        self.catalog_df = catalog_df.copy()
        self.item_metadata = catalog_df.set_index("item_id").to_dict(orient="index")

        # Record user history to avoid recommending already purchased items if desired
        self.user_history = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

        # Recency-weighted score computation
        max_time = train_df["timestamp"].max()
        time_diffs_days = (max_time - train_df["timestamp"]).dt.total_seconds() / (24 * 3600.0)
        
        # Exponential time decay: weight * (0.5)^(days / half_life)
        decay_factors = np.power(0.5, time_diffs_days / self.decay_half_life_days)
        weighted_interactions = train_df["event_weight"] * decay_factors

        df_scored = train_df.copy()
        df_scored["decayed_score"] = weighted_interactions

        # Aggregate total score per item
        item_scores = df_scored.groupby("item_id")["decayed_score"].sum().to_dict()

        # Merge with catalog (items with 0 interactions get score 0)
        scored_catalog = []
        for item_id, meta in self.item_metadata.items():
            score = float(item_scores.get(item_id, 0.0))
            scored_catalog.append({
                "item_id": item_id,
                "title": meta["title"],
                "category": meta["category"],
                "sub_category": meta["sub_category"],
                "brand": meta["brand"],
                "price": meta["price"],
                "score": score
            })

        # Sort globally
        self.global_ranked_items = sorted(scored_catalog, key=lambda x: x["score"], reverse=True)

        # Sort within each category
        self.category_ranked_items = {}
        for category in catalog_df["category"].unique():
            cat_items = [it for it in self.global_ranked_items if it["category"] == category]
            self.category_ranked_items[category] = cat_items

        return self

    def recommend(
        self,
        user_id: Optional[str] = None,
        top_k: int = 10,
        category: Optional[str] = None,
        filter_history: bool = True
    ) -> List[Dict[str, Any]]:
        """Return Top-K popular items, optionally conditioned on category or filtered by user history."""
        seen_items = self.user_history.get(user_id, set()) if (filter_history and user_id) else set()

        candidate_pool = self.category_ranked_items.get(category, self.global_ranked_items) if category else self.global_ranked_items

        recommendations = []
        for rank, item in enumerate(candidate_pool):
            if item["item_id"] in seen_items:
                continue

            rec = dict(item)
            rec["rank"] = len(recommendations) + 1
            rec["explanation"] = (
                f"Popular in {item['category']}: trending with high overall customer engagement."
                if category else
                "Store Bestseller: high customer engagement and volume across catalog."
            )
            rec["model"] = "PopularityBaseline"
            recommendations.append(rec)
            if len(recommendations) >= top_k:
                break

        return recommendations

