"""
Algions AI Labs - E-Commerce Recommendation Engine
Explainability & Decision-Support Module:
Generates human-interpretable feature signals, anchor item attributions,
confidence calibrations, and responsible decision-support advisories.
"""

from typing import Dict, Any, List, Optional


class RecommendationExplainer:
    """
    Translates raw recommendation scores into human-understandable signals
    for both shoppers and merchandising decision-makers.
    """

    def __init__(self, catalog_df, user_history_dict: Optional[Dict[str, Dict[str, float]]] = None):
        self.catalog_map = catalog_df.set_index("item_id").to_dict(orient="index")
        self.user_history = user_history_dict or {}

    def explain_prediction(
        self,
        user_id: str,
        recommended_item: Dict[str, Any],
        user_demographics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Produce a comprehensive explainability card for a specific recommended item.
        """
        item_id = recommended_item["item_id"]
        meta = self.catalog_map.get(item_id, {})
        user_items = self.user_history.get(user_id, {})

        # Find most influential anchor item from user's history
        anchor_item = None
        anchor_score = -1.0
        shared_tags = []

        rec_tags = set(t.strip() for t in meta.get("tags", "").split(",")) if meta.get("tags") else set()

        for past_item_id, weight in user_items.items():
            past_meta = self.catalog_map.get(past_item_id, {})
            past_tags = set(t.strip() for t in past_meta.get("tags", "").split(",")) if past_meta.get("tags") else set()
            overlap = rec_tags.intersection(past_tags)
            overlap_score = weight * (len(overlap) + (2.0 if past_meta.get("category") == meta.get("category") else 0.5))

            if overlap_score > anchor_score:
                anchor_score = overlap_score
                anchor_item = past_meta
                shared_tags = list(overlap)

        # Confidence level calculation
        raw_score = recommended_item.get("score", 0.5)
        num_user_interactions = len(user_items)

        if num_user_interactions >= 6 and raw_score > 0.4:
            confidence = "High"
            uncertainty_reason = "Well-established user behavior with multiple supporting interactions."
        elif num_user_interactions >= 2:
            confidence = "Medium"
            uncertainty_reason = "Moderate browsing signal; subject to dynamic preference drift."
        else:
            confidence = "Low (Cold-Start)"
            uncertainty_reason = "Limited or no prior interactions; recommendation relies on demographic/popularity priors."

        explanation_card = {
            "item_id": item_id,
            "title": meta.get("title", recommended_item.get("title", "Product")),
            "category": meta.get("category"),
            "sub_category": meta.get("sub_category"),
            "brand": meta.get("brand"),
            "price": meta.get("price"),
            "confidence_level": confidence,
            "uncertainty_assessment": uncertainty_reason,
            "primary_rationale": recommended_item.get("explanation", "Suggested based on overall catalog affinity."),
            "anchor_product": anchor_item.get("title") if anchor_item else "General Browsing Cohort",
            "matching_features": shared_tags if shared_tags else [meta.get("category", "General")],
            "decision_support_advisory": (
                "Algorithmic decision-support advisory: Review current inventory stock, "
                "active promotional campaigns, and return rates before manual campaign deployment."
            )
        }
        return explanation_card

