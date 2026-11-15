"""
Algions AI Labs - E-Commerce Recommendation Engine
Model: Content-Based Recommender with TF-IDF Vectorization and User Profile Aggregation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Content-Based Recommender utilizing TF-IDF representations of product metadata
    (titles, categories, brands, tags, and descriptions) and user historical affinity vectors.
    """

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
        self.catalog_df: Optional[pd.DataFrame] = None
        self.item_metadata: Dict[str, Dict[str, Any]] = {}
        self.item_tfidf_matrix = None
        self.item2idx: Dict[str, int] = {}
        self.idx2item: Dict[int, str] = {}
        self.user_profiles: Dict[str, np.ndarray] = {}
        self.user_history: Dict[str, Dict[str, float]] = {}
        self.user_preferred_cat: Dict[str, str] = {}

    def _build_text_corpus(self, catalog_df: pd.DataFrame) -> List[str]:
        """Combine item metadata fields into a structured text document."""
        corpus = []
        for _, row in catalog_df.iterrows():
            text = (
                f"{row['title']} {row['category']} {row['sub_category']} "
                f"{row['brand']} {row['tags']} {row['description']}"
            )
            corpus.append(text)
        return corpus

    def fit(self, train_df: pd.DataFrame, catalog_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None):
        """Fit TF-IDF matrix and construct user historical preference vectors."""
        self.catalog_df = catalog_df.copy().reset_index(drop=True)
        self.item_metadata = self.catalog_df.set_index("item_id").to_dict(orient="index")

        for idx, row in self.catalog_df.iterrows():
            self.item2idx[row["item_id"]] = idx
            self.idx2item[idx] = row["item_id"]

        # 1. Fit TF-IDF on product text representations
        corpus = self._build_text_corpus(self.catalog_df)
        self.item_tfidf_matrix = self.vectorizer.fit_transform(corpus)  # Shape: (num_items, num_features)

        # 2. Cache user history with implicit weights
        for _, row in train_df.iterrows():
            u = row["user_id"]
            i = row["item_id"]
            w = float(row["event_weight"])
            if u not in self.user_history:
                self.user_history[u] = {}
            self.user_history[u][i] = self.user_history[u].get(i, 0.0) + w

        # 3. Cache user demographic preferred categories if available
        if users_df is not None:
            self.user_preferred_cat = users_df.set_index("user_id")["primary_category"].to_dict()

        # 4. Construct user profile vectors
        num_features = self.item_tfidf_matrix.shape[1]
        for user_id, items_dict in self.user_history.items():
            user_vec = np.zeros(num_features, dtype=np.float32)
            total_weight = 0.0

            for item_id, weight in items_dict.items():
                if item_id in self.item2idx:
                    item_idx = self.item2idx[item_id]
                    # Dense vector of item
                    item_vec = self.item_tfidf_matrix[item_idx].toarray().flatten()
                    user_vec += weight * item_vec
                    total_weight += weight

            if total_weight > 0:
                user_vec /= total_weight
                # L2 normalize user profile vector
                norm = np.linalg.norm(user_vec)
                if norm > 0:
                    user_vec /= norm
                self.user_profiles[user_id] = user_vec

        return self

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        category: Optional[str] = None,
        filter_history: bool = True
    ) -> List[Dict[str, Any]]:
        """Recommend items with highest cosine similarity to user's metadata preference vector."""
        seen_items = set(self.user_history.get(user_id, {}).keys()) if filter_history else set()

        # Handle cold-start users (no prior history)
        if user_id not in self.user_profiles:
            return self._cold_start_recommend(user_id=user_id, top_k=top_k, category=category)

        user_vec = self.user_profiles[user_id].reshape(1, -1)
        # Compute cosine similarity against all catalog items
        sim_scores = cosine_similarity(user_vec, self.item_tfidf_matrix).flatten()

        # Rank items
        ranked_indices = np.argsort(-sim_scores)

        # Identify user's most influential historical item for explanation
        user_top_history_item_id = max(self.user_history[user_id].items(), key=lambda x: x[1])[0] if self.user_history.get(user_id) else None
        anchor_title = self.item_metadata.get(user_top_history_item_id, {}).get("title", "previous purchases") if user_top_history_item_id else "your profile"

        recommendations = []
        for idx in ranked_indices:
            item_id = self.idx2item[idx]
            if item_id in seen_items:
                continue

            meta = self.item_metadata[item_id]
            if category and meta["category"] != category:
                continue

            score = float(sim_scores[idx])
            rec = {
                "item_id": item_id,
                "title": meta["title"],
                "category": meta["category"],
                "sub_category": meta["sub_category"],
                "brand": meta["brand"],
                "price": meta["price"],
                "score": round(score, 4),
                "rank": len(recommendations) + 1,
                "model": "ContentBased",
                "explanation": f"Similar product attributes and category tags to {anchor_title[:45]}..."
            }
            recommendations.append(rec)
            if len(recommendations) >= top_k:
                break

        return recommendations

    def _cold_start_recommend(self, user_id: str, top_k: int, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fallback for cold-start users based on preferred demographic category or top ratings."""
        target_cat = category or self.user_preferred_cat.get(user_id, None)

        pool = self.catalog_df
        if target_cat:
            cat_pool = pool[pool["category"] == target_cat]
            if not cat_pool.empty:
                pool = cat_pool

        # Rank by rating * log(review_count)
        pool = pool.copy()
        pool["quality_score"] = pool["rating"] * np.log1p(pool["review_count"])
        top_pool = pool.sort_values(by="quality_score", ascending=False).head(top_k)

        recommendations = []
        for idx, row in top_pool.reset_index(drop=True).iterrows():
            recommendations.append({
                "item_id": row["item_id"],
                "title": row["title"],
                "category": row["category"],
                "sub_category": row["sub_category"],
                "brand": row["brand"],
                "price": row["price"],
                "score": round(float(row["quality_score"]), 4),
                "rank": idx + 1,
                "model": "ContentBased (ColdStart Fallback)",
                "explanation": f"Top-rated selection in {row['category']} for new users."
            })
        return recommendations

    def recommend_similar_items(self, item_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Item-to-item recommendation based on semantic similarity of metadata."""
        if item_id not in self.item2idx:
            return []

        item_idx = self.item2idx[item_id]
        item_vec = self.item_tfidf_matrix[item_idx]
        sim_scores = cosine_similarity(item_vec, self.item_tfidf_matrix).flatten()

        ranked_indices = np.argsort(-sim_scores)
        similar_items = []
        for idx in ranked_indices:
            cand_id = self.idx2item[idx]
            if cand_id == item_id:
                continue
            meta = self.item_metadata[cand_id]
            similar_items.append({
                "item_id": cand_id,
                "title": meta["title"],
                "category": meta["category"],
                "sub_category": meta["sub_category"],
                "brand": meta["brand"],
                "price": meta["price"],
                "similarity_score": round(float(sim_scores[idx]), 4),
                "rank": len(similar_items) + 1
            })
            if len(similar_items) >= top_k:
                break
        return similar_items

