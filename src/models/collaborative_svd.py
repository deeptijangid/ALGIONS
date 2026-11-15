"""
Algions AI Labs - E-Commerce Recommendation Engine
Model: Collaborative Filtering using Matrix Factorization via Truncated SVD.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Set
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class CollaborativeSVDRecommender:
    """
    Collaborative Filtering recommender utilizing Truncated Singular Value Decomposition (SVD)
    on the sparse implicit interaction feedback matrix.
    """

    def __init__(self, n_components: int = 30, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        
        self.user2idx: Dict[str, int] = {}
        self.idx2user: Dict[int, str] = {}
        self.item2idx: Dict[str, int] = {}
        self.idx2item: Dict[int, str] = {}
        
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.item_metadata: Dict[str, Dict[str, Any]] = {}
        self.user_history: Dict[str, Set[str]] = {}
        self.global_item_scores: Dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame, catalog_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None):
        """Build sparse interaction matrix and decompose into user/item latent factor spaces."""
        self.item_metadata = catalog_df.set_index("item_id").to_dict(orient="index")

        # Map all catalog items to fixed indices
        unique_items = sorted(catalog_df["item_id"].unique())
        self.item2idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx2item = {idx: item for idx, item in enumerate(unique_items)}

        # Map users observed in train
        unique_users = sorted(train_df["user_id"].unique())
        self.user2idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx2user = {idx: user for idx, user in enumerate(unique_users)}

        # Aggregate user-item weights
        grouped = train_df.groupby(["user_id", "item_id"])["event_weight"].sum().reset_index()
        grouped["rating"] = np.log1p(grouped["event_weight"])

        # Cache user history
        self.user_history = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

        # Compute global popularity backup
        self.global_item_scores = grouped.groupby("item_id")["rating"].sum().to_dict()

        num_users = len(unique_users)
        num_items = len(unique_items)

        row_indices = [self.user2idx[u] for u in grouped["user_id"]]
        col_indices = [self.item2idx[i] for i in grouped["item_id"]]
        data_values = grouped["rating"].values

        interaction_matrix = csr_matrix((data_values, (row_indices, col_indices)), shape=(num_users, num_items))

        # Adjust n_components if matrix rank is smaller
        actual_components = min(self.n_components, num_users - 1, num_items - 1)
        if actual_components < self.n_components:
            self.svd = TruncatedSVD(n_components=actual_components, random_state=self.random_state)

        # Decompose R = U * Sigma * V^T
        user_latent = self.svd.fit_transform(interaction_matrix)  # U * Sigma
        item_latent = self.svd.components_.T                     # V

        # Symmetrically distribute singular values
        sigma_sqrt = np.sqrt(self.svd.singular_values_)
        self.user_factors = user_latent / sigma_sqrt
        self.item_factors = item_latent * sigma_sqrt

        return self

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        category: Optional[str] = None,
        filter_history: bool = True
    ) -> List[Dict[str, Any]]:
        """Compute dot-product predicted ratings for the user against all item latent vectors."""
        seen_items = self.user_history.get(user_id, set()) if filter_history else set()

        if user_id not in self.user2idx:
            # Cold-start fallback to global popularity
            return self._cold_start_recommend(top_k=top_k, category=category)

        u_idx = self.user2idx[user_id]
        u_vector = self.user_factors[u_idx]

        # Predicted ratings: user_vector @ item_factors^T
        predicted_scores = np.dot(u_vector, self.item_factors.T)

        ranked_indices = np.argsort(-predicted_scores)

        recommendations = []
        for idx in ranked_indices:
            item_id = self.idx2item[idx]
            if item_id in seen_items:
                continue

            meta = self.item_metadata[item_id]
            if category and meta["category"] != category:
                continue

            score = float(predicted_scores[idx])
            rec = {
                "item_id": item_id,
                "title": meta["title"],
                "category": meta["category"],
                "sub_category": meta["sub_category"],
                "brand": meta["brand"],
                "price": meta["price"],
                "score": round(score, 4),
                "rank": len(recommendations) + 1,
                "model": "CollaborativeFiltering_SVD",
                "explanation": "Customers with purchasing habits similar to yours frequently engaged with this product."
            }
            recommendations.append(rec)
            if len(recommendations) >= top_k:
                break

        return recommendations

    def _cold_start_recommend(self, top_k: int, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fallback for unseen users in collaborative model."""
        sorted_items = sorted(self.global_item_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        for item_id, score in sorted_items:
            meta = self.item_metadata.get(item_id)
            if not meta:
                continue
            if category and meta["category"] != category:
                continue

            recommendations.append({
                "item_id": item_id,
                "title": meta["title"],
                "category": meta["category"],
                "sub_category": meta["sub_category"],
                "brand": meta["brand"],
                "price": meta["price"],
                "score": round(float(score), 4),
                "rank": len(recommendations) + 1,
                "model": "CollaborativeFiltering (ColdStart Fallback)",
                "explanation": "High collective engagement among active shoppers across store."
            })
            if len(recommendations) >= top_k:
                break

        return recommendations

