"""
Algions AI Labs - E-Commerce Recommendation Engine
Preprocessor: Temporal train/test splitting, implicit feedback aggregation,
index mapping, and data quality validation with strict leakage prevention.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any, Optional


class RecommenderDataPreprocessor:
    """
    Handles data ingestion, validation, chronological train/test splitting,
    and user-item implicit interaction matrix preparation.
    """

    def __init__(self, time_split_quantile: float = 0.80):
        self.time_split_quantile = time_split_quantile
        self.user2idx: Dict[str, int] = {}
        self.idx2user: Dict[int, str] = {}
        self.item2idx: Dict[str, int] = {}
        self.idx2item: Dict[int, str] = {}
        self.split_timestamp: Optional[pd.Timestamp] = None
        self.summary_stats: Dict[str, Any] = {}

    def validate_data_quality(
        self,
        catalog_df: pd.DataFrame,
        users_df: pd.DataFrame,
        interactions_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check for nulls, duplicates, invalid foreign keys, and chronological monotonicity."""
        catalog_nulls = catalog_df.isnull().sum().to_dict()
        users_nulls = users_df.isnull().sum().to_dict()
        interactions_nulls = interactions_df.isnull().sum().to_dict()

        duplicate_interactions = int(interactions_df.duplicated(subset=["session_id", "user_id", "item_id", "event_type"]).sum())
        orphan_users = set(interactions_df["user_id"]) - set(users_df["user_id"])
        orphan_items = set(interactions_df["item_id"]) - set(catalog_df["item_id"])

        is_monotonic = interactions_df["timestamp"].is_monotonic_increasing

        quality_report = {
            "total_products": len(catalog_df),
            "total_users": len(users_df),
            "total_interactions": len(interactions_df),
            "catalog_null_counts": catalog_nulls,
            "users_null_counts": users_nulls,
            "interactions_null_counts": interactions_nulls,
            "duplicate_interactions": duplicate_interactions,
            "orphan_users_count": len(orphan_users),
            "orphan_items_count": len(orphan_items),
            "timestamp_is_sorted": bool(is_monotonic),
            "min_timestamp": str(interactions_df["timestamp"].min()),
            "max_timestamp": str(interactions_df["timestamp"].max())
        }
        return quality_report

    def chronological_split(
        self,
        interactions_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split interactions strictly chronologically to prevent temporal data leakage.
        Interactions occurring before the split timestamp form the train set;
        interactions at or after the split timestamp form the held-out test set.
        """
        sorted_df = interactions_df.sort_values(by="timestamp").copy()
        cutoff_idx = int(len(sorted_df) * self.time_split_quantile)
        self.split_timestamp = sorted_df.iloc[cutoff_idx]["timestamp"]

        train_df = sorted_df[sorted_df["timestamp"] < self.split_timestamp].copy()
        test_df = sorted_df[sorted_df["timestamp"] >= self.split_timestamp].copy()

        # Compute split statistics
        train_users = set(train_df["user_id"].unique())
        test_users = set(test_df["user_id"].unique())
        train_items = set(train_df["item_id"].unique())
        test_items = set(test_df["item_id"].unique())

        cold_start_users_in_test = test_users - train_users
        cold_start_items_in_test = test_items - train_items

        self.summary_stats = {
            "split_timestamp": str(self.split_timestamp),
            "train_interaction_count": len(train_df),
            "test_interaction_count": len(test_df),
            "train_unique_users": len(train_users),
            "test_unique_users": len(test_users),
            "train_unique_items": len(train_items),
            "test_unique_items": len(test_items),
            "cold_start_test_users": len(cold_start_users_in_test),
            "cold_start_test_items": len(cold_start_items_in_test)
        }
        return train_df, test_df

    def build_index_mappings(self, users_df: pd.DataFrame, catalog_df: pd.DataFrame):
        """Build bidirectional integer mappings for users and items across full catalog."""
        unique_users = sorted(users_df["user_id"].unique())
        unique_items = sorted(catalog_df["item_id"].unique())

        self.user2idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx2user = {idx: u for idx, u in enumerate(unique_users)}
        self.item2idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx2item = {idx: i for idx, i in enumerate(unique_items)}

    def aggregate_implicit_feedback(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multiple interactions between a user and an item into a consolidated
        implicit preference score: sum(event_weight) with log-damping.
        """
        grouped = interactions_df.groupby(["user_id", "item_id"]).agg(
            total_weight=("event_weight", "sum"),
            interaction_count=("event_weight", "count"),
            last_timestamp=("timestamp", "max")
        ).reset_index()

        # Log transform for diminishing returns on repeated clicks
        grouped["rating"] = np.log1p(grouped["total_weight"])
        return grouped

    def compute_sparsity(self, num_users: int, num_items: int, num_interactions: int) -> float:
        """Compute the matrix sparsity percentage."""
        total_possible = num_users * num_items
        if total_possible == 0:
            return 1.0
        return 1.0 - (num_interactions / float(total_possible))

