"""
Algions AI Labs - E-Commerce Recommendation Engine
Evaluation Metrics Suite: Precision@K, Recall@K, NDCG@K, MAP@K,
Catalog Coverage, Novelty, and Cold-Start Stratified Evaluation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Any


def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Compute Precision at rank K."""
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / float(k)


def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Compute Recall at rank K."""
    if not relevant or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / float(len(relevant))


def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K with binary relevance."""
    if not relevant or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    dcg = 0.0
    for idx, item in enumerate(rec_k):
        if item in relevant:
            dcg += 1.0 / np.log2(idx + 2)  # idx is 0-indexed, so rank = idx + 1, log2(rank + 1) = log2(idx + 2)

    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(relevant))))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Compute Average Precision at rank K."""
    if not relevant or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = 0
    sum_precisions = 0.0
    for idx, item in enumerate(rec_k):
        if item in relevant:
            hits += 1
            sum_precisions += hits / float(idx + 1)
    
    num_eval = min(k, len(relevant))
    return sum_precisions / float(num_eval) if num_eval > 0 else 0.0


def evaluate_recommender(
    model,
    ground_truth_test: Dict[str, Set[str]],
    catalog_item_ids: List[str],
    train_user_set: Set[str],
    k_list: List[int] = [5, 10],
    item_category_map: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Perform rigorous evaluation of a recommender model across all test users.
    Computes overall ranking metrics, catalog coverage, category diversity,
    and stratified metrics for warm vs cold-start users.
    """
    metrics = {f"Precision@{k}": [] for k in k_list}
    metrics.update({f"Recall@{k}": [] for k in k_list})
    metrics.update({f"NDCG@{k}": [] for k in k_list})
    metrics.update({f"MAP@{k}": [] for k in k_list})

    # Separate tracking for warm vs cold users
    warm_metrics = {f"NDCG@{k}": [] for k in k_list}
    cold_metrics = {f"NDCG@{k}": [] for k in k_list}

    all_recommended_items_top10: Set[str] = set()
    category_diversity_scores: List[float] = []

    total_test_users = len(ground_truth_test)
    max_k = max(k_list)

    for user_id, relevant_items in ground_truth_test.items():
        if not relevant_items:
            continue

        # Generate top recommendation candidates
        recommendations = model.recommend(user_id=user_id, top_k=max_k)
        rec_item_ids = [r["item_id"] for r in recommendations]

        all_recommended_items_top10.update(rec_item_ids[:10])

        is_cold = user_id not in train_user_set

        # Compute metric for each k
        for k in k_list:
            p_k = precision_at_k(rec_item_ids, relevant_items, k)
            r_k = recall_at_k(rec_item_ids, relevant_items, k)
            ndcg_k = ndcg_at_k(rec_item_ids, relevant_items, k)
            map_k = average_precision_at_k(rec_item_ids, relevant_items, k)

            metrics[f"Precision@{k}"].append(p_k)
            metrics[f"Recall@{k}"].append(r_k)
            metrics[f"NDCG@{k}"].append(ndcg_k)
            metrics[f"MAP@{k}"].append(map_k)

            if is_cold:
                cold_metrics[f"NDCG@{k}"].append(ndcg_k)
            else:
                warm_metrics[f"NDCG@{k}"].append(ndcg_k)

        # Intra-list category diversity
        if item_category_map and rec_item_ids:
            rec_categories = [item_category_map.get(iid, "Unknown") for iid in rec_item_ids[:10]]
            unique_cats = len(set(rec_categories))
            category_diversity_scores.append(unique_cats / min(10, len(rec_categories)))

    # Compute macro-averages
    results = {}
    for metric_name, values in metrics.items():
        results[metric_name] = round(float(np.mean(values)), 4) if values else 0.0

    # Warm vs Cold breakdown
    for k in k_list:
        results[f"Warm_Users_NDCG@{k}"] = round(float(np.mean(warm_metrics[f"NDCG@{k}"])), 4) if warm_metrics[f"NDCG@{k}"] else 0.0
        results[f"Cold_Users_NDCG@{k}"] = round(float(np.mean(cold_metrics[f"NDCG@{k}"])), 4) if cold_metrics[f"NDCG@{k}"] else 0.0

    # Catalog Coverage @ 10
    total_catalog_size = len(catalog_item_ids)
    coverage = len(all_recommended_items_top10) / float(total_catalog_size) if total_catalog_size > 0 else 0.0
    results["Catalog_Coverage@10"] = round(float(coverage), 4)
    results["Catalog_Recommended_Items_Count"] = len(all_recommended_items_top10)

    # Category Diversity
    if category_diversity_scores:
        results["Avg_Category_Diversity@10"] = round(float(np.mean(category_diversity_scores)), 4)

    return results

