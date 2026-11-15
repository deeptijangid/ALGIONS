"""
Algions AI Labs - E-Commerce Recommendation Engine
Script: Ingestion, Validation, Chronological Split, and Model Training Pipeline.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor import RecommenderDataPreprocessor
from src.models import (
    PopularityRecommender,
    ContentBasedRecommender,
    CollaborativeSVDRecommender,
    HybridRecommender
)

RANDOM_SEED = 42

def generate_eda_plots(catalog_df: pd.DataFrame, interactions_df: pd.DataFrame, output_dir: str = "artifacts/figures"):
    """Generate and save publication-quality EDA charts."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Power-law interaction distribution per item
    plt.figure(figsize=(9, 5))
    item_counts = interactions_df["item_id"].value_counts().values
    plt.plot(np.arange(1, len(item_counts) + 1), item_counts, color="#2b5c8f", lw=2.5)
    plt.fill_between(np.arange(1, len(item_counts) + 1), item_counts, color="#2b5c8f", alpha=0.2)
    plt.title("Item Interaction Distribution (Long-Tail / Power-Law)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Item Rank (by Popularity)", fontsize=11)
    plt.ylabel("Number of Interactions", fontsize=11)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path_1 = os.path.join(output_dir, "interaction_distribution.png")
    plt.savefig(plot_path_1, dpi=200)
    plt.close()
    print(f"Saved EDA plot: {plot_path_1}")

    # 2. Conversion Funnel distribution
    plt.figure(figsize=(8, 4.5))
    funnel_counts = interactions_df["event_type"].value_counts()[["view", "cart", "purchase"]]
    colors = ["#4385f4", "#ea4335", "#34a853"]
    bars = plt.bar(funnel_counts.index, funnel_counts.values, color=colors, width=0.55, edgecolor="black", lw=1.2)
    plt.title("E-Commerce Conversion Funnel Volume", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Interaction Stage", fontsize=11)
    plt.ylabel("Total Events", fontsize=11)
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / len(interactions_df)) * 100
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(funnel_counts)*0.015), f"{yval:,} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")
    plt.ylim(0, max(funnel_counts) * 1.15)
    plt.tight_layout()
    plot_path_2 = os.path.join(output_dir, "funnel_distribution.png")
    plt.savefig(plot_path_2, dpi=200)
    plt.close()
    print(f"Saved EDA plot: {plot_path_2}")


def main():
    print("=" * 60)
    print("STEP 2: Training Pipeline Execution")
    print("=" * 60)

    # 1. Load raw data
    data_dir = "data/raw"
    catalog_path = os.path.join(data_dir, "products.csv")
    users_path = os.path.join(data_dir, "users.csv")
    interactions_path = os.path.join(data_dir, "interactions.csv")

    if not (os.path.exists(catalog_path) and os.path.exists(users_path) and os.path.exists(interactions_path)):
        print("Raw data not found. Running data generator first...")
        from scripts.generate_data import main as run_gen
        run_gen()

    catalog_df = pd.read_csv(catalog_path)
    users_df = pd.read_csv(users_path)
    interactions_df = pd.read_csv(interactions_path)
    interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])

    print(f"Loaded {len(catalog_df)} products, {len(users_df)} users, {len(interactions_df)} interactions.")

    # 2. Validation & Preprocessing
    preprocessor = RecommenderDataPreprocessor(time_split_quantile=0.80)
    quality_report = preprocessor.validate_data_quality(catalog_df, users_df, interactions_df)
    print(f"Data Quality Report: Monotonic={quality_report['timestamp_is_sorted']}, "
          f"Duplicates={quality_report['duplicate_interactions']}, "
          f"Orphan Users={quality_report['orphan_users_count']}")

    # 3. Generate EDA Plots
    generate_eda_plots(catalog_df, interactions_df, output_dir="artifacts/figures")

    # 4. Strict Chronological Train/Test Split
    print("Performing strict chronological split (80% train, 20% test)...")
    train_df, test_df = preprocessor.chronological_split(interactions_df)
    preprocessor.build_index_mappings(users_df, catalog_df)

    os.makedirs("data/processed", exist_ok=True)
    train_path = "data/processed/train_interactions.csv"
    test_path = "data/processed/test_interactions.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved train interactions: {len(train_df)} ({train_path})")
    print(f"Saved test interactions:  {len(test_df)}  ({test_path})")
    print(f"Split cutoff timestamp:   {preprocessor.split_timestamp}")

    # Sparsity
    sparsity = preprocessor.compute_sparsity(len(users_df), len(catalog_df), len(train_df))
    print(f"Matrix Sparsity: {sparsity * 100:.2f}%")

    # 5. Model Training
    os.makedirs("artifacts", exist_ok=True)

    # A. Popularity Baseline
    print("\n--- Training Model 1: Popularity Baseline ---")
    pop_model = PopularityRecommender(decay_half_life_days=30.0)
    pop_model.fit(train_df, catalog_df, users_df)
    pop_artifact_path = "artifacts/popularity_model.joblib"
    joblib.dump(pop_model, pop_artifact_path)
    print(f"Saved Popularity Model -> {pop_artifact_path}")

    # B. Content-Based
    print("\n--- Training Model 2: Content-Based Filtering ---")
    content_model = ContentBasedRecommender(max_features=5000, ngram_range=(1, 2))
    content_model.fit(train_df, catalog_df, users_df)
    content_artifact_path = "artifacts/content_model.joblib"
    joblib.dump(content_model, content_artifact_path)
    print(f"Saved Content-Based Model -> {content_artifact_path}")

    # C. Collaborative SVD
    print("\n--- Training Model 3: Collaborative Filtering (SVD) ---")
    cf_model = CollaborativeSVDRecommender(n_components=30, random_state=RANDOM_SEED)
    cf_model.fit(train_df, catalog_df, users_df)
    cf_artifact_path = "artifacts/collaborative_model.joblib"
    joblib.dump(cf_model, cf_artifact_path)
    print(f"Saved Collaborative SVD Model -> {cf_artifact_path}")

    # D. Adaptive Hybrid Recommender
    print("\n--- Training Model 4: Adaptive Hybrid Recommender ---")
    hybrid_model = HybridRecommender(
        cf_weight=0.65,
        svd_components=30,
        decay_half_life_days=30.0,
        random_state=RANDOM_SEED
    )
    hybrid_model.fit(train_df, catalog_df, users_df)
    hybrid_artifact_path = "artifacts/hybrid_model.joblib"
    joblib.dump(hybrid_model, hybrid_artifact_path)
    print(f"Saved Hybrid Model -> {hybrid_artifact_path}")

    # 6. Save Model Metadata
    metadata = {
        "training_timestamp": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "dataset_statistics": {
            "total_products": len(catalog_df),
            "total_users": len(users_df),
            "total_raw_interactions": len(interactions_df),
            "train_interactions": len(train_df),
            "test_interactions": len(test_df),
            "split_timestamp": str(preprocessor.split_timestamp),
            "matrix_sparsity_pct": round(sparsity * 100, 2),
            "cold_start_users_in_test": preprocessor.summary_stats.get("cold_start_test_users", 0),
            "cold_start_items_in_test": preprocessor.summary_stats.get("cold_start_test_items", 0)
        },
        "models": {
            "popularity": {"artifact": pop_artifact_path, "decay_half_life_days": 30.0},
            "content_based": {"artifact": content_artifact_path, "max_features": 5000, "ngram_range": [1, 2]},
            "collaborative_svd": {"artifact": cf_artifact_path, "n_components": 30},
            "hybrid": {"artifact": hybrid_artifact_path, "default_cf_weight": 0.65, "dynamic_routing": True}
        },
        "target_definition": {
            "prediction_unit": "User-Item Interaction",
            "prediction_horizon": "Next session window in held-out test timeframe (post-split)",
            "implicit_weights": {"view": 1.0, "cart": 2.5, "purchase": 5.0}
        }
    }

    metadata_path = "artifacts/model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"\nSaved metadata -> {metadata_path}")
    print("[SUCCESS] Training pipeline finished successfully.")


if __name__ == "__main__":
    main()

