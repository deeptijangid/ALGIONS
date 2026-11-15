"""
Algions AI Labs - E-Commerce Recommendation Engine
Script: Comprehensive Evaluation of Candidate Models vs Popularity Baseline.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics import evaluate_recommender


def plot_model_comparison(results: dict, output_path: str = "artifacts/figures/model_comparison.png"):
    """Plot benchmark comparison bar chart across key ranking metrics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid")

    models = list(results.keys())
    metrics_to_plot = ["NDCG@10", "Precision@10", "Recall@10", "Catalog_Coverage@10"]
    labels = ["NDCG@10", "Precision@10", "Recall@10", "Catalog Coverage"]

    x = np.arange(len(metrics_to_plot))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = ["#80868b", "#fbbc04", "#4285f4", "#34a853"]

    for i, model_name in enumerate(models):
        values = [results[model_name].get(m, 0.0) for m in metrics_to_plot]
        ax.bar(x + (i - 1.5) * width, values, width, label=model_name, color=palette[i % len(palette)], edgecolor="black", lw=0.8)

    ax.set_ylabel("Score / Ratio", fontsize=11, fontweight="bold")
    ax.set_title("E-Commerce Recommender Benchmark: Baseline vs Candidate Models", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.legend(title="Recommendation Engine", fontsize=10, loc="upper right")
    ax.set_ylim(0, max([max(results[m].get(k, 0.0) for k in metrics_to_plot) for m in models]) * 1.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved benchmark figure: {output_path}")


def main():
    print("=" * 60)
    print("STEP 3: Offline Model Evaluation on Held-Out Test Set")
    print("=" * 60)

    # 1. Load test data
    test_path = "data/processed/test_interactions.csv"
    train_path = "data/processed/train_interactions.csv"
    catalog_path = "data/raw/products.csv"

    if not (os.path.exists(test_path) and os.path.exists(catalog_path)):
        print("Processed data not found. Please run training pipeline first!")
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)
    catalog_df = pd.read_csv(catalog_path)

    catalog_item_ids = catalog_df["item_id"].tolist()
    item_category_map = catalog_df.set_index("item_id")["category"].to_dict()
    train_user_set = set(train_df["user_id"].unique())

    # Build ground truth test dictionary: user_id -> set of interacted item_ids in test timeframe
    ground_truth_test = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    print(f"Total test users with post-split ground truth: {len(ground_truth_test):,}")

    # 2. Load models
    model_paths = {
        "Popularity Baseline": "artifacts/popularity_model.joblib",
        "Content-Based": "artifacts/content_model.joblib",
        "Collaborative SVD": "artifacts/collaborative_model.joblib",
        "Adaptive Hybrid": "artifacts/hybrid_model.joblib"
    }

    results = {}
    k_list = [5, 10]

    for model_name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"Warning: Artifact {path} not found. Skipping {model_name}.")
            continue
        print(f"Evaluating {model_name}...")
        model = joblib.load(path)
        metrics = evaluate_recommender(
            model=model,
            ground_truth_test=ground_truth_test,
            catalog_item_ids=catalog_item_ids,
            train_user_set=train_user_set,
            k_list=k_list,
            item_category_map=item_category_map
        )
        results[model_name] = metrics

    # 3. Save evaluation metrics
    output_json = "artifacts/evaluation_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved evaluation metrics -> {output_json}")

    # 4. Print comparison table
    print("\n" + "=" * 90)
    print("RANKING QUALITY BENCHMARK (HELD-OUT TEST SET)")
    print("=" * 90)
    df_metrics = pd.DataFrame(results).T
    cols_to_display = [
        "Precision@5", "Precision@10",
        "Recall@5", "Recall@10",
        "NDCG@5", "NDCG@10",
        "MAP@10",
        "Catalog_Coverage@10",
        "Cold_Users_NDCG@10"
    ]
    existing_cols = [c for c in cols_to_display if c in df_metrics.columns]
    print(df_metrics[existing_cols].to_string())

    # 5. Generate plot
    plot_model_comparison(results, output_path="artifacts/figures/model_comparison.png")
    print("\n[SUCCESS] Evaluation complete.")


if __name__ == "__main__":
    main()
