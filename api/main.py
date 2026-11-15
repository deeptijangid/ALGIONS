"""
Algions AI Labs - E-Commerce Recommendation Engine
FastAPI Service: Exposes real model inference, explainability signals,
cold-start routing, and catalog discovery.
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem,
    SimilarItemsRequest,
    SimilarItemsResponse,
    SimilarItem,
    UserProfileResponse,
    HealthResponse
)
from src.explainability import RecommendationExplainer

app = FastAPI(
    title="Algions AI Labs - E-Commerce Recommendation Engine API",
    description="Production-grade decision-support recommendation service supporting Hybrid, CF, Content-Based, and Popularity models.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for artifacts and metadata
MODELS: Dict[str, Any] = {}
CATALOG_DF: Optional[pd.DataFrame] = None
USERS_DF: Optional[pd.DataFrame] = None
TRAIN_INTERACTIONS_DF: Optional[pd.DataFrame] = None
EXPLAINER: Optional[RecommendationExplainer] = None
EVALUATION_RESULTS: Dict[str, Any] = {}


def load_artifacts():
    """Load pre-trained models, catalog, and user tables into memory on startup."""
    global MODELS, CATALOG_DF, USERS_DF, TRAIN_INTERACTIONS_DF, EXPLAINER, EVALUATION_RESULTS

    # 1. Catalog & Users
    cat_path = "data/raw/products.csv"
    users_path = "data/raw/users.csv"
    train_path = "data/processed/train_interactions.csv"

    if os.path.exists(cat_path):
        CATALOG_DF = pd.read_csv(cat_path)
    if os.path.exists(users_path):
        USERS_DF = pd.read_csv(users_path)
    if os.path.exists(train_path):
        TRAIN_INTERACTIONS_DF = pd.read_csv(train_path)

    # 2. Models
    model_paths = {
        "popularity": "artifacts/popularity_model.joblib",
        "content": "artifacts/content_model.joblib",
        "collaborative": "artifacts/collaborative_model.joblib",
        "hybrid": "artifacts/hybrid_model.joblib"
    }

    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                MODELS[name] = joblib.load(path)
                print(f"[API] Loaded model artifact: {name} from {path}")
            except Exception as e:
                print(f"[API] Failed to load {name}: {e}")

    # 3. Explainer
    if CATALOG_DF is not None and TRAIN_INTERACTIONS_DF is not None:
        user_history = {}
        for _, row in TRAIN_INTERACTIONS_DF.iterrows():
            u = row["user_id"]
            i = row["item_id"]
            w = float(row["event_weight"])
            if u not in user_history:
                user_history[u] = {}
            user_history[u][i] = user_history[u].get(i, 0.0) + w
        EXPLAINER = RecommendationExplainer(CATALOG_DF, user_history)

    # 4. Evaluation results
    eval_path = "artifacts/evaluation_results.json"
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            EVALUATION_RESULTS = json.load(f)


@app.on_event("startup")
def startup_event():
    load_artifacts()


# Mount static assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def get_dashboard():
    """Serve the interactive web demonstration interface."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Algions Recommendation Engine API. Interactive UI assets not found; visit /docs for API documentation."}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health check and loaded artifact status."""
    return HealthResponse(
        status="HEALTHY",
        version="1.0.0",
        models_loaded={k: (v is not None) for k, v in MODELS.items()},
        catalog_item_count=len(CATALOG_DF) if CATALOG_DF is not None else 0,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/recommend", response_model=RecommendationResponse, tags=["Inference"])
def get_recommendations(req: RecommendationRequest):
    """
    Generate Top-N real recommendations for a user with explainable feature signals.
    """
    start_time = time.perf_counter()

    # Determine model
    model_key = req.model_type.lower()
    if model_key not in MODELS:
        # Fallback to hybrid or available model
        model_key = "hybrid" if "hybrid" in MODELS else (list(MODELS.keys())[0] if MODELS else None)

    if not model_key or model_key not in MODELS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation models are not loaded. Please execute training pipeline first."
        )

    model = MODELS[model_key]

    # Generate raw recommendations
    try:
        raw_recs = model.recommend(
            user_id=req.user_id,
            top_k=req.top_k,
            category=req.category,
            filter_history=req.filter_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # User segmentation determination
    user_seen_count = 0
    if TRAIN_INTERACTIONS_DF is not None:
        user_seen_count = len(TRAIN_INTERACTIONS_DF[TRAIN_INTERACTIONS_DF["user_id"] == req.user_id])
    
    if user_seen_count == 0:
        user_segment = "Cold-Start User (No Prior Interactions)"
    elif user_seen_count < 5:
        user_segment = "Casual / Sparse User (1-4 Interactions)"
    else:
        user_segment = f"High-Activity Shopper ({user_seen_count} Interactions)"

    # Format recommendations with explainability
    formatted_items = []
    user_demog = None
    if USERS_DF is not None:
        match = USERS_DF[USERS_DF["user_id"] == req.user_id]
        if not match.empty:
            user_demog = match.iloc[0].to_dict()

    for item in raw_recs:
        explanation_card = None
        if EXPLAINER:
            explanation_card = EXPLAINER.explain_prediction(req.user_id, item, user_demog)

        confidence = explanation_card["confidence_level"] if explanation_card else "Medium"
        formatted_items.append(RecommendationItem(
            item_id=item["item_id"],
            title=item["title"],
            category=item["category"],
            sub_category=item.get("sub_category", "General"),
            brand=item.get("brand", "Standard"),
            price=float(item.get("price", 0.0)),
            score=float(item.get("score", 0.0)),
            rank=int(item.get("rank", len(formatted_items) + 1)),
            model=item.get("model", model_key),
            explanation=item.get("explanation", "Recommended item."),
            confidence_level=confidence,
            primary_driver=item.get("primary_driver", "Ensemble")
        ))

    latency = round((time.perf_counter() - start_time) * 1000.0, 2)

    return RecommendationResponse(
        user_id=req.user_id,
        user_segment=user_segment,
        model_used=model_key.capitalize(),
        top_k=len(formatted_items),
        latency_ms=latency,
        recommendations=formatted_items,
        decision_support_advisory=(
            "Decision-Support Notice: Recommendations are generated based on historical browsing patterns. "
            "Merchandisers should verify available inventory stock levels and promotional policies before launching targeted campaigns."
        )
    )


@app.post("/api/recommend_similar", response_model=SimilarItemsResponse, tags=["Inference"])
def get_similar_products(req: SimilarItemsRequest):
    """Find semantically similar products to a given anchor item."""
    if "content" not in MODELS:
        raise HTTPException(status_code=503, detail="Content-based model not loaded.")

    content_model = MODELS["content"]
    similar_list = content_model.recommend_similar_items(req.item_id, top_k=req.top_k)

    anchor_meta = CATALOG_DF[CATALOG_DF["item_id"] == req.item_id] if CATALOG_DF is not None else None
    anchor_title = anchor_meta.iloc[0]["title"] if (anchor_meta is not None and not anchor_meta.empty) else req.item_id

    formatted = [
        SimilarItem(
            item_id=it["item_id"],
            title=it["title"],
            category=it["category"],
            sub_category=it["sub_category"],
            brand=it["brand"],
            price=float(it["price"]),
            similarity_score=float(it["similarity_score"]),
            rank=int(it["rank"])
        )
        for it in similar_list
    ]

    return SimilarItemsResponse(
        anchor_item_id=req.item_id,
        anchor_title=anchor_title,
        top_k=len(formatted),
        similar_items=formatted
    )


@app.get("/api/user_profile/{user_id}", response_model=UserProfileResponse, tags=["Catalog & Cohort"])
def get_user_profile(user_id: str):
    """Retrieve user demographic profile and recent interaction history."""
    if USERS_DF is None or CATALOG_DF is None:
        raise HTTPException(status_code=503, detail="Data tables not loaded.")

    user_match = USERS_DF[USERS_DF["user_id"] == user_id]
    demographics = user_match.iloc[0].to_dict() if not user_match.empty else {"user_id": user_id, "status": "Synthetic/Cold"}

    interactions = []
    if TRAIN_INTERACTIONS_DF is not None:
        user_events = TRAIN_INTERACTIONS_DF[TRAIN_INTERACTIONS_DF["user_id"] == user_id].sort_values(by="timestamp", ascending=False).head(10)
        for _, row in user_events.iterrows():
            item_meta = CATALOG_DF[CATALOG_DF["item_id"] == row["item_id"]]
            title = item_meta.iloc[0]["title"] if not item_meta.empty else row["item_id"]
            interactions.append({
                "item_id": row["item_id"],
                "title": title,
                "event_type": row["event_type"],
                "event_weight": float(row["event_weight"]),
                "timestamp": str(row["timestamp"])
            })

    return UserProfileResponse(
        user_id=user_id,
        demographics=demographics,
        interaction_count=len(interactions),
        recent_interactions=interactions,
        is_cold_start=(len(interactions) == 0)
    )


@app.get("/api/catalog", tags=["Catalog & Cohort"])
def get_catalog_sample(
    category: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = Query(default=20, le=100)
):
    """Search and browse products in the catalog."""
    if CATALOG_DF is None:
        raise HTTPException(status_code=503, detail="Catalog not loaded.")

    df = CATALOG_DF.copy()
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if query:
        df = df[df["title"].str.contains(query, case=False, na=False) | df["tags"].str.contains(query, case=False, na=False)]

    records = df.head(limit).to_dict(orient="records")
    return {"total_matching": len(df), "limit": limit, "items": records}


@app.get("/api/metrics", tags=["Evaluation"])
def get_metrics_benchmark():
    """Return offline benchmark evaluation results on the held-out test set."""
    if not EVALUATION_RESULTS:
        eval_path = "artifacts/evaluation_results.json"
        if os.path.exists(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"message": "Benchmark not executed yet. Run python scripts/evaluate.py"}
    return EVALUATION_RESULTS

