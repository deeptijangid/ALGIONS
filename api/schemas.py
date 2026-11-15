"""
Algions AI Labs - E-Commerce Recommendation Engine
Pydantic Schemas for API Requests, Responses, and Explainability Signals.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., example="USER_0042", description="User identifier to generate recommendations for.")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of items to recommend.")
    category: Optional[str] = Field(default=None, example="Electronics", description="Optional category filter.")
    filter_history: bool = Field(default=True, description="Filter out items already interacted with.")
    model_type: Optional[str] = Field(default="hybrid", description="Model variant: 'hybrid', 'collaborative', 'content', or 'popularity'.")


class RecommendationItem(BaseModel):
    item_id: str
    title: str
    category: str
    sub_category: str
    brand: str
    price: float
    score: float
    rank: int
    model: str
    explanation: str
    confidence_level: Optional[str] = "Medium"
    primary_driver: Optional[str] = "Ensemble"


class RecommendationResponse(BaseModel):
    user_id: str
    user_segment: str
    model_used: str
    top_k: int
    latency_ms: float
    recommendations: List[RecommendationItem]
    decision_support_advisory: str


class SimilarItemsRequest(BaseModel):
    item_id: str = Field(..., example="PROD_0001", description="Anchor item ID to find similar items for.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of similar items to return.")


class SimilarItem(BaseModel):
    item_id: str
    title: str
    category: str
    sub_category: str
    brand: str
    price: float
    similarity_score: float
    rank: int


class SimilarItemsResponse(BaseModel):
    anchor_item_id: str
    anchor_title: str
    top_k: int
    similar_items: List[SimilarItem]


class UserProfileResponse(BaseModel):
    user_id: str
    demographics: Dict[str, Any]
    interaction_count: int
    recent_interactions: List[Dict[str, Any]]
    is_cold_start: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, bool]
    catalog_item_count: int
    timestamp: str

