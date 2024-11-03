from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Union, Dict, Any
from enum import Enum
from uuid import UUID
from datetime import datetime

class SearchType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"

class Attribute(BaseModel):
    Size: str

class Product(BaseModel):
    id: UUID
    title: str
    brand: str
    price: float
    attributes: List[Dict[str, str]]
    category: str
    description: str
    image_url: List[HttpUrl]

    class Config:
        json_schema_extra = {
            "example": {
                "id": "fff0834a-280e-4bbb-b7d8-6f0a43495029",
                "title": "Kurta Dupatta - 1879",
                "brand": "Zellbury",
                "price": 0,
                "attributes": [{"Size": "XL"}],
                "category": "Stitched",
                "description": "Expertly crafted for women...",
                "image_url": ["https://zellbury.com/cdn/shop/files/WPS2421879-1.jpg"]
            }
        }

class SearchQuery(BaseModel):
    query_type: SearchType
    query: str  # For text: string, for image/audio: base64 encoded string
    num_results: Optional[int] = 5
    min_similarity: Optional[float] = 0.0

class SearchResult(BaseModel):
    product: Product
    similarity_score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int
    search_time: float
    service_type: Optional[str] = None
    query_understanding: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional information about how the query was understood"
    )

class UserPreferences(BaseModel):
    brand_weights: Optional[Dict[str, float]] = None
    price_range: Optional[tuple[float, float]] = None
    preferred_colors: Optional[List[str]] = None
    category_weights: Optional[Dict[str, float]] = None
    seasonal_preference: Optional[str] = None

class SearchMetrics(BaseModel):
    ndcg_score: float
    precision_at_k: float
    recall_at_k: float
    map_score: float
    diversity_score: float
    timestamp: datetime

class EvaluationResult(BaseModel):
    legacy_metrics: SearchMetrics
    imagebind_metrics: SearchMetrics
    timestamp: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "legacy_metrics": {
                    "ndcg_score": 0.85,
                    "precision_at_k": 0.75,
                    "recall_at_k": 0.80,
                    "map_score": 0.82,
                    "diversity_score": 0.70,
                    "timestamp": "2024-03-11T10:00:00"
                },
                "imagebind_metrics": {
                    "ndcg_score": 0.88,
                    "precision_at_k": 0.78,
                    "recall_at_k": 0.83,
                    "map_score": 0.85,
                    "diversity_score": 0.72,
                    "timestamp": "2024-03-11T10:00:00"
                },
                "timestamp": "2024-03-11T10:00:00"
            }
        }
