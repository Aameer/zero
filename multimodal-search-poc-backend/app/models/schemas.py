# app/models/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Union
from enum import Enum

class SearchType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"

class Product(BaseModel):
    id: int
    title: str
    brand: str
    price: float
    color: str
    category: str
    description: str
    image_url: str

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
