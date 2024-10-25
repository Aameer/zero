from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Union, Dict
from enum import Enum
from uuid import UUID

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
