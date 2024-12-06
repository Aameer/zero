# app/models/schemas.py
from pydantic import BaseModel, HttpUrl, Field, validator
from typing import List, Optional, Dict, Union, Tuple, Any
from enum import Enum
from uuid import UUID
from datetime import datetime

class SearchType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"

class AttributeType(str, Enum):
    COLOR = "Color"
    SIZE = "Size"
    FABRIC = "Fabric"
    SEASON = "Season"
    STYLE = "Style"
    DESIGN = "Design"

class UserPreferences(BaseModel):
    brand_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Brand preferences with weights between 0 and 1"
    )
    price_range: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Min and max price range"
    )
    preferred_colors: Optional[List[str]] = Field(
        default=None,
        description="List of preferred colors"
    )
    category_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Category preferences with weights between 0 and 1"
    )
    delivery_speed: Optional[int] = Field(
        default=None,
        description="Preferred delivery time in days"
    )
    seasonal_preference: Optional[str] = Field(
        default=None,
        description="Preferred season (SPRING, SUMMER, FALL, WINTER)"
    )
    size_preference: Optional[List[str]] = Field(
        default=None,
        description="Preferred sizes"
    )
    fabric_preference: Optional[List[str]] = Field(
        default=None,
        description="Preferred fabric types"
    )

    @validator('brand_weights', 'category_weights')
    def validate_weights(cls, v):
        if v is not None:
            for weight in v.values():
                if not 0 <= weight <= 1:
                    raise ValueError("Weights must be between 0 and 1")
        return v

    @validator('price_range')
    def validate_price_range(cls, v):
        if v is not None:
            if len(v) != 2 or v[0] > v[1]:
                raise ValueError("Price range must be [min, max] with min <= max")
        return v

class Product(BaseModel):
    id: str  
    title: str
    brand: str
    price: float
    attributes: List[Dict[str, List[str]]]  # Updated for nested list structure
    category: str
    description: str
    image_url: List[HttpUrl]
    created_at: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Product creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Product last update timestamp"
    )
    url: HttpUrl  # Added field for product URL
    homePage: Optional[HttpUrl] = None  # Optional field for homepage
    returns: Optional[HttpUrl] = None  # Optional field for returns URL

    class Config:
        schema_extra = {
            "example": {
                "id": "673dcd1a57abf185743f9611",
                "title": "1 PC Printed Lawn Dupatta",
                "brand": "alkaramstudio",
                "price": 927.00,
                "attributes": [
                    {"Color": ["Grey"]},
                    {"Fabric": ["Lawn"]}
                ],
                "category": "Alkaram Studio - Kurti",
                "description": "1 PC Printed Lawn Dupatta",
                "image_url": [
                    "https://cdn.shopify.com/s/files/1/0623/6481/1444/products/SLRD-01-23-3-Grey-1.jpg?v=1718486171",
                    "https://cdn.shopify.com/s/files/1/0623/6481/1444/products/SLRD-01-23-3-Grey-2.jpg?v=1718486174",
                    "https://cdn.shopify.com/s/files/1/0623/6481/1444/products/SLRD-01-23-3-Grey-3.jpg?v=1718486177"
                ],
                "url": "https://www.alkaramstudio.com/products/1-pc-printed-lawn-dupatta-slrd-01-23-3-grey",
                "homePage": "https://www.alkaramstudio.com/",
                "returns": "https://www.alkaramstudio.com/pages/return-and-exchange-policy",
                "created_at": "2024-12-03T10:30:00",
                "updated_at": "2024-12-03T10:30:00"
            }
        }

class SearchResult(BaseModel):
    product: Product
    similarity_score: float
    attribute_matches: Optional[Dict[str, float]] = Field(
        default=None,
        description="Matching scores for different attributes"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Explanation of why this result matches"
    )

class SearchQuery(BaseModel):
    query_type: SearchType
    query: str
    num_results: Optional[int] = 5
    min_similarity: Optional[float] = 0.0
    preferences: Optional[UserPreferences] = None
    filter_attributes: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Filter results by specific attributes"
    )
    user_id: Optional[int] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int
    search_time: float
    query_understanding: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Details about how the query was understood"
    )
