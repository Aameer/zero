# app/__init__.py
from .services.search_service import EnhancedSearchService
from .models.schemas import SearchType, SearchQuery, SearchResult, SearchResponse, Product

__version__ = "0.1.0"
