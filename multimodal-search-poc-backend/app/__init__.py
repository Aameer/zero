## app/__init__.py
#from .services.search_service import SearchService
#from .models.schemas import SearchType, SearchQuery, SearchResult, SearchResponse, Product
#
#__version__ = "0.1.0"
#__all__ = [
#    "SearchService",
#    "SearchType",
#    "SearchQuery",
#    "SearchResult",
#    "SearchResponse",
#    "Product"
#]
# app/__init__.py
from .services.search_service import SearchService
from .models.schemas import SearchType, SearchQuery, SearchResult, SearchResponse, Product

__version__ = "0.1.0"
