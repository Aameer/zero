# app/main.py
import urllib3
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import json
import time
import logging
from fastapi import Form

from app.models.schemas import (
    Product, SearchQuery, SearchResponse, SearchType,
    UserPreferences, SearchResult
)
from app.services.search_service import EnhancedSearchService
from app.services.imagebind_service import ImageBindSearchService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multimodal Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
legacy_search_service: Optional[EnhancedSearchService] = None
imagebind_search_service: Optional[ImageBindSearchService] = None
search_service: Optional[EnhancedSearchService] = None

async def initialize_service():
    """Initialize service safely"""
    global search_service
    try:
        with open("app/data/catalog.json", "r") as f:
            catalog = json.load(f)
            # Start with just 2 items
            catalog = catalog[:2]

        search_service = await EnhancedSearchService.create(catalog)
        logger.info("Search service initialized successfully")

    except Exception as e:
        logger.error(f"Service initialization failed: {str(e)}")
        if search_service:
            await search_service.cleanup()
        search_service = None
        raise

@app.on_event("startup")
async def startup_event():
    """Startup handler"""
    await initialize_service()

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown handler"""
    global search_service
    if search_service:
        await search_service.cleanup()
        search_service = None

"""
@app.on_event("startup")
async def startup_event():
    await initialize_services()

@app.on_event("shutdown")
async def shutdown_event():
    global legacy_search_service, imagebind_search_service
    try:
        if legacy_search_service:
            await legacy_search_service.cleanup()
        if imagebind_search_service:
            await imagebind_search_service.cleanup()
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
    finally:
        legacy_search_service = None
        imagebind_search_service = None
"""
@app.get("/products", response_model=List[Product])
async def get_products():
    """Get all products endpoint"""
    try:
        with open("app/data/catalog.json", "r") as f:
            catalog = json.load(f)
        return catalog
    except Exception as e:
        logger.error(f"Error fetching products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """General search endpoint"""
    if not legacy_search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    start_time = time.time()
    try:
        results = await legacy_search_service.search(
            query_type=query.query_type,
            query=query.query,
            num_results=query.num_results,
            min_similarity=query.min_similarity,
            user_preferences=query.preferences
        )
        search_time = time.time() - start_time

        return SearchResponse(
            results=results,
            total_results=len(results),
            search_time=search_time
        )
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search/image", response_model=SearchResponse)
async def image_search(
    file: UploadFile = File(...),
    preferences: str = Form(None),
    num_results: int = 5,
    min_similarity: float = 0.0
):
    """Image search endpoint"""
    if not legacy_search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    start_time = time.time()
    try:
        # Parse preferences if provided
        user_preferences = None
        if preferences:
            try:
                preferences_dict = json.loads(preferences)
                user_preferences = UserPreferences(**preferences_dict)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid preferences JSON")

        contents = await file.read()
        results = await legacy_search_service.search(
            query_type=SearchType.IMAGE,
            query=contents,
            num_results=num_results,
            min_similarity=min_similarity,
            user_preferences=user_preferences
        )
        search_time = time.time() - start_time

        return SearchResponse(
            results=results,
            total_results=len(results),
            search_time=search_time
        )
    except Exception as e:
        logger.error(f"Error in image search: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Add similar endpoints for audio search if needed...
# Add similar modifications for audio search and other endpoints...
"""
# New endpoint to get search results with similarity scores
@app.post("/search/detailed", response_model=List[SearchResult])
async def detailed_search(query: SearchQuery):
    if not search_service:
        raise HTTPException(status_code=500, detail="Search service not initialized")

    try:
        results = search_service.search(
            query_type=query.query_type,
            query=query.query,
            num_results=query.num_results,
            min_similarity=query.min_similarity,
            user_preferences=query.preferences
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/search/detailed", response_model=List[SearchResult])
async def detailed_search(query: SearchQuery):
    if not search_service:
        raise HTTPException(status_code=500, detail="Search service not initialized")

    try:
        return search_service.search(
            query_type=query.query_type,
            query=query.query,
            num_results=query.num_results,
            min_similarity=query.min_similarity,
            user_preferences=query.preferences
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search/image/detailed", response_model=List[SearchResult])
async def detailed_image_search(
    file: UploadFile = File(...),
    preferences: str = Form(None),
    num_results: int = 5,
    min_similarity: float = 0.0
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        user_preferences = None
        if preferences:
            try:
                preferences_dict = json.loads(preferences)
                user_preferences = UserPreferences(**preferences_dict)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid preferences JSON")

        contents = await file.read()
        return search_service.search(
            query_type=SearchType.IMAGE,
            query=contents,
            num_results=num_results,
            min_similarity=min_similarity,
            user_preferences=user_preferences
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search/audio", response_model=List[Product])
async def audio_search(
    file: UploadFile = File(...),
    preferences: str = Form(None),
    num_results: int = 5,
    min_similarity: float = 0.0
):
    # List of allowed audio MIME types
    allowed_audio_types = [
        'audio/mpeg',        # .mp3
        'audio/mp3',         # alternative for .mp3
        'audio/wav',         # .wav
        'audio/wave',        # alternative for .wav
        'audio/x-wav',       # alternative for .wav
        'audio/aac',         # .aac
        'audio/ogg',         # .ogg
        'audio/webm',        # .webm
        'audio/x-m4a',       # .m4a
        'audio/mp4'          # .mp4 audio
    ]

    # Check content type and filename extension
    file_extension = file.filename.lower().split('.')[-1]
    is_valid_extension = file_extension in ['mp3', 'wav', 'aac', 'ogg', 'webm', 'm4a', 'mp4']
    is_valid_mime = file.content_type in allowed_audio_types

    if not (is_valid_extension or is_valid_mime):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format. Supported formats: mp3, wav, aac, ogg, webm, m4a, mp4. Got content-type: {file.content_type}"
        )

    try:
        # Parse preferences JSON if provided
        user_preferences = None
        if preferences:
            try:
                preferences_dict = json.loads(preferences)
                user_preferences = UserPreferences(**preferences_dict)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid preferences JSON")

        contents = await file.read()
        search_results = search_service.search(
            query_type=SearchType.AUDIO,
            query=contents,
            num_results=num_results,
            min_similarity=min_similarity,
            user_preferences=user_preferences
        )

        # Extract just the products from the search results
        products = [result.product for result in search_results]
        return products

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search/audio/detailed", response_model=List[SearchResult])
async def detailed_audio_search(
    file: UploadFile = File(...),
    preferences: str = Form(None),
    num_results: int = 5,
    min_similarity: float = 0.0
):
    # List of allowed audio MIME types
    allowed_audio_types = [
        'audio/mpeg',        # .mp3
        'audio/mp3',         # alternative for .mp3
        'audio/wav',         # .wav
        'audio/wave',        # alternative for .wav
        'audio/x-wav',       # alternative for .wav
        'audio/aac',         # .aac
        'audio/ogg',         # .ogg
        'audio/webm',        # .webm
        'audio/x-m4a',       # .m4a
        'audio/mp4'          # .mp4 audio
    ]

    # Check content type and filename extension
    file_extension = file.filename.lower().split('.')[-1]
    is_valid_extension = file_extension in ['mp3', 'wav', 'aac', 'ogg', 'webm', 'm4a', 'mp4']
    is_valid_mime = file.content_type in allowed_audio_types

    if not (is_valid_extension or is_valid_mime):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format. Supported formats: mp3, wav, aac, ogg, webm, m4a, mp4. Got content-type: {file.content_type}"
        )

    try:
        # Parse preferences JSON if provided
        user_preferences = None
        if preferences:
            try:
                preferences_dict = json.loads(preferences)
                user_preferences = UserPreferences(**preferences_dict)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid preferences JSON")

        contents = await file.read()
        return search_service.search(
            query_type=SearchType.AUDIO,
            query=contents,
            num_results=num_results,
            min_similarity=min_similarity,
            user_preferences=user_preferences
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search/compare", response_model=List[SearchResponse])
async def compare_search(query: SearchQuery):
    if not legacy_search_service or not imagebind_search_service:
        raise HTTPException(status_code=500, detail="Search services not initialized")

    results = []

    # Get results from both implementations
    for service in [legacy_search_service, imagebind_search_service]:
        start_time = time.time()
        try:
            service_results = service.search(
                query_type=query.query_type,
                query=query.query,
                num_results=query.num_results,
                min_similarity=query.min_similarity
            )
            search_time = time.time() - start_time

            results.append(SearchResponse(
                results=service_results,
                total_results=len(service_results),
                search_time=search_time,
                service_type=service.__class__.__name__
            ))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return results

@app.post("/evaluate")
async def evaluate_search(use_weights: bool = False) -> EvaluationResult:
    try:
        evaluator = SearchEvaluator("app/data/ground_truth.json")

        legacy_metrics = evaluator.evaluate_search(legacy_search_service, use_weights)
        imagebind_metrics = evaluator.evaluate_search(imagebind_search_service, use_weights)

        return EvaluationResult(
            legacy_metrics=legacy_metrics,
            imagebind_metrics=imagebind_metrics,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""
