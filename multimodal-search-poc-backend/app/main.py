# app/main.py
# app/main.py
import urllib3
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from fastapi import Form
import json
import time
import logging

from app.models.schemas import (
    Product, SearchQuery, SearchResponse, SearchType,
    UserPreferences, SearchResult
)
from app.config.search_config import SearchConfig
from app.services.search_service import EnhancedSearchService

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Multimodal Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search service
search_service = None

@app.on_event("startup")
async def startup_event():
    global search_service
    try:
        logger.info("Starting up the application...")
        with open("app/data/catalog.json", "r") as f:
            catalog = json.load(f)
        search_service = EnhancedSearchService(catalog)
        logger.info("Search service created, initializing indexes...")
        await search_service.initialize()
        logger.info("Startup completed successfully!")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.get("/products", response_model=List[Product])
async def get_products():
    try:
        with open("app/data/catalog.json", "r") as f:
            catalog = json.load(f)
        return catalog
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[Product])
async def search(
    query: SearchQuery,
    user_id: Optional[int] = Header(None),
    authorization: Optional[str] = Header(None)
):
    try:
        # Extract token from authorization header
        token = None
        if authorization and authorization.startswith("Token "):
            token = authorization.split(" ")[1]

        search_results = await search_service.search_with_auth(
            query_type=query.query_type,
            query=query.query,
            num_results=query.num_results,
            min_similarity=query.min_similarity,
            user_preferences=query.preferences,
            user_id=user_id,
            auth_token=token
        )
        
        return [result.product for result in search_results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# @app.post("/search", response_model=List[Product])
# async def search(query: SearchQuery):
#     if not search_service:
#         raise HTTPException(status_code=500, detail="Search service not initialized")

#     try:
#         search_results = search_service.search(
#             query_type=query.query_type,
#             query=query.query,
#             num_results=query.num_results,
#             min_similarity=query.min_similarity,
#             user_preferences=query.preferences
#         )

#         # Extract just the products from the search results
#         products = [result.product for result in search_results]
#         return products

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

@app.post("/search/image", response_model=List[Product])
async def image_search(
    file: UploadFile = File(...),
    preferences: str = Form(None),
    num_results: int = 5,
    min_similarity: float = 0.0
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

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
            query_type=SearchType.IMAGE,
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
