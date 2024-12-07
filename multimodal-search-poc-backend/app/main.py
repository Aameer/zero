# app/main.py
import urllib3
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Union
import json
import time
import logging
from typing import List, Optional
from fastapi import Query, HTTPException


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
    allow_origins=["http://localhost:3000", "http://localhost:8081", "http://localhost:19006", "https://app.zerotab.app"],
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
        
        # Import settings and pass BACKEND_URL to EnhancedSearchService
        from app.config.settings import settings
        logger.info(f"Using backend URL: {settings.BACKEND_URL}")
        
        search_service = EnhancedSearchService(
            catalog=catalog,
            base_url=settings.BACKEND_URL
        )
        
        logger.info("Search service created, initializing indexes...")
        await search_service.initialize()
        logger.info("Startup completed successfully!")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is up and running"}

@app.get("/")
async def home():
    return {
        "service": "Search API",
        "status": "operational",
        "message": "Welcome to Search API service"
    }

@app.get("/products", response_model=List[Product])
async def get_products():
    try:
        with open("app/data/catalog.json", "r") as f:
            catalog = json.load(f)
        return catalog
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products", response_model=List[Product])
async def get_products(
    skip: int = Query(default=0, ge=0, description="Number of products to skip"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of products to return")
):
    try:
        with open("app/data/catalog.json", "r") as f:
            catalog = json.load(f)
        return catalog[skip : skip + limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Text Search Endpoints
@app.post("/search", response_model=List[Product])
async def search(
    query: SearchQuery,
    authorization: Optional[str] = Header(None)
):
    try:
        token = None
        if authorization and authorization.startswith("Token "):
            token = authorization.split(" ")[1]
            
        search_results = await search_service.search_with_auth(
            query_type=query.query_type,
            query=query.query,
            num_results=query.num_results,
            min_similarity=query.min_similarity,
            user_preferences=query.preferences,
            user_id=query.user_id,
            auth_token=token
        )

        return [result.product for result in search_results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/detailed", response_model=List[SearchResult])
async def detailed_search(
    query: SearchQuery,
    authorization: Optional[str] = Header(None)
):
    if not search_service:
        raise HTTPException(status_code=500, detail="Search service not initialized")

    try:
        token = None
        if authorization and authorization.startswith("Token "):
            token = authorization.split(" ")[1]

        return await search_service.search_with_auth(
            query_type=query.query_type,
            query=query.query,
            num_results=query.num_results,
            min_similarity=query.min_similarity,
            user_preferences=query.preferences,
            user_id=query.user_id,
            auth_token=token
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Image Search Endpoints
@app.post("/search/image", response_model=List[Product])
async def image_search(
    file: UploadFile = File(...),
    preferences: str = Form(None),
    num_results: int = Form(10),
    min_similarity: float = Form(0.0),
    user_id: Optional[int] = Form(None),
    authorization: Optional[str] = Header(None)
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

        # Extract token from authorization header
        token = None
        if authorization and authorization.startswith("Token "):
            token = authorization.split(" ")[1]

        contents = await file.read()
        search_results = await search_service.search_with_auth(
            query_type=SearchType.IMAGE,
            query=contents,
            num_results=num_results,
            min_similarity=min_similarity,
            user_preferences=user_preferences,
            user_id=user_id,
            auth_token=token
        )

        return [result.product for result in search_results]

    except Exception as e:
        logger.error(f"Error in image search: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search/image/detailed", response_model=List[SearchResult])
async def detailed_image_search(
    file: UploadFile = File(...),
    preferences: str = Form(None),
    num_results: int = Form(10),
    min_similarity: float = Form(0.0),
    user_id: Optional[int] = Form(None),
    authorization: Optional[str] = Header(None)
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

        # Extract token from authorization header
        token = None
        if authorization and authorization.startswith("Token "):
            token = authorization.split(" ")[1]

        contents = await file.read()
        return await search_service.search_with_auth(
            query_type=SearchType.IMAGE,
            query=contents,
            num_results=num_results,
            min_similarity=min_similarity,
            user_preferences=user_preferences,
            user_id=user_id,
            auth_token=token
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Audio Search Endpoints
@app.post("/search/audio", response_model=List[Product])
async def audio_search(
    file: UploadFile = File(...),
    preferences: str = Form(None),
    num_results: int = Form(10),
    min_similarity: float = Form(0.0),
    user_id: Optional[int] = Form(None),
    authorization: Optional[str] = Header(None)
):
    allowed_audio_types = [
        'audio/mpeg',        # .mp3
        'audio/mp3',         # alternative for .mp3
        'audio/wav',         # .wav
        'audio/wave',        # alternative for .wav
        'audio/x-wav',       # alternative for .wav
        'audio/webm',        # .webm
        'audio/aac',         # .aac
        'audio/ogg',         # .ogg
        'audio/x-m4a',       # .m4a
        'audio/mp4',         # .mp4 audio
        'application/octet-stream'  # Generic binary data
    ]

    content_type = file.content_type or 'application/octet-stream'
    file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''

    is_valid_extension = file_extension in ['mp3', 'wav', 'aac', 'ogg', 'webm', 'm4a', 'mp4']
    is_valid_mime = any(
        allowed_type in content_type.lower()
        for allowed_type in allowed_audio_types
    )

    if not (is_valid_extension or is_valid_mime):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format. Supported formats: mp3, wav, aac, ogg, webm, m4a, mp4. Got content-type: {content_type}"
        )

    try:
        user_preferences = None
        if preferences:
            try:
                preferences_dict = json.loads(preferences)
                user_preferences = UserPreferences(**preferences_dict)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid preferences JSON")

        token = None
        if authorization and authorization.startswith("Token "):
            token = authorization.split(" ")[1]

        contents = await file.read()
        logger.info(f"Processing audio file: size={len(contents)}, type={content_type}")

        search_results = await search_service.search_with_auth(
            query_type=SearchType.AUDIO,
            query=contents,
            num_results=num_results,
            min_similarity=min_similarity,
            user_preferences=user_preferences,
            user_id=user_id,
            auth_token=token
        )

        return [result.product for result in search_results]

    except Exception as e:
        logger.error(f"Error in audio search: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search/audio/detailed", response_model=List[SearchResult])
async def detailed_audio_search(
    file: UploadFile = File(...),
    preferences: str = Form(None),
    num_results: int = Form(10),
    min_similarity: float = Form(0.0),
    user_id: Optional[int] = Form(None),
    authorization: Optional[str] = Header(None)
):
    allowed_audio_types = [
        'audio/mpeg',
        'audio/mp3',
        'audio/wav',
        'audio/wave',
        'audio/x-wav',
        'audio/aac',
        'audio/ogg',
        'audio/webm',
        'audio/x-m4a',
        'audio/mp4'
    ]

    file_extension = file.filename.lower().split('.')[-1]
    is_valid_extension = file_extension in ['mp3', 'wav', 'aac', 'ogg', 'webm', 'm4a', 'mp4']
    is_valid_mime = file.content_type in allowed_audio_types

    if not (is_valid_extension or is_valid_mime):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format. Supported formats: mp3, wav, aac, ogg, webm, m4a, mp4. Got content-type: {file.content_type}"
        )

    try:
        user_preferences = None
        if preferences:
            try:
                preferences_dict = json.loads(preferences)
                user_preferences = UserPreferences(**preferences_dict)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid preferences JSON")

        token = None
        if authorization and authorization.startswith("Token "):
            token = authorization.split(" ")[1]

        contents = await file.read()
        return await search_service.search_with_auth(
            query_type=SearchType.AUDIO,
            query=contents,
            num_results=num_results,
            min_similarity=min_similarity,
            user_preferences=user_preferences,
            user_id=user_id,
            auth_token=token
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))