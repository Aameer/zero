# app/main.py
import urllib3
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json
import time

from app.models.schemas import Product, SearchQuery, SearchResponse, SearchType
from app.services.search_service import SearchService

app = FastAPI(title="Multimodal Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
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
        with open("app/data/catalog.json", "r") as f:
            catalog = json.load(f)
        search_service = SearchService(catalog)
    except Exception as e:
        print(f"Error during startup: {e}")
        raise

@app.get("/products", response_model=List[Product])
async def get_products():
    try:
        with open("app/data/catalog.json", "r") as f:
            catalog = json.load(f)
        return catalog
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    if not search_service:
        raise HTTPException(status_code=500, detail="Search service not initialized")

    start_time = time.time()
    try:
        results = search_service.search(
            query_type=query.query_type,
            query=query.query,
            num_results=query.num_results,
            min_similarity=query.min_similarity
        )
        search_time = time.time() - start_time

        return SearchResponse(
            results=results,
            total_results=len(results),
            search_time=search_time
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search/image")
async def image_search(
    file: UploadFile = File(...),
    num_results: int = 5,
    min_similarity: float = 0.0
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    start_time = time.time()
    try:
        contents = await file.read()
        results = search_service.image_search(contents, num_results, min_similarity)
        search_time = time.time() - start_time

        return SearchResponse(
            results=results,
            total_results=len(results),
            search_time=search_time
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search/audio")
async def audio_search(
    file: UploadFile = File(...),
    num_results: int = 5,
    min_similarity: float = 0.0
):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio")

    start_time = time.time()
    try:
        contents = await file.read()
        results = search_service.audio_search(contents, num_results, min_similarity)
        search_time = time.time() - start_time

        return SearchResponse(
            results=results,
            total_results=len(results),
            search_time=search_time
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

