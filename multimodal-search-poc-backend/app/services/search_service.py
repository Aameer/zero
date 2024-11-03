# app/services/search_service.py
from sentence_transformers import SentenceTransformer
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import faiss
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import io
from PIL import Image
from scipy.io import wavfile
import logging
import aiohttp
from datetime import datetime
from functools import lru_cache
import json
from collections import defaultdict
import Levenshtein  # for fuzzy matching
import soundfile as sf
import librosa
import asyncio
import requests
import gc
from app.models.schemas import SearchType, SearchResult, Product, UserPreferences

# Add this after the existing imports (around line 22)
def release_memory():
    """Helper function to release memory"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Rest of the imports and SeasonalWeights/SearchWeights classes remain the same
logger = logging.getLogger(__name__)

class SeasonalWeights:
    """Seasonal weightings for product relevance"""
    SEASONS = {
        'SPRING': {'months': [3, 4, 5], 'boost': 1.3},
        'SUMMER': {'months': [6, 7, 8], 'boost': 1.3},
        'FALL': {'months': [9, 10, 11], 'boost': 1.3},
        'WINTER': {'months': [12, 1, 2], 'boost': 1.3}
    }

    @staticmethod
    def get_current_season() -> str:
        current_month = datetime.now().month
        for season, data in SeasonalWeights.SEASONS.items():
            if current_month in data['months']:
                return season
        return 'NONE'

class SearchWeights:
    """Configuration for various search weight factors"""
    BASE_WEIGHTS = {
        'similarity': 1.0,
        'brand': 0.8,
        'price': 0.6,
        'color': 0.7,
        'category': 0.5,
        'seasonal': 0.4,
        'attribute': 0.3
    }

    COLOR_SIMILARITY = {
        'pink': ['rose', 'magenta', 'fuchsia'],
        'red': ['maroon', 'crimson', 'scarlet'],
        'blue': ['navy', 'azure', 'cobalt'],
        # Add more color mappings
    }

class EnhancedSearchService:
    def __init__(self, catalog: List[Dict]):
        """Initialize search service with minimal setup"""
        self.catalog = catalog[:2]  # Start with just 2 items
        self.products = [Product(**p) for p in self.catalog]

        # Initialize state flags
        self._models_loaded = False
        self._indexes_created = False
        self.is_initialized = False

        # Initialize dimensions
        self.text_dimension = 384
        self.clip_dimension = 512
        self.combined_dimension = self.text_dimension + self.clip_dimension

        # Initialize holders
        self.text_model = None
        self.clip_model = None
        self.clip_processor = None
        self.audio_model = None
        self.audio_processor = None
        self.combined_embeddings = None
        self.combined_index = None
        self.attribute_embeddings = {}
        self.attribute_indexes = {}

    async def initialize(self):
        """Main initialization sequence"""
        if self.is_initialized:
            return

        try:
            # Load models first
            await self._load_models()

            # Clean up memory after loading models
            release_memory()

            # Initialize indexes
            await self._init_multimodal_indexes()

            self.is_initialized = True
            logger.info("Service initialization complete!")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            await self.cleanup()
            raise

    @classmethod
    async def create(cls, catalog: List[Dict]):
        """Factory method for creating service instance"""
        service = cls(catalog)
        try:
            await service.initialize()
            return service
        except Exception as e:
            await service.cleanup()
            raise RuntimeError(f"Failed to create service: {str(e)}")

    async def _load_models(self):
        """Load AI models with careful memory management"""
        if self._models_loaded:
            return

        try:
            # Load text model
            logger.info("Loading text model...")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            release_memory()

            # Load CLIP model
            logger.info("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            release_memory()

            # Load audio model
            logger.info("Loading audio model...")
            self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            self.audio_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            release_memory()

            self._models_loaded = True
            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            await self.cleanup()
            raise

    async def _init_multimodal_indexes(self):
        """Initialize indexes with careful memory management"""
        if self._indexes_created:
            return

        try:
            logger.info("Creating multimodal embeddings...")

            # Process text embeddings
            text_embeddings = []
            for product in self.products:
                try:
                    text = f"{product.title} {product.brand} {product.description}"
                    text_emb = self.text_model.encode([text])[0]
                    text_embeddings.append(text_emb.astype(np.float32))
                    release_memory()
                except Exception as e:
                    logger.error(f"Error processing text for product {product.id}: {e}")
                    text_embeddings.append(np.zeros(self.text_dimension, dtype=np.float32))

            # Process image embeddings
            clip_embeddings = []
            async with aiohttp.ClientSession() as session:
                for product in self.products:
                    try:
                        if product.image_url:
                            async with session.get(str(product.image_url[0]), timeout=10) as response:
                                if response.status == 200:
                                    image_data = await response.read()
                                    image = Image.open(io.BytesIO(image_data))
                                    inputs = self.clip_processor(images=image, return_tensors="pt")

                                    with torch.no_grad():
                                        features = self.clip_model.get_image_features(**inputs)
                                        clip_emb = features[0].cpu().numpy()
                                        clip_embeddings.append(clip_emb.astype(np.float32))

                                    # Clean up
                                    del image_data
                                    del image
                                    del inputs
                                    del features
                                else:
                                    clip_embeddings.append(np.zeros(self.clip_dimension, dtype=np.float32))
                        else:
                            clip_embeddings.append(np.zeros(self.clip_dimension, dtype=np.float32))
                    except Exception as e:
                        logger.error(f"Error processing image for product {product.id}: {e}")
                        clip_embeddings.append(np.zeros(self.clip_dimension, dtype=np.float32))
                    release_memory()

            # Convert to arrays and normalize
            text_embeddings = np.vstack(text_embeddings)
            clip_embeddings = np.vstack(clip_embeddings)

            faiss.normalize_L2(text_embeddings)
            faiss.normalize_L2(clip_embeddings)

            # Create combined embeddings
            self.combined_embeddings = np.hstack([text_embeddings, clip_embeddings])
            del text_embeddings
            del clip_embeddings
            release_memory()

            # Create FAISS index
            logger.info("Creating FAISS index...")
            self.combined_index = faiss.IndexFlatL2(self.combined_dimension)
            self.combined_index.add(self.combined_embeddings.astype(np.float32))

            self._indexes_created = True
            logger.info("Index creation complete")

        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
            self.combined_embeddings = None
            self.combined_index = None
            self._indexes_created = False
            raise
        finally:
            release_memory()

    async def search(
        self,
        query_type: SearchType,
        query: Union[str, bytes],
        num_results: int = 5,
        min_similarity: float = 0.0,
        user_preferences: Optional[UserPreferences] = None
    ) -> List[SearchResult]:
        """Enhanced multimodal search with memory management"""
        if not self.is_initialized:
            raise RuntimeError("Service not initialized")

        try:
            # Get search results based on type
            if query_type == SearchType.TEXT:
                expanded_query = self._expand_query(query)
                results = await self._text_search(expanded_query, num_results)
            elif query_type == SearchType.IMAGE:
                results = await self._image_search(query, num_results)
            elif query_type == SearchType.AUDIO:
                transcription = await self._transcribe_audio(query)
                expanded_query = self._expand_query(transcription)
                results = await self._text_search(expanded_query, num_results)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")

            # Create search results
            search_results = []
            for idx, similarity in results:
                if similarity >= min_similarity and idx < len(self.products):
                    search_results.append(
                        SearchResult(
                            product=self.products[idx],
                            similarity_score=similarity
                        )
                    )

            release_memory()
            return search_results

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    async def _text_search(self, query: str, num_results: int) -> List[Tuple[int, float]]:
        """Text search with memory management"""
        try:
            query_text_embedding = self.text_model.encode([query])

            clip_inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
            with torch.no_grad():
                query_clip_embedding = self.clip_model.get_text_features(**clip_inputs)
                query_clip_embedding = query_clip_embedding.cpu().numpy()

            # Create combined query
            query_embedding = np.hstack([
                query_text_embedding.astype(np.float32),
                query_clip_embedding.astype(np.float32)
            ])
            faiss.normalize_L2(query_embedding)

            # Search
            D, I = self.combined_index.search(query_embedding, num_results)

            results = list(zip(I[0], D[0]))
            release_memory()
            return results

        except Exception as e:
            logger.error(f"Text search error: {str(e)}")
            raise

    async def cleanup(self):
        """Enhanced cleanup with thorough memory management"""
        try:
            # Clear models
            self.text_model = None
            self.clip_model = None
            self.clip_processor = None
            self.audio_model = None
            self.audio_processor = None

            # Clear embeddings and indexes
            self.combined_embeddings = None
            self.combined_index = None
            self.attribute_embeddings.clear()
            self.attribute_indexes.clear()

            # Reset flags
            self._models_loaded = False
            self._indexes_created = False
            self.is_initialized = False

            # Force memory cleanup
            release_memory()
            logger.info("Cleanup completed successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

    def __del__(self):
        """Destructor with safe cleanup"""
        try:
            logger.info("Cleaning up search service...")
            release_memory()
        except Exception as e:
            logger.error(f"Error in destructor: {str(e)}")

    # Keep your existing helper methods (_expand_query, _diversify_results, etc.)
    # but add release_memory() calls where appropriate

    def _expand_query(self, query: str) -> str:
        """Query expansion helper"""
        # Existing implementation remains the same
        pass

    def _diversify_results(self, results: List[Tuple[int, float]],
                          diversity_threshold: float = 0.3) -> List[Tuple[int, float]]:
        """Result diversification with memory management"""
        try:
            # Existing implementation remains the same
            diversified = []
            for idx, score in results:
                # ... existing diversification logic ...
                pass
            return diversified
        finally:
            release_memory()

    def get_embeddings(self, product_ids: List[str]) -> np.ndarray:
        """Get embeddings with memory management"""
        if not self.is_initialized:
            raise RuntimeError("Service not initialized")

        try:
            indices = [
                idx for idx, product in enumerate(self.products)
                if str(product.id) in product_ids
            ]
            return self.combined_embeddings[indices].copy()
        finally:
            release_memory()
