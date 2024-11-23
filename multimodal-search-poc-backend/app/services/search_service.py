# app/services/search_service.py
# Standard library imports
import ssl
import io
import json
import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from collections import defaultdict
from typing import List, Dict, Optional, Union, Tuple, Any, cast

# Third-party imports
from pydantic import BaseModel, Field
import numpy as np
import torch
import faiss
import aiohttp
import requests
import librosa
import soundfile as sf
import Levenshtein
from PIL import Image
from tqdm import tqdm
from scipy.io import wavfile
from tqdm.asyncio import tqdm_asyncio
from sentence_transformers import SentenceTransformer
from transformers import (
    CLIPProcessor,
    CLIPModel,
    WhisperProcessor,
    WhisperForConditionalGeneration
)

# Local imports
from app.models.schemas import SearchType, SearchResult, Product, UserPreferences
from app.config.search_config import SearchConfig  # Make sure this import works
from app.utils.cache import CacheManager
from app.config.settings import settings
# Set up logger
logger = logging.getLogger(__name__)

# You can also create constants for these values to avoid SearchConfig references
IMAGE_BATCH_SIZE = 16
MAX_IMAGES_PER_PRODUCT = 3

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

class StoredPreferences(BaseModel):
    """Model for stored user preferences from API"""
    brand_affinities: Dict[str, float] = Field(default_factory=dict)
    color_preferences: List[str] = Field(default_factory=list)
    size_preferences: List[str] = Field(default_factory=list)
    fabric_preferences: List[str] = Field(default_factory=list)
    category_preferences: Dict[str, float] = Field(default_factory=dict)
    price_range: Optional[Dict[str, float]] = None
    style_preferences: List[str] = Field(default_factory=list)
    seasonal_preferences: List[str] = Field(default_factory=list)
    purchase_history_categories: Dict[str, int] = Field(default_factory=dict)

class PreferencesFetcher:
    """Service to fetch user preferences from API"""
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
        logger.info(f"Initialized PreferencesFetcher with base URL: {base_url}")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_user_preferences(self, user_id: int, token: str) -> Optional[StoredPreferences]:
        """Fetch user preferences from API"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            headers = {
                "Authorization": f"Token {token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            async with self.session.get(
                f"{self.base_url}/api/users/{user_id}/",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    shopping_prefs = data.get("shopping_preferences", {})
                    logger.info(f"Retrieved preferences for user {user_id}: {shopping_prefs}")
                    return StoredPreferences(**shopping_prefs)
                else:
                    logger.error(f"Failed to fetch preferences: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Error fetching user preferences: {str(e)}")
            return None
        finally:
            if self.session:
                await self.session.close()
                self.session = None

class EnhancedSearchService:
    def __init__(self, catalog: List[Dict], base_url: str = "http://localhost:8000"):
        """Initialize search service with API integration"""
        self.catalog = catalog
        self.products = [Product(**p) for p in catalog]
        
        # Initialize preferences fetcher with base_url
        logger.info(f"Initializing EnhancedSearchService with base URL: {base_url}")
        self.preferences_fetcher = PreferencesFetcher(base_url)
        
        # Initialize cache manager
        from app.config.settings import settings
        self.cache_manager = CacheManager(settings)
        
        # Set batch size constants
        self.IMAGE_BATCH_SIZE = getattr(SearchConfig, 'IMAGE_BATCH_SIZE', 16)
        self.MAX_IMAGES_PER_PRODUCT = getattr(SearchConfig, 'MAX_IMAGES_PER_PRODUCT', 3)
        
        # Initialize models
        self._initialize_models()
    
    def _convert_stored_to_user_preferences(
        self,
        stored_prefs: StoredPreferences
    ) -> UserPreferences:
        """Convert stored preferences to search preferences"""
        price_range = None
        if stored_prefs.price_range:
            price_range = (
                stored_prefs.price_range["min"],
                stored_prefs.price_range["max"]
            )

        return UserPreferences(
            brand_weights=stored_prefs.brand_affinities,
            price_range=price_range,
            preferred_colors=stored_prefs.color_preferences,
            category_weights=stored_prefs.category_preferences,
            size_preference=stored_prefs.size_preferences,
            fabric_preference=stored_prefs.fabric_preferences,
            seasonal_preference=stored_prefs.seasonal_preferences[0] if stored_prefs.seasonal_preferences else None
        )
    
    def _calculate_user_affinity_score(
        self,
        product: Product,
        stored_preferences: StoredPreferences
    ) -> float:
        """Calculate affinity score with enhanced historical data"""
        affinity_score = 1.0

        if not stored_preferences:
            return affinity_score

        # Brand affinity with purchase history boost
        if product.brand in stored_preferences.brand_affinities:
            affinity_score *= (1 + stored_preferences.brand_affinities[product.brand])

        # Category affinity with purchase history
        if product.category in stored_preferences.category_preferences:
            base_score = stored_preferences.category_preferences[product.category]
            # Boost based on purchase history
            purchase_count = stored_preferences.purchase_history_categories.get(product.category, 0)
            history_boost = min(purchase_count * 0.1, 0.5)  # Cap the boost at 0.5
            affinity_score *= (1 + base_score + history_boost)

        # Style affinity
        product_styles = [attr['Style'] for attr in product.attributes if 'Style' in attr]
        if any(style in stored_preferences.style_preferences for style in product_styles):
            affinity_score *= 1.2

        # Seasonal relevance
        if stored_preferences.seasonal_preferences:
            product_season = next((attr['Season'] for attr in product.attributes if 'Season' in attr), None)
            if product_season in stored_preferences.seasonal_preferences:
                affinity_score *= 1.3

        return affinity_score

    async def initialize(self):
        """Async initialization method with enhanced caching"""
        try:
            logger.info("Starting async initialization...")

            # Check if we need to refresh embeddings
            refresh_needed = self.cache_manager.should_refresh_embeddings(self.catalog)
            
            if not refresh_needed:
                cached_embeddings = self.cache_manager.load_embeddings('multimodal', self.catalog)
                if cached_embeddings is not None:
                    logger.info("Loading embeddings from cache...")
                    text_embeddings = cached_embeddings['text_embeddings']
                    clip_embeddings = cached_embeddings['image_embeddings']
                    
                    # Create combined embeddings
                    self.combined_embeddings = np.hstack([text_embeddings, clip_embeddings])
                    self.combined_embeddings = np.ascontiguousarray(self.combined_embeddings, dtype=np.float32)
                    
                    # Create FAISS index
                    self.combined_index = faiss.IndexFlatIP(self.combined_dimension)
                    if len(self.combined_embeddings) > 0:
                        self.combined_index.add(self.combined_embeddings)
                    
                    logger.info("Successfully loaded cached embeddings")
                    return

            logger.info("Computing new embeddings...")
            
            # Create text embeddings
            product_texts = []
            for p in self.products:
                attribute_text = ' '.join([
                    f"{key} {value}"
                    for attr in p.attributes
                    for key, value in attr.items()
                ])
                text = f"{p.title} {p.brand} {p.description} {attribute_text}"
                product_texts.append(text)

            # Process text embeddings in batches
            batch_size = 32
            text_embeddings = []
            for i in range(0, len(product_texts), batch_size):
                batch = product_texts[i:i + batch_size]
                batch_embeddings = self.text_model.encode(batch)
                text_embeddings.append(batch_embeddings)
            
            text_embeddings = np.vstack(text_embeddings).astype(np.float32)
            
            # Cache images and compute CLIP embeddings
            logger.info("Processing and caching images...")
            await self.cache_manager.cache_images_batch(self.products)
            
            # Process images with cached files
            clip_embeddings = []
            for product in self.products:
                image_path = self.cache_manager.get_image_path(product.id)
                if image_path.exists():
                    try:
                        image = Image.open(image_path).convert('RGB')
                        inputs = self.clip_processor(images=image, return_tensors="pt")
                        with torch.no_grad():
                            image_embedding = self.clip_model.get_image_features(**inputs)
                        clip_embeddings.append(image_embedding[0].cpu().numpy())
                    except Exception as e:
                        logger.error(f"Error processing cached image for {product.id}: {e}")
                        clip_embeddings.append(np.zeros(self.clip_dimension))
                else:
                    clip_embeddings.append(np.zeros(self.clip_dimension))
            
            clip_embeddings = np.vstack(clip_embeddings).astype(np.float32)
            
            # Save embeddings to cache
            self.cache_manager.save_embeddings(
                {
                    'text_embeddings': text_embeddings,
                    'image_embeddings': clip_embeddings
                },
                'multimodal',
                self.catalog
            )
            
            # Create combined embeddings and index
            self.combined_embeddings = np.hstack([text_embeddings, clip_embeddings])
            self.combined_embeddings = np.ascontiguousarray(self.combined_embeddings, dtype=np.float32)
            
            # Create FAISS index
            self.combined_index = faiss.IndexFlatIP(self.combined_dimension)
            if len(self.combined_embeddings) > 0:
                self.combined_index.add(self.combined_embeddings)
            
            logger.info("Search service initialization complete!")
            
        except Exception as e:
            logger.error(f"Error in initialization: {e}")
            raise

    def _initialize_models(self):
        """Initialize all required models with error handling"""
        try:
            logger.info("Starting model initialization...")

            logger.info("Downloading and initializing text model...")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.text_dimension = 384

            logger.info("Downloading and initializing CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_dimension = 512

            logger.info("Downloading and initializing Whisper model...")
            self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            self.audio_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            self.combined_dimension = self.text_dimension + self.clip_dimension
            logger.info("All models initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    async def _init_multimodal_indexes(self):
        """Initialize combined text and image embeddings index with actual images"""
        try:
            logger.info("Creating multimodal embeddings...")
            logger.info(f"Processing {len(self.products)} products...")

            # Try to load embeddings from cache
            cached_embeddings = self.cache_manager.load_embeddings('multimodal')

            if cached_embeddings is not None:
                logger.info("Using cached embeddings")
                text_embeddings = cached_embeddings['text_embeddings']
                clip_embeddings = cached_embeddings['image_embeddings']
                logger.info("Successfully loaded embeddings from cache")
            else:
                logger.info("Computing new embeddings...")
                # Create text embeddings with attribute awareness
                product_texts = []
                for p in self.products:
                    attribute_text = ' '.join([
                        f"{key} {value}"
                        for attr in p.attributes
                        for key, value in attr.items()
                    ])
                    text = f"{p.title} {p.brand} {p.description} {attribute_text}"
                    product_texts.append(text)

                logger.info(f"Computing text embeddings for {len(product_texts)} products...")

                # Process text embeddings in batches with progress bar
                batch_size = 32
                text_embeddings = []

                with tqdm(total=len(product_texts), desc="Computing text embeddings") as pbar:
                    for i in range(0, len(product_texts), batch_size):
                        batch = product_texts[i:i + batch_size]
                        batch_embeddings = self.text_model.encode(batch)
                        text_embeddings.append(batch_embeddings)
                        pbar.update(len(batch))

                text_embeddings = np.vstack(text_embeddings).astype(np.float32)
                logger.info("Text embeddings computed successfully")

                # Process images with progress tracking
                logger.info("Starting image processing...")
                batch_size = self.IMAGE_BATCH_SIZE
                total_batches = (len(self.products) + batch_size - 1) // batch_size

                image_url_batches = [
                    [p.image_url for p in self.products[i:i + batch_size]]
                    for i in range(0, len(self.products), batch_size)
                ]

                # Create progress bar for image batches
                pbar = tqdm(
                    total=len(image_url_batches),
                    desc="Processing image batches",
                    unit="batch"
                )

                all_clip_embeddings = []
                async def process_image_batch(batch, batch_num):
                    embeddings = await self._process_single_batch(batch, batch_num, total_batches)
                    pbar.update(1)
                    pbar.set_postfix({"Batch": f"{batch_num}/{total_batches}"})
                    return embeddings

                # Process batches with progress tracking
                try:
                    tasks = [
                        process_image_batch(batch, i+1)
                        for i, batch in enumerate(image_url_batches)
                    ]
                    all_clip_embeddings = await asyncio.gather(*tasks)
                finally:
                    pbar.close()

                if not all_clip_embeddings:
                    logger.warning("No image embeddings were generated, using zeros")
                    clip_embeddings = np.zeros((len(self.products), self.clip_dimension), dtype=np.float32)
                else:
                    clip_embeddings = np.vstack(all_clip_embeddings)

                # Save embeddings to cache
                logger.info("Saving embeddings to cache...")
                self.cache_manager.save_embeddings(
                    {
                        'text_embeddings': text_embeddings,
                        'image_embeddings': clip_embeddings
                    },
                    'multimodal',
                    catalog_data=self.catalog
                )
                logger.info("Embeddings saved to cache")

            # Ensure all embeddings are float32 and contiguous
            text_embeddings = np.ascontiguousarray(text_embeddings, dtype=np.float32)
            clip_embeddings = np.ascontiguousarray(clip_embeddings, dtype=np.float32)

            # Normalize embeddings
            logger.info("Normalizing embeddings...")
            if len(text_embeddings) > 0:
                text_embeddings_copy = text_embeddings.copy()
                faiss.normalize_L2(text_embeddings_copy)
                text_embeddings = text_embeddings_copy

            if len(clip_embeddings) > 0:
                clip_embeddings_copy = clip_embeddings.copy()
                faiss.normalize_L2(clip_embeddings_copy)
                clip_embeddings = clip_embeddings_copy

            # Create combined embeddings
            logger.info("Creating combined embeddings...")
            self.combined_embeddings = np.hstack([text_embeddings, clip_embeddings])
            self.combined_embeddings = np.ascontiguousarray(self.combined_embeddings, dtype=np.float32)

            # Create FAISS index
            logger.info("Creating FAISS index...")
            self.combined_index = faiss.IndexFlatIP(self.combined_dimension)
            if len(self.combined_embeddings) > 0:
                self.combined_index.add(self.combined_embeddings)
                logger.info("FAISS index created successfully")

            logger.info("Multimodal index creation completed successfully!")

        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to create search indexes: {str(e)}")

    async def _process_single_batch(self, image_urls_batch, batch_num, total_batches):
        """Process a single batch of images"""
        batch_embeddings = []
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(ssl=False)

        # Create progress bar for images within the batch
        batch_pbar = tqdm(
            total=len(image_urls_batch),
            desc=f"Batch {batch_num}/{total_batches}",
            leave=False,
            unit="img"
        )

        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                for i, urls in enumerate(image_urls_batch):
                    product_embeddings = []
                    for url_obj in urls[:self.MAX_IMAGES_PER_PRODUCT]:
                        for attempt in range(3):
                            try:
                                url_str = str(url_obj)
                                if '?' in url_str:
                                    base_url = url_str.split('?')[0]
                                    url_str = f"{base_url}?quality=80"

                                async with session.get(url_str) as response:
                                    if response.status == 200:
                                        image_data = await response.read()
                                        image = Image.open(io.BytesIO(image_data)).convert('RGB')
                                        inputs = self.clip_processor(images=image, return_tensors="pt")

                                        with torch.no_grad():
                                            image_embedding = self.clip_model.get_image_features(**inputs)
                                        product_embeddings.append(image_embedding[0].cpu().numpy())
                                        break
                                    else:
                                        logger.warning(f"Failed to fetch image, status: {response.status}")

                            except Exception as e:
                                logger.error(f"Error processing image URL {url_str}: {str(e)}")
                                if attempt == 2:
                                    continue
                                await asyncio.sleep(1)

                    if product_embeddings:
                        avg_embedding = np.mean(product_embeddings, axis=0)
                        batch_embeddings.append(avg_embedding.astype(np.float32))
                    else:
                        batch_embeddings.append(np.zeros(self.clip_dimension, dtype=np.float32))

                    batch_pbar.update(1)

        finally:
            batch_pbar.close()
        return np.array(batch_embeddings, dtype=np.float32)
    def _init_attribute_indexes(self):
        """Initialize separate indexes for specific attributes"""
        self.attribute_embeddings = {}
        self.attribute_indexes = {}

        # Create separate indexes for important attributes
        attributes_to_index = ['Size', 'Color', 'Fabric', 'Season']

        for attr in attributes_to_index:
            # Collect all unique values for this attribute
            values = set()
            for product in self.products:
                for attribute in product.attributes:
                    if attr in attribute:
                        values.add(attribute[attr])

            # Create embeddings for attribute values
            if values:
                embeddings = self.text_model.encode(list(values))
                self.attribute_embeddings[attr] = embeddings

                # Create FAISS index for this attribute
                index = faiss.IndexFlatIP(self.text_dimension)
                faiss.normalize_L2(embeddings)
                index.add(embeddings.astype('float32'))
                self.attribute_indexes[attr] = index

    @lru_cache(maxsize=1000)
    def _get_cached_embeddings(self, key: str, compute_func) -> np.ndarray:
        """Get embeddings from cache or compute them"""
        if key not in self.cache:
            self.cache[key] = compute_func()
        return self.cache[key]

    def _process_product_images(self, image_urls: List[str]) -> np.ndarray:
        """Process product images and return combined embedding"""
        image_embeddings = []

        for url in image_urls[:3]:  # Process up to 3 images per product
            try:
                response = requests.get(url)
                image = Image.open(io.BytesIO(response.content))
                inputs = self.clip_processor(images=image, return_tensors="pt")

                with torch.no_grad():
                    image_embedding = self.clip_model.get_image_features(**inputs)
                image_embeddings.append(image_embedding[0].cpu().numpy())
            except Exception as e:
                logger.error(f"Error processing image URL {url}: {str(e)}")
                continue

        if not image_embeddings:
            return np.zeros(self.clip_dimension)

        # Combine multiple image embeddings
        combined = np.mean(image_embeddings, axis=0)
        return combined

    def _diversify_results(self, results: List[Tuple[int, float]],
                          diversity_threshold: float = 0.3) -> List[Tuple[int, float]]:
        """Ensure results aren't too similar to each other"""
        diversified = []
        for idx, score in results:
            if not diversified:
                diversified.append((idx, score))
                continue

            # Check similarity with existing results
            product_embedding = self.combined_embeddings[idx]
            is_diverse = True

            for div_idx, _ in diversified:
                div_embedding = self.combined_embeddings[div_idx]
                similarity = np.dot(product_embedding, div_embedding)

                if similarity > diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                diversified.append((idx, score))

        return diversified
    
    def _get_attribute_matches(
        self,
        product: Product,
        preferences: Optional[UserPreferences]
    ) -> Dict[str, float]:
        """Calculate how well each product attribute matches the user preferences"""
        matches = {}
        
        if not preferences:
            return matches

        # Brand match
        if preferences.brand_weights and product.brand in preferences.brand_weights:
            matches['brand'] = preferences.brand_weights[product.brand]

        # Price match
        if preferences.price_range:
            min_price, max_price = preferences.price_range
            if min_price <= product.price <= max_price:
                matches['price'] = 1.0
            else:
                matches['price'] = 0.0

        # Color match with semantic matching
        if preferences.preferred_colors:
            product_colors = [
                attr['Color'] for attr in product.attributes
                if 'Color' in attr
            ]
            color_matches = []
            for preferred_color in preferences.preferred_colors:
                similar_colors = SearchWeights.COLOR_SIMILARITY.get(preferred_color.lower(), [])
                if any(color.lower() in [preferred_color.lower()] + similar_colors
                    for color in product_colors):
                    color_matches.append(1.0)
                else:
                    color_matches.append(0.0)
            if color_matches:
                matches['color'] = max(color_matches)  # Use the best color match

        # Category match
        if preferences.category_weights and product.category in preferences.category_weights:
            matches['category'] = preferences.category_weights[product.category]

        # Size match
        if preferences.size_preference:
            product_sizes = [
                attr['Size'] for attr in product.attributes
                if 'Size' in attr
            ]
            if any(size in preferences.size_preference for size in product_sizes):
                matches['size'] = 1.0
            else:
                matches['size'] = 0.0

        # Fabric match
        if preferences.fabric_preference:
            product_fabrics = [
                attr['Fabric'] for attr in product.attributes
                if 'Fabric' in attr
            ]
            if any(fabric in preferences.fabric_preference for fabric in product_fabrics):
                matches['fabric'] = 1.0
            else:
                matches['fabric'] = 0.0

        # Season match
        if preferences.seasonal_preference:
            product_season = next(
                (attr['Season'] for attr in product.attributes if 'Season' in attr),
                None
            )
            if product_season and product_season.upper() == preferences.seasonal_preference.upper():
                matches['season'] = 1.0
            else:
                matches['season'] = 0.0

        # Calculate average match score
        if matches:
            matches['overall'] = sum(matches.values()) / len(matches)

        return matches

    def _combine_preferences(
        self,
        stored_preferences: Optional[UserPreferences],
        request_preferences: Optional[UserPreferences]
    ) -> UserPreferences:
        """Combine stored user preferences with request-specific preferences"""
        combined_prefs = UserPreferences(
            brand_weights={},
            price_range=None,
            preferred_colors=[],
            category_weights={},
            seasonal_preference=None,
            size_preference=[],
            fabric_preference=[]
        )

        if stored_preferences:
            # Combine brand weights with stored weight factor
            for brand, weight in (stored_preferences.brand_weights or {}).items():
                combined_prefs.brand_weights[brand] = weight * SearchWeights.BASE_WEIGHTS['brand']

            # Add stored price range if no request price range
            if stored_preferences.price_range:
                combined_prefs.price_range = stored_preferences.price_range

            # Add stored colors
            if stored_preferences.preferred_colors:
                combined_prefs.preferred_colors.extend(stored_preferences.preferred_colors)

            # Combine category weights with stored weight factor
            for category, weight in (stored_preferences.category_weights or {}).items():
                combined_prefs.category_weights[category] = weight * SearchWeights.BASE_WEIGHTS['category']

            # Add stored seasonal preference if no request preference
            if stored_preferences.seasonal_preference:
                combined_prefs.seasonal_preference = stored_preferences.seasonal_preference

            # Add stored size preferences
            if stored_preferences.size_preference:
                combined_prefs.size_preference.extend(stored_preferences.size_preference)

            # Add stored fabric preferences
            if stored_preferences.fabric_preference:
                combined_prefs.fabric_preference.extend(stored_preferences.fabric_preference)

        if request_preferences:
            # Combine brand weights with request weight factor
            for brand, weight in (request_preferences.brand_weights or {}).items():
                current_weight = combined_prefs.brand_weights.get(brand, 0)
                combined_prefs.brand_weights[brand] = max(
                    current_weight,
                    weight * SearchWeights.BASE_WEIGHTS['brand']
                )

            # Request price range overrides stored price range
            if request_preferences.price_range:
                combined_prefs.price_range = request_preferences.price_range

            # Add request colors
            if request_preferences.preferred_colors:
                combined_prefs.preferred_colors.extend(request_preferences.preferred_colors)

            # Combine category weights with request weight factor
            for category, weight in (request_preferences.category_weights or {}).items():
                current_weight = combined_prefs.category_weights.get(category, 0)
                combined_prefs.category_weights[category] = max(
                    current_weight,
                    weight * SearchWeights.BASE_WEIGHTS['category']
                )

            # Request seasonal preference overrides stored preference
            if request_preferences.seasonal_preference:
                combined_prefs.seasonal_preference = request_preferences.seasonal_preference

            # Add request size preferences
            if request_preferences.size_preference:
                combined_prefs.size_preference.extend(request_preferences.size_preference)

            # Add request fabric preferences
            if request_preferences.fabric_preference:
                combined_prefs.fabric_preference.extend(request_preferences.fabric_preference)

        # Remove duplicates while preserving order
        combined_prefs.preferred_colors = list(dict.fromkeys(combined_prefs.preferred_colors))
        combined_prefs.size_preference = list(dict.fromkeys(combined_prefs.size_preference))
        combined_prefs.fabric_preference = list(dict.fromkeys(combined_prefs.fabric_preference))

        return combined_prefs

    def _apply_preferences(
        self,
        base_scores: np.ndarray,
        indices: np.ndarray,
        preferences: UserPreferences
    ) -> List[tuple[int, float]]:
        """Apply enhanced user preferences with multiple factors"""
        if not preferences:
            return [(idx, score) for score, idx in zip(base_scores[0], indices[0])]

        results = []
        current_season = SeasonalWeights.get_current_season()

        for score, idx in zip(base_scores[0], indices[0]):
            if idx >= len(self.products):
                continue

            product = self.products[idx]
            final_score = float(score)  # Base similarity score
            boost_multiplier = 1.0  # Start with neutral multiplier

            # Apply weighted preferences as boosters
            weights = SearchWeights.BASE_WEIGHTS

            # Brand preference
            if preferences.brand_weights and product.brand in preferences.brand_weights:
                logger.info(f"boosting {preferences.brand_weights[product.brand]} with {weights['brand']}, boost_multiplier :{boost_multiplier}")
                brand_boost = preferences.brand_weights[product.brand] * weights['brand']
                boost_multiplier += brand_boost

            # Price range preference - soft boost instead of strict filtering
            if preferences.price_range:
                min_price, max_price = preferences.price_range
                if min_price <= product.price <= max_price:
                    boost_multiplier += weights['price']
                else:
                    # Small penalty for being outside range, but don't exclude
                    price_distance = min(abs(product.price - min_price), abs(product.price - max_price))
                    price_penalty = min(0.5, price_distance / max_price)  # Cap the penalty
                    boost_multiplier *= (1.0 - price_penalty)

            # Color preference with semantic matching
            if preferences.preferred_colors:
                product_colors = [
                    attr['Color'] for attr in product.attributes
                    if 'Color' in attr
                ]

                for preferred_color in preferences.preferred_colors:
                    similar_colors = SearchWeights.COLOR_SIMILARITY.get(preferred_color.lower(), [])
                    if any(color.lower() in [preferred_color.lower()] + similar_colors
                        for color in product_colors):
                        boost_multiplier += weights['color']
                        break  # Only boost once for color match

            # Category preference
            if preferences.category_weights and product.category in preferences.category_weights:
                category_boost = preferences.category_weights[product.category] * weights['category']
                boost_multiplier += category_boost

            # Seasonal boost
            product_season = next((attr['Season'] for attr in product.attributes
                                if 'Season' in attr), None)
            if product_season and product_season.upper() == current_season:
                boost_multiplier += SeasonalWeights.SEASONS[current_season]['boost'] - 1.0  # Adjust boost to be additive

            # Apply final boost multiplier to score
            final_score *= boost_multiplier

            results.append((idx, final_score))

        # Sort by final score but keep all results
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Only apply diversity to maintain order but not remove items
        diversity_threshold = 0.95  # Higher threshold to be more lenient
        diversified_order = []
        remaining = set(range(len(results)))
        
        # First, add highly similar items
        while remaining:
            max_score = -1
            best_idx = None
            
            for i in remaining:
                idx, score = results[i]
                if score > max_score:
                    max_score = score
                    best_idx = i
                    
            if best_idx is not None:
                diversified_order.append(results[best_idx])
                remaining.remove(best_idx)
        
        return diversified_order

    def _expand_query(self, query: str) -> str:
        """Expand query with related terms"""
        # Add color variations
        expanded_terms = [query]

        # Add color synonyms
        for color, variations in SearchWeights.COLOR_SIMILARITY.items():
            if color in query.lower():
                expanded_terms.extend(variations)

        # Add size variations (S -> Small, etc.)
        size_mappings = {
            'S': 'Small',
            'M': 'Medium',
            'L': 'Large',
            'XL': 'Extra Large'
        }

        for abbrev, full in size_mappings.items():
            if abbrev in query.upper():
                expanded_terms.append(full)

        return ' '.join(expanded_terms)
    
    async def search_with_auth(
        self,
        query_type: SearchType,
        query: Union[str, bytes],
        num_results: int = 5,
        min_similarity: float = 0.0,
        user_preferences: Optional[UserPreferences] = None,
        user_id: Optional[int] = None,
        auth_token: Optional[str] = None
    ) -> List[SearchResult]:
        """Enhanced search incorporating API-fetched preferences"""
        stored_preferences = None

        #logger.info(f"Main search checking if user_id and auth_token are present?")
        if user_id and auth_token:
            #logger.info(f" found user id : {user_id} and auth token : {auth_token}\n")
            async with PreferencesFetcher(base_url=settings.BACKEND_URL) as fetcher:
                stored_prefs = await fetcher.get_user_preferences(user_id, auth_token)
                if stored_prefs:
                    stored_preferences = self._convert_stored_to_user_preferences(stored_prefs)
                    #logger.info(f"gotten stored preferences {stored_preferences}\n")

        # Combine stored and request preferences
        # TODO: combined preferences arent working well - for now diabling them - get from users stored preferences only.
        # combined_preferences = self._combine_preferences(stored_preferences, user_preferences)
        # logger.info(f"gotten combined preferences {combined_preferences}\n")

        # Use the combined preferences in search
        return self.search(
            query_type=query_type,
            query=query,
            num_results=num_results,
            min_similarity=min_similarity,
            user_preferences=stored_preferences
        )
    
    def search(
        self,
        query_type: SearchType,
        query: Union[str, bytes],
        num_results: int = 5,
        min_similarity: float = 0.0,
        user_preferences: Optional[UserPreferences] = None
    ) -> List[SearchResult]:
        """Enhanced multimodal search with query expansion and result diversification"""
        try:
            logger.info(f"\n-->Searching with preferences: {user_preferences}\n")
        
            # Get raw search results based on query type
            if query_type == SearchType.TEXT:
                expanded_query = self._expand_query(query)
                # Get more initial results to allow for filtering
                initial_results = self._text_search(expanded_query, num_results * 4)
            elif query_type == SearchType.IMAGE:
                initial_results = self._image_search(query, num_results * 4)
            elif query_type == SearchType.AUDIO:
                transcription = self._transcribe_audio(query)
                expanded_query = self._expand_query(transcription)
                initial_results = self._text_search(expanded_query, num_results * 4)
            else:
                raise ValueError(f"Unsupported search type: {query_type}")
            
            logger.info(f"initial results: {initial_results}")
            # Apply preferences if available
            if user_preferences:
                # Convert results to numpy arrays for _apply_preferences
                base_scores = np.array([[score for _, score in initial_results]])
                indices = np.array([[idx for idx, _ in initial_results]])
                
                # Get weighted results
                weighted_results = self._apply_preferences(base_scores, indices, user_preferences)
            else:
                weighted_results = initial_results

            logger.info(f"weighted  results: {weighted_results}")
            # Create SearchResult objects
            search_results = []
            processed_indices = set()  # To avoid duplicates
            
            for idx, score in weighted_results:
                if (score >= min_similarity and 
                    idx < len(self.products) and 
                    idx not in processed_indices and 
                    len(search_results) < num_results):
                    
                    product = self.products[idx]
                    processed_indices.add(idx)
                    
                    search_results.append(
                        SearchResult(
                            product=product,
                            similarity_score=score,
                            attribute_matches=self._get_attribute_matches(
                                product, user_preferences
                            ) if user_preferences else None
                        )
                    )

            logger.info(f"search  results: {search_results}")
            # Sort by final score while maintaining diversity
            search_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.info(f"Found {len(search_results)} results after applying preferences")
            return search_results[:num_results]  # Ensure we don't exceed requested number

        except Exception as e:
                logger.error(f"Error in search: {str(e)}")
                raise

    # [Previous methods _text_search, _image_search, _transcribe_audio remain unchanged]
    def _text_search(self, query: str, num_results: int) -> List[tuple[int, float]]:
        """Perform text search using combined embeddings"""
        try:
            # Create text embedding
            query_text_embedding = self.text_model.encode([query])

            # Create CLIP embedding from text
            clip_inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
            with torch.no_grad():
                query_clip_embedding = self.clip_model.get_text_features(**clip_inputs)
                query_clip_embedding = query_clip_embedding.cpu().numpy()

            # Normalize embeddings
            faiss.normalize_L2(query_text_embedding)
            faiss.normalize_L2(query_clip_embedding)

            # Combine embeddings
            query_embedding = np.hstack([query_text_embedding, query_clip_embedding])

            # Search combined index
            similarities, indices = self.combined_index.search(
                query_embedding.astype('float32'),
                num_results
            )
            # print(
            #     ">"*100, 
            #     len(
            #         list(
            #             zip(
            #                 indices[0],
            #                 similarities[0]
            #             )
            #         )
            #     )
            # )
            return list(zip(indices[0], similarities[0]))

        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            raise

    def _image_search(self, image_data: bytes, num_results: int) -> List[tuple[int, float]]:
        """Perform image search using combined embeddings"""
        try:
            image = Image.open(io.BytesIO(image_data))

            # Get CLIP image features
            clip_inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**clip_inputs)

            # Extract features from image using just CLIP
            image_features = image_features.cpu().numpy()
            faiss.normalize_L2(image_features)

            # Create a query embedding that matches our index dimensions
            # Pad with zeros where we would have had text embeddings
            text_padding = np.zeros((1, self.text_dimension), dtype=np.float32)
            query_embedding = np.hstack([text_padding, image_features])

            # Search combined index
            similarities, indices = self.combined_index.search(
                query_embedding.astype('float32'),
                num_results
            )

            return list(zip(indices[0], similarities[0]))

        except Exception as e:
            logger.error(f"Error in image search: {str(e)}")
            raise

    def _transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio to text, handling different audio formats including WebM"""
        try:
            # First try to determine the format and convert if necessary
            try:
                # Try to load with librosa which handles multiple formats
                import io
                import soundfile as sf
                import numpy as np
                from pydub import AudioSegment
                import tempfile

                # Create a temporary file to save the audio
                with tempfile.NamedTemporaryFile(suffix='.webm', delete=True) as temp_webm:
                    temp_webm.write(audio_bytes)
                    temp_webm.flush()

                    # Convert WebM to WAV using pydub
                    audio = AudioSegment.from_file(temp_webm.name)

                    # Create another temporary file for WAV
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
                        audio.export(temp_wav.name, format='wav')

                        # Now load the WAV file with librosa
                        audio_array, sample_rate = librosa.load(
                            temp_wav.name,
                            sr=16000  # Whisper expects 16kHz
                        )

            except Exception as e:
                # If conversion fails, try direct loading
                logger.warning(f"WebM conversion failed, trying direct loading: {str(e)}")
                audio_array, sample_rate = librosa.load(
                    io.BytesIO(audio_bytes),
                    sr=16000
                )

            # Convert to float32 and normalize
            audio_array = audio_array.astype(np.float32)

            # Create input features for Whisper
            input_features = self.audio_processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features

            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.audio_model.generate(input_features)
                transcription = self.audio_processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]

            logger.info(f"Transcribed Audio: {transcription}")
            return transcription

        except Exception as e:
            logger.error(f"Error in audio transcription: {str(e)}")
            raise

    def _preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess audio data for the model"""
        try:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                from scipy import signal
                number_of_samples = round(len(audio_data) * 16000 / sample_rate)
                audio_data = signal.resample(audio_data, number_of_samples)

            return audio_data

        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            raise
