# app/services/search_service.py
import asyncio
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
from dataclasses import dataclass
import soundfile as sf
import librosa
import requests
from datetime import datetime
from functools import lru_cache
import json
from collections import defaultdict
import Levenshtein  # for fuzzy matching

from app.models.schemas import SearchType, SearchResult, Product, UserPreferences

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
        """Initialize search service with multiple models and indexes"""
        self.catalog = catalog
        self.products = [Product(**p) for p in catalog]
        self.cache = {}  # Simple cache for embeddings

        # Initialize models with error handling
        self._initialize_models()

        # Initialize indexes and cache
        asyncio.run(self._init_async())  # Run async initialization
        logger.info("Search service initialization complete!")

    async def _init_async(self):
        """Async initialization wrapper"""
        await self._init_multimodal_indexes()
        self._init_attribute_indexes()

    def _initialize_models(self):
        """Initialize all required models with error handling"""
        try:
            logger.info("Initializing text model...")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.text_dimension = 384

            logger.info("Initializing CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_dimension = 512

            logger.info("Initializing audio model...")
            self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            self.audio_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            # Combined dimension for the unified embedding space
            self.combined_dimension = self.text_dimension + self.clip_dimension
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    async def _process_image_batch(self, session, products: List[Product]) -> List[np.ndarray]:
        """Process a batch of product images"""
        batch_embeddings = []
        for product in products:
            try:
                # Process first image only for now
                if product.image_url:
                    async with session.get(str(product.image_url[0])) as response:
                        if response.status == 200:
                            image_data = await response.read()
                            image = Image.open(io.BytesIO(image_data))
                            inputs = self.clip_processor(images=image, return_tensors="pt")

                            with torch.no_grad():
                                image_embedding = self.clip_model.get_image_features(**inputs)
                            batch_embeddings.append(image_embedding[0].cpu().numpy())
                        else:
                            batch_embeddings.append(np.zeros(self.clip_dimension))
                else:
                    batch_embeddings.append(np.zeros(self.clip_dimension))
            except Exception as e:
                logger.error(f"Error processing image for product {product.id}: {str(e)}")
                batch_embeddings.append(np.zeros(self.clip_dimension))

        return batch_embeddings

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

            # Apply weighted preferences
            weights = SearchWeights.BASE_WEIGHTS

            # Brand preference
            if preferences.brand_weights and product.brand in preferences.brand_weights:
                brand_boost = preferences.brand_weights[product.brand] * weights['brand']
                final_score *= (1 + brand_boost)

            # Price range preference
            if preferences.price_range:
                min_price, max_price = preferences.price_range
                if min_price <= product.price <= max_price:
                    final_score *= (1 + weights['price'])

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
                        final_score *= (1 + weights['color'])

            # Category preference
            if preferences.category_weights and product.category in preferences.category_weights:
                category_boost = preferences.category_weights[product.category] * weights['category']
                final_score *= (1 + category_boost)

            # Seasonal boost
            product_season = next((attr['Season'] for attr in product.attributes
                                 if 'Season' in attr), None)
            if product_season and product_season.upper() == current_season:
                final_score *= SeasonalWeights.SEASONS[current_season]['boost']

            results.append((idx, final_score))

        # Sort by final score and apply diversity
        results.sort(key=lambda x: x[1], reverse=True)
        return self._diversify_results(results)

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

    async def search(
        self,
        query_type: SearchType,
        query: Union[str, bytes],
        num_results: int = 5,
        min_similarity: float = 0.0,
        user_preferences: Optional[UserPreferences] = None
    ) -> List[SearchResult]:
        """Enhanced multimodal search with query expansion and result diversification"""
        try:
            # Get raw search results based on query type
            if query_type == SearchType.TEXT:
                expanded_query = self._expand_query(query)
                results = await self._text_search(expanded_query, num_results * 2)
            elif query_type == SearchType.IMAGE:
                results = await self._image_search(query, num_results * 2)
            elif query_type == SearchType.AUDIO:
                transcription = await self._transcribe_audio(query)
                expanded_query = self._expand_query(transcription)
                results = await self._text_search(expanded_query, num_results * 2)
            else:
                raise ValueError(f"Unsupported search type: {query_type}")

            # Apply preferences and create SearchResult objects
            search_results = []
            for idx, similarity in results[:num_results]:
                if similarity >= min_similarity and idx < len(self.products):
                    search_results.append(
                        SearchResult(
                            product=self.products[idx],
                            similarity_score=similarity
                        )
                    )

            return search_results

        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            raise

    async def _text_search(self, query: str, num_results: int) -> List[tuple[int, float]]:
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

            return list(zip(indices[0], similarities[0]))

        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            raise

# Fix 4: Make _image_search async
    async def _image_search(self, image_data: bytes, num_results: int) -> List[tuple[int, float]]:
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

# Fix 5: Make _transcribe_audio async
    async def _transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio to text, handling different audio formats"""
        try:
            # Convert audio bytes to numpy array
            try:
                # First try loading with librosa which handles multiple formats
                audio_array, sample_rate = librosa.load(
                    io.BytesIO(audio_bytes),
                    sr=16000,  # Whisper expects 16kHz
                    mono=True
                )
            except Exception as e:
                logger.error(f"Error loading audio with librosa: {str(e)}")
                raise RuntimeError(f"Could not process audio file: {str(e)}")

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
