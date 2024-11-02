# app/services/search_service.py
from sentence_transformers import SentenceTransformer
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import faiss
import numpy as np
from typing import List, Dict, Optional, Union
import io
from PIL import Image
from scipy.io import wavfile
import logging
from dataclasses import dataclass
import soundfile as sf
import io
import librosa

from app.models.schemas import SearchType, SearchResult, Product, UserPreferences

logger = logging.getLogger(__name__)

class EnhancedSearchService:
    def __init__(self, catalog: List[Dict]):
        """Initialize the enhanced search service with multiple models and indexes"""
        self.catalog = catalog
        self.products = [Product(**p) for p in catalog]

        # Initialize models with error handling
        self._initialize_models()

        # Initialize indexes
        self._init_multimodal_indexes()
        logger.info("Search service initialization complete!")

    def _initialize_models(self):
        """Initialize all required models with proper error handling"""
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

    def _init_multimodal_indexes(self):
        """Initialize combined text and image embeddings index"""
        try:
            logger.info("Creating multimodal embeddings...")

            # Create text embeddings
            product_texts = [
                f"{p.title} {p.brand} {p.description} {' '.join([str(attr) for attr in p.attributes])}"
                for p in self.products
            ]
            text_embeddings = self.text_model.encode(product_texts)

            # Create CLIP embeddings from product descriptions and images
            clip_embeddings = []
            for product in self.products:
                try:
                    # Process text descriptions with CLIP
                    inputs = self.clip_processor(
                        text=[product.description],
                        return_tensors="pt",
                        padding=True
                    )
                    with torch.no_grad():
                        clip_embedding = self.clip_model.get_text_features(**inputs)
                    clip_embeddings.append(clip_embedding[0].cpu().numpy())
                except Exception as e:
                    logger.error(f"Error processing product {product.id}: {str(e)}")
                    clip_embeddings.append(np.zeros(self.clip_dimension))

            # Stack and normalize embeddings
            clip_embeddings = np.vstack(clip_embeddings)
            faiss.normalize_L2(text_embeddings)
            faiss.normalize_L2(clip_embeddings)

            # Combine embeddings
            self.combined_embeddings = np.hstack([text_embeddings, clip_embeddings])

            # Create FAISS index
            self.combined_index = faiss.IndexFlatIP(self.combined_dimension)
            self.combined_index.add(self.combined_embeddings.astype('float32'))

            logger.info("Multimodal index creation complete")

        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
            raise RuntimeError(f"Failed to create search indexes: {str(e)}")

    def _apply_preferences(
        self,
        base_scores: np.ndarray,
        indices: np.ndarray,
        preferences: UserPreferences
    ) -> List[tuple[int, float]]:
        """Apply user preferences to modify search results ranking"""
        if not preferences:
            return [(idx, score) for score, idx in zip(base_scores[0], indices[0])]

        results = []
        for score, idx in zip(base_scores[0], indices[0]):
            if idx >= len(self.products):
                continue

            product = self.products[idx]
            preference_score = float(score)  # Start with base similarity score

            # Apply brand preferences
            if preferences.brand_weights and product.brand in preferences.brand_weights:
                preference_score *= (1 + preferences.brand_weights[product.brand])

            # Apply price range preferences
            if preferences.price_range:
                min_price, max_price = preferences.price_range
                if min_price <= product.price <= max_price:
                    preference_score *= 1.2

            # Apply color preferences
            if preferences.preferred_colors:
                product_colors = [
                    attr['Color'] for attr in product.attributes
                    if 'Color' in attr
                ]
                if any(color in preferences.preferred_colors for color in product_colors):
                    preference_score *= 1.1

            # Apply category preferences
            if preferences.category_weights and product.category in preferences.category_weights:
                preference_score *= (1 + preferences.category_weights[product.category])

            results.append((idx, preference_score))

        # Sort by final preference score
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search(
        self,
        query_type: SearchType,
        query:  Union[str, bytes],
        num_results: int = 5,
        min_similarity: float = 0.0,
        user_preferences: Optional[UserPreferences] = None
    ) -> List[SearchResult]:
        """
        Perform multimodal search with user preferences
        Returns list of SearchResult objects with products and similarity scores
        """
        try:
            # Get raw search results based on query type
            if query_type == SearchType.TEXT:
                results = self._text_search(query, num_results * 2)
            elif query_type == SearchType.IMAGE:
                results = self._image_search(query, num_results * 2)
            elif query_type == SearchType.AUDIO:
                transcription = self._transcribe_audio(query)
                results = self._text_search(transcription, num_results * 2)
            else:
                raise ValueError(f"Unsupported search type: {query_type}")

            # Filter by minimum similarity and create SearchResult objects
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

    # Then update the _transcribe_audio method in the EnhancedSearchService class:
    def _transcribe_audio(self, audio_bytes: bytes) -> str:
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
