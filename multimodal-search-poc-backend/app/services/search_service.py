# app/services/search_service.py
from sentence_transformers import SentenceTransformer
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import faiss
import numpy as np
from typing import List, Dict
import io
from PIL import Image
from scipy.io import wavfile

from app.models.schemas import SearchType, SearchResult, Product

class SearchService:
    def __init__(self, catalog: List[Dict]):
        self.catalog = catalog
        self.products = [Product(**p) for p in catalog]

        # Initialize text search
        print("Initializing text model...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_dimension = 384

        # Initialize image search
        print("Initializing image model...")
        self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Initialize audio transcription
        print("Initializing audio model...")
        self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.audio_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

        # Initialize FAISS indexes
        print("Initializing search indexes...")
        self._init_indexes()
        print("Search service initialization complete!")

    def _init_indexes(self):
        # Text embeddings
        print("Creating text embeddings...")
        #TODO: add more attributes here.
        product_texts = [
            f"{p.title} {p.brand} {p.description}"
            for p in self.products
        ]
        text_embeddings = self.text_model.encode(product_texts)
        self.text_index = faiss.IndexFlatIP(self.text_dimension)
        faiss.normalize_L2(text_embeddings)
        self.text_index.add(text_embeddings.astype('float32'))

        # Image embeddings using CLIP
        print("Creating image embeddings...")
        image_texts = [p.description for p in self.products]
        inputs = self.image_processor(text=image_texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.image_model.get_text_features(**inputs)

        # Initialize image index
        self.image_dimension = text_features.shape[1]
        self.image_index = faiss.IndexFlatIP(self.image_dimension)
        text_features = text_features.cpu().numpy()
        faiss.normalize_L2(text_features)
        self.image_index.add(text_features.astype('float32'))

    def search(self, query_type: SearchType, query: str, num_results: int = 5, min_similarity: float = 0.0) -> List[SearchResult]:
        try:
            if query_type == SearchType.TEXT:
                return self.text_search(query, num_results, min_similarity)
            elif query_type == SearchType.IMAGE:
                return self.image_search(query, num_results, min_similarity)
            elif query_type == SearchType.AUDIO:
                return self.audio_search(query, num_results, min_similarity)
            else:
                raise ValueError(f"Unsupported search type: {query_type}")
        except Exception as e:
            print(f"Error in search: {str(e)}")
            raise

    def text_search(self, query: str, num_results: int = 5, min_similarity: float = 0.0) -> List[SearchResult]:
        try:
            # Generate embedding for the query
            print(f"Processing text query: {query}")
            query_embedding = self.text_model.encode([query])
            faiss.normalize_L2(query_embedding)

            # Search the index
            similarities, indices = self.text_index.search(query_embedding.astype('float32'), num_results)

            # Filter and format results
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if similarity >= min_similarity and idx < len(self.products):
                    results.append(SearchResult(
                        product=self.products[idx],
                        similarity_score=float(similarity)
                    ))

            print(f"Found {len(results)} results for text search")
            return results
        except Exception as e:
            print(f"Error in text search: {str(e)}")
            raise

    def image_search(self, image_data: bytes, num_results: int = 5, min_similarity: float = 0.0) -> List[SearchResult]:
        try:
            # Process image
            print("Processing image query...")
            image = Image.open(io.BytesIO(image_data))
            inputs = self.image_processor(images=image, return_tensors="pt")

            # Generate image embedding
            with torch.no_grad():
                image_features = self.image_model.get_image_features(**inputs)

            # Normalize and search
            image_features = image_features.cpu().numpy()
            faiss.normalize_L2(image_features)

            similarities, indices = self.image_index.search(image_features.astype('float32'), num_results)

            # Filter and format results
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if similarity >= min_similarity and idx < len(self.products):
                    results.append(SearchResult(
                        product=self.products[idx],
                        similarity_score=float(similarity)
                    ))

            print(f"Found {len(results)} results for image search")
            return results
        except Exception as e:
            print(f"Error in image search: {str(e)}")
            raise

    def audio_search(self, audio_bytes: bytes, num_results: int = 5, min_similarity: float = 0.0) -> List[SearchResult]:
        try:
            print("Processing audio query...")
            # Convert audio bytes to tensor
            rate, audio_data = wavfile.read(io.BytesIO(audio_bytes))

            # Convert to float32 if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0

            # Prepare audio input for Whisper
            input_features = self.audio_processor(
                audio_data,
                sampling_rate=rate,
                return_tensors="pt"
            ).input_features

            # Generate transcription
            print("Transcribing audio...")
            with torch.no_grad():
                predicted_ids = self.audio_model.generate(input_features)
                transcription = self.audio_processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]

            print(f"Transcribed Audio: {transcription}")

            # Use the transcribed text for searching
            return self.text_search(transcription, num_results, min_similarity)

        except Exception as e:
            print(f"Error in audio search: {str(e)}")
            raise RuntimeError(f"Error processing audio: {str(e)}")

    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
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
            print(f"Error preprocessing audio: {str(e)}")
            raise
