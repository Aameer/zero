# app/utils/cache.py
import os
import pickle
import numpy as np
from pathlib import Path
import requests
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, settings):
        self.settings = settings
        self._init_directories()

    def _init_directories(self):
        """Initialize all required directories"""
        directories = [
            self.settings.CACHE_DIR,
            self.settings.IMAGE_CACHE_DIR,
            self.settings.EMBEDDING_CACHE_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_image_path(self, product_id: str) -> Path:
        """Get the path where an image should be cached"""
        return self.settings.IMAGE_CACHE_DIR / f"{product_id}.jpg"

    def get_embedding_path(self, embedding_type: str) -> Path:
        """Get the path where embeddings should be cached"""
        return self.settings.EMBEDDING_CACHE_DIR / f"{embedding_type}_embeddings.pkl"

    def cache_image(self, product_id: str, image_url: str) -> Path:
        """Download and cache an image"""
        image_path = self.get_image_path(product_id)

        if not image_path.exists():
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()

                image = Image.open(io.BytesIO(response.content)).convert('RGB')
                image.save(image_path)
                logger.info(f"Cached image for product {product_id}")
            except Exception as e:
                logger.error(f"Error caching image for product {product_id}: {str(e)}")
                return None

        return image_path

    def save_embeddings(self, embeddings: np.ndarray, embedding_type: str):
        """Save embeddings to cache"""
        path = self.get_embedding_path(embedding_type)
        with open(path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"Saved {embedding_type} embeddings to cache")

    def load_embeddings(self, embedding_type: str) -> np.ndarray:
        """Load embeddings from cache"""
        path = self.get_embedding_path(embedding_type)
        if path.exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
