# app/utils/cache.py
import os
import pickle
import numpy as np
from pathlib import Path
import requests
import aiohttp
import asyncio
from PIL import Image
import io
import logging
import json
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, settings):
        self.settings = settings
        self.metadata_path = self.settings.CACHE_DIR / "cache_metadata.json"
        self._init_directories()
        self.metadata = self._load_metadata()

    def _init_directories(self):
        """Initialize all required directories"""
        directories = [
            self.settings.CACHE_DIR,
            self.settings.IMAGE_CACHE_DIR,
            self.settings.EMBEDDING_CACHE_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self):
        """Load cache metadata"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache metadata: {e}")
                return {}
        return {}

    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")

    def _generate_cache_key(self, data) -> str:
        """Generate a cache key based on the data"""
        if isinstance(data, list):
            # For product data, create a hash of relevant fields
            content = json.dumps(data, sort_keys=True)
            return hashlib.md5(content.encode()).hexdigest()
        return hashlib.md5(str(data).encode()).hexdigest()

    def get_image_path(self, product_id: str) -> Path:
        """Get the path where an image should be cached"""
        return self.settings.IMAGE_CACHE_DIR / f"{product_id}.jpg"

    def get_embedding_path(self, embedding_type: str) -> Path:
        """Get the path where embeddings should be cached"""
        return self.settings.EMBEDDING_CACHE_DIR / f"{embedding_type}_embeddings.npz"

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
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            # Clear image cache
            for file in self.settings.IMAGE_CACHE_DIR.glob("*"):
                file.unlink()

            # Clear embeddings cache
            for file in self.settings.EMBEDDING_CACHE_DIR.glob("*"):
                file.unlink()

            # Clear metadata
            self.metadata = {}
            self._save_metadata()

            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def should_refresh_embeddings(self, catalog_data) -> bool:
        """Determine if embeddings need to be refreshed based on settings and catalog changes"""
        if getattr(self.settings, 'REFRESH_EMBEDDINGS', False):
            logger.info("Forced refresh of embeddings due to REFRESH_EMBEDDINGS setting")
            return True

        current_hash = self._generate_cache_key(catalog_data)
        stored_hash = self.metadata.get('catalog_hash')

        if stored_hash != current_hash:
            logger.info("Catalog data has changed, refreshing embeddings required")
            return True

        # Check if we have valid embeddings
        if not (self.settings.EMBEDDING_CACHE_DIR / "multimodal_embeddings.npz").exists():
            logger.info("No cached embeddings found")
            return True

        return False

    async def cache_images_batch(self, products, batch_size=16):
        """Cache images in batches with retry logic"""
        async def download_single_image(session, url, product_id):
            image_path = self.get_image_path(product_id)
            if image_path.exists():
                return True

            for attempt in range(3):
                try:
                    async with session.get(url, timeout=30) as response:
                        if response.status == 200:
                            data = await response.read()
                            image = Image.open(io.BytesIO(data)).convert('RGB')
                            image.save(image_path)
                            return True
                except Exception as e:
                    if attempt == 2:
                        logger.error(f"Failed to download image for {product_id} from {url}: {e}")
                    await asyncio.sleep(1)
            return False

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=False),
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            tasks = []
            for product in products:
                if isinstance(product.image_url, list) and product.image_url:
                    url = str(product.image_url[0])
                    if '?' in url:
                        url = f"{url.split('?')[0]}?quality=80"
                    tasks.append(download_single_image(session, url, product.id))
                
                if len(tasks) >= batch_size:
                    await asyncio.gather(*tasks)
                    tasks = []
            
            if tasks:
                await asyncio.gather(*tasks)

    def save_embeddings(self, embeddings_dict: dict, embedding_type: str, catalog_data=None):
        """Enhanced save embeddings with catalog hash"""
        try:
            path = self.get_embedding_path(embedding_type)
            np.savez_compressed(path, **embeddings_dict)

            # Update metadata with catalog hash
            cache_key = self._generate_cache_key(catalog_data) if catalog_data else embedding_type
            self.metadata[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'embedding_type': embedding_type,
                'num_items': len(next(iter(embeddings_dict.values()))),
                'path': str(path)
            }
            self.metadata['catalog_hash'] = cache_key
            self._save_metadata()

            logger.info(f"Saved {embedding_type} embeddings to cache with catalog hash")
        except Exception as e:
            logger.error(f"Error saving embeddings to cache: {e}")

    def load_embeddings(self, embedding_type: str, catalog_data=None) -> dict:
        """Enhanced load embeddings with catalog hash check"""
        try:
            path = self.get_embedding_path(embedding_type)
            if not path.exists():
                return None

            if catalog_data:
                current_hash = self._generate_cache_key(catalog_data)
                stored_hash = self.metadata.get('catalog_hash')
                if stored_hash != current_hash:
                    logger.info("Catalog has changed, cached embeddings invalid")
                    return None

            cache_key = self._generate_cache_key(catalog_data) if catalog_data else embedding_type
            metadata = self.metadata.get(cache_key, {})

            if metadata:
                cache_time = datetime.fromisoformat(metadata.get('timestamp', '2000-01-01'))
                if (datetime.now() - cache_time).days < 7:
                    with np.load(path) as data:
                        return {k: data[k] for k in data.files}
                else:
                    logger.info(f"Cache for {embedding_type} is too old")
            return None

        except Exception as e:
            logger.error(f"Error loading embeddings from cache: {e}")
            return None

