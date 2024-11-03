from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import torch
import numpy as np
from typing import List, Dict, Optional, Union
import io
from PIL import Image
import logging
from app.models.schemas import SearchType, SearchResult, Product, UserPreferences

logger = logging.getLogger(__name__)

class ImageBindSearchService:
    def __init__(self, catalog: List[Dict]):
        """Initialize search service with ImageBind model"""
        self.catalog = catalog
        self.products = [Product(**p) for p in catalog]

        # Initialize ImageBind
        logger.info("Initializing ImageBind model...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

        # Initialize embeddings
        self._init_product_embeddings()

    def _init_product_embeddings(self):
        """Initialize unified embeddings for all products"""
        logger.info("Creating unified embeddings...")
        self.embeddings = []
        batch_size = 32

        for i in range(0, len(self.products), batch_size):
            batch = self.products[i:i + batch_size]
            batch_embeddings = self._create_product_embeddings(batch)
            self.embeddings.extend(batch_embeddings)

        self.embeddings = np.vstack(self.embeddings)

        # Create FAISS index
        logger.info("Creating FAISS index...")
        self.embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        normalized_embeddings = self.embeddings.copy()
        faiss.normalize_L2(normalized_embeddings)
        self.index.add(normalized_embeddings.astype('float32'))

    def _create_product_embeddings(self, products: List[Product]) -> np.ndarray:
        """Create unified embeddings for a batch of products"""
        inputs = {
            ModalityType.TEXT: [],
            ModalityType.VISION: [],
        }

        for product in products:
            # Prepare text input
            text = f"{product.title} {product.brand} {product.description}"
            inputs[ModalityType.TEXT].append(text)

            # Prepare image input
            try:
                response = requests.get(product.image_url[0])
                image = Image.open(io.BytesIO(response.content))
                inputs[ModalityType.VISION].append(self._preprocess_image(image))
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                inputs[ModalityType.VISION].append(torch.zeros((3, 224, 224)))

        # Convert inputs to tensors
        inputs = {
            modal: data.load_and_transform_data(modal, inputs[modal], device=self.device)
            for modal in inputs
        }

        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(inputs)

        # Combine embeddings from different modalities
        combined = torch.cat([
            embeddings[ModalityType.TEXT],
            embeddings[ModalityType.VISION]
        ], dim=1)

        return combined.cpu().numpy()

    def search(
        self,
        query_type: SearchType,
        query: Union[str, bytes],
        num_results: int = 5,
        min_similarity: float = 0.0,
        user_preferences: Optional[UserPreferences] = None
    ) -> List[SearchResult]:
        """Perform unified multimodal search"""
        try:
            # Process query based on type
            if query_type == SearchType.TEXT:
                query_embedding = self._process_text_query(query)
            elif query_type == SearchType.IMAGE:
                query_embedding = self._process_image_query(query)
            elif query_type == SearchType.AUDIO:
                text_query = self._transcribe_audio(query)
                query_embedding = self._process_text_query(text_query)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")

            # Search index
            faiss.normalize_L2(query_embedding)
            similarities, indices = self.index.search(
                query_embedding.astype('float32'),
                num_results
            )

            # Create results
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if similarity >= min_similarity and idx < len(self.products):
                    results.append(SearchResult(
                        product=self.products[idx],
                        similarity_score=float(similarity)
                    ))

            return results

        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            raise

    def _process_text_query(self, query: str) -> np.ndarray:
        """Process text query using ImageBind"""
        inputs = {
            ModalityType.TEXT: [query]
        }
        inputs = {
            modal: data.load_and_transform_data(modal, inputs[modal], device=self.device)
            for modal in inputs
        }

        with torch.no_grad():
            embeddings = self.model(inputs)
        return embeddings[ModalityType.TEXT].cpu().numpy()

    def _process_image_query(self, image_data: bytes) -> np.ndarray:
        """Process image query using ImageBind"""
        image = Image.open(io.BytesIO(image_data))
        inputs = {
            ModalityType.VISION: [self._preprocess_image(image)]
        }
        inputs = {
            modal: data.load_and_transform_data(modal, inputs[modal], device=self.device)
            for modal in inputs
        }

        with torch.no_grad():
            embeddings = self.model(inputs)
        return embeddings[ModalityType.VISION].cpu().numpy()

    def get_embeddings(self, product_ids: List[str]) -> np.ndarray:
        """Get embeddings for specific products - used for evaluation"""
        indices = [
            idx for idx, product in enumerate(self.products)
            if str(product.id) in product_ids
        ]
        return self.embeddings[indices]
