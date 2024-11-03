# app/config/search_config.py

# app/config/search_config.py

class SearchConfig:
    # Model configurations
    TEXT_MODEL = 'all-MiniLM-L6-v2'
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    WHISPER_MODEL = "openai/whisper-base"

    # Search weights
    WEIGHT_FACTORS = {
        'similarity': 1.0,
        'brand': 0.8,
        'price': 0.6,
        'color': 0.7,
        'category': 0.5,
        'seasonal': 0.4,
        'attribute': 0.3
    }

    # Search parameters
    DEFAULT_NUM_RESULTS = 5
    MIN_SIMILARITY_THRESHOLD = 0.0
    DIVERSITY_THRESHOLD = 0.3
    MAX_QUERY_EXPANSION_TERMS = 5

    # Image processing
    IMAGE_BATCH_SIZE = 16       # Number of images to process in one batch
    MAX_IMAGES_PER_PRODUCT = 3  # Maximum number of images to process per product

    # Cache settings
    EMBEDDING_CACHE_SIZE = 1000
    IMAGE_CACHE_SIZE = 500
