# app/config/search_config.py
from typing import Dict

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

    # Color similarity mappings
    COLOR_SIMILARITY: Dict[str, list] = {
        'pink': ['rose', 'magenta', 'fuchsia'],
        'red': ['maroon', 'crimson', 'scarlet'],
        'blue': ['navy', 'azure', 'cobalt', 'turquoise'],
        'green': ['olive', 'lime', 'emerald', 'sage'],
        'purple': ['violet', 'lavender', 'mauve', 'plum'],
        'yellow': ['gold', 'mustard', 'amber'],
        'brown': ['tan', 'beige', 'khaki', 'chocolate'],
        'black': ['jet', 'ebony', 'onyx'],
        'white': ['ivory', 'cream', 'pearl', 'off-white'],
        'grey': ['silver', 'charcoal', 'slate']
    }

    # Seasonal configurations
    SEASONAL_BOOSTS = {
        'SPRING': {'months': [3, 4, 5], 'boost': 1.3},
        'SUMMER': {'months': [6, 7, 8], 'boost': 1.3},
        'FALL': {'months': [9, 10, 11], 'boost': 1.3},
        'WINTER': {'months': [12, 1, 2], 'boost': 1.3}
    }

    # Search parameters
    DEFAULT_NUM_RESULTS = 5
    MIN_SIMILARITY_THRESHOLD = 0.0
    DIVERSITY_THRESHOLD = 0.3
    MAX_QUERY_EXPANSION_TERMS = 5

    # Cache settings
    EMBEDDING_CACHE_SIZE = 1000
    IMAGE_CACHE_SIZE = 500

    # Image processing
    MAX_IMAGES_PER_PRODUCT = 3  # Maximum number of images to process per product
    IMAGE_BATCH_SIZE = 16       # Number of images to process in one batch
