# Multimodal Search API
# Copyright (c) 2024, Aameer Rafiq Wani
A FastAPI-based backend service that provides multimodal search capabilities for e-commerce products using state-of-the-art AI models.

## Features

- 🔍 Text-based semantic search using BERT embeddings
- 🖼️ Image similarity search using CLIP
- 🎤 Audio-to-text search using Whisper
- ⚡ Fast similarity search using FAISS
- 🔄 Real-time processing and response
- 📊 Similarity scores and rankings

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for better performance)
- 4GB+ RAM
- 2GB+ disk space for models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimodal-search-poc-backend.git
cd multimodal-search-poc-backend
```

2. Create virtual environment and Install dependencies 
```bash
python3.9 -m venv <virtual-env-name> # please note i have used python3.9.6, tried with python3.9.20 it was throwing some issues ( I am using mac with apple chip)
source <virtual-env-name>/bin/activate
pip install -r requirements.txt
```
3. How to run server
```bash
cd /Users/aameer/Documents/Aameer/eco/zero/multimodal-search-poc-backend
source ~/Documents/Aameer/eco/venv/bin/activate
uvicorn app.main:app --host localhost --port 9000
```


## Curls:
``` bash
# 1. TEXT SEARCH
# Regular text search with enhanced preferences
curl -X 'POST' \
  'http://localhost:9000/search' \
  -H 'Content-Type: application/json' \
  -d '{
  "query_type": "text",
  "query": "pink girl dress",
  "num_results": 5,
  "preferences": {
    "brand_weights": {"Zellbury": 0.8, "Junaid Jamshed": 0.6},
    "price_range": [1000, 5000],
    "preferred_colors": ["Pink", "Red"],
    "category_weights": {"Stitched": 0.7},
    "seasonal_preference": "SUMMER",
    "size_preference": ["M", "L"],
    "fabric_preference": ["Cotton", "Lawn"]
  },
  "filter_attributes": {
    "Size": ["M", "L"],
    "Fabric": ["Cotton", "Lawn"]
  }
}'

# Detailed text search with enhanced features
curl -X 'POST' \
  'http://localhost:9000/search/detailed' \
  -H 'Content-Type: application/json' \
  -d '{
  "query_type": "text",
  "query": "pink girl dress",
  "num_results": 5,
  "preferences": {
    "brand_weights": {"Zellbury": 0.8, "Junaid Jamshed": 0.6},
    "price_range": [1000, 5000],
    "preferred_colors": ["Pink", "Red"],
    "category_weights": {"Stitched": 0.7},
    "seasonal_preference": "SUMMER",
    "size_preference": ["M", "L"],
    "fabric_preference": ["Cotton", "Lawn"]
  },
  "filter_attributes": {
    "Size": ["M", "L"],
    "Fabric": ["Cotton", "Lawn"]
  }
}'

# 2. IMAGE SEARCH
# Regular image search with enhanced preferences
curl -X 'POST' \
  'http://localhost:9000/search/image' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/Users/aameer/Desktop/WPC2412841-1.jpg' \
  -F 'num_results=5' \
  -F 'preferences={
    "brand_weights":{"Zellbury":0.8,"Junaid Jamshed":0.6},
    "price_range":[1000,5000],
    "preferred_colors":["Pink","Red"],
    "category_weights":{"Stitched":0.7},
    "seasonal_preference":"SUMMER",
    "size_preference":["M","L"],
    "fabric_preference":["Cotton","Lawn"]
  }' \
  -F 'filter_attributes={"Size":["M","L"],"Fabric":["Cotton","Lawn"]}'

# Detailed image search with enhanced features
curl -X 'POST' \
  'http://localhost:9000/search/image/detailed' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/Users/aameer/Desktop/WPC2412841-1.jpg' \
  -F 'num_results=5' \
  -F 'preferences={
    "brand_weights":{"Zellbury":0.8,"Junaid Jamshed":0.6},
    "price_range":[1000,5000],
    "preferred_colors":["Pink","Red"],
    "category_weights":{"Stitched":0.7},
    "seasonal_preference":"SUMMER",
    "size_preference":["M","L"],
    "fabric_preference":["Cotton","Lawn"]
  }' \
  -F 'filter_attributes={"Size":["M","L"],"Fabric":["Cotton","Lawn"]}'

# 3. AUDIO SEARCH
# Regular audio search with enhanced preferences
curl -X 'POST' \
  'http://localhost:9000/search/audio' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/Users/aameer/Downloads/green_dress_audio_test.mp3;type=audio/mpeg' \
  -F 'num_results=5' \
  -F 'preferences={
    "brand_weights":{"Zellbury":0.8,"Junaid Jamshed":0.6},
    "price_range":[1000,5000],
    "preferred_colors":["Pink","Red"],
    "category_weights":{"Stitched":0.7},
    "seasonal_preference":"SUMMER",
    "size_preference":["M","L"],
    "fabric_preference":["Cotton","Lawn"]
  }' \
  -F 'filter_attributes={"Size":["M","L"],"Fabric":["Cotton","Lawn"]}'

# Detailed audio search with enhanced features
curl -X 'POST' \
  'http://localhost:9000/search/audio/detailed' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/Users/aameer/Downloads/green_dress_audio_test.mp3;type=audio/mpeg' \
  -F 'num_results=5' \
  -F 'preferences={
    "brand_weights":{"Zellbury":0.8,"Junaid Jamshed":0.6},
    "price_range":[1000,5000],
    "preferred_colors":["Pink","Red"],
    "category_weights":{"Stitched":0.7},
    "seasonal_preference":"SUMMER",
    "size_preference":["M","L"],
    "fabric_preference":["Cotton","Lawn"]
  }' \
  -F 'filter_attributes={"Size":["M","L"],"Fabric":["Cotton","Lawn"]}'
```


# Technical Deep Dive: Enhanced Search Service Architecture

## Core Technologies & Models
1. **AI/ML Models**
   - **Text Processing**: Uses `SentenceTransformer` (all-MiniLM-L6-v2) for text embedding
   - **Image Processing**: Uses OpenAI's `CLIP` model for image understanding
   - **Audio Processing**: Uses OpenAI's `Whisper` model for speech-to-text
   - **Vector Search**: Uses `FAISS` for efficient similarity search

## Key Components

### 1. Embedding System
- **Combined Embeddings**: Creates a unified search space by concatenating:
  - Text embeddings (384 dimensions)
  - Image embeddings (512 dimensions)
- **Normalization**: Uses L2 normalization on all embeddings for consistent similarity calculations

### 2. Search Index Architecture
- **Primary Index**: FAISS IndexFlatIP (Inner Product) for the combined embeddings
- **Attribute Indexes**: Separate indexes for specific attributes like Size, Color, Fabric, Season
- **Caching System**: LRU cache for frequently accessed embeddings

### 3. Search Processing Pipeline

#### Text Search Flow:
```
Query → Text Expansion → Text Embedding → CLIP Text Embedding → Combined Search → Results
```

#### Image Search Flow:
```
Image → CLIP Processing → Image Embedding → Zero-Padded Text Vector → Combined Search → Results
```

#### Audio Search Flow:
```
Audio → Whisper Transcription → Text Processing Pipeline → Results
```

## Advanced Features

### 1. Query Enhancement
- **Query Expansion**: Adds related terms (e.g., color variations, size mappings)
- **Semantic Understanding**: Handles color similarities and attribute relationships

### 2. Result Processing
- **Score Calculation**:
  ```python
  final_score = base_similarity * (1 + brand_boost) * (1 + seasonal_boost) * (1 + category_boost)
  ```
- **Result Diversification**: Ensures results aren't too similar using similarity thresholds

### 3. Performance Optimizations
- **Batch Processing**: Handles images in batches of 16
- **Async Operations**: Uses asyncio for parallel image processing
- **Error Resilience**: Multiple retry attempts for image downloads
- **Caching Strategy**: Two-level caching:
  - In-memory LRU cache for embeddings
  - Persistent cache for computed embeddings

## Implementation Details

### 1. Data Structures
```python
Combined Embedding Dimensions = 896 (384 text + 512 image)
Maximum Images Per Product = 3
Batch Size = 16
```

### 2. Critical Algorithms

#### Diversity Algorithm:
```python
for each result:
    if similarity_to_existing_results < threshold:
        add_to_results
    else:
        skip_result
```

#### Preference Weighting:
```python
Weights = {
    'similarity': 1.0,
    'brand': 0.8,
    'price': 0.6,
    'color': 0.7,
    'category': 0.5,
    'seasonal': 0.4,
    'attribute': 0.3
}
```

### 3. Error Handling
- Graceful degradation for missing images
- Automatic retry mechanism for failed requests
- Fallback to text-only search if image processing fails

## Performance Considerations

### 1. Memory Management
- Efficient batch processing to control memory usage
- Use of numpy's ascontiguousarray for optimal FAISS performance
- Careful management of PyTorch tensors with context managers

### 2. Scaling Considerations
- Async processing for I/O-bound operations
- Caching strategy for frequently accessed data
- Batched processing for large datasets

### 3. Optimization Points
- Image preprocessing and resizing
- Embedding caching
- Query expansion limits
- Batch size tuning

## Integration Points

### 1. Required Dependencies
- FAISS for vector search
- PyTorch for deep learning models
- Sentence Transformers for text embeddings
- CLIP and Whisper for multimodal processing

### 2. Configuration Requirements
- Model paths and configurations
- Cache settings
- Performance tuning parameters
- Error retry policies
