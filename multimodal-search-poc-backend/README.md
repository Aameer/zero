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

2. How to run
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
