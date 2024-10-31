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
