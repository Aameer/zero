# Multimodal Search API
# Copyright (c) 2024, Aameer Rafiq Wani
A FastAPI-based backend service that provides multimodal search capabilities for e-commerce products using state-of-the-art AI models. The project implements two approaches for multimodal search: a traditional multi-model approach and a unified ImageBind approach, allowing for performance comparison and gradual migration.

## Features

### Enhanced Search Implementation
- 🔍 Text-based semantic search using BERT embeddings
- 🖼️ Image similarity search using CLIP
- 🎤 Audio-to-text search using Whisper
- ⚡ Fast similarity search using FAISS
- 🎯 Attribute-based weighting system
- 🔄 Seasonal and contextual boosting
- 📊 Configurable similarity thresholds

### ImageBind Implementation
- 🎯 Unified multimodal embeddings using ImageBind
- 🔄 Single embedding space for text, images, and audio
- 🎨 Better cross-modal understanding
- 📈 Improved zero-shot performance
- 🔍 Native support for multimodal queries
- 🚀 Streamlined architecture with single model

### Evaluation Framework
- 📊 NDCG (Normalized Discounted Cumulative Gain) scoring
- 🎯 Precision and recall metrics
- 📈 MAP (Mean Average Precision) evaluation
- 🔄 Diversity scoring
- ⚖️ Direct comparison between implementations
- 📈 Support for weighted and unweighted evaluation

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, but recommended for better performance)
- 4GB+ RAM
- 2GB+ disk space for models
- macOS/Linux/Windows

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimodal-search-poc-backend.git
cd multimodal-search-poc-backend
```

2. Create and activate virtual environment:
```bash
python -m venv msearch
source msearch/bin/activate  # On Unix/macOS
# or
.\msearch\Scripts\activate  # On Windows
```

3. Upgrade pip:
```bash
python -m pip install --upgrade pip
```

4. Install PyTorch dependencies first:
```bash
pip install torch==1.13.1 torchvision==0.14.1
```

5. Install ImageBind:
```bash
# Clone ImageBind repository
git clone https://github.com/facebookresearch/ImageBind.git
cd ImageBind
pip install -e .
cd ..
```

6. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

7. Verify the installation:
```bash
# Check torch version
python -c "import torch; print(torch.__version__)"

# Verify ImageBind installation
python -c "from imagebind import data; print('ImageBind imported successfully')"
```

## Project Structure
```
multimodal-search-poc-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application
│   ├── data/
│   │   ├── catalog.json             # Product catalog
│   │   └── ground_truth.json        # Evaluation data
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py               # Data models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── search_service.py        # Enhanced search implementation
│   │   ├── imagebind_service.py     # ImageBind implementation
│   │   └── evaluation_service.py    # Evaluation framework
│   └── utils/
│       ├── __init__.py
│       └── metrics.py               # Evaluation metrics
├── tests/
│   ├── __init__.py
│   └── evaluation/
│       ├── __init__.py
│       ├── test_evaluation.py       # Evaluation scripts
│       └── sample_queries/          # Test data
└── requirements.txt
```

## Running the Application

1. Start the server:
```bash
cd /path/to/multimodal-search-poc-backend
source msearch/bin/activate  # Activate virtual environment
uvicorn app.main:app --host localhost --port 8000
```

2. Access the API documentation:
```
http://localhost:8000/docs
```

## API Endpoints

- `POST /search`: General search endpoint supporting text, image, and audio queries
- `POST /search/image`: Dedicated image search endpoint
- `POST /search/audio`: Dedicated audio search endpoint
- `GET /products`: Retrieve product catalog
- `POST /search/compare`: Compare results from both implementations
- `POST /evaluate`: Run evaluation framework

## Evaluation

To compare the performance of different search implementations:

```bash
# Run evaluation without weights
python -m tests.evaluation.test_evaluation

# Run evaluation with attribute weights
python -m tests.evaluation.test_evaluation --use-weights
```

### Evaluation Metrics

- **NDCG**: Measures ranking quality
- **Precision@K**: Accuracy of top K results
- **Recall@K**: Coverage of relevant items
- **MAP**: Overall precision across all recall levels
- **Diversity Score**: Variety in results

## Implementation Comparison

### Enhanced Search
- Better for fine-grained control
- Explicit attribute matching
- More configurable weights and boosts
- Higher computational overhead

### ImageBind
- Better cross-modal understanding
- Simpler architecture
- Improved zero-shot capabilities
- More efficient resource usage
- Better scaling with multiple modalities

## Troubleshooting

1. SSL/Git Clone Issues with ImageBind:
   ```bash
   # Alternative: Download as ZIP
   curl -L https://github.com/facebookresearch/ImageBind/archive/refs/heads/main.zip -o ImageBind.zip
   unzip ImageBind.zip
   cd ImageBind-main
   pip install -e .
   cd ..
   ```

2. CUDA/GPU Issues:
   - Make sure your CUDA version is compatible with PyTorch 1.13.1
   - For CPU-only installation, no additional configuration is needed

3. Memory Issues:
   - Consider reducing batch sizes in configuration
   - Use CPU version of FAISS if GPU memory is limited

4. Import Errors:
   - Ensure you're in the correct virtual environment
   - Verify all dependencies are installed
   - Check Python path includes project root

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

Copyright (c) 2024, Aameer Rafiq Wani. All rights reserved.
