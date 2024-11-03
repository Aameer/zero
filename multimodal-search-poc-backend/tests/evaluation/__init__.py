from pathlib import Path

# Set up test data paths
EVALUATION_DIR = Path(__file__).parent
SAMPLE_QUERIES_DIR = EVALUATION_DIR / "sample_queries"
TEST_DATA_DIR = EVALUATION_DIR / "test_data"

# Create directories if they don't exist
SAMPLE_QUERIES_DIR.mkdir(exist_ok=True)
TEST_DATA_DIR.mkdir(exist_ok=True)

# Constants for evaluation
DEFAULT_K = 10
SIMILARITY_THRESHOLD = 0.5
