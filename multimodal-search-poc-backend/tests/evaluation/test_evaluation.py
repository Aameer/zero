import os
import sys
from pathlib import Path
import asyncio
import torch
import json
import logging
from app.services.search_service import EnhancedSearchService
import gc

# Add project root to Python path
project_root = str(Path(__file__).parents[2])
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_catalog():
    catalog_path = os.path.join(project_root, "app", "data", "catalog.json")
    with open(catalog_path, "r") as f:
        return json.load(f)

def load_ground_truth():
    ground_truth_path = os.path.join(project_root, "app", "data", "ground_truth.json")
    with open(ground_truth_path, "r") as f:
        return json.load(f)

async def simple_evaluation():
    service = None
    try:
        # Load data
        catalog = load_catalog()
        ground_truth = load_ground_truth()

        # Initialize service using create method
        logger.info("Initializing search service...")
        service = await EnhancedSearchService.create(catalog)

        # Get first test query
        test_query = ground_truth["queries"][0]

        # Perform search
        logger.info(f"Testing search with query: {test_query['query']}")
        results = await service.search(
            query_type=test_query["type"],
            query=test_query["query"],
            num_results=5
        )

        # Print results
        logger.info("Search Results:")
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Product ID: {result.product.id}")
            print(f"  Title: {result.product.title}")
            print(f"  Similarity: {result.similarity_score:.3f}")
            print()

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

    finally:
        if service:
            logger.info("Cleaning up service...")
            await service.cleanup()
            del service

        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    try:
        asyncio.run(simple_evaluation())
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
