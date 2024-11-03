from typing import List, Dict, Tuple
from sklearn.metrics import ndcg_score
import numpy as np
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import torch
from PIL import Image
import io
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SearchEvalMetrics:
    ndcg_score: float
    precision_at_k: float
    recall_at_k: float
    map_score: float
    diversity_score: float
    timestamp: datetime

class SearchEvaluator:
    def __init__(self, ground_truth_file: str):
        """
        Initialize evaluator with ground truth data.

        ground_truth_file format:
        {
            "queries": [
                {
                    "query": "blue summer dress",
                    "type": "text",
                    "relevant_products": ["product_id1", "product_id2"],
                    "relevance_scores": [5, 3]  # 5 = perfect match, 1 = slight relevance
                }
            ]
        }
        """
        with open(ground_truth_file, 'r') as f:
            self.ground_truth = json.load(f)

        self.relevance_threshold = 0.5

    def calculate_ndcg(self, predicted_results: List[str],
                      relevant_products: List[str],
                      relevance_scores: List[int],
                      k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        y_true = np.zeros(len(predicted_results))
        for idx, prod_id in enumerate(predicted_results):
            if prod_id in relevant_products:
                rel_idx = relevant_products.index(prod_id)
                y_true[idx] = relevance_scores[rel_idx]

        y_pred = np.ones(len(predicted_results))  # Predicted relevance
        return ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1))

    def calculate_precision_recall(self, predicted_results: List[str],
                                 relevant_products: List[str],
                                 k: int = 10) -> Tuple[float, float]:
        """Calculate Precision@K and Recall@K"""
        pred_set = set(predicted_results[:k])
        rel_set = set(relevant_products)

        true_positives = len(pred_set.intersection(rel_set))
        precision = true_positives / k if k > 0 else 0
        recall = true_positives / len(rel_set) if rel_set else 0

        return precision, recall

    def calculate_diversity(self, results: List[Dict], embeddings: np.ndarray) -> float:
        """Calculate diversity score based on embedding distances"""
        if len(results) < 2:
            return 1.0

        pairwise_distances = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                pairwise_distances.append(dist)

        return np.mean(pairwise_distances)

    def evaluate_search(self, search_service, k: int = 10) -> SearchEvalMetrics:
        """Evaluate search performance across all test queries"""
        total_ndcg = 0
        total_precision = 0
        total_recall = 0
        total_diversity = 0
        query_count = len(self.ground_truth["queries"])

        for query_data in self.ground_truth["queries"]:
            # Perform search
            results = search_service.search(
                query_type=query_data["type"],
                query=query_data["query"],
                num_results=k
            )

            # Extract product IDs from results
            predicted_ids = [str(r.product.id) for r in results]

            # Calculate metrics
            ndcg = self.calculate_ndcg(
                predicted_ids,
                query_data["relevant_products"],
                query_data["relevance_scores"],
                k
            )

            precision, recall = self.calculate_precision_recall(
                predicted_ids,
                query_data["relevant_products"],
                k
            )

            # Get embeddings for diversity calculation
            result_embeddings = search_service.get_embeddings(predicted_ids)
            diversity = self.calculate_diversity(results, result_embeddings)

            total_ndcg += ndcg
            total_precision += precision
            total_recall += recall
            total_diversity += diversity

        return SearchEvalMetrics(
            ndcg_score=total_ndcg / query_count,
            precision_at_k=total_precision / query_count,
            recall_at_k=total_recall / query_count,
            map_score=total_precision / query_count,  # Simplified MAP
            diversity_score=total_diversity / query_count,
            timestamp=datetime.now()
        )
