from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import ndcg_score
from scipy.spatial.distance import cdist

def calculate_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain

    Args:
        y_true: Ground truth relevance scores
        y_pred: Predicted relevance scores
        k: Number of items to consider
    """
    if k is not None:
        y_true = y_true[:, :k]
        y_pred = y_pred[:, :k]
    return float(ndcg_score(y_true, y_pred))

def calculate_precision_recall(predicted_ids: List[str],
                             relevant_ids: List[str],
                             k: int) -> tuple[float, float]:
    """
    Calculate Precision@k and Recall@k

    Args:
        predicted_ids: List of predicted product IDs
        relevant_ids: List of relevant product IDs
        k: Number of items to consider
    """
    if not relevant_ids:
        return 0.0, 0.0

    predicted_set = set(predicted_ids[:k])
    relevant_set = set(relevant_ids)

    intersection = len(predicted_set.intersection(relevant_set))

    precision = intersection / k if k > 0 else 0.0
    recall = intersection / len(relevant_set) if relevant_set else 0.0

    return precision, recall

def calculate_map(predicted_ids: List[str],
                 relevant_ids: List[str],
                 k: int) -> float:
    """
    Calculate Mean Average Precision

    Args:
        predicted_ids: List of predicted product IDs
        relevant_ids: List of relevant product IDs
        k: Number of items to consider
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    cumulative_precision = 0.0
    num_relevant_found = 0

    for i, pred_id in enumerate(predicted_ids[:k], 1):
        if pred_id in relevant_set:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / i
            cumulative_precision += precision_at_i

    return cumulative_precision / len(relevant_set) if relevant_set else 0.0

def calculate_diversity(embeddings: np.ndarray) -> float:
    """
    Calculate diversity score based on embedding distances

    Args:
        embeddings: Matrix of embeddings
    """
    if len(embeddings) < 2:
        return 1.0

    # Calculate pairwise distances
    distances = cdist(embeddings, embeddings, metric='cosine')

    # Get upper triangle of distance matrix (excluding diagonal)
    upper_tri = distances[np.triu_indices_from(distances, k=1)]

    return float(np.mean(upper_tri))

def weighted_score(base_score: float, weights: Dict[str, float],
                  attributes: Dict[str, Any]) -> float:
    """
    Apply weights to base similarity score based on attributes

    Args:
        base_score: Base similarity score
        weights: Dictionary of attribute weights
        attributes: Product attributes
    """
    score = base_score

    for attr, weight in weights.items():
        if attr in attributes:
            score *= (1 + weight)

    return score
