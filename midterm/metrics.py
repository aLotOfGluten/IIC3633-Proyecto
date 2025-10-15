import numpy as np
import torch
from typing import List, Dict, Set, Optional
from scipy.spatial.distance import cosine


def mean_reciprocal_rank(recommendations: List[List[int]],
                         ground_truth: List[Set[int]]) -> float:
    if len(recommendations) != len(ground_truth):
        raise ValueError("Recommendations and ground truth must have same length")

    reciprocal_ranks = []
    for rec_list, true_items in zip(recommendations, ground_truth):
        rank = None
        for i, item in enumerate(rec_list, 1):
            if item in true_items:
                rank = i
                break
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    return np.mean(reciprocal_ranks)


def inter_list_diversity(recommendations: List[List[int]]) -> float:
    if len(recommendations) < 2:
        return 0.0

    similarities = []
    for i in range(len(recommendations)):
        for j in range(i + 1, len(recommendations)):
            set_i = set(recommendations[i])
            set_j = set(recommendations[j])
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)

            jaccard = intersection / union if union > 0 else 0.0
            similarities.append(jaccard)

    return 1.0 - np.mean(similarities)


def cosine_diversity(embeddings: torch.Tensor,
                     recommendations: List[List[int]]) -> float:
    if len(recommendations) == 0 or len(recommendations[0]) < 2:
        return 0.0

    diversities = []
    for rec_list in recommendations:
        rec_embeddings = embeddings[rec_list]

        pairwise_similarities = []
        for i in range(len(rec_list)):
            for j in range(i + 1, len(rec_list)):
                sim = 1.0 - cosine(
                    rec_embeddings[i].numpy(),
                    rec_embeddings[j].numpy()
                )
                pairwise_similarities.append(sim)

        diversities.append(1.0 - np.mean(pairwise_similarities))

    return np.mean(diversities)


def coverage(recommendations: List[List[int]], catalog_size: int) -> float:
    recommended_items = set()
    for rec_list in recommendations:
        recommended_items.update(rec_list)

    return len(recommended_items) / catalog_size


def propagation_reach(infected_nodes: Set[int], total_nodes: int) -> float:
    return len(infected_nodes) / total_nodes


def propagation_depth(infection_rounds: List[Set[int]]) -> int:
    return len(infection_rounds)


def propagation_speed(infection_rounds: List[Set[int]]) -> float:
    if len(infection_rounds) <= 1:
        return 0.0

    speeds = []
    for i in range(1, len(infection_rounds)):
        new_infections = len(infection_rounds[i])
        if new_infections > 0:
            speeds.append(new_infections)

    return np.mean(speeds) if speeds else 0.0


class PropagationMetrics:
    def __init__(self, infection_rounds: List[Set[int]], total_nodes: int):
        self.infection_rounds = infection_rounds
        self.total_nodes = total_nodes
        self.all_infected = set().union(*infection_rounds) if infection_rounds else set()

    def compute_all(self) -> Dict[str, float]:
        return {
            'reach': propagation_reach(self.all_infected, self.total_nodes),
            'depth': propagation_depth(self.infection_rounds),
            'speed': propagation_speed(self.infection_rounds),
            'total_infected': len(self.all_infected)
        }

    def expected_propagation(self, num_simulations: int = 100) -> float:
        return len(self.all_infected) / self.total_nodes


class RecommendationMetrics:
    def __init__(self, recommendations: List[List[int]],
                 ground_truth: List[Set[int]],
                 catalog_size: int,
                 embeddings: Optional[torch.Tensor] = None):
        self.recommendations = recommendations
        self.ground_truth = ground_truth
        self.catalog_size = catalog_size
        self.embeddings = embeddings

    def compute_all(self) -> Dict[str, float]:
        metrics = {
            'mrr': mean_reciprocal_rank(self.recommendations, self.ground_truth),
            'ild': inter_list_diversity(self.recommendations),
            'coverage': coverage(self.recommendations, self.catalog_size)
        }

        if self.embeddings is not None:
            metrics['cosine_diversity'] = cosine_diversity(
                self.embeddings,
                self.recommendations
            )

        return metrics
