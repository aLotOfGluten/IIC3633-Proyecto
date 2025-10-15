import torch
import numpy as np
from typing import Set, List, Optional, Dict
from torch_geometric.data import Data


class LinearThresholdModel:
    def __init__(self, graph: Data, seed: int = 42):
        self.graph = graph
        self.num_nodes = graph.num_nodes
        self.seed = seed
        self._build_adjacency()

    def _build_adjacency(self):
        # For LTM, we need both outgoing neighbors (for spreading) and incoming neighbors (for influence)
        self.out_neighbors = [[] for _ in range(self.num_nodes)]
        self.in_neighbors = [[] for _ in range(self.num_nodes)]
        self.edge_weights = {}

        edge_index = self.graph.edge_index.numpy()
        weights = self.graph.edge_weight.numpy() if hasattr(self.graph, 'edge_weight') else None

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            weight = weights[i] if weights is not None else 1.0

            self.out_neighbors[src].append(dst)
            self.in_neighbors[dst].append(src)
            self.edge_weights[(src, dst)] = weight

    def _normalize_influence(self, node: int) -> Dict[int, float]:
        # Calculate normalized incoming influence to this node
        if not self.in_neighbors[node]:
            return {}

        influences = {}
        total_weight = sum(
            self.edge_weights.get((neighbor, node), 1.0)
            for neighbor in self.in_neighbors[node]
        )

        if total_weight == 0:
            return {}

        for neighbor in self.in_neighbors[node]:
            weight = self.edge_weights.get((neighbor, node), 1.0)
            influences[neighbor] = weight / total_weight

        return influences

    def simulate(self,
                seed_nodes: Set[int],
                thresholds: Optional[np.ndarray] = None,
                max_iterations: int = 100) -> List[Set[int]]:

        if thresholds is None:
            np.random.seed(self.seed)
            thresholds = np.random.uniform(0.0, 1.0, self.num_nodes)

        activated = seed_nodes.copy()
        infection_rounds = [seed_nodes.copy()]
        newly_activated = seed_nodes.copy()

        for iteration in range(max_iterations):
            if not newly_activated:
                break

            candidates = set()
            for node in newly_activated:
                candidates.update(self.out_neighbors[node])

            candidates -= activated

            next_activated = set()
            for candidate in candidates:
                influence = self._normalize_influence(candidate)
                total_influence = sum(
                    influence.get(neighbor, 0.0)
                    for neighbor in influence
                    if neighbor in activated
                )

                if total_influence >= thresholds[candidate]:
                    next_activated.add(candidate)

            if not next_activated:
                break

            activated.update(next_activated)
            infection_rounds.append(next_activated.copy())
            newly_activated = next_activated

        return infection_rounds

    def expected_propagation(self,
                            seed_nodes: Set[int],
                            num_simulations: int = 100) -> float:
        np.random.seed(self.seed)
        total_activated = []

        for sim in range(num_simulations):
            thresholds = np.random.uniform(0.0, 1.0, self.num_nodes)
            infection_rounds = self.simulate(seed_nodes, thresholds, max_iterations=50)

            all_infected = set().union(*infection_rounds) if infection_rounds else set()
            total_activated.append(len(all_infected))

        return np.mean(total_activated)

    def monte_carlo_propagation(self,
                               seed_nodes: Set[int],
                               num_simulations: int = 100) -> Dict[str, float]:
        np.random.seed(self.seed)
        reaches = []
        depths = []

        for sim in range(num_simulations):
            thresholds = np.random.uniform(0.0, 1.0, self.num_nodes)
            infection_rounds = self.simulate(seed_nodes, thresholds, max_iterations=50)

            all_infected = set().union(*infection_rounds) if infection_rounds else set()
            reaches.append(len(all_infected) / self.num_nodes)
            depths.append(len(infection_rounds))

        return {
            'mean_reach': np.mean(reaches),
            'std_reach': np.std(reaches),
            'mean_depth': np.mean(depths),
            'std_depth': np.std(depths),
            'max_reach': np.max(reaches),
            'min_reach': np.min(reaches)
        }


def load_ltm_from_graph(graph_path: str, seed: int = 42) -> LinearThresholdModel:
    graph = torch.load(graph_path)
    return LinearThresholdModel(graph, seed=seed)
