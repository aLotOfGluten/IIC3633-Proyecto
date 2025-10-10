import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict
from scipy.sparse import csr_matrix


class DataLoader:
    def __init__(self, base_path: str = "../data_processing/processed_h1"):
        self.base_path = Path(base_path)

    def load_interactions(self) -> pd.DataFrame:
        return pd.read_csv(self.base_path / "train_interactions_idx.csv")

    def load_mappings(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        user_map = pd.read_csv(self.base_path / "user_map.csv")
        item_map = pd.read_csv(self.base_path / "item_map.csv")
        return user_map, item_map

    def load_labels(self) -> pd.DataFrame:
        return pd.read_csv(self.base_path / "item_labels.csv")


class GraphStatistics:
    @staticmethod
    def compute_graph_stats(num_nodes: int, edge_index: torch.Tensor) -> Dict[str, float]:
        num_edges = edge_index.size(1)
        density = num_edges / (num_nodes * (num_nodes - 1))

        degrees = torch.bincount(edge_index[0], minlength=num_nodes)
        avg_degree = degrees.float().mean().item()
        max_degree = degrees.max().item()
        min_degree = degrees.min().item()

        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'min_degree': min_degree
        }

    @staticmethod
    def compute_bipartite_stats(num_users: int, num_items: int,
                               edge_index: torch.Tensor) -> Dict[str, float]:
        user_to_item_mask = edge_index[0] < num_users
        user_to_item_edges = edge_index[:, user_to_item_mask]

        num_edges = user_to_item_edges.size(1)
        density = num_edges / (num_users * num_items)

        user_degrees = torch.bincount(user_to_item_edges[0], minlength=num_users)
        item_degrees = torch.bincount(user_to_item_edges[1] - num_users, minlength=num_items)

        return {
            'num_users': num_users,
            'num_items': num_items,
            'num_edges': num_edges,
            'density': density,
            'avg_user_degree': user_degrees.float().mean().item(),
            'avg_item_degree': item_degrees.float().mean().item(),
            'max_user_degree': user_degrees.max().item(),
            'max_item_degree': item_degrees.max().item()
        }


def build_user_item_matrix(interactions: pd.DataFrame,
                          num_users: int,
                          num_items: int) -> csr_matrix:
    row = interactions['user_idx'].values
    col = interactions['item_idx'].values
    data = np.ones(len(interactions))
    return csr_matrix((data, (row, col)), shape=(num_users, num_items))
