import torch
import numpy as np
from pathlib import Path
from typing import Tuple
from torch_geometric.data import Data
from graph_utils import DataLoader, GraphStatistics, build_user_item_matrix


class BipartiteGraphBuilder:
    def __init__(self, interactions, num_users: int, num_items: int):
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items

    def build(self) -> Data:
        user_indices = torch.tensor(self.interactions['user_idx'].values, dtype=torch.long)
        item_indices = torch.tensor(self.interactions['item_idx'].values, dtype=torch.long)
        item_indices_shifted = item_indices + self.num_users

        edge_index = torch.stack([
            torch.cat([user_indices, item_indices_shifted]),
            torch.cat([item_indices_shifted, user_indices])
        ], dim=0)

        return Data(
            edge_index=edge_index,
            num_users=self.num_users,
            num_items=self.num_items
        )


class SocialGraphBuilder:
    def __init__(self, interactions, num_users: int, num_items: int,
                 min_common_items: int = 3):
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items
        self.min_common_items = min_common_items

    def build(self) -> Data:
        user_item_matrix = build_user_item_matrix(
            self.interactions,
            self.num_users,
            self.num_items
        )

        common_items_matrix = user_item_matrix @ user_item_matrix.T
        common_items_matrix.setdiag(0)
        common_items_matrix.eliminate_zeros()

        coo = common_items_matrix.tocoo()
        mask = coo.data >= self.min_common_items

        edge_index = torch.tensor(
            np.vstack([coo.row[mask], coo.col[mask]]),
            dtype=torch.long
        )
        edge_weight = torch.tensor(coo.data[mask], dtype=torch.float)

        return Data(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=self.num_users
        )


class GraphPipeline:
    def __init__(self, data_path: str = "../data_processing/processed_h1",
                 output_path: str = "graphs"):
        self.loader = DataLoader(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)

    def run(self, min_common_items: int = 3) -> Tuple[Data, Data]:
        print("Loading data...")
        interactions = self.loader.load_interactions()
        user_map, item_map = self.loader.load_mappings()

        num_users = len(user_map)
        num_items = len(item_map)

        print(f"Dataset: {num_users} users, {num_items} items, {len(interactions)} interactions")

        print("\nBuilding bipartite graph...")
        bipartite_builder = BipartiteGraphBuilder(interactions, num_users, num_items)
        bipartite_graph = bipartite_builder.build()

        print("\nBuilding social graph...")
        social_builder = SocialGraphBuilder(
            interactions, num_users, num_items, min_common_items
        )
        social_graph = social_builder.build()

        self._save_graphs(bipartite_graph, social_graph)
        self._save_statistics(bipartite_graph, social_graph, min_common_items)

        return bipartite_graph, social_graph

    def _save_graphs(self, bipartite_graph: Data, social_graph: Data):
        torch.save(bipartite_graph, self.output_path / "bipartite_graph.pt")
        torch.save(social_graph, self.output_path / "social_graph.pt")
        print(f"Graphs saved to {self.output_path}/")

    def _save_statistics(self, bipartite_graph: Data, social_graph: Data,
                        min_common_items: int):
        stats_file = self.output_path / "graph_stats.txt"

        with open(stats_file, 'w') as f:
            f.write(f"{'=' * 60}\n")
            f.write("BIPARTITE GRAPH (User-Item)\n")
            f.write(f"{'=' * 60}\n\n")

            bipartite_stats = GraphStatistics.compute_bipartite_stats(
                bipartite_graph.num_users,
                bipartite_graph.num_items,
                bipartite_graph.edge_index
            )

            for key, value in bipartite_stats.items():
                if isinstance(value, float) and value < 1:
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")

            f.write(f"\n{'=' * 60}\n")
            f.write(f"SOCIAL GRAPH (User-User, min_common_items={min_common_items})\n")
            f.write(f"{'=' * 60}\n\n")

            social_stats = GraphStatistics.compute_graph_stats(
                social_graph.num_nodes,
                social_graph.edge_index
            )

            for key, value in social_stats.items():
                if isinstance(value, float) and value < 1:
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")

        print(f"Statistics saved to {stats_file}")


def main():
    pipeline = GraphPipeline()
    bipartite_graph, social_graph = pipeline.run(min_common_items=3)

    num_nodes = bipartite_graph.num_users + bipartite_graph.num_items
    print(f"\n{'=' * 60}")
    print("Graph construction completed successfully")
    print(f"{'=' * 60}")
    print(f"\nBipartite graph: {num_nodes} nodes, "
          f"{bipartite_graph.edge_index.size(1)} edges")
    print(f"Social graph: {social_graph.num_nodes} nodes, "
          f"{social_graph.edge_index.size(1)} edges")


if __name__ == "__main__":
    main()
