import torch
from pathlib import Path
from graph_utils import DataLoader, load_embeddings
from linear_threshold_model import LinearThresholdModel
from metrics import RecommendationMetrics, PropagationMetrics
from node_features import NodeFeatureGenerator, RandomEncoder


def test_graphs():
    print("=" * 60)
    print("TEST 1: Verificar grafos")
    print("=" * 60)

    bipartite = torch.load('graphs/bipartite_graph.pt', weights_only=False)
    social = torch.load('graphs/social_graph.pt', weights_only=False)

    print(f"✓ Grafo bipartito cargado: {bipartite.num_users} users, "
          f"{bipartite.num_items} items")
    print(f"✓ Grafo social cargado: {social.num_nodes} nodes, "
          f"{social.edge_index.size(1)} edges")

    assert bipartite.num_users == 4856
    assert bipartite.num_items == 2308
    assert social.num_nodes == 4856

    return bipartite, social


def test_data_loading():
    print("\n" + "=" * 60)
    print("TEST 2: Verificar carga de datos")
    print("=" * 60)

    loader = DataLoader()
    interactions = loader.load_interactions()
    user_map, item_map = loader.load_mappings()
    labels = loader.load_labels()
    item_text = loader.load_item_text()

    print(f"✓ Interacciones: {len(interactions)}")
    print(f"✓ Usuarios: {len(user_map)}")
    print(f"✓ Items: {len(item_map)}")
    print(f"✓ Labels: {len(labels)}")
    print(f"✓ Items con texto: {len(item_text)}")

    assert len(user_map) == 4856
    assert len(item_map) == 2308

    return interactions, user_map, item_map, labels


def test_embeddings():
    print("\n" + "=" * 60)
    print("TEST 3: Verificar generación de embeddings")
    print("=" * 60)

    generator = NodeFeatureGenerator()
    encoder = RandomEncoder(embedding_dim=16, seed=42)

    item_embeddings = generator.generate_item_embeddings(encoder)
    user_embeddings = generator.generate_user_embeddings(4856, 16, seed=42)

    print(f"✓ Item embeddings: {item_embeddings.shape}")
    print(f"✓ User embeddings: {user_embeddings.shape}")

    assert item_embeddings.shape == (2308, 16)
    assert user_embeddings.shape == (4856, 16)

    return item_embeddings, user_embeddings


def test_linear_threshold_model(social):
    print("\n" + "=" * 60)
    print("TEST 4: Verificar Linear Threshold Model")
    print("=" * 60)

    ltm = LinearThresholdModel(social, seed=42)

    seed_nodes = {0, 1, 2, 3, 4}
    infection_rounds = ltm.simulate(seed_nodes, max_iterations=10)

    total_infected = sum(len(r) for r in infection_rounds)

    print(f"✓ LTM inicializado con {ltm.num_nodes} nodos")
    print(f"✓ Simulación completada: {len(infection_rounds)} rondas")
    print(f"✓ Total infectados: {total_infected}")

    assert len(infection_rounds) > 0
    assert total_infected >= len(seed_nodes)

    return infection_rounds


def test_metrics(item_embeddings, infection_rounds):
    print("\n" + "=" * 60)
    print("TEST 5: Verificar métricas")
    print("=" * 60)

    recommendations = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    ground_truth = [{2, 3}, {6}]

    rec_metrics = RecommendationMetrics(
        recommendations=recommendations,
        ground_truth=ground_truth,
        catalog_size=2308,
        embeddings=item_embeddings
    )
    rec_results = rec_metrics.compute_all()

    print("✓ Métricas de recomendación:")
    for key, value in rec_results.items():
        print(f"  {key}: {value:.4f}")

    prop_metrics = PropagationMetrics(infection_rounds, 4856)
    prop_results = prop_metrics.compute_all()

    print("\n✓ Métricas de propagación:")
    for key, value in prop_results.items():
        print(f"  {key}: {value}")

    assert 'mrr' in rec_results
    assert 'ild' in rec_results
    assert 'reach' in prop_results
    assert 'depth' in prop_results


def main():
    print("PIPELINE DE VERIFICACIÓN - MIDTERM")
    print("=" * 60)

    try:
        bipartite, social = test_graphs()
        interactions, user_map, item_map, labels = test_data_loading()
        item_embeddings, user_embeddings = test_embeddings()
        infection_rounds = test_linear_threshold_model(social)
        test_metrics(item_embeddings, infection_rounds)

        print("\n" + "=" * 60)
        print("✓ TODOS LOS TESTS PASARON CORRECTAMENTE")
        print("=" * 60)
        print("\nEl pipeline está listo para implementar modelos GNN.")
        print("Ver README.md sección 'Guía para Implementar Modelos GNN'")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
