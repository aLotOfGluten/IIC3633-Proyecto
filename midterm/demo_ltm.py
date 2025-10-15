import torch
import numpy as np
from pathlib import Path
from linear_threshold_model import LinearThresholdModel
from metrics import PropagationMetrics


def load_social_graph(graph_path: str = "graphs/social_graph.pt"):
    return torch.load(graph_path, weights_only=False)


def print_simulation_results(ltm: LinearThresholdModel,
                            seed_nodes: set,
                            infection_rounds: list,
                            title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Seed nodes: {len(seed_nodes)}")
    print(f"Total rounds: {len(infection_rounds)}")

    total_infected = set().union(*infection_rounds) if infection_rounds else set()
    print(f"Total infected nodes: {len(total_infected)}")
    print(f"Reach: {len(total_infected) / ltm.num_nodes * 100:.2f}%")

    print("\nInfection progression:")
    cumulative = 0
    for i, round_nodes in enumerate(infection_rounds):
        cumulative += len(round_nodes)
        print(f"  Round {i}: {len(round_nodes)} new infections "
              f"(cumulative: {cumulative})")


def demo_basic_simulation():
    print("\n" + "=" * 60)
    print("DEMO: Basic Linear Threshold Model Simulation")
    print("=" * 60)

    print("\nLoading social graph...")
    graph = load_social_graph()
    print("Initializing Linear Threshold Model...")
    ltm = LinearThresholdModel(graph, seed=42)

    print(f"\nSocial graph loaded:")
    print(f"  Nodes: {ltm.num_nodes}")
    print(f"  Edges: {graph.edge_index.size(1)}")

    seed_nodes = {0, 1, 2, 3, 4}
    print(f"\nRunning simulation with {len(seed_nodes)} seed nodes...")
    infection_rounds = ltm.simulate(seed_nodes, max_iterations=50)
    print("Simulation complete!")

    print_simulation_results(
        ltm,
        seed_nodes,
        infection_rounds,
        "Simulation with random thresholds (uniform [0, 1])"
    )


def demo_custom_thresholds():
    print("\n" + "=" * 60)
    print("DEMO: Custom Threshold Configuration")
    print("=" * 60)

    print("\nLoading social graph...")
    graph = load_social_graph()
    print("Initializing Linear Threshold Model...")
    ltm = LinearThresholdModel(graph, seed=42)

    seed_nodes = {0, 1, 2}
    print(f"\nUsing {len(seed_nodes)} seed nodes for both simulations")

    print("\n[1/2] Running simulation with LOW thresholds (0.1)...")
    low_thresholds = np.ones(ltm.num_nodes) * 0.1
    infection_rounds_low = ltm.simulate(
        seed_nodes,
        thresholds=low_thresholds,
        max_iterations=50
    )
    print("Low threshold simulation complete!")

    print_simulation_results(
        ltm,
        seed_nodes,
        infection_rounds_low,
        "Simulation with LOW thresholds (0.1)"
    )

    print("\n[2/2] Running simulation with HIGH thresholds (0.8)...")
    high_thresholds = np.ones(ltm.num_nodes) * 0.8
    infection_rounds_high = ltm.simulate(
        seed_nodes,
        thresholds=high_thresholds,
        max_iterations=50
    )
    print("High threshold simulation complete!")

    print_simulation_results(
        ltm,
        seed_nodes,
        infection_rounds_high,
        "Simulation with HIGH thresholds (0.8)"
    )


def demo_expected_propagation():
    print("\n" + "=" * 60)
    print("DEMO: Expected Propagation (Monte Carlo)")
    print("=" * 60)

    print("\nLoading social graph...")
    graph = load_social_graph()
    print("Initializing Linear Threshold Model...")
    ltm = LinearThresholdModel(graph, seed=42)

    seed_sizes = [1, 5, 10, 20]

    print(f"\nTesting {len(seed_sizes)} different seed sizes (10 simulations each)...")
    print("\nExpected propagation for different seed set sizes:")
    print(f"{'Seed Size':<15} {'Expected Infected':<20} {'Expected Reach'}")
    print("-" * 60)

    for idx, size in enumerate(seed_sizes, 1):
        print(f"[{idx}/{len(seed_sizes)}] Computing for seed size {size}...", end=" ", flush=True)
        seed_nodes = set(range(size))
        expected = ltm.expected_propagation(seed_nodes, num_simulations=10)
        reach = expected / ltm.num_nodes * 100

        print(f"\r{size:<15} {expected:<20.2f} {reach:.2f}%")

    print("\nExpected propagation analysis complete!")


def demo_monte_carlo_analysis():
    print("\n" + "=" * 60)
    print("DEMO: Monte Carlo Propagation Analysis")
    print("=" * 60)

    print("\nLoading social graph...")
    graph = load_social_graph()
    print("Initializing Linear Threshold Model...")
    ltm = LinearThresholdModel(graph, seed=42)

    seed_nodes = set(range(10))

    print(f"\nRunning Monte Carlo analysis with {len(seed_nodes)} seed nodes...")
    print("(20 simulations)")
    print("Progress: ", end="", flush=True)

    results = ltm.monte_carlo_propagation(seed_nodes, num_simulations=20)
    print("✓ Complete!")

    print("\nResults:")
    print(f"  Mean reach: {results['mean_reach'] * 100:.2f}% "
          f"(±{results['std_reach'] * 100:.2f}%)")
    print(f"  Max reach: {results['max_reach'] * 100:.2f}%")
    print(f"  Min reach: {results['min_reach'] * 100:.2f}%")
    print(f"  Mean depth: {results['mean_depth']:.2f} rounds "
          f"(±{results['std_depth']:.2f})")


def demo_propagation_metrics():
    print("\n" + "=" * 60)
    print("DEMO: Propagation Metrics")
    print("=" * 60)

    print("\nLoading social graph...")
    graph = load_social_graph()
    print("Initializing Linear Threshold Model...")
    ltm = LinearThresholdModel(graph, seed=42)

    seed_nodes = set(range(15))
    print(f"\nRunning simulation with {len(seed_nodes)} seed nodes...")
    infection_rounds = ltm.simulate(seed_nodes, max_iterations=50)
    print("Simulation complete!")

    print("\nComputing propagation metrics...")
    metrics_calculator = PropagationMetrics(infection_rounds, ltm.num_nodes)
    metrics = metrics_calculator.compute_all()
    print("Metrics computed!")

    print(f"\nSeed nodes: {len(seed_nodes)}")
    print("\nPropagation metrics:")
    print(f"  Reach: {metrics['reach'] * 100:.2f}%")
    print(f"  Depth: {metrics['depth']} rounds")
    print(f"  Speed: {metrics['speed']:.2f} infections/round")
    print(f"  Total infected: {metrics['total_infected']} nodes")


def main():
    print("=" * 60)
    print("Linear Threshold Model - Demonstration Suite")
    print("=" * 60)

    demos = [
        ("1", "Basic simulation", demo_basic_simulation),
        ("2", "Custom thresholds", demo_custom_thresholds),
        ("3", "Expected propagation", demo_expected_propagation),
        ("4", "Monte Carlo analysis", demo_monte_carlo_analysis),
        ("5", "Propagation metrics", demo_propagation_metrics),
    ]

    print("\nAvailable demos:")
    for num, name, _ in demos:
        print(f"  {num}. {name}")
    print("  all. Run all demos")

    choice = input("\nSelect demo (1-5 or 'all'): ").strip()

    if choice == "all":
        for _, _, demo_func in demos:
            demo_func()
    else:
        for num, _, demo_func in demos:
            if choice == num:
                demo_func()
                break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
