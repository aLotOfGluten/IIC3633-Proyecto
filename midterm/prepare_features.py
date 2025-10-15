import torch
import argparse
from pathlib import Path
from node_features import BERTEncoder, RandomEncoder, NodeFeatureGenerator
from graph_utils import DataLoader, save_embeddings


class Config:
    DATA_PATH = "../data_processing/processed_h1"
    OUTPUT_DIR = "graphs"
    BERT_MODEL = "all-MiniLM-L6-v2"
    RANDOM_DIM = 64
    SEED = 42


def generate_bert_embeddings(output_dir: Path, data_path: str, model_name: str):
    print("=" * 60)
    print("Generating BERT embeddings for items")
    print("=" * 60)

    encoder = BERTEncoder(model_name=model_name)
    generator = NodeFeatureGenerator(data_path=data_path)

    item_embeddings = generator.generate_item_embeddings(encoder)

    output_path = output_dir / "item_embeddings_bert.pt"
    save_embeddings(item_embeddings, output_path)

    print(f"\nBERT embeddings saved to {output_path}")
    print(f"Shape: {item_embeddings.shape}")
    print(f"Embedding dimension: {encoder.embedding_dim}")


def generate_random_embeddings(output_dir: Path, data_path: str,
                               embedding_dim: int, seed: int):
    print("\n" + "=" * 60)
    print("Generating random embeddings for items (baseline)")
    print("=" * 60)

    encoder = RandomEncoder(embedding_dim=embedding_dim, seed=seed)
    generator = NodeFeatureGenerator(data_path=data_path)

    item_embeddings = generator.generate_item_embeddings(encoder)

    output_path = output_dir / "item_embeddings_random.pt"
    save_embeddings(item_embeddings, output_path)

    print(f"\nRandom embeddings saved to {output_path}")
    print(f"Shape: {item_embeddings.shape}")


def generate_user_embeddings(output_dir: Path, data_path: str,
                             embedding_dim: int, seed: int):
    print("\n" + "=" * 60)
    print("Generating learnable user embeddings")
    print("=" * 60)

    loader = DataLoader(data_path)
    user_map, _ = loader.load_mappings()
    num_users = len(user_map)

    generator = NodeFeatureGenerator(data_path=data_path)
    user_embeddings = generator.generate_user_embeddings(
        num_users,
        embedding_dim,
        seed=seed
    )

    output_path = output_dir / "user_embeddings_init.pt"
    save_embeddings(user_embeddings, output_path)

    print(f"\nUser embeddings saved to {output_path}")
    print(f"Shape: {user_embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate node feature embeddings for GNN models"
    )
    parser.add_argument(
        "--bert",
        action="store_true",
        help="Generate BERT embeddings for items"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Generate random embeddings for items (baseline)"
    )
    parser.add_argument(
        "--users",
        action="store_true",
        help="Generate initial user embeddings"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all embeddings"
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default=Config.BERT_MODEL,
        help=f"BERT model name (default: {Config.BERT_MODEL})"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=Config.RANDOM_DIM,
        help=f"Embedding dimension for random/user embeddings (default: {Config.RANDOM_DIM})"
    )

    args = parser.parse_args()

    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    if args.all:
        generate_bert_embeddings(
            output_dir,
            Config.DATA_PATH,
            args.bert_model
        )
        generate_random_embeddings(
            output_dir,
            Config.DATA_PATH,
            args.dim,
            Config.SEED
        )
        generate_user_embeddings(
            output_dir,
            Config.DATA_PATH,
            args.dim,
            Config.SEED
        )
    else:
        if args.bert:
            generate_bert_embeddings(
                output_dir,
                Config.DATA_PATH,
                args.bert_model
            )
        if args.random:
            generate_random_embeddings(
                output_dir,
                Config.DATA_PATH,
                args.dim,
                Config.SEED
            )
        if args.users:
            generate_user_embeddings(
                output_dir,
                Config.DATA_PATH,
                args.dim,
                Config.SEED
            )

        if not (args.bert or args.random or args.users):
            print("No embedding type specified. Use --bert, --random, --users, or --all")
            parser.print_help()
            return

    print("\n" + "=" * 60)
    print("Feature generation completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
