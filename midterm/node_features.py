import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path


class TextEncoder(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass


class BERTEncoder(TextEncoder):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings.cpu()

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class RandomEncoder(TextEncoder):
    def __init__(self, embedding_dim: int = 64, seed: int = 42):
        self._embedding_dim = embedding_dim
        self.seed = seed

    def encode(self, texts: List[str]) -> torch.Tensor:
        np.random.seed(self.seed)
        embeddings = np.random.randn(len(texts), self._embedding_dim)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return torch.from_numpy(embeddings).float()

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class NodeFeatureGenerator:
    def __init__(self, data_path: str = "../data_processing/processed_h1"):
        self.data_path = Path(data_path)

    def load_item_texts(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path / "item_text_clean.csv")

    def generate_item_embeddings(self, encoder: TextEncoder) -> torch.Tensor:
        item_map = pd.read_csv(self.data_path / "item_map.csv")
        item_texts = self.load_item_texts()

        merged = item_map.merge(
            item_texts[['item_idx', 'text_clean']],
            on='item_idx',
            how='left'
        )
        merged = merged.sort_values('item_idx').reset_index(drop=True)
        texts = merged['text_clean'].fillna('').tolist()

        print(f"Encoding {len(texts)} items with {encoder.__class__.__name__}...")
        print(f"  Items with text: {merged['text_clean'].notna().sum()}")
        print(f"  Items without text: {merged['text_clean'].isna().sum()}")

        embeddings = encoder.encode(texts)

        print(f"Generated embeddings: {embeddings.shape}")
        return embeddings

    def generate_user_embeddings(self, num_users: int, embedding_dim: int,
                                seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.nn.init.xavier_uniform_(torch.empty(num_users, embedding_dim))
