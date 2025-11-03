"""
Utilidades de entrenamiento optimizadas para GNN RecSys
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


class InteractionDataset(Dataset):
    """
    Dataset para interacciones user-item con negative sampling.
    Mucho más rápido que iterar con pandas.
    """
    def __init__(self, interactions_df, neg_dict, num_items):
        """
        Args:
            interactions_df: DataFrame con columnas ['user_idx', 'item_idx']
            neg_dict: Dict[user_idx -> List[negative_item_idx]]
            num_items: Número total de items para random sampling
        """
        self.users = interactions_df['user_idx'].values
        self.items = interactions_df['item_idx'].values
        self.neg_dict = neg_dict
        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]

        # Sample negative item
        if user in self.neg_dict and len(self.neg_dict[user]) > 0:
            neg_item = np.random.choice(self.neg_dict[user])
        else:
            neg_item = np.random.randint(0, self.num_items)

        return user, pos_item, neg_item


def train_epoch_batched(model, edge_index, item_features, train_loader, optimizer, device, is_lightgcn=False):
    """
    Entrena un epoch usando mini-batches.
    MUCHO más rápido que iterar con pandas.

    Args:
        model: Modelo GNN
        edge_index: Grafo de entrenamiento
        item_features: Features de items (None para LightGCN)
        train_loader: DataLoader con triplets (user, pos_item, neg_item)
        optimizer: Optimizador
        device: 'cuda' o 'cpu'
        is_lightgcn: Si es True, no usa item_features

    Returns:
        loss promedio del epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    # Forward pass: obtener embeddings UNA VEZ por epoch
    with torch.no_grad():
        if is_lightgcn:
            user_emb, item_emb = model(edge_index)
        else:
            user_emb, item_emb = model(edge_index, item_features)

    # Iterar por mini-batches
    for batch_users, batch_pos_items, batch_neg_items in train_loader:
        batch_users = batch_users.to(device)
        batch_pos_items = batch_pos_items.to(device)
        batch_neg_items = batch_neg_items.to(device)

        optimizer.zero_grad()

        # Obtener embeddings del batch (lookup rápido)
        batch_user_emb = user_emb[batch_users]
        batch_pos_item_emb = item_emb[batch_pos_items]
        batch_neg_item_emb = item_emb[batch_neg_items]

        # Calcular scores
        pos_scores = (batch_user_emb * batch_pos_item_emb).sum(dim=1)
        neg_scores = (batch_user_emb * batch_neg_item_emb).sum(dim=1)

        # BPR loss
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def train_epoch_full_gradient(model, edge_index, item_features, train_df, neg_dict, optimizer, device, is_lightgcn=False):
    """
    Entrena un epoch usando gradiente completo (sin mini-batches).
    Más rápido que la versión con pandas iterrows() pero más lento que batched.

    Usa vectorización de numpy para preparar datos.
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    if is_lightgcn:
        user_emb, item_emb = model(edge_index)
    else:
        user_emb, item_emb = model(edge_index, item_features)

    # Preparar datos vectorizados (sin pandas iterrows!)
    users = train_df['user_idx'].values
    pos_items = train_df['item_idx'].values

    # Vectorizar negative sampling
    neg_items = np.zeros(len(users), dtype=np.int64)
    for idx, user in enumerate(users):
        if user in neg_dict and len(neg_dict[user]) > 0:
            neg_items[idx] = np.random.choice(neg_dict[user])
        else:
            neg_items[idx] = np.random.randint(0, model.num_items)

    # Convertir a tensors
    users_t = torch.LongTensor(users).to(device)
    pos_items_t = torch.LongTensor(pos_items).to(device)
    neg_items_t = torch.LongTensor(neg_items).to(device)

    # Calcular scores
    pos_scores = (user_emb[users_t] * item_emb[pos_items_t]).sum(dim=1)
    neg_scores = (user_emb[users_t] * item_emb[neg_items_t]).sum(dim=1)

    # BPR loss
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

    loss.backward()
    optimizer.step()

    return loss.item()


def create_train_loader(train_df, neg_dict, num_items, batch_size=1024, shuffle=True, num_workers=0):
    """
    Crea un DataLoader optimizado para entrenamiento.

    Args:
        train_df: DataFrame con columnas ['user_idx', 'item_idx']
        neg_dict: Dict[user_idx -> List[negative_item_idx]]
        num_items: Número total de items
        batch_size: Tamaño del batch (default: 1024)
        shuffle: Si mezclar los datos (default: True)
        num_workers: Número de workers para carga paralela (default: 0)

    Returns:
        DataLoader
    """
    dataset = InteractionDataset(train_df, neg_dict, num_items)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return loader
