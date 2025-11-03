import pandas as pd
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.sparse import csr_matrix

DATA_PATH = 'data_processing/processed_round2_per_user_temporal'
OUTPUT_PATH = 'midterm/graphs_per_user_temporal'
MAX_EDGE_CAP = 5
MIN_COMMON_ITEMS = 3

def load_and_merge():
    df15 = pd.read_csv(f'{DATA_PATH}/twitter15_processed.csv', sep=';')
    df16 = pd.read_csv(f'{DATA_PATH}/twitter16_processed.csv', sep=';')
    df = pd.concat([df15, df16], ignore_index=True)
    return df

def create_mappings(df, train_only=True):
    """
    Crea mappings usando SOLO train set (train_only=True)
    o usando todo el dataset (train_only=False).

    Para evitar data leakage, debemos usar train_only=True.
    """
    if train_only:
        train_df = df[df['split'] == 'train']
        unique_users = sorted(train_df['child_user_id'].unique())
        unique_items = sorted(train_df['source_tree_id'].unique())
        print(f"\n  IMPORTANTE: Mappings creados SOLO con train set")
        print(f"  Esto evita data leakage pero puede causar cold-start en val/test")
    else:
        unique_users = sorted(df['child_user_id'].unique())
        unique_items = sorted(df['source_tree_id'].unique())
        print(f"\n  IMPORTANTE: Mappings creados con todo el dataset")

    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}

    return user_to_idx, item_to_idx

def apply_edge_cap(df, user_to_idx, item_to_idx):
    df = df.copy()
    df['user_idx'] = df['child_user_id'].map(user_to_idx)
    df['item_idx'] = df['source_tree_id'].map(item_to_idx)

    grouped = df.groupby(['user_idx', 'item_idx']).size().reset_index(name='count')
    grouped['count'] = grouped['count'].clip(upper=MAX_EDGE_CAP)

    expanded_rows = []
    for _, row in grouped.iterrows():
        for _ in range(row['count']):
            expanded_rows.append({
                'user_idx': row['user_idx'],
                'item_idx': row['item_idx']
            })

    return pd.DataFrame(expanded_rows)

def build_bipartite(interactions_df, num_users, num_items):
    user_indices = torch.tensor(interactions_df['user_idx'].values, dtype=torch.long)
    item_indices = torch.tensor(interactions_df['item_idx'].values, dtype=torch.long)
    item_indices_shifted = item_indices + num_users

    edge_index = torch.stack([
        torch.cat([user_indices, item_indices_shifted]),
        torch.cat([item_indices_shifted, user_indices])
    ], dim=0)

    graph_data = {
        'edge_index': edge_index,
        'num_users': num_users,
        'num_items': num_items
    }

    return graph_data

def build_social(interactions_df, num_users, num_items):
    row_idx = interactions_df['user_idx'].values
    col_idx = interactions_df['item_idx'].values
    data = np.ones(len(interactions_df), dtype=np.float32)

    user_item_matrix = csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(num_users, num_items)
    )

    common_items_matrix = user_item_matrix @ user_item_matrix.T
    common_items_matrix.setdiag(0)
    common_items_matrix.eliminate_zeros()

    coo = common_items_matrix.tocoo()
    mask = coo.data >= MIN_COMMON_ITEMS

    edge_index = torch.tensor(
        np.vstack([coo.row[mask], coo.col[mask]]),
        dtype=torch.long
    )
    edge_weight = torch.tensor(coo.data[mask], dtype=torch.float)

    graph_data = {
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'num_nodes': num_users
    }

    return graph_data

def save_mappings(user_to_idx, item_to_idx, output_path):
    user_df = pd.DataFrame([
        {'user_id': uid, 'user_idx': idx}
        for uid, idx in user_to_idx.items()
    ])

    item_df = pd.DataFrame([
        {'item_id': iid, 'item_idx': idx}
        for iid, idx in item_to_idx.items()
    ])

    user_df.to_csv(output_path / 'user_map.csv', index=False)
    item_df.to_csv(output_path / 'item_map.csv', index=False)

def main():
    print('Loading temporal data...')
    df = load_and_merge()

    print(f'Total interactions: {len(df)}')
    print(f'Unique users: {df["child_user_id"].nunique()}')
    print(f'Unique items: {df["source_tree_id"].nunique()}')

    print('Creating mappings (using ONLY train set)...')
    user_to_idx, item_to_idx = create_mappings(df, train_only=True)
    num_users = len(user_to_idx)
    num_items = len(item_to_idx)

    print(f'Mapped: {num_users} users, {num_items} items')

    # Filtrar val/test para eliminar cold-start ITEMS (pero permitir cold-start users)
    print('\nFiltering val/test to remove cold-start ITEMS (allowing new users)...')
    train_items = set(item_to_idx.keys())
    all_users = df['child_user_id'].unique()

    # Agregar usuarios de val/test a los mappings si no están
    for user_id in all_users:
        if user_id not in user_to_idx:
            user_to_idx[user_id] = len(user_to_idx)

    num_users = len(user_to_idx)  # Actualizar número de usuarios

    val_df_orig = df[df['split'] == 'val']
    test_df_orig = df[df['split'] == 'test']

    # Filtrar SOLO por items (no por usuarios)
    val_df_filtered = val_df_orig[val_df_orig['source_tree_id'].isin(train_items)]
    test_df_filtered = test_df_orig[test_df_orig['source_tree_id'].isin(train_items)]

    print(f'  Val: {len(val_df_orig)} → {len(val_df_filtered)} (removed {len(val_df_orig) - len(val_df_filtered)} cold-start items)')
    print(f'  Test: {len(test_df_orig)} → {len(test_df_filtered)} (removed {len(test_df_orig) - len(test_df_filtered)} cold-start items)')
    print(f'  Total users (including new users from val/test): {num_users}')

    # Actualizar df con los splits filtrados
    df = pd.concat([
        df[df['split'] == 'train'],
        val_df_filtered,
        test_df_filtered
    ])

    print('Applying edge cap...')
    train_df = df[df['split'] == 'train']
    interactions_capped = apply_edge_cap(train_df, user_to_idx, item_to_idx)

    print(f'Interactions after cap: {len(interactions_capped)}')

    print('Building bipartite graph...')
    bipartite = build_bipartite(interactions_capped, num_users, num_items)

    print('Building social graph...')
    social = build_social(interactions_capped, num_users, num_items)

    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    print('Saving graphs...')
    torch.save(bipartite, output_path / 'bipartite_graph.pt')
    torch.save(social, output_path / 'social_graph.pt')

    print('Saving mappings...')
    save_mappings(user_to_idx, item_to_idx, output_path)

    print('Saving splits...')
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split].copy()
        split_df['user_idx'] = split_df['child_user_id'].map(user_to_idx)
        split_df['item_idx'] = split_df['source_tree_id'].map(item_to_idx)
        split_df[['user_idx', 'item_idx']].to_csv(
            output_path / f'{split}_interactions.csv',
            index=False
        )

    print('\nBipartite graph:')
    print(f'  Nodes: {num_users + num_items}')
    print(f'  Edges: {bipartite["edge_index"].shape[1]}')

    print('\nSocial graph:')
    print(f'  Nodes: {social["num_nodes"]}')
    print(f'  Edges: {social["edge_index"].shape[1]}')

    print(f'\nSaved to {output_path}/')

if __name__ == '__main__':
    main()
