"""
Script para crear split RANDOM (no temporal) para RecSys
Divide interacciones aleatoriamente garantizando overlap de items.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def main():
    print("="*80)
    print("CREANDO RANDOM SPLIT PARA RECSYS")
    print("="*80)

    # Cargar datos
    print("\n1. Cargando datos...")
    df15 = pd.read_csv('data_processing/processed_round2/twitter15_processed.csv', sep=';')
    df16 = pd.read_csv('data_processing/processed_round2/twitter16_processed.csv', sep=';')
    df = pd.concat([df15, df16], ignore_index=True)

    print(f"   Total interacciones: {len(df):,}")
    print(f"   Usuarios únicos: {df['child_user_id'].nunique():,}")
    print(f"   Items únicos: {df['source_tree_id'].nunique():,}")

    # Crear split aleatorio estratificado por item
    # Para cada item, dividir sus interacciones 80/10/10
    print("\n2. Creando split aleatorio estratificado por item...")
    print("   (Cada item tendrá interacciones en train, val y test)")

    np.random.seed(42)
    df['split'] = 'train'  # default

    # Agrupar por item
    item_groups = df.groupby('source_tree_id')

    train_indices = []
    val_indices = []
    test_indices = []

    for item_id, group in item_groups:
        n = len(group)
        indices = group.index.tolist()
        np.random.shuffle(indices)

        # Calcular tamaños
        train_size = max(1, int(0.8 * n))  # al menos 1 en train
        val_size = int(0.1 * n)

        train_indices.extend(indices[:train_size])
        val_indices.extend(indices[train_size:train_size+val_size])
        test_indices.extend(indices[train_size+val_size:])

    df.loc[train_indices, 'split'] = 'train'
    df.loc[val_indices, 'split'] = 'val'
    df.loc[test_indices, 'split'] = 'test'

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    print(f"   Train: {len(train_df):,} interacciones ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df):,} interacciones ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df):,} interacciones ({len(test_df)/len(df)*100:.1f}%)")

    # Verificar overlap de items
    print("\n3. Verificando overlap de items entre splits...")
    train_items = set(train_df['source_tree_id'].unique())
    val_items = set(val_df['source_tree_id'].unique())
    test_items = set(test_df['source_tree_id'].unique())

    print(f"   Items únicos en train: {len(train_items):,}")
    print(f"   Items únicos en val: {len(val_items):,}")
    print(f"   Items únicos en test: {len(test_items):,}")
    print(f"   Items en train Y val: {len(train_items & val_items):,}")
    print(f"   Items en train Y test: {len(train_items & test_items):,}")
    print(f"   Items en TODOS los splits: {len(train_items & val_items & test_items):,}")

    # Verificar que todos los items estén en train
    all_items_in_train = len(train_items) == df['source_tree_id'].nunique()
    print(f"\n   ✓ TODOS los items aparecen en train: {all_items_in_train}")

    # Verificar overlap de usuarios
    print("\n4. Verificando overlap de usuarios entre splits...")
    train_users = set(train_df['child_user_id'].unique())
    val_users = set(val_df['child_user_id'].unique())
    test_users = set(test_df['child_user_id'].unique())

    print(f"   Usuarios únicos en train: {len(train_users):,}")
    print(f"   Usuarios únicos en val: {len(val_users):,}")
    print(f"   Usuarios únicos en test: {len(test_users):,}")
    print(f"   Usuarios en train Y val: {len(train_users & val_users):,}")
    print(f"   Usuarios en train Y test: {len(train_users & test_users):,}")

    # Guardar datasets
    print("\n5. Guardando datasets con split random...")
    output_dir = Path('data_processing/processed_round2_random')
    output_dir.mkdir(exist_ok=True)

    # Separar por dataset original
    twitter15_items = set(df15['source_tree_id'].unique())
    twitter16_items = set(df16['source_tree_id'].unique())

    df15_new = df[df['source_tree_id'].isin(twitter15_items)]
    df16_new = df[df['source_tree_id'].isin(twitter16_items)]

    df15_new.to_csv(output_dir / 'twitter15_processed.csv', sep=';', index=False)
    df16_new.to_csv(output_dir / 'twitter16_processed.csv', sep=';', index=False)

    print(f"   ✓ Guardado: {output_dir / 'twitter15_processed.csv'}")
    print(f"   ✓ Guardado: {output_dir / 'twitter16_processed.csv'}")

    # Estadísticas por item
    print("\n6. Estadísticas de interacciones por item...")
    item_counts = df.groupby('source_tree_id').size()
    print(f"   Media interacciones por item: {item_counts.mean():.2f}")
    print(f"   Mediana: {item_counts.median():.0f}")
    print(f"   Min: {item_counts.min()}")
    print(f"   Max: {item_counts.max()}")

    items_with_all_splits = 0
    for item_id in df['source_tree_id'].unique():
        item_df = df[df['source_tree_id'] == item_id]
        splits = set(item_df['split'].unique())
        if len(splits) == 3:
            items_with_all_splits += 1

    print(f"\n   Items que aparecen en los 3 splits: {items_with_all_splits} ({items_with_all_splits/df['source_tree_id'].nunique()*100:.1f}%)")

    print("\n" + "="*80)
    print("✓ RANDOM SPLIT COMPLETADO")
    print("="*80)
    print(f"\nArchivos generados en: {output_dir}/")
    print("\nSiguientes pasos:")
    print("  1. Actualizar build_temporal_graphs.py para usar 'processed_round2_random'")
    print("  2. Ejecutar build_temporal_graphs.py")
    print("  3. Re-entrenar modelos")

if __name__ == '__main__':
    main()
