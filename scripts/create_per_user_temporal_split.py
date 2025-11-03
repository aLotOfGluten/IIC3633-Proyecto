"""
Per-User Temporal Split: Mantiene temporalidad SIN cold-start
Divide las interacciones de CADA usuario por tiempo (80/10/10)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("="*80)
    print("PER-USER TEMPORAL SPLIT")
    print("="*80)

    # Cargar datos
    print("\n1. Cargando datos...")
    df15 = pd.read_csv('data_processing/processed_round2/twitter15_processed.csv', sep=';')
    df16 = pd.read_csv('data_processing/processed_round2/twitter16_processed.csv', sep=';')
    df = pd.concat([df15, df16], ignore_index=True)

    print(f"   Total interacciones: {len(df):,}")
    print(f"   Usuarios únicos: {df['child_user_id'].nunique():,}")
    print(f"   Items únicos: {df['source_tree_id'].nunique():,}")

    # Convertir timestamps
    print("\n2. Ordenando por tiempo...")
    df['child_datetime'] = pd.to_datetime(df['child_datetime'])
    df = df.sort_values(['child_user_id', 'child_datetime']).reset_index(drop=True)

    # Per-user temporal split
    print("\n3. Aplicando per-user temporal split...")
    print("   (Cada usuario divide SUS interacciones 80/10/10 por tiempo)")

    df['split'] = 'train'

    user_groups = df.groupby('child_user_id')
    train_indices = []
    val_indices = []
    test_indices = []

    users_with_enough_data = 0
    users_filtered = 0

    for user_id, group in user_groups:
        n = len(group)

        # Filtrar usuarios con pocas interacciones
        if n < 5:
            users_filtered += 1
            continue

        users_with_enough_data += 1
        indices = group.index.tolist()

        # Split temporal por usuario
        train_size = max(1, int(0.8 * n))
        val_size = max(0, int(0.1 * n))

        train_indices.extend(indices[:train_size])
        val_indices.extend(indices[train_size:train_size+val_size])
        test_indices.extend(indices[train_size+val_size:])

    df.loc[train_indices, 'split'] = 'train'
    df.loc[val_indices, 'split'] = 'val'
    df.loc[test_indices, 'split'] = 'test'

    # Filtrar usuarios con < 5 interacciones
    df = df[df['split'].isin(['train', 'val', 'test'])].reset_index(drop=True)

    print(f"   Usuarios con >= 5 interacciones: {users_with_enough_data:,}")
    print(f"   Usuarios filtrados (< 5 interacciones): {users_filtered:,}")

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    print(f"\n   Train: {len(train_df):,} interacciones ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df):,} interacciones ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df):,} interacciones ({len(test_df)/len(df)*100:.1f}%)")

    # Verificar overlap de items
    print("\n4. Verificando overlap de items...")
    train_items = set(train_df['source_tree_id'].unique())
    val_items = set(val_df['source_tree_id'].unique())
    test_items = set(test_df['source_tree_id'].unique())

    print(f"   Items únicos en train: {len(train_items):,}")
    print(f"   Items únicos en val: {len(val_items):,}")
    print(f"   Items únicos en test: {len(test_items):,}")
    print(f"   Items en train Y val: {len(train_items & val_items):,} ({len(train_items & val_items)/len(val_items)*100:.1f}%)")
    print(f"   Items en train Y test: {len(train_items & test_items):,} ({len(train_items & test_items)/len(test_items)*100:.1f}%)")

    # Verificar temporalidad
    print("\n5. Verificando orden temporal...")
    violations = 0
    for user_id in df['child_user_id'].unique():
        user_df = df[df['child_user_id'] == user_id]

        train_user = user_df[user_df['split'] == 'train']
        val_user = user_df[user_df['split'] == 'val']
        test_user = user_df[user_df['split'] == 'test']

        if len(train_user) > 0 and len(val_user) > 0:
            if train_user['child_datetime'].max() > val_user['child_datetime'].min():
                violations += 1

        if len(val_user) > 0 and len(test_user) > 0:
            if val_user['child_datetime'].max() > test_user['child_datetime'].min():
                violations += 1

    print(f"   Violaciones de orden temporal: {violations}")
    print(f"   ✓ Orden temporal respetado: {violations == 0}")

    # Guardar
    print("\n6. Guardando datasets...")
    output_dir = Path('data_processing/processed_round2_per_user_temporal')
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

    # Estadísticas temporales
    print("\n7. Estadísticas temporales...")
    print(f"   Train timestamps: {train_df['child_datetime'].min()} → {train_df['child_datetime'].max()}")
    print(f"   Val timestamps:   {val_df['child_datetime'].min()} → {val_df['child_datetime'].max()}")
    print(f"   Test timestamps:  {test_df['child_datetime'].min()} → {test_df['child_datetime'].max()}")

    # Análisis de cold-start
    val_cold = len(val_items - train_items)
    test_cold = len(test_items - train_items)

    print(f"\n8. Análisis de cold-start...")
    print(f"   Items cold-start en val: {val_cold} ({val_cold/len(val_items)*100:.1f}%)")
    print(f"   Items cold-start en test: {test_cold} ({test_cold/len(test_items)*100:.1f}%)")

    print("\n" + "="*80)
    print("✓ PER-USER TEMPORAL SPLIT COMPLETADO")
    print("="*80)
    print(f"\nArchivos en: {output_dir}/")
    print("\nCaracterísticas:")
    print("  ✓ Mantiene orden temporal POR USUARIO")
    print("  ✓ Predice el futuro de cada usuario basado en su pasado")
    print(f"  ✓ Item overlap train-test: ~{len(train_items & test_items)/len(test_items)*100:.0f}%")
    print("\nPróximo paso:")
    print("  1. Actualizar build_temporal_graphs.py para usar 'processed_round2_per_user_temporal'")
    print("  2. Regenerar grafos")
    print("  3. Re-entrenar modelos")

if __name__ == '__main__':
    main()
