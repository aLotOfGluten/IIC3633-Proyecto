"""
Script para crear split temporal CORRECTO
Divide INTERACCIONES por tiempo, no items.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("="*80)
    print("CREANDO SPLIT TEMPORAL CORRECTO")
    print("="*80)

    # Cargar datos procesados
    print("\n1. Cargando datos...")
    df15 = pd.read_csv('data_processing/processed_round2/twitter15_processed.csv', sep=';')
    df16 = pd.read_csv('data_processing/processed_round2/twitter16_processed.csv', sep=';')
    df = pd.concat([df15, df16], ignore_index=True)

    print(f"   Total interacciones: {len(df):,}")
    print(f"   Usuarios únicos: {df['child_user_id'].nunique():,}")
    print(f"   Items únicos: {df['source_tree_id'].nunique():,}")

    # Convertir timestamps a datetime
    print("\n2. Procesando timestamps...")
    df['child_datetime'] = pd.to_datetime(df['child_datetime'])
    df = df.sort_values('child_datetime').reset_index(drop=True)

    print(f"   Fecha más antigua: {df['child_datetime'].min()}")
    print(f"   Fecha más reciente: {df['child_datetime'].max()}")

    # Split temporal 80/10/10
    print("\n3. Creando split temporal 80/10/10...")
    n = len(df)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)

    df['split'] = 'test'
    df.loc[:train_size, 'split'] = 'train'
    df.loc[train_size:train_size+val_size, 'split'] = 'val'

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    print(f"   Train: {len(train_df):,} interacciones ({len(train_df)/n*100:.1f}%)")
    print(f"   Val:   {len(val_df):,} interacciones ({len(val_df)/n*100:.1f}%)")
    print(f"   Test:  {len(test_df):,} interacciones ({len(test_df)/n*100:.1f}%)")

    # Verificar overlap de items
    print("\n4. Verificando overlap de items entre splits...")
    train_items = set(train_df['source_tree_id'].unique())
    val_items = set(val_df['source_tree_id'].unique())
    test_items = set(test_df['source_tree_id'].unique())

    print(f"   Items únicos en train: {len(train_items):,}")
    print(f"   Items únicos en val: {len(val_items):,}")
    print(f"   Items únicos en test: {len(test_items):,}")
    print(f"   Items en train Y val: {len(train_items & val_items):,}")
    print(f"   Items en train Y test: {len(train_items & test_items):,}")
    print(f"   Items en val Y test: {len(val_items & test_items):,}")

    # Verificar overlap de usuarios
    print("\n5. Verificando overlap de usuarios entre splits...")
    train_users = set(train_df['child_user_id'].unique())
    val_users = set(val_df['child_user_id'].unique())
    test_users = set(test_df['child_user_id'].unique())

    print(f"   Usuarios únicos en train: {len(train_users):,}")
    print(f"   Usuarios únicos en val: {len(val_users):,}")
    print(f"   Usuarios únicos en test: {len(test_users):,}")
    print(f"   Usuarios en train Y val: {len(train_users & val_users):,}")
    print(f"   Usuarios en train Y test: {len(train_users & test_users):,}")

    # Guardar datasets con el nuevo split
    print("\n6. Guardando datasets con nuevo split temporal...")
    output_dir = Path('data_processing/processed_round2_temporal')
    output_dir.mkdir(exist_ok=True)

    # Guardar separados por dataset original usando la columna 'dataset' que ya existe
    df15_new = df15[['source_tree_id', 'is_new_post', 'parent_user_id', 'parent_tweet_id',
                      'parent_delay_min', 'child_user_id', 'child_tweet_id', 'child_delay_min',
                      'parent_label', 'source_datetime', 'child_datetime', 'text',
                      'tree_start_time', 'tree_label', 'split']].copy()
    df15_new['split'] = df15_new['source_tree_id'].map(dict(zip(df['source_tree_id'], df['split'])))

    df16_new = df16[['source_tree_id', 'is_new_post', 'parent_user_id', 'parent_tweet_id',
                      'parent_delay_min', 'child_user_id', 'child_tweet_id', 'child_delay_min',
                      'parent_label', 'source_datetime', 'child_datetime', 'text',
                      'tree_start_time', 'tree_label', 'split']].copy()
    df16_new['split'] = df16_new['source_tree_id'].map(dict(zip(df['source_tree_id'], df['split'])))

    df15_new.to_csv(output_dir / 'twitter15_processed.csv', sep=';', index=False)
    df16_new.to_csv(output_dir / 'twitter16_processed.csv', sep=';', index=False)

    print(f"   ✓ Guardado: {output_dir / 'twitter15_processed.csv'}")
    print(f"   ✓ Guardado: {output_dir / 'twitter16_processed.csv'}")

    # Estadísticas de cold-start
    print("\n7. Análisis de cold-start...")
    val_cold_items = val_items - train_items
    test_cold_items = test_items - train_items

    val_cold_interactions = val_df[val_df['source_tree_id'].isin(val_cold_items)]
    test_cold_interactions = test_df[test_df['source_tree_id'].isin(test_cold_items)]

    print(f"   Items cold-start en val: {len(val_cold_items):,} ({len(val_cold_items)/len(val_items)*100:.1f}%)")
    print(f"   Items cold-start en test: {len(test_cold_items):,} ({len(test_cold_items)/len(test_items)*100:.1f}%)")
    print(f"   Interacciones cold-start en val: {len(val_cold_interactions):,} ({len(val_cold_interactions)/len(val_df)*100:.1f}%)")
    print(f"   Interacciones cold-start en test: {len(test_cold_interactions):,} ({len(test_cold_interactions)/len(test_df)*100:.1f}%)")

    print("\n" + "="*80)
    print("✓ SPLIT TEMPORAL COMPLETADO")
    print("="*80)
    print(f"\nArchivos generados en: {output_dir}/")
    print("\nSiguientes pasos:")
    print("  1. Ejecutar build_temporal_graphs.py con los nuevos datos")
    print("  2. Re-entrenar los modelos con los grafos corregidos")

if __name__ == '__main__':
    main()
