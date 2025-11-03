import pandas as pd
import numpy as np
from pathlib import Path

def decode_snowflake_timestamp(tweet_id):
    try:
        timestamp_ms = ((int(tweet_id) >> 22) + 1288834974657)
        return pd.to_datetime(timestamp_ms, unit='ms')
    except:
        return pd.NaT

def process_dataset(dataset_name, cutoff_percentile=0.8):
    print(f"\n{'='*80}")
    print(f"PROCESANDO {dataset_name.upper()}")
    print(f"{'='*80}")

    df = pd.read_csv(f'data_processing/processed_round2/{dataset_name}_processed.csv', sep=';')

    print(f"\nDatos originales:")
    print(f"  Interacciones: {len(df):,}")
    print(f"  Usuarios: {df['child_user_id'].nunique():,}")
    print(f"  Items: {df['source_tree_id'].nunique():,}")

    df['timestamp'] = df['child_tweet_id'].apply(decode_snowflake_timestamp)
    df['child_datetime'] = pd.to_datetime(df['child_datetime'])
    df['timestamp'] = df['timestamp'].fillna(df['child_datetime'])

    print(f"\nTimestamps: {df['timestamp'].min()} → {df['timestamp'].max()}")

    df_collapsed = df.groupby(['child_user_id', 'source_tree_id']).agg({
        'timestamp': 'min',
        'text': 'first',
        'parent_label': 'first',
        'tree_label': 'first',
        'child_user_id': 'size'
    }).reset_index()

    df_collapsed.columns = ['child_user_id', 'source_tree_id', 'timestamp', 'text',
                            'parent_label', 'tree_label', 'interaction_count']

    CAP = 10
    df_collapsed['interaction_count'] = df_collapsed['interaction_count'].clip(upper=CAP)

    print(f"\nColapsando cascadas:")
    print(f"  Antes: {len(df):,} interacciones")
    print(f"  Después: {len(df_collapsed):,} interacciones únicas")
    print(f"  Cap: {CAP} max por user-item")

    sorted_times = df_collapsed['timestamp'].sort_values()
    cutoff_T = sorted_times.quantile(cutoff_percentile)

    print(f"\nCutoff temporal T (percentil {cutoff_percentile*100:.0f}%): {cutoff_T}")

    df_collapsed['split'] = 'test'
    df_collapsed.loc[df_collapsed['timestamp'] < cutoff_T, 'split'] = 'train'

    train_df = df_collapsed[df_collapsed['split'] == 'train']
    test_df = df_collapsed[df_collapsed['split'] == 'test']

    print(f"\nSplit temporal:")
    print(f"  Train: {len(train_df):,} ({len(train_df)/len(df_collapsed)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/len(df_collapsed)*100:.1f}%)")

    train_users = set(train_df['child_user_id'].unique())
    test_users = set(test_df['child_user_id'].unique())
    cold_start_users = test_users - train_users

    train_items = set(train_df['source_tree_id'].unique())
    test_items = set(test_df['source_tree_id'].unique())
    cold_start_items = test_items - train_items

    print(f"\nAnálisis de cold-start:")
    print(f"  Items en train: {len(train_items):,}")
    print(f"  Items en test: {len(test_items):,}")
    print(f"  Cold-start items: {len(cold_start_items):,} ({len(cold_start_items)/len(test_items)*100:.1f}%)")
    print(f"  Cold-start users: {len(cold_start_users):,} ({len(cold_start_users)/len(test_users)*100:.1f}%)")

    output_dir = Path(f'data_processing/processed_{dataset_name}_global_temporal')
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'{dataset_name}_processed.csv'
    df_collapsed.to_csv(output_file, sep=';', index=False)

    print(f"\nGuardado en: {output_file}")

    return {
        'dataset': dataset_name,
        'total_interactions': len(df_collapsed),
        'train_interactions': len(train_df),
        'test_interactions': len(test_df),
        'cold_start_items': len(cold_start_items),
        'cutoff_T': cutoff_T
    }

def main():
    print("="*80)
    print("SPLIT TEMPORAL GLOBAL - COLD-START DE ITEMS")
    print("="*80)
    print("\nObjetivo: Recomendar tweets NUEVOS (escenario online real)")
    print("Split por timestamp de tweets, no por usuario")

    results = []
    for dataset in ['twitter15', 'twitter16']:
        result = process_dataset(dataset, cutoff_percentile=0.8)
        results.append(result)

    print(f"\n{'='*80}")
    print("RESUMEN")
    print(f"{'='*80}")

    for result in results:
        print(f"\n{result['dataset'].upper()}:")
        print(f"  Train: {result['train_interactions']:,} interacciones")
        print(f"  Test:  {result['test_interactions']:,} interacciones")
        print(f"  Cold-start items: {result['cold_start_items']:,}")

    print(f"\n{'='*80}")
    print("Próximo paso: Entrenar con grafo social + BERT")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
