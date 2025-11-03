import pandas as pd
import numpy as np
from collections import defaultdict

DATA_PATH = 'data_processing/processed_round2'
OUTPUT_PATH = 'data_processing/processed_round2'
N_NEGATIVES = 12
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

def load_data():
    df15 = pd.read_csv(f'{DATA_PATH}/twitter15_processed.csv', sep=';')
    df16 = pd.read_csv(f'{DATA_PATH}/twitter16_processed.csv', sep=';')

    ua15 = pd.read_csv(f'{DATA_PATH}/twitter15_user_activity.csv', sep=';')
    ua16 = pd.read_csv(f'{DATA_PATH}/twitter16_user_activity.csv', sep=';')

    df = pd.concat([df15, df16], ignore_index=True)
    user_activity = pd.concat([ua15, ua16], ignore_index=True)

    df['source_datetime'] = pd.to_datetime(df['source_datetime'])
    user_activity['first_activity'] = pd.to_datetime(user_activity['first_activity'])
    user_activity['last_activity'] = pd.to_datetime(user_activity['last_activity'])

    return df, user_activity

def build_structures(df):
    user_items = defaultdict(set)
    for _, row in df.iterrows():
        user_items[row['child_user_id']].add(row['source_tree_id'])

    item_timestamps = df.groupby('source_tree_id')['source_datetime'].min().to_dict()

    train_df = df[df['split'] == 'train']
    item_popularity = train_df.groupby('source_tree_id').size().to_dict()

    return user_items, item_timestamps, item_popularity

def sample_negatives(user_id, user_items, item_timestamps, item_popularity, user_activity):
    user_window = user_activity[user_activity['child_user_id'] == user_id]
    if len(user_window) == 0:
        return []

    start = user_window.iloc[0]['first_activity']
    end = user_window.iloc[0]['last_activity']

    available_items = [
        item_id for item_id, ts in item_timestamps.items()
        if start <= ts <= end and item_id not in user_items[user_id]
    ]

    if len(available_items) < N_NEGATIVES:
        return available_items

    sorted_items = sorted(available_items, key=lambda x: item_popularity.get(x, 0), reverse=True)

    n_popular = N_NEGATIVES // 2
    n_unpopular = N_NEGATIVES - n_popular

    popular_pool = sorted_items[:len(sorted_items)//2]
    unpopular_pool = sorted_items[len(sorted_items)//2:]

    popular_samples = np.random.choice(
        popular_pool,
        size=min(n_popular, len(popular_pool)),
        replace=False
    ).tolist()

    unpopular_samples = np.random.choice(
        unpopular_pool,
        size=min(n_unpopular, len(unpopular_pool)),
        replace=False
    ).tolist()

    return popular_samples + unpopular_samples

def main():
    print('Loading data...')
    df, user_activity = load_data()

    print(f'Total interactions: {len(df)}')
    print(f'Unique users: {df["child_user_id"].nunique()}')
    print(f'Unique items: {df["source_tree_id"].nunique()}')

    print('Building structures...')
    user_items, item_timestamps, item_popularity = build_structures(df)

    print('Generating negative samples...')
    all_negatives = []

    for user_id in user_activity['child_user_id'].unique():
        negatives = sample_negatives(
            user_id,
            user_items,
            item_timestamps,
            item_popularity,
            user_activity
        )

        for item_id in negatives:
            all_negatives.append({
                'user_id': user_id,
                'item_id': item_id,
                'label': 0
            })

    neg_df = pd.DataFrame(all_negatives)

    print(f'Generated {len(neg_df)} negative samples')
    print(f'Avg negatives per user: {len(neg_df) / user_activity["child_user_id"].nunique():.2f}')

    output_file = f'{OUTPUT_PATH}/negative_samples.csv'
    neg_df.to_csv(output_file, index=False)
    print(f'Saved to {output_file}')

if __name__ == '__main__':
    main()
