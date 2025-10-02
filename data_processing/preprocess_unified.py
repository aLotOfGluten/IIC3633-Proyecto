"""
Pipeline de Preprocesamiento Unificado: Twitter15 + Twitter16
Prepara el dataset combinado para sistemas de recomendación y GNNs.
"""

import re
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Set, List, DefaultDict


class Config:
    DATASETS = ["twitter15", "twitter16"]
    OUTPUT_DIR = "processed_h1"
    CONTENT_DIR = "content"
    # Nota: Con MIN_INTERACTIONS=8, usuarios con exactamente 8 interacciones tendrán
    # 7 en train y 1 en test (última interacción va a test)
    MIN_INTERACTIONS = 8
    RANDOM_STATE = 42
    LABEL_STRATEGY = '4class'  # '4class', '3class', 'binary_strict', 'binary_all'


class StatsLogger:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.stats_file = self.output_dir / "stats_summary.txt"
        self.logs: List[str] = []

    def log(self, message: str) -> None:
        print(message)
        self.logs.append(message)

    def save(self) -> None:
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.logs))
        print(f"\n* Estadísticas guardadas en: {self.stats_file}")


def clean_text(text: str) -> str:
    """Limpia y normaliza el texto de tweets."""
    text = re.sub(r'http\S+|www\.\S+', 'URL', text)
    text = re.sub(r'@\w+', '@USER', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s@]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def load_data(config: Config, logger: StatsLogger) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, str]]:
    """Carga labels y contenido de tweets desde ambos datasets."""
    logger.log("\n" + "─" * 80)
    logger.log("PASO 1: Cargando labels y contenido de tweets")
    logger.log("─" * 80)

    labels_dict: Dict[str, Tuple[str, str]] = {}
    tweets_text: Dict[str, str] = {}
    label_counts_by_dataset: Dict[str, Dict[str, int]] = {}

    for dataset in config.DATASETS:
        label_file = Path(dataset) / "label.txt"
        tweets_file = Path(config.CONTENT_DIR) / dataset / "source_tweets.txt"

        if not label_file.exists():
            logger.log(f"WARN Advertencia: {label_file} no encontrado, saltando...")
            continue

        dataset_labels: Dict[str, int] = {}
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    label, item_id = line.split(':', 1)
                    full_item_id = f"{dataset}_{item_id}"
                    labels_dict[full_item_id] = (label, dataset)
                    dataset_labels[label] = dataset_labels.get(label, 0) + 1

        if tweets_file.exists():
            with open(tweets_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        item_id, text = parts
                        full_item_id = f"{dataset}_{item_id}"
                        tweets_text[full_item_id] = clean_text(text)

        label_counts_by_dataset[dataset] = dataset_labels
        logger.log(f"\n{dataset}: {sum(dataset_labels.values())} items cargados")
        for label, count in sorted(dataset_labels.items(), key=lambda x: -x[1]):
            logger.log(f"  • {label:12s}: {count:4d} ({count/sum(dataset_labels.values())*100:5.1f}%)")

    logger.log(f"\n* Total combinado: {len(labels_dict)} items únicos")
    logger.log(f"* Tweets con texto: {len(tweets_text)}")

    global_label_counts: DefaultDict[str, int] = defaultdict(int)
    for label, dataset in labels_dict.values():
        global_label_counts[label] += 1

    logger.log(f"\nDistribución global de labels:")
    for label, count in sorted(global_label_counts.items(), key=lambda x: -x[1]):
        logger.log(f"  • {label:12s}: {count:4d} ({count/len(labels_dict)*100:5.1f}%)")

    return labels_dict, tweets_text


def extract_interactions(
    config: Config,
    logger: StatsLogger,
    labels_dict: Dict[str, Tuple[str, str]]
) -> Tuple[pd.DataFrame, DefaultDict[str, List[Tuple[str, float]]], DefaultDict[str, Set[str]]]:
    """Extrae interacciones user-item desde árboles de propagación."""
    logger.log("\n" + "─" * 80)
    logger.log("PASO 2: Extrayendo interacciones desde árboles de propagación")
    logger.log("─" * 80)

    pattern = re.compile(r"\['([^']+)',\s*'([^']+)',\s*'([^']+)'\]")
    user_items: DefaultDict[str, List[Tuple[str, float]]] = defaultdict(list)
    item_users: DefaultDict[str, Set[str]] = defaultdict(set)

    users_list: List[str] = []
    items_list: List[str] = []
    timestamps_list: List[float] = []
    interactions_list: List[int] = []
    datasets_list: List[str] = []

    total_trees = 0
    processed_trees = 0
    skipped_trees = 0

    for dataset in config.DATASETS:
        tree_dir = Path(dataset) / "tree"

        if not tree_dir.exists():
            logger.log(f"WARN Advertencia: {tree_dir} no encontrado, saltando...")
            continue

        tree_files = sorted([f.name for f in tree_dir.glob('*.txt')])
        total_trees += len(tree_files)
        logger.log(f"\nProcesando {dataset}: {len(tree_files)} árboles...")

        for idx, tree_file in enumerate(tree_files, 1):
            if idx % 200 == 0:
                logger.log(f"  {dataset}: {idx}/{len(tree_files)} árboles procesados...")

            raw_item_id = tree_file.replace('.txt', '')
            full_item_id = f"{dataset}_{raw_item_id}"

            if full_item_id not in labels_dict:
                skipped_trees += 1
                continue

            tree_path = tree_dir / tree_file
            tree_users: List[Tuple[str, float]] = []

            with open(tree_path, 'r', encoding='utf-8') as f:
                for line in f:
                    matches = pattern.findall(line)
                    for match in matches:
                        user_id, tweet_id, timestamp = match
                        if user_id == 'ROOT':
                            continue
                        try:
                            ts_float = float(timestamp)
                            tree_users.append((user_id, ts_float))
                        except ValueError:
                            tree_users.append((user_id, 0.0))

            user_last_ts: Dict[str, float] = {}
            for user_id, timestamp in tree_users:
                if user_id not in user_last_ts or timestamp > user_last_ts[user_id]:
                    user_last_ts[user_id] = timestamp

            for user_id, timestamp in user_last_ts.items():
                user_items[user_id].append((full_item_id, timestamp))
                item_users[full_item_id].add(user_id)

                users_list.append(user_id)
                items_list.append(full_item_id)
                timestamps_list.append(timestamp)
                interactions_list.append(1)
                datasets_list.append(dataset)

            processed_trees += 1

    all_interactions_df = pd.DataFrame({
        'user_id': users_list,
        'item_id': items_list,
        'timestamp': timestamps_list,
        'interaction': interactions_list,
        'dataset': datasets_list
    })

    logger.log(f"\n* Total de árboles: {total_trees}")
    logger.log(f"* Árboles procesados: {processed_trees}")
    logger.log(f"* Árboles sin label (ignorados): {skipped_trees}")
    logger.log(f"* Interacciones extraídas: {len(all_interactions_df)}")
    logger.log(f"* Usuarios únicos: {len(user_items)}")
    logger.log(f"* Items únicos: {len(item_users)}")

    dataset_interactions = all_interactions_df['dataset'].value_counts()

    logger.log(f"\nInteracciones por dataset:")
    for dataset, count in dataset_interactions.items():
        logger.log(f"  • {dataset}: {count} ({count/len(all_interactions_df)*100:.1f}%)")

    return all_interactions_df, user_items, item_users


def filter_users(
    config: Config,
    logger: StatsLogger,
    all_interactions_df: pd.DataFrame,
    user_items: DefaultDict[str, List[Tuple[str, float]]]
) -> Tuple[pd.DataFrame, Set[str]]:
    """Filtra usuarios con menos de MIN_INTERACTIONS interacciones."""
    logger.log("\n" + "─" * 80)
    logger.log(f"PASO 3: Filtrando usuarios con < {config.MIN_INTERACTIONS} interacciones")
    logger.log("─" * 80)

    user_counts = {user: len(items) for user, items in user_items.items()}
    counts_list = list(user_counts.values())

    logger.log(f"\nEstadísticas ANTES del filtrado:")
    logger.log(f"  • Total usuarios: {len(user_counts)}")
    logger.log(f"  • Media interacciones: {np.mean(counts_list):.2f}")
    logger.log(f"  • Mediana interacciones: {np.median(counts_list):.0f}")
    logger.log(f"  • Min interacciones: {min(counts_list)}")
    logger.log(f"  • Max interacciones: {max(counts_list)}")

    valid_users = {user for user, count in user_counts.items()
                   if count >= config.MIN_INTERACTIONS}

    logger.log(f"\nEstadísticas DESPUÉS del filtrado:")
    logger.log(f"  • Usuarios válidos: {len(valid_users)}")
    logger.log(f"  • Usuarios eliminados: {len(user_counts) - len(valid_users)}")
    logger.log(f"  • % retenido: {len(valid_users)/len(user_counts)*100:.1f}%")

    filtered_interactions_df = all_interactions_df[all_interactions_df['user_id'].isin(valid_users)]

    logger.log(f"  • Interacciones válidas: {len(filtered_interactions_df)}")
    logger.log(f"  • Interacciones eliminadas: {len(all_interactions_df) - len(filtered_interactions_df)}")

    filtered_counts = [user_counts[user] for user in valid_users]
    if filtered_counts:
        logger.log(f"\nEstadísticas de usuarios filtrados:")
        logger.log(f"  • Media interacciones: {np.mean(filtered_counts):.2f}")
        logger.log(f"  • Mediana interacciones: {np.median(filtered_counts):.0f}")
        logger.log(f"  • Min interacciones: {min(filtered_counts)}")
        logger.log(f"  • Max interacciones: {max(filtered_counts)}")

    return filtered_interactions_df, valid_users


def split_train_test(
    logger: StatsLogger,
    filtered_interactions_df: pd.DataFrame,
    valid_users: Set[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en train/test usando leave-one-out por usuario."""
    logger.log("\n" + "─" * 80)
    logger.log("PASO 4: División Train/Test (última interacción de cada usuario → test)")
    logger.log("─" * 80)

    last_interaction_indices = filtered_interactions_df.loc[
        filtered_interactions_df.groupby('user_id')['timestamp'].idxmax()
    ].index

    test_df = filtered_interactions_df.loc[last_interaction_indices]
    train_df = filtered_interactions_df.drop(last_interaction_indices)

    train_df = train_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    test_df = test_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    logger.log(f"\nResultado del split:")
    logger.log(f"  • Train: {len(train_df)} interacciones")
    logger.log(f"  • Test:  {len(test_df)} interacciones")
    logger.log(f"  • Ratio: {len(test_df)/len(filtered_interactions_df)*100:.1f}% test")

    test_users = set(test_df['user_id'])
    logger.log(f"\n  • Usuarios en test: {len(test_users)}")
    logger.log(f"  • Usuarios válidos: {len(valid_users)}")
    logger.log(f"  • * Verificación (cada usuario tiene 1 test): {len(test_users) == len(valid_users)}")

    return train_df, test_df


def create_mappings(
    config: Config,
    logger: StatsLogger,
    valid_users: Set[str],
    item_users: DefaultDict[str, Set[str]]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Crea mapeos de IDs a índices numéricos."""
    logger.log("\n" + "─" * 80)
    logger.log("PASO 5: Generando mapeos globales user_id → user_idx e item_id → item_idx")
    logger.log("─" * 80)

    unique_users = sorted(valid_users)
    unique_items = sorted(item_users.keys())
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}

    logger.log(f"\n* {len(user_to_idx)} usuarios mapeados (0 a {len(user_to_idx)-1})")
    logger.log(f"* {len(item_to_idx)} items mapeados (0 a {len(item_to_idx)-1})")

    user_map_df = pd.DataFrame({'user_id': unique_users, 'user_idx': range(len(unique_users))})
    item_map_df = pd.DataFrame({'item_id': unique_items, 'item_idx': range(len(unique_items))})

    output_dir = Path(config.OUTPUT_DIR)
    user_map_path = output_dir / "user_map.csv"
    item_map_path = output_dir / "item_map.csv"
    user_map_df.to_csv(user_map_path, index=False)
    item_map_df.to_csv(item_map_path, index=False)

    logger.log(f"\n* Guardado: {user_map_path}")
    logger.log(f"* Guardado: {item_map_path}")

    return user_to_idx, item_to_idx


def save_interactions(
    config: Config,
    logger: StatsLogger,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_to_idx: Dict[str, int],
    item_to_idx: Dict[str, int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Guarda interacciones train/test con IDs e índices."""
    logger.log("\n" + "─" * 80)
    logger.log("PASO 6: Guardando interacciones train/test")
    logger.log("─" * 80)

    output_dir = Path(config.OUTPUT_DIR)

    train_path = output_dir / "train_interactions.csv"
    test_path = output_dir / "test_interactions.csv"
    train_df[['user_id', 'item_id', 'interaction']].to_csv(train_path, index=False)
    test_df[['user_id', 'item_id', 'interaction']].to_csv(test_path, index=False)

    logger.log(f"\n* Guardado: {train_path}")
    logger.log(f"* Guardado: {test_path}")

    train_df['user_idx'] = train_df['user_id'].map(user_to_idx)
    train_df['item_idx'] = train_df['item_id'].map(item_to_idx)
    test_df['user_idx'] = test_df['user_id'].map(user_to_idx)
    test_df['item_idx'] = test_df['item_id'].map(item_to_idx)

    train_idx_path = output_dir / "train_interactions_idx.csv"
    test_idx_path = output_dir / "test_interactions_idx.csv"
    train_df[['user_idx', 'item_idx', 'interaction']].to_csv(train_idx_path, index=False)
    test_df[['user_idx', 'item_idx', 'interaction']].to_csv(test_idx_path, index=False)

    logger.log(f"* Guardado: {train_idx_path}")
    logger.log(f"* Guardado: {test_idx_path}")

    return train_df, test_df


def get_label_info(label_str: str, strategy: str) -> Tuple[int, bool]:
    """Obtiene índice de label según estrategia."""
    if strategy == '4class':
        label_map = {'true': 0, 'false': 1, 'unverified': 2, 'non-rumor': 3}
        return label_map.get(label_str, -1), True
    elif strategy == '3class':
        if label_str == 'true':
            return 0, True
        elif label_str == 'false':
            return 1, True
        else:
            return 2, True
    elif strategy == 'binary_strict':
        if label_str == 'true':
            return 1, True
        elif label_str == 'false':
            return 0, True
        else:
            return -1, False
    elif strategy == 'binary_all':
        return (1, True) if label_str == 'true' else (0, True)
    return -1, False


def save_labels_and_text(
    config: Config,
    logger: StatsLogger,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels_dict: Dict[str, Tuple[str, str]],
    tweets_text: Dict[str, str],
    item_to_idx: Dict[str, int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Guarda labels y texto limpio de tweets."""
    logger.log("\n" + "─" * 80)
    logger.log(f"PASO 7: Generando labels - Estrategia: {config.LABEL_STRATEGY}")
    logger.log("─" * 80)

    items_in_data = set(train_df['item_id']) | set(test_df['item_id'])
    item_labels_list: List[Dict] = []

    excluded_count = 0

    for item_id in sorted(items_in_data):
        if item_id in labels_dict:
            label_str, dataset_origin = labels_dict[item_id]
            label_idx, include = get_label_info(label_str, config.LABEL_STRATEGY)

            if not include:
                excluded_count += 1
                continue

            binary_label = 1 if label_str == 'true' else 0
            multiclass_label = {'true': 0, 'false': 1, 'unverified': 2, 'non-rumor': 3}.get(label_str, -1)

            item_labels_list.append({
                'item_id': item_id,
                'item_idx': item_to_idx.get(item_id, -1),
                'label': label_str,
                'binary_label': binary_label,
                'multiclass_label': multiclass_label,
                'dataset': dataset_origin
            })

    labels_df = pd.DataFrame(item_labels_list)
    output_dir = Path(config.OUTPUT_DIR)
    labels_path = output_dir / "item_labels.csv"
    labels_df.to_csv(labels_path, index=False)

    text_clean_list: List[Dict] = []
    for item_id in sorted(items_in_data):
        if item_id in tweets_text:
            text_clean_list.append({
                'item_id': item_id,
                'item_idx': item_to_idx.get(item_id, -1),
                'text_clean': tweets_text[item_id]
            })

    text_df = pd.DataFrame(text_clean_list)
    text_path = output_dir / "item_text_clean.csv"
    text_df.to_csv(text_path, index=False)

    logger.log(f"\n* {len(labels_df)} labels guardadas")
    logger.log(f"* {len(text_df)} tweets con texto guardados")
    if excluded_count > 0:
        logger.log(f"* {excluded_count} items excluidos por estrategia {config.LABEL_STRATEGY}")
    logger.log(f"* Guardado: {labels_path}")
    logger.log(f"* Guardado: {text_path}")

    logger.log(f"\nDistribución de labels ({config.LABEL_STRATEGY}):")
    label_dist = labels_df['label'].value_counts()

    if config.LABEL_STRATEGY == '4class':
        label_names = {0: 'true', 1: 'false', 2: 'unverified', 3: 'non-rumor'}
        for idx in range(4):
            label = label_names[idx]
            count = label_dist.get(label, 0)
            logger.log(f"  • Clase {idx} ({label:12s}): {count:4d} ({count/len(labels_df)*100:5.1f}%)")
    elif config.LABEL_STRATEGY == '3class':
        logger.log(f"  • Clase 0 (true):       {label_dist.get('true', 0):4d}")
        logger.log(f"  • Clase 1 (false):      {label_dist.get('false', 0):4d}")
        ambiguous = label_dist.get('unverified', 0) + label_dist.get('non-rumor', 0)
        logger.log(f"  • Clase 2 (ambiguous):  {ambiguous:4d}")
    elif config.LABEL_STRATEGY in ['binary_strict', 'binary_all']:
        idx_dist = labels_df['label_idx'].value_counts().sort_index()
        logger.log(f"  • Clase 0 (fake):  {idx_dist.get(0, 0):4d} ({idx_dist.get(0, 0)/len(labels_df)*100:5.1f}%)")
        logger.log(f"  • Clase 1 (real):  {idx_dist.get(1, 0):4d} ({idx_dist.get(1, 0)/len(labels_df)*100:5.1f}%)")

    logger.log(f"\nDistribución por dataset origen:")
    dataset_dist = labels_df['dataset'].value_counts()
    for dataset, count in dataset_dist.items():
        logger.log(f"  • {dataset}: {count:4d} ({count/len(labels_df)*100:5.1f}%)")

    return labels_df, text_df


def print_final_stats(
    config: Config,
    logger: StatsLogger,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    filtered_interactions_df: pd.DataFrame,
    valid_users: Set[str],
    labels_df: pd.DataFrame,
    text_df: pd.DataFrame,
    user_to_idx: Dict[str, int],
    item_to_idx: Dict[str, int]
) -> None:
    """Imprime estadísticas finales del dataset."""
    logger.log("\n" + "=" * 80)
    logger.log("PREPROCESAMIENTO COMPLETADO *")
    logger.log("=" * 80)

    logger.log(f"\nArchivos generados en '{config.OUTPUT_DIR}/':")
    logger.log(f"  1. train_interactions.csv       → {len(train_df)} interacciones")
    logger.log(f"  2. test_interactions.csv        → {len(test_df)} interacciones")
    logger.log(f"  3. train_interactions_idx.csv   → {len(train_df)} interacciones (índices)")
    logger.log(f"  4. test_interactions_idx.csv    → {len(test_df)} interacciones (índices)")
    logger.log(f"  5. user_map.csv                 → {len(user_to_idx)} usuarios")
    logger.log(f"  6. item_map.csv                 → {len(item_to_idx)} items")
    logger.log(f"  7. item_labels.csv              → {len(labels_df)} labels")
    logger.log(f"  8. item_text_clean.csv          → {len(text_df)} tweets con texto")

    logger.log(f"\n{'=' * 80}")
    logger.log("ESTADÍSTICAS DEL DATASET COMBINADO")
    logger.log("=" * 80)

    items_in_data = set(train_df['item_id']) | set(test_df['item_id'])
    num_users = len(valid_users)
    num_items = len(items_in_data)
    num_interactions = len(filtered_interactions_df)
    density = (num_interactions / (num_users * num_items)) * 100

    logger.log(f"\n* Métricas generales:")
    logger.log(f"  • Número de usuarios: {num_users}")
    logger.log(f"  • Número de items (tweets raíz): {num_items}")
    logger.log(f"  • Número de interacciones: {num_interactions}")
    logger.log(f"  • Densidad de matriz user–item: {density:.4f}%")
    logger.log(f"  • Interacciones promedio por usuario: {num_interactions/num_users:.2f}")
    logger.log(f"  • Interacciones promedio por item: {num_interactions/num_items:.2f}")

    logger.log(f"\n* Distribución Train/Test:")
    logger.log(f"  • Train: {len(train_df)} ({len(train_df)/num_interactions*100:.1f}%)")
    logger.log(f"  • Test:  {len(test_df)} ({len(test_df)/num_interactions*100:.1f}%)")
    logger.log(f"  • Estrategia: última interacción de cada usuario → test")

    label_dist = labels_df['label'].value_counts()
    logger.log(f"\n* Distribución de etiquetas:")
    if config.LABEL_STRATEGY == '4class':
        label_names = {0: 'true', 1: 'false', 2: 'unverified', 3: 'non-rumor'}
        for idx in range(4):
            label = label_names[idx]
            count = label_dist.get(label, 0)
            logger.log(f"  • Clase {idx} ({label:12s}): {count:4d} ({count/len(labels_df)*100:5.1f}%)")
    else:
        idx_dist = labels_df['label_idx'].value_counts().sort_index()
        for idx, count in idx_dist.items():
            logger.log(f"  • Clase {idx}: {count:4d} ({count/len(labels_df)*100:5.1f}%)")

    dataset_dist = labels_df['dataset'].value_counts()
    logger.log(f"\n* Contribución por dataset:")
    for dataset, count in sorted(dataset_dist.items()):
        logger.log(f"  • {dataset}: {count:4d} items ({count/len(labels_df)*100:5.1f}%)")

    logger.log("\n" + "=" * 80)
    logger.log("* Dataset listo para entrenar sistemas de recomendación:")
    logger.log("  • Baselines H1: Random, Most Popular, User-KNN")
    logger.log("  • Midterm: GNNs (LightGCN, GAT), Matrix Factorization, NCF")
    logger.log("=" * 80 + "\n")


def main() -> None:
    """Función principal que ejecuta el pipeline completo."""
    config = Config()

    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    logger = StatsLogger(config.OUTPUT_DIR)

    logger.log("=" * 80)
    logger.log("PREPROCESAMIENTO UNIFICADO: TWITTER15 + TWITTER16")
    logger.log("=" * 80)
    logger.log(f"\nFecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"\nConfiguración:")
    logger.log(f"  • Datasets: {', '.join(config.DATASETS)}")
    logger.log(f"  • Mínimo interacciones por usuario: {config.MIN_INTERACTIONS}")
    logger.log(f"  • Estrategia de labels: {config.LABEL_STRATEGY}")
    logger.log(f"  • Salida: {config.OUTPUT_DIR}/")

    labels_dict, tweets_text = load_data(config, logger)

    all_interactions_df, user_items, item_users = extract_interactions(config, logger, labels_dict)

    filtered_interactions_df, valid_users = filter_users(config, logger, all_interactions_df, user_items)

    train_df, test_df = split_train_test(logger, filtered_interactions_df, valid_users)

    user_to_idx, item_to_idx = create_mappings(config, logger, valid_users, item_users)

    train_df, test_df = save_interactions(config, logger, train_df, test_df, user_to_idx, item_to_idx)

    labels_df, text_df = save_labels_and_text(
        config, logger, train_df, test_df, labels_dict, tweets_text, item_to_idx
    )

    print_final_stats(
        config, logger, train_df, test_df, filtered_interactions_df,
        valid_users, labels_df, text_df, user_to_idx, item_to_idx
    )

    logger.save()
    logger.log(f"\nOK Pipeline completado exitosamente!")
    logger.log(f"Revisa {config.OUTPUT_DIR}/ para todos los archivos generados.\n")


if __name__ == "__main__":
    main()
