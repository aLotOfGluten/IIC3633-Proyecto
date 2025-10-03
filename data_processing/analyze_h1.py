import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, deque

try:
    from scipy.stats import mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

FAKE_COLOR = '#FF6B6B'
REAL_COLOR = '#51CF66'
USER_COLOR = '#4A90E2'
ITEM_COLOR = '#E94B3C'
USER_INTERACTION_FILTER_THRESHOLD = 8
ALPHA_SIGNIFICANCE = 0.05

BASE_DIR = Path('.')
PROCESSED_DIR = BASE_DIR / 'processed_h1'
OUTPUT_DIR = BASE_DIR / 'plots_and_reports'
OUTPUT_DIR.mkdir(exist_ok=True)

TWITTER15_TREE = BASE_DIR / 'twitter15' / 'tree'
TWITTER16_TREE = BASE_DIR / 'twitter16' / 'tree'


def style_boxplot(bp, colors):
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)


def load_data():
    print("Cargando datos...")

    data = {
        'train_inter': pd.read_csv(PROCESSED_DIR / 'train_interactions.csv'),
        'test_inter': pd.read_csv(PROCESSED_DIR / 'test_interactions.csv'),
        'user_map': pd.read_csv(PROCESSED_DIR / 'user_map.csv'),
        'item_map': pd.read_csv(PROCESSED_DIR / 'item_map.csv'),
        'item_labels': pd.read_csv(PROCESSED_DIR / 'item_labels.csv'),
        'item_text': pd.read_csv(PROCESSED_DIR / 'item_text_clean.csv')
    }

    print(f"  ✓ Train: {len(data['train_inter']):,} | Test: {len(data['test_inter']):,}")
    print(f"  ✓ Users: {len(data['user_map']):,} | Items: {len(data['item_map']):,}")

    return data


def compute_global_stats(data):
    print("\n=== 1. Estadísticas Globales ===")

    n_users = len(data['user_map'])
    n_items = len(data['item_map'])
    n_interactions_train = len(data['train_inter'])
    n_interactions_test = len(data['test_inter'])
    n_interactions_total = n_interactions_train + n_interactions_test

    density = n_interactions_total / (n_users * n_items) * 100
    all_interactions = pd.concat([data['train_inter'], data['test_inter']])

    labels = data['item_labels']
    n_real = (labels['binary_label'] == 1).sum()
    n_fake = (labels['binary_label'] == 0).sum()
    pct_real = n_real / len(labels) * 100
    pct_fake = n_fake / len(labels) * 100

    label_counts = labels['label'].value_counts()
    n_false = label_counts.get('false', 0)
    n_true = label_counts.get('true', 0)
    n_unverified = label_counts.get('unverified', 0)
    n_nonrumor = label_counts.get('non-rumor', 0)

    interactions_per_user = all_interactions.groupby('user_id').size()
    interactions_per_item = all_interactions.groupby('item_id').size()

    stats = {
        'n_users': n_users,
        'n_items': n_items,
        'n_interactions_total': n_interactions_total,
        'n_interactions_train': n_interactions_train,
        'n_interactions_test': n_interactions_test,
        'density': density,
        'n_real': n_real,
        'n_fake': n_fake,
        'pct_real': pct_real,
        'pct_fake': pct_fake,
        'interactions_per_user': interactions_per_user,
        'interactions_per_item': interactions_per_item,
        'all_interactions': all_interactions
    }

    summary_df = pd.DataFrame([{
        'n_usuarios': n_users,
        'n_items': n_items,
        'n_interacciones': n_interactions_total,
        'densidad_%': round(density, 4),
        'n_false': n_false,
        'n_true': n_true,
        'n_unverified': n_unverified,
        'n_nonrumor': n_nonrumor,
        'pct_false': round(n_false / len(labels) * 100, 2),
        'pct_true': round(n_true / len(labels) * 100, 2),
        'pct_unverified': round(n_unverified / len(labels) * 100, 2),
        'pct_nonrumor': round(n_nonrumor / len(labels) * 100, 2)
    }])
    summary_df.to_csv(OUTPUT_DIR / 'resumen_stats.csv', index=False)

    print(f"  ✓ Usuarios: {n_users:,} | Items: {n_items:,}")
    print(f"  ✓ Interacciones: {n_interactions_total:,} | Densidad: {density:.4f}%")
    print(f"  ✓ Balance: False={n_false} | True={n_true} | Unverified={n_unverified} | Non-rumor={n_nonrumor}")
    print(f"  ✓ Guardado: resumen_stats.csv")

    return stats


def plot_interaction_distributions(stats):
    print("\n=== 2. Distribución de Interacciones ===")

    inter_user = stats['interactions_per_user']
    inter_item = stats['interactions_per_item']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(inter_user, bins=50, edgecolor='black', alpha=0.7, color=USER_COLOR)
    ax.set_xlabel('Número de Interacciones')
    ax.set_ylabel('Número de Usuarios')
    ax.set_title('Distribución de Interacciones por Usuario')
    ax.axvline(inter_user.median(), color='red', linestyle='--',
               label=f'Mediana={inter_user.median():.0f}', linewidth=2)
    ax.axvline(USER_INTERACTION_FILTER_THRESHOLD, color='orange', linestyle='--',
               label=f'Filtro mínimo={USER_INTERACTION_FILTER_THRESHOLD}', linewidth=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'usuarios_hist.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(inter_item, bins=50, edgecolor='black', alpha=0.7, color=ITEM_COLOR)
    ax.set_xlabel('Número de Interacciones')
    ax.set_ylabel('Número de Items')
    ax.set_title('Distribución de Interacciones por Item (Long-tail)')
    ax.axvline(inter_item.median(), color='red', linestyle='--',
               label=f'Mediana={inter_item.median():.0f}', linewidth=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'items_hist.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ usuarios_hist.png - Mediana: {inter_user.median():.0f}")
    print(f"  ✓ items_hist.png - Long-tail evidente (mediana={inter_item.median():.0f}, max={inter_item.max()})")


def plot_label_balance(data):
    print("\n=== 3. Balance de Clases ===")

    labels = data['item_labels']
    label_counts = labels['label'].value_counts()

    label_order = ['false', 'true', 'unverified', 'non-rumor']
    label_counts = label_counts.reindex(label_order)

    label_colors = {
        'false': '#FF6B6B',
        'true': '#51CF66',
        'unverified': '#FFD93D',
        'non-rumor': '#6C5CE7'
    }

    colors = [label_colors[l] for l in label_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(label_order, label_counts.values,
                   edgecolor='black', alpha=0.8, color=colors)
    ax.set_ylabel('Número de Items')
    ax.set_title('Balance de Clases (Multiclase)')
    ax.set_xlabel('Categoría')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (label, count) in enumerate(zip(label_order, label_counts.values)):
        pct = count / len(labels) * 100
        ax.text(i, count, f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'balance_labels.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ balance_labels.png - False: {label_counts['false']}, True: {label_counts['true']}, Unverified: {label_counts['unverified']}, Non-rumor: {label_counts['non-rumor']}")


def analyze_cascades(data):
    print("\n=== 4. Cascadas de Difusión ===")

    cascade_sizes = []
    cascade_depths = []
    cascade_info = {}

    tree_files = list(TWITTER15_TREE.glob('*.txt')) + list(TWITTER16_TREE.glob('*.txt'))
    print(f"  Procesando {len(tree_files)} cascadas...")

    labels_map = {str(row['item_id']): row['binary_label']
                  for _, row in data['item_labels'].iterrows()}

    cascade_sizes_fake = []
    cascade_depths_fake = []
    cascade_sizes_real = []
    cascade_depths_real = []

    for tree_file in tree_files:
        tree_id = tree_file.stem

        try:
            with open(tree_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                continue

            users = set()
            edges = []

            for line in lines:
                line = line.strip()
                if not line or '->' not in line:
                    continue

                parts = line.split('->')
                if len(parts) != 2:
                    continue

                src = parts[0].strip().strip("'\"")
                dst = parts[1].strip().strip("'\"")

                users.add(src)
                users.add(dst)
                edges.append((src, dst))

            size = len(users)

            if edges:
                graph = defaultdict(list)
                in_degree = defaultdict(int)

                for src, dst in edges:
                    graph[src].append(dst)
                    in_degree[dst] += 1

                all_nodes = set(graph.keys()) | set(in_degree.keys())
                roots = [n for n in all_nodes if in_degree[n] == 0]

                if roots:
                    root = roots[0]
                    queue = deque([(root, 0)])
                    max_depth = 0
                    visited = set()

                    while queue:
                        node, depth = queue.popleft()
                        if node in visited:
                            continue
                        visited.add(node)
                        max_depth = max(max_depth, depth)

                        for child in graph[node]:
                            if child not in visited:
                                queue.append((child, depth + 1))

                    depth = max_depth
                else:
                    depth = 0
            else:
                depth = 0

            cascade_sizes.append(size)
            cascade_depths.append(depth)
            cascade_info[tree_id] = {'size': size, 'depth': depth}

            if tree_id in labels_map:
                if labels_map[tree_id] == 0:
                    cascade_sizes_fake.append(size)
                    cascade_depths_fake.append(depth)
                else:
                    cascade_sizes_real.append(size)
                    cascade_depths_real.append(depth)

        except (IndexError, ValueError, IOError) as e:
            print(f"⚠️  Skipping {tree_file.name}: {type(e).__name__}")
            continue

    cascade_sizes = np.array(cascade_sizes)
    cascade_depths = np.array(cascade_depths)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cascade_sizes, bins=50, edgecolor='black', alpha=0.7, color='#4A90E2')
    ax.set_xlabel('Tamaño de la Cascada (Nº de usuarios)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Tamaño de Cascadas')
    ax.axvline(cascade_sizes.mean(), color='red', linestyle='--',
               label=f'Promedio={cascade_sizes.mean():.1f}', linewidth=2)
    ax.axvline(np.median(cascade_sizes), color='orange', linestyle='--',
               label=f'Mediana={np.median(cascade_sizes):.1f}', linewidth=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cascadas_tamano_hist.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cascade_depths, bins=30, edgecolor='black', alpha=0.7, color='#9B59B6')
    ax.set_xlabel('Profundidad de la Cascada (Niveles)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Profundidad de Cascadas')
    ax.axvline(cascade_depths.mean(), color='red', linestyle='--',
               label=f'Promedio={cascade_depths.mean():.1f}', linewidth=2)
    ax.axvline(np.median(cascade_depths), color='orange', linestyle='--',
               label=f'Mediana={np.median(cascade_depths):.1f}', linewidth=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cascadas_profundidad_hist.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Tamaño promedio: {cascade_sizes.mean():.1f} usuarios (max={cascade_sizes.max()})")
    print(f"  ✓ Profundidad promedio: {cascade_depths.mean():.1f} niveles (max={cascade_depths.max()})")
    if cascade_sizes_fake and cascade_sizes_real:
        print(f"  ✓ Fake - Tamaño: {np.mean(cascade_sizes_fake):.1f} | Profundidad: {np.mean(cascade_depths_fake):.1f}")
        print(f"  ✓ Real - Tamaño: {np.mean(cascade_sizes_real):.1f} | Profundidad: {np.mean(cascade_depths_real):.1f}")
    print(f"  ✓ cascadas_tamano_hist.png")
    print(f"  ✓ cascadas_profundidad_hist.png")

    return cascade_info


def analyze_virality(data, stats):
    print("\n=== 5. Viralidad: Fake vs Real ===")

    all_inter = stats['all_interactions']
    labels = data['item_labels'][['item_id', 'binary_label']]

    inter_with_labels = all_inter.merge(labels, on='item_id', how='left')
    inter_with_labels = inter_with_labels[inter_with_labels['binary_label'].notna()]

    item_inter_counts = inter_with_labels.groupby(['item_id', 'binary_label']).size().reset_index(name='n_interactions')

    fake_inter = item_inter_counts[item_inter_counts['binary_label'] == 0]['n_interactions']
    real_inter = item_inter_counts[item_inter_counts['binary_label'] == 1]['n_interactions']

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Noticias Falsas', 'Noticias Verdaderas']
    means = [fake_inter.mean(), real_inter.mean()]
    std_devs = [fake_inter.std(), real_inter.std()]
    colors = [FAKE_COLOR, REAL_COLOR]

    bars = ax.bar(categories, means, yerr=std_devs, capsize=10,
                   alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Promedio de Interacciones')
    ax.set_title('Viralidad: Noticias Falsas vs Verdaderas\n(Promedio de Interacciones por Item)')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (mean, std) in enumerate(zip(means, std_devs)):
        ax.text(i, mean + std + 5, f'μ={mean:.1f}\nσ={std:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'viralidad_fake_vs_real.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Fake promedio: {fake_inter.mean():.2f} interacciones (σ={fake_inter.std():.2f})")
    print(f"  ✓ Real promedio: {real_inter.mean():.2f} interacciones (σ={real_inter.std():.2f})")

    if SCIPY_AVAILABLE:
        stat, p_value = mannwhitneyu(fake_inter, real_inter, alternative='two-sided')
        print(f"  ✓ Mann-Whitney U test p-value: {p_value:.4e}")
        if p_value < ALPHA_SIGNIFICANCE:
            print(f"  ✓ Diferencia estadísticamente significativa (p < {ALPHA_SIGNIFICANCE})")
        else:
            print(f"  ✓ Diferencia NO significativa (p >= {ALPHA_SIGNIFICANCE})")
    else:
        print(f"  ⚠️  scipy no disponible - test estadístico omitido")

    print(f"  ✓ viralidad_fake_vs_real.png")

    return item_inter_counts


def generate_top_items(data, stats, item_inter_counts):
    print("\n=== 6. Top Items Más Virales ===")

    top_items = item_inter_counts.nlargest(10, 'n_interactions').copy()
    labels_full = data['item_labels'][['item_id', 'label']].copy()
    top_items = top_items.merge(labels_full, on='item_id', how='left')
    top_items['tipo'] = top_items['binary_label'].map({0: 'Fake', 1: 'Real'})

    top_items_final = top_items[['item_id', 'n_interactions', 'tipo', 'label']].copy()
    top_items_final.columns = ['item_id', 'interacciones', 'tipo', 'categoria']
    top_items_final.to_csv(OUTPUT_DIR / 'top_items.csv', index=False)

    fake_count = (top_items_final['tipo'] == 'Fake').sum()
    real_count = (top_items_final['tipo'] == 'Real').sum()
    print(f"  ✓ top_items.csv - Top 10: {fake_count} Fake, {real_count} Real")
    print(f"  ✓ Más viral: {top_items_final.iloc[0]['item_id']} ({top_items_final.iloc[0]['tipo']}, {top_items_final.iloc[0]['interacciones']} interacciones)")


def generate_top_users(stats):
    print("\n=== 7. Top Usuarios Más Activos ===")

    inter_per_user = stats['interactions_per_user']
    top_users = inter_per_user.nlargest(10)

    top_users_df = pd.DataFrame({
        'user_id': top_users.index,
        'n_interacciones': top_users.values
    })

    top_users_df.to_csv(OUTPUT_DIR / 'top_usuarios.csv', index=False)

    print(f"  ✓ top_usuarios.csv - Top usuario: {top_users_df.iloc[0]['user_id']} ({top_users_df.iloc[0]['n_interacciones']} interacciones)")


def analyze_user_behavior(data, stats):
    print("\n=== 8. Comportamiento de Usuarios (Fake vs Real) ===")

    all_inter = stats['all_interactions']
    labels = data['item_labels'][['item_id', 'binary_label']]

    inter_with_labels = all_inter.merge(labels, on='item_id', how='left')
    inter_with_labels = inter_with_labels[inter_with_labels['binary_label'].notna()]

    users_fake = inter_with_labels[inter_with_labels['binary_label'] == 0]['user_id'].unique()
    users_real = inter_with_labels[inter_with_labels['binary_label'] == 1]['user_id'].unique()

    fake_user_activity = inter_with_labels[inter_with_labels['user_id'].isin(users_fake)].groupby('user_id').size()
    real_user_activity = inter_with_labels[inter_with_labels['user_id'].isin(users_real)].groupby('user_id').size()

    print(f"  ✓ Usuarios que comparten Fake: {len(users_fake):,} (actividad promedio: {fake_user_activity.mean():.2f})")
    print(f"  ✓ Usuarios que comparten Real: {len(users_real):,} (actividad promedio: {real_user_activity.mean():.2f})")


def analyze_text_basic(data):
    print("\n=== 9. Análisis Básico de Texto ===")

    texts = data['item_text']
    labels = data['item_labels'][['item_id', 'binary_label']]

    text_with_labels = texts.merge(labels, on='item_id', how='left')
    text_with_labels = text_with_labels[text_with_labels['binary_label'].notna()]

    text_with_labels['length'] = text_with_labels['text_clean'].str.len()

    fake_lengths = text_with_labels[text_with_labels['binary_label'] == 0]['length']
    real_lengths = text_with_labels[text_with_labels['binary_label'] == 1]['length']

    print(f"  ✓ Longitud promedio: {text_with_labels['length'].mean():.1f} caracteres")
    print(f"  ✓ Fake: {fake_lengths.mean():.1f} chars | Real: {real_lengths.mean():.1f} chars")
    print(f"  ✓ Diferencia: {abs(fake_lengths.mean() - real_lengths.mean()):.1f} caracteres")


def main():
    print("=" * 70)
    print("ANÁLISIS H1: DESINFORMACIÓN Y VIRALIDAD")
    print("Dataset: Twitter15/16")
    print("=" * 70)

    data = load_data()
    stats = compute_global_stats(data)
    plot_interaction_distributions(stats)
    plot_label_balance(data)
    cascade_info = analyze_cascades(data)
    item_inter_counts = analyze_virality(data, stats)
    generate_top_items(data, stats, item_inter_counts)
    generate_top_users(stats)
    analyze_user_behavior(data, stats)
    analyze_text_basic(data)

    print("\n" + "=" * 70)
    print("✓ ANÁLISIS COMPLETADO")
    print("=" * 70)
    print(f"\nArchivos generados en: {OUTPUT_DIR}/")
    print("\n📊 Gráficos:")
    print("  - usuarios_hist.png")
    print("  - items_hist.png")
    print("  - balance_labels.png")
    print("  - cascadas_tamano_hist.png")
    print("  - cascadas_profundidad_hist.png")
    print("  - viralidad_fake_vs_real.png")
    print("\n📋 Tablas:")
    print("  - resumen_stats.csv")
    print("  - top_usuarios.csv")
    print("  - top_items.csv")


if __name__ == '__main__':
    main()
