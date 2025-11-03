# Midterm - GNN Recommender Systems

Sistema de recomendación basado en Graph Neural Networks (GNNs) aplicado al dataset Twitter15/16. Incluye construcción de grafos, node embeddings, modelos GCN/LightGCN, Linear Threshold Model, y análisis de propagación de fake news.

**Objetivo:** Comparar sistemas de recomendación clásicos vs GNN-based en términos de precisión (MRR) y amplificación de desinformación (propagación en grafo social).

---

## ⚠️ Estado Actual y Pendientes

### ✅ Completado (Midterm)

- Grafos bipartito y social construidos (`graphs/`)
- Node embeddings: BERT, Random, User init
- Modelos implementados: GCN-BERT, GCN-Random, LightGCN
- Linear Threshold Model implementado y funcional
- Notebooks comparativos completos (Semana2-4)
- Análisis de propagación de fake news
- **Datos usados:** `../data_processing/processed_h1/` (sin filtro temporal)

### ✅ Completado (Grafos Temporales)

- **Grafos temporales construidos** (`graphs_temporal/`)
  - Script: `../build_temporal_graphs.py`
  - Bipartito: 39,958 users × 1,386 items = 176,996 edges
  - Social: 39,958 nodos, 311,020 edges
  - Cap de 5 en edges duplicados implementado
  - Splits: 80% train / 10% val / 10% test

- **Negative Sampling implementado**
  - Script: `../negative_sampling.py`
  - 382,391 samples (~11 por usuario)
  - Ventana temporal respetada
  - Balance 50/50 popular/unpopular

### ⏳ Pendientes para Versión Final

1. **Re-entrenamiento de modelos GNN**
   - Re-entrenar GCN-BERT, GCN-Random, LightGCN con:
     - Grafos temporales (`graphs_temporal/`)
     - Negative sampling implementado
     - 3 capas (actualmente 2)
   - Comparar métricas con modelos baseline

2. **Integración final Parte 2**
   - Ejecutar LTM con grafos temporales actualizados
   - Analizar propagación con nuevos datos
   - Generar visualizaciones finales
   - Consolidar resultados para informe

---

## Quick Setup

```bash
cd midterm
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python test_pipeline.py
```

## Estructura

```
midterm/
├── build_graphs.py              # Grafos bipartito y social
├── prepare_features.py          # Node embeddings (BERT/random)
├── linear_threshold_model.py    # Simulación de propagación
├── metrics.py                   # MRR, ILD, propagation metrics
├── graph_utils.py               # DataLoader y utilidades
├── node_features.py             # Encoders
├── demo_ltm.py                  # Demos interactivas
└── test_pipeline.py             # Tests de verificación
```

## Pipeline

```bash
# 1. Construir grafos
python build_graphs.py

# 2. Generar embeddings
python prepare_features.py --all  # BERT + random + users
# o
python prepare_features.py --random --users  # Solo random (más rápido)

# 3. Demo LTM
python demo_ltm.py
```

## Componentes

### Grafos

**Bipartito (User-Item)**
- 4,856 users × 2,308 items = 117,988 edges
- Nodos: `[user_0...user_N, item_0...item_M]`

**Social (User-User)**
- Grafo implícito por co-interacciones (≥3 items comunes)
- 4,856 nodes, 295,660 edges
- Pesos = items compartidos

```python
bipartite = torch.load('graphs/bipartite_graph.pt')
social = torch.load('graphs/social_graph.pt')
```

### Embeddings

**BERT**: `all-MiniLM-L6-v2` (384-dim) para items
**Random/Learnable**: Xavier init para users

```bash
python prepare_features.py --bert    # Items con BERT
python prepare_features.py --users   # Users con Xavier
```

### Linear Threshold Model

```python
from linear_threshold_model import LinearThresholdModel

ltm = LinearThresholdModel(torch.load('graphs/social_graph.pt'))
seed_nodes = {0, 1, 2}
rounds = ltm.simulate(seed_nodes, max_iterations=50)
stats = ltm.monte_carlo_propagation(seed_nodes, num_simulations=100)
```

### Métricas

**Recomendación**: MRR, ILD, Coverage, Cosine Diversity
**Propagación**: Reach, Depth, Speed

```python
from metrics import RecommendationMetrics, PropagationMetrics

rec_metrics = RecommendationMetrics(recs, gt, num_items, embeddings).compute_all()
prop_metrics = PropagationMetrics(rounds, num_users).compute_all()
```

## Implementación de Modelos

### Archivos necesarios

```python
bipartite = torch.load('midterm/graphs/bipartite_graph.pt')
social = torch.load('midterm/graphs/social_graph.pt')
item_features = load_embeddings('midterm/graphs/item_embeddings_bert.pt')
train = pd.read_csv('data_processing/processed_h1/train_interactions_idx.csv')
test = pd.read_csv('data_processing/processed_h1/test_interactions_idx.csv')
labels = pd.read_csv('data_processing/processed_h1/item_labels.csv')
```

**Nota**: Cada modelo debe entrenarse en dos versiones:
1. Con BERT embeddings (`item_embeddings_bert.pt`)
2. Sin BERT embeddings (`item_embeddings_random.pt`)

Esto permite comparar el impacto de embeddings semánticos vs random en recomendación y propagación.

### 1. GCN Baseline

```python
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNRecommender(nn.Module):
    def __init__(self, num_users, num_items, item_feature_dim,
                 embedding_dim=64, hidden_dim=32):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_projection = nn.Linear(item_feature_dim, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)

    def forward(self, edge_index, item_features):
        user_emb = self.user_embedding.weight
        item_emb = self.item_projection(item_features)
        x = torch.cat([user_emb, item_emb], dim=0)

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        return x[:self.num_users], x[self.num_users:]
```

**Training (BPR Loss)**

```python
def train_epoch(model, edge_index, item_features, interactions, optimizer):
    model.train()
    optimizer.zero_grad()

    user_emb, item_emb = model(edge_index, item_features)

    pos_users = torch.LongTensor(interactions['user_idx'].values)
    pos_items = torch.LongTensor(interactions['item_idx'].values)
    neg_items = torch.randint(0, model.num_items, (len(pos_users),))

    pos_scores = (user_emb[pos_users] * item_emb[pos_items]).sum(dim=1)
    neg_scores = (user_emb[pos_users] * item_emb[neg_items]).sum(dim=1)

    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
    loss.backward()
    optimizer.step()

    return loss.item()
```

**Evaluación**

```python
@torch.no_grad()
def evaluate(model, edge_index, item_features, test_df, k=10):
    model.eval()
    user_emb, item_emb = model(edge_index, item_features)

    recommendations = []
    ground_truth = []

    for user_idx in range(model.num_users):
        scores = user_emb[user_idx] @ item_emb.T
        _, top_items = torch.topk(scores, k)
        recommendations.append(top_items.tolist())

        true_items = test_df[test_df['user_idx'] == user_idx]['item_idx'].values
        ground_truth.append(set(true_items))

    from metrics import RecommendationMetrics
    metrics = RecommendationMetrics(
        recommendations, ground_truth,
        model.num_items, item_features
    ).compute_all()

    return metrics
```

### 2. LightGCN

```python
from torch_geometric.nn import LGConv

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, edge_index):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embeddings = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            all_embeddings.append(x)

        final_emb = torch.stack(all_embeddings, dim=0).mean(dim=0)
        return final_emb[:self.num_users], final_emb[self.num_users:]
```

**Hiperparámetros**: layers=3-4, dim=64-128, lr=0.001, epochs=50-100

### 3. Análisis de Propagación de Fake News

```python
from linear_threshold_model import LinearThresholdModel
from metrics import PropagationMetrics

ltm = LinearThresholdModel(social, seed=42)

def analyze_misinformation_spread(recommendations, model_name):
    seed_users = set()

    for user_idx, rec_list in enumerate(recommendations):
        for item_idx in rec_list[:10]:
            item_label = labels[labels['item_idx'] == item_idx]['label'].values
            if len(item_label) > 0 and item_label[0] == 'false':
                seed_users.add(user_idx)
                break

    print(f"\n{model_name}: {len(seed_users)} users exposed to fake news")

    if len(seed_users) == 0:
        return None

    infection_rounds = ltm.simulate(seed_users, max_iterations=50)
    metrics = PropagationMetrics(infection_rounds, ltm.num_nodes).compute_all()

    print(f"  Reach: {metrics['reach']*100:.2f}%")
    print(f"  Depth: {metrics['depth']} rounds")
    print(f"  Infected: {metrics['total_infected']} users")

    return metrics

gcn_metrics = analyze_misinformation_spread(gcn_recs, "GCN")
lgcn_metrics = analyze_misinformation_spread(lightgcn_recs, "LightGCN")
```

## Dataset Stats

### Versión Baseline (`graphs/`) - Sin Filtro Temporal

**Grafos generados:**
- **Bipartite:** 4,856 users × 2,308 items, 117,988 edges (density: 0.526%)
  - Avg user degree: 12.15
  - Avg item degree: 25.56
- **Social:** 4,856 nodes, 295,660 edges (density: 1.25%)
  - Construido con threshold: ≥3 items compartidos
  - Avg degree: 60.89

**Splits (Leave-One-Out)**
- Train: 58,994 interacciones (~92.4%)
- Test: 4,856 interacciones (~7.6%, 1 por usuario)

**Labels (Balanceadas)**
- true: 25.1% (579 items)
- false: 24.9% (575 items)
- unverified: 24.9% (575 items)
- non-rumor: 25.1% (579 items)

**Datos completos:** Ver `graphs/graph_stats.txt`

---

### Versión Temporal (`graphs_temporal/`) - Twitter15 + Twitter16 Unificado

**Dataset procesado:**
- Total interacciones: 110,732
- Usuarios únicos: 39,958
- Items únicos: 1,386
- Filtro temporal: datos ≤ Marzo 2015

**Grafos generados:**
- **Bipartite:** 39,958 users × 1,386 items = 176,996 edges
  - Cap de 5 en edges duplicados implementado
  - Avg user degree: 4.43
  - Avg item degree: 127.7
- **Social:** 39,958 nodes, 311,020 edges
  - Threshold: ≥3 items compartidos
  - Avg degree: 7.78

**Splits (Temporal 80/10/10)**
- Train: 88,620 interacciones (80.0%)
- Val: 11,534 interacciones (10.4%)
- Test: 10,578 interacciones (9.6%)

**Negative Sampling:**
- Total samples: 382,391
- Usuarios con samples: 34,618
- Avg samples por usuario: 11.05

---

## Roadmap y Estado por Semana

### ✓ Semana 1 (Oct 6-12): Construcción de Grafos

**Objetivo:** Construir grafos bipartito usuario-ítem y grafo social implícito a partir de las interacciones.

**Completado:**
- Grafo bipartito: 4,856 users × 2,308 items (117,988 edges)
- Grafo social: 4,856 users (295,660 edges, threshold ≥3 co-interacciones)
- Scripts: `build_graphs.py`, `graph_utils.py`
- Outputs: `graphs/bipartite_graph.pt`, `graphs/social_graph.pt`

---

### ✓ Semana 2 (Oct 13-19): GCN Baseline + LTM

**Objetivo:** Implementar GCN como baseline, Linear Threshold Model, y métricas de evaluación.

**Notebook:** [`Semana2_GCN.ipynb`](Semana2_GCN.ipynb)

**Completado:**
- Node embeddings con BERT (384-dim) y Random (64-dim): `prepare_features.py`
- GCN implementado con dos variantes:
  - **GCN-BERT**: MRR=0.0481, Propagation Reach=64.93%
  - **GCN-Random**: MRR=0.0376, Propagation Reach=39.02%
- Linear Threshold Model: `linear_threshold_model.py`
- Métricas: MRR, ILD, Coverage, Propagation (Reach, Depth, Speed): `metrics.py`
- Análisis de exposición a fake news en top-10 recomendaciones

**Hallazgos:**
- BERT mejora precisión (+28% MRR) pero amplifica fake news (65% vs 39% reach)
- 51.73% de usuarios expuestos a fake news en top-10 (GCN-BERT)

---

### ✓ Semana 3 (Oct 20-26): LightGCN

**Objetivo:** Implementar LightGCN (estado del arte) y comparar con GCN.

**Notebook:** [`Semana3_LightGCN.ipynb`](Semana3_LightGCN.ipynb)

**Completado:**
- LightGCN implementado (3 layers, embedding_dim=64)
- Resultados:
  - **LightGCN**: MRR=0.0508, Propagation Reach=37.99%
  - Mejor trade-off: máxima precisión con mínima amplificación de fake news
- Comparación directa con GCN-BERT y GCN-Random

**Hallazgos:**
- LightGCN supera a ambas versiones de GCN en precisión (+5.6% vs GCN-BERT)
- Reduce propagación de fake news en 42% vs GCN-BERT
- Sesga hacia contenido "non-rumor" (64.8%), reduciendo fake news directas en 80%

---

### ✓ Semana 4 (Oct 27-30): Evaluación Comparativa

**Objetivo:** Comparar todos los modelos (clásicos + GNNs) y analizar trade-offs precisión vs amplificación.

**Notebook:** [`Semana4_Comparative.ipynb`](Semana4_Comparative.ipynb)

**Completado:**
- Implementación de modelos clásicos (User-KNN, Random, Most Popular)
- Análisis de propagación para TODOS los modelos (6 en total)
- Comparación completa: MRR, ILD, Coverage, Propagation (Reach, Depth, Speed)
- Visualización de trade-off: Precisión vs Amplificación de fake news
- Distribución de labels en recomendaciones por modelo
- Ejemplos de recomendación para usuarios específicos
- Conclusiones sobre modelos óptimos

**Modelos comparados:**
1. Random (baseline) - MRR: ~0.001, Reach: variable
2. Most Popular (baseline) - MRR: ~0.007, Reach: variable
3. User-KNN (clásico colaborativo) - MRR: ~0.125, Reach: calculado
4. GCN-BERT (GNN + embeddings semánticos) - MRR: 0.0481, Reach: 64.93%
5. GCN-Random (GNN sin features) - MRR: 0.0376, Reach: 39.02%
6. LightGCN (GNN estado del arte) - MRR: 0.0508, Reach: 37.99%

**Hallazgos clave:**
- LightGCN logra el mejor balance: máxima precisión (MRR) con mínima propagación de fake news
- Existe un trade-off claro entre precisión y amplificación de desinformación
- BERT amplifica contenido viral pero también más peligroso
- Los modelos clásicos tienen comportamientos diversos en propagación

**Nota:** Sheaf4Rec se deja como trabajo futuro (complejidad de implementación).

---

## 📋 Especificaciones Técnicas Implementadas

### ✅ Construcción de Grafos Temporales

**Script:** `../build_temporal_graphs.py`

**Implementación:**
- 1 nodo por usuario (sin duplicados)
- 1 nodo por item/tree (rumor source)
- Cap de 5 en edges duplicados usuario-item
- Unificación de Twitter15 + Twitter16 con split temporal

**Código clave:**
```python
grouped = df.groupby(['user_idx', 'item_idx']).size().reset_index(name='count')
grouped['count'] = grouped['count'].clip(upper=MAX_EDGE_CAP)
```

**Output:**
- `graphs_temporal/bipartite_graph.pt`
- `graphs_temporal/social_graph.pt`
- Splits: train/val/test (80/10/10)
- Mapeos: user_map.csv, item_map.csv

### ✅ Negative Sampling Temporal

**Script:** `../negative_sampling.py`

**Implementación:**
- ~11 samples por usuario (382K total)
- Ventana temporal respetada usando `user_activity.csv`
- Balance 50/50 entre items populares y no populares
- Exclusión de items ya interactuados

**Código clave:**
```python
available_items = [
    item_id for item_id, ts in item_timestamps.items()
    if start <= ts <= end and item_id not in user_items[user_id]
]

sorted_items = sorted(available_items, key=lambda x: item_popularity.get(x, 0), reverse=True)
popular_pool = sorted_items[:len(sorted_items)//2]
unpopular_pool = sorted_items[len(sorted_items)//2:]
```

**Output:**
- `data_processing/processed_round2/negative_samples.csv`

### Re-entrenamiento GNNs (3 capas)

**Modelos a re-entrenar:**
1. GCN-BERT (3 capas)
2. GCN-Random (3 capas)
3. LightGCN (3 capas)

**Cambios respecto a versión actual:**
- Actualmente: 2 capas GCN/LightGCN
- Nuevo: 3 capas (mejor capacidad de agregación de vecindario)
- Usar BPR Loss + negative samples generados

**Hiperparámetros sugeridos:**
```python
num_layers = 3
embedding_dim = 64
learning_rate = 0.001
epochs = 50-100
batch_size = 1024
```

---

## Resultados Comparativos (Versión Actual)

### Métricas de Recomendación

| Modelo      | MRR    | ILD    | Coverage | Usuarios Expuestos a Fake News |
|-------------|--------|--------|----------|-------------------------------|
| GCN-BERT    | 0.0481 | 0.9004 | 0.1088   | 2,512 (51.73%)                |
| GCN-Random  | 0.0376 | 0.8951 | 0.1096   | 1,776 (36.57%)                |
| **LightGCN**| **0.0508** | 0.8493 | **0.1655** | **1,691 (34.82%)** |

### Métricas de Propagación de Fake News

| Modelo      | Reach  | Depth (rounds) | Speed (users/round) |
|-------------|--------|----------------|---------------------|
| GCN-BERT    | 64.93% | 13             | 53.42               |
| GCN-Random  | 39.02% | 4              | 39.67               |
| **LightGCN**| **37.99%** | 5          | **38.50**           |

### Hallazgos Clave

1. **LightGCN es el mejor modelo**: Máxima precisión (MRR) con mínima propagación de fake news
2. **BERT amplifica desinformación**: GCN-BERT tiene mejor MRR que GCN-Random, pero propaga fake news 66% más
3. **Trade-off precisión-seguridad**: LightGCN rompe este trade-off, logrando el mejor balance
4. **Distribución de contenido**: LightGCN sesga hacia contenido "non-rumor" (64.8%) en lugar de "unverified", reduciendo fake news directas en 80% vs baseline
