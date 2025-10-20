# Midterm - GNN Recommender Systems

Pipeline para sistemas de recomendación basados en GNNs sobre Twitter15/16. Incluye construcción de grafos, embeddings, Linear Threshold Model, y métricas de evaluación.

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

**Grafos**
- Bipartite: 4,856 users × 2,308 items, 117,988 edges (0.53% density)
- Social: 4,856 nodes, 295,660 edges (1.25% density)

**Splits**
- Train: 58,994 interactions
- Test: 4,856 interactions (leave-one-out)

**Labels** (balanced)
- true: 24.6%
- false: 23.5%
- unverified: 25.5%
- non-rumor: 26.4%

Ver `graphs/graph_stats.txt` para detalles.

## Estado

- ✓ Semana 1: Construcción de grafos
- ✓ Semana 2: BERT embeddings + LTM + métricas + **GCN implementado**
  - GCN-BERT: MRR=0.0481, Propagation Reach=64.93%
  - GCN-Random: MRR=0.0376, Propagation Reach=39.02%
- ✓ Semana 3: **LightGCN implementado**
  - LightGCN: MRR=0.0508, Propagation Reach=37.99%
  - Mejor trade-off: máxima precisión con mínima amplificación de fake news
- Semana 4: Análisis comparativo de todos los modelos + informe

## Notebooks Implementados

### Semana 2: GCN Baseline
**Archivo:** `Semana2_GCN.ipynb`

Implementación de Graph Convolutional Network para recomendación de tweets con dos versiones:
- **GCN-BERT**: Usa embeddings semánticos de BERT (384-dim) para items
- **GCN-Random**: Usa embeddings random (64-dim) para items

Incluye:
- Entrenamiento con BPR loss
- Evaluación con MRR, ILD, Coverage
- Análisis de propagación de fake news con Linear Threshold Model
- Distribución de labels en recomendaciones

### Semana 3: LightGCN
**Archivo:** `Semana3_LightGCN.ipynb`

Implementación de LightGCN (arquitectura simplificada sin transformaciones ni activaciones):
- Embeddings aprendibles para users e items (sin features pre-entrenadas)
- 3 capas de propagación con layer combination
- Comparación directa con GCN-BERT y GCN-Random

Incluye análisis completo de recomendación y propagación.

## Resultados Comparativos

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
