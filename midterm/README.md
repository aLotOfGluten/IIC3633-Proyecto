# Midterm - GNN-based Recommender Systems

Pipeline completo para sistemas de recomendación basados en Graph Neural Networks sobre datasets Twitter15/16. Incluye construcción de grafos, generación de node embeddings, Linear Threshold Model para simulación de propagación, y métricas de evaluación.

## Estructura del Proyecto

```
midterm/
├── build_graphs.py              # Construcción de grafos bipartito y social
├── prepare_features.py          # Generación de node embeddings
├── linear_threshold_model.py    # Simulación de propagación (LTM)
├── metrics.py                   # MRR, ILD, propagation metrics
├── graph_utils.py               # Utilidades de carga y estadísticas
├── node_features.py             # Encoders BERT y random
├── demo_ltm.py                  # Suite de demos interactivas
├── test_pipeline.py             # Tests de verificación
└── graphs/                      # Salidas generadas
    ├── bipartite_graph.pt
    ├── social_graph.pt
    ├── graph_stats.txt
    ├── item_embeddings_bert.pt
    ├── item_embeddings_random.pt
    └── user_embeddings_init.pt
```

## Setup Rápido

### 1. Instalar dependencias

```bash
pip install torch torch-geometric scipy pandas numpy sentence-transformers
```

### 2. Construir grafos

```bash
cd midterm
python build_graphs.py
```

### 3. Generar embeddings

```bash
python prepare_features.py --bert --users
```

### 4. Verificar pipeline

```bash
python test_pipeline.py
```

## Componentes Principales

### Grafos

**Bipartito (User-Item)**
- Conecta usuarios con ítems según interacciones de entrenamiento
- Nodos: `[user_0, ..., user_N, item_0, ..., item_M]`
- Aristas bidireccionales
- Estadísticas: 4,856 users × 2,308 items = 117,988 edges (densidad 0.53%)

**Social (User-User)**
- Grafo implícito basado en co-interacciones
- Conexión si usuarios comparten ≥3 items en común
- Pesos: número de items compartidos
- Estadísticas: 4,856 nodes, 295,660 edges (densidad 1.25%)

```python
import torch

bipartite = torch.load('graphs/bipartite_graph.pt')
social = torch.load('graphs/social_graph.pt')

print(bipartite.num_users, bipartite.num_items)
print(social.edge_index.shape, social.edge_weight.shape)
```

### Node Embeddings

**BERT Embeddings (items)**
- Modelo: `all-MiniLM-L6-v2` (384-dim)
- Embeddings semánticos del contenido de tweets
- Maneja automáticamente items sin texto (130 de 2308)

**Random/Learnable Embeddings (users)**
- Inicialización Xavier para embeddings entrenables
- Dimensión configurable (default: 64)

```bash
# Generar BERT embeddings
python prepare_features.py --bert

# Generar user embeddings
python prepare_features.py --users --dim 64

# Generar todos
python prepare_features.py --all
```

```python
from graph_utils import load_embeddings

item_emb = load_embeddings('graphs/item_embeddings_bert.pt')
user_emb = load_embeddings('graphs/user_embeddings_init.pt')
```

### Linear Threshold Model

Simulación de propagación de información sobre el grafo social usando el modelo clásico de umbrales.

**Características:**
- Thresholds configurables (uniform random, custom, etc.)
- Simulación estocástica con múltiples rondas
- Análisis Monte Carlo para esperanza de propagación

```python
from linear_threshold_model import LinearThresholdModel
import torch

ltm = LinearThresholdModel(torch.load('graphs/social_graph.pt'))

seed_nodes = {0, 1, 2, 3, 4}
infection_rounds = ltm.simulate(seed_nodes, max_iterations=50)

results = ltm.monte_carlo_propagation(seed_nodes, num_simulations=100)
print(f"Mean reach: {results['mean_reach']*100:.2f}%")
```

**Demo interactiva:**
```bash
python demo_ltm.py
```

### Métricas de Evaluación

**Recomendación:**
- **MRR** (Mean Reciprocal Rank): Calidad del ranking
- **ILD** (Inter-List Diversity): Diversidad entre usuarios
- **Coverage**: % del catálogo recomendado
- **Cosine Diversity**: Diversidad basada en embeddings

**Propagación:**
- **Reach**: % de nodos infectados
- **Depth**: Profundidad de cascada
- **Speed**: Velocidad de propagación
- **Total Infected**: Nodos alcanzados

```python
from metrics import RecommendationMetrics, PropagationMetrics

rec_metrics = RecommendationMetrics(
    recommendations=recs,
    ground_truth=gt,
    catalog_size=num_items,
    embeddings=item_emb
).compute_all()

prop_metrics = PropagationMetrics(
    infection_rounds=rounds,
    total_nodes=num_users
).compute_all()
```

---

## Guía de Implementación de Modelos

### Archivos Requeridos

| Archivo | Descripción |
|---------|-------------|
| `graphs/bipartite_graph.pt` | Grafo user-item para entrenamiento |
| `graphs/social_graph.pt` | Grafo social para propagación |
| `graphs/item_embeddings_bert.pt` | Node features de ítems (384-dim) |
| `../data_processing/processed_h1/train_interactions_idx.csv` | Train set |
| `../data_processing/processed_h1/test_interactions_idx.csv` | Test set |
| `../data_processing/processed_h1/item_labels.csv` | Labels (true/false/unverified/non-rumor) |

---

### 1. GCN Baseline (Semana 2)

#### Setup

```python
import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.nn import GCNConv

import sys
sys.path.insert(0, 'midterm')
from graph_utils import load_embeddings
from metrics import RecommendationMetrics

bipartite = torch.load('midterm/graphs/bipartite_graph.pt')
item_features = load_embeddings('midterm/graphs/item_embeddings_bert.pt')
train = pd.read_csv('data_processing/processed_h1/train_interactions_idx.csv')
test = pd.read_csv('data_processing/processed_h1/test_interactions_idx.csv')

num_users = bipartite.num_users
num_items = bipartite.num_items
edge_index = bipartite.edge_index
```

#### Modelo

```python
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

#### Entrenamiento (BPR Loss)

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

model = GCNRecommender(num_users, num_items, item_features.size(1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    loss = train_epoch(model, edge_index, item_features, train, optimizer)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

#### Evaluación

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

    metrics_calc = RecommendationMetrics(
        recommendations=recommendations,
        ground_truth=ground_truth,
        catalog_size=model.num_items,
        embeddings=item_features
    )
    return metrics_calc.compute_all()

metrics = evaluate(model, edge_index, item_features, test)
print(f"MRR: {metrics['mrr']:.4f}")
print(f"ILD: {metrics['ild']:.4f}")
print(f"Coverage: {metrics['coverage']:.4f}")
```

---

### 2. LightGCN (Semana 3)

Modelo simplificado que elimina transformaciones no lineales y features explícitas.

```python
from torch_geometric.nn import LGConv

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

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

**Hiperparámetros recomendados:**
- Layers: 3-4
- Embedding dim: 64-128
- Learning rate: 0.001
- Epochs: 50-100

---

### 3. Sheaf4Rec (Semana 3 - Opcional)

Modelo avanzado usando sheaf neural networks. Consultar paper original para implementación.

**Referencias:**
- "Sheaf Neural Networks for Graph-based Recommendations"
- PyTorch Geometric: sheaf convolution layers

---

### 4. Análisis de Propagación de Desinformación (Semana 4)

#### Objetivo

Medir cómo cada modelo de recomendación amplifica la propagación de fake news en la red social.

#### Metodología

1. Generar recomendaciones top-10 para todos los usuarios
2. Identificar usuarios expuestos a fake news (label='false')
3. Simular propagación usando Linear Threshold Model
4. Comparar alcance de desinformación entre modelos

```python
from linear_threshold_model import LinearThresholdModel
from metrics import PropagationMetrics
import pandas as pd
import torch

labels = pd.read_csv('data_processing/processed_h1/item_labels.csv')
social = torch.load('midterm/graphs/social_graph.pt')
ltm = LinearThresholdModel(social, seed=42)

def analyze_misinformation_spread(recommendations, model_name):
    seed_users = set()

    for user_idx, rec_list in enumerate(recommendations):
        for item_idx in rec_list[:10]:
            item_label = labels[labels['item_idx'] == item_idx]['label'].values
            if len(item_label) > 0 and item_label[0] == 'false':
                seed_users.add(user_idx)
                break

    print(f"\n{model_name}:")
    print(f"  Usuarios con fake news en top-10: {len(seed_users)}")

    if len(seed_users) == 0:
        print("  No hay usuarios expuestos a fake news")
        return None

    infection_rounds = ltm.simulate(seed_users, max_iterations=50)
    prop_metrics = PropagationMetrics(infection_rounds, ltm.num_nodes)
    metrics = prop_metrics.compute_all()

    print(f"  Alcance: {metrics['reach']*100:.2f}%")
    print(f"  Profundidad: {metrics['depth']} rondas")
    print(f"  Infectados: {metrics['total_infected']} usuarios")

    return metrics

gcn_metrics = analyze_misinformation_spread(gcn_recommendations, "GCN")
lightgcn_metrics = analyze_misinformation_spread(lightgcn_recommendations, "LightGCN")
```

#### Modelos a Comparar

| Modelo | Tipo | Objetivo |
|--------|------|----------|
| Random | Baseline | Control aleatorio |
| Most Popular | Baseline | Popularidad |
| User-KNN | Colaborativo | Baseline clásico |
| GCN | GNN | Modelo baseline |
| LightGCN | GNN | Estado del arte simplificado |
| Sheaf4Rec | GNN | Estado del arte avanzado |

#### Métricas de Análisis

**Recomendación:**
- MRR (calidad)
- ILD (diversidad)
- Coverage (amplitud)

**Propagación:**
- Reach (alcance)
- Depth (profundidad)
- Speed (velocidad)

**Análisis crítico:**
- ¿Qué modelo tiene mejor MRR?
- ¿Qué modelo es más diverso (ILD)?
- ¿Qué modelo amplifica más fake news?
- ¿Existe trade-off entre precisión y propagación de desinformación?

---

## Consideraciones Técnicas

### Embeddings

- **Items con texto:** 2178 de 2308 items tienen contenido textual
- **Items sin texto:** 130 items reciben embedding BERT de string vacío
- **Alternativa:** Usar embeddings híbridos (BERT + learnable layer)

### Evaluación

- **Estrategia:** Leave-one-out (último item de cada usuario → test)
- **Exclusión:** Eliminar items de train al generar recomendaciones
- **Métricas:** Promediar sobre todos los usuarios

### Balance de Clases

Dataset balanceado: ~25% por clase (true/false/unverified/non-rumor)

**Análisis de bias:**
- Comparar distribución en recomendaciones vs dataset base
- Medir exposure de fake news vs true news
- Identificar si modelos amplifican desinformación sistemáticamente

---

## Referencia de APIs

### DataLoader

```python
from graph_utils import DataLoader

loader = DataLoader(data_path="../data_processing/processed_h1")
interactions = loader.load_interactions()
user_map, item_map = loader.load_mappings()
labels = loader.load_labels()
item_text = loader.load_item_text()
```

### Embeddings

```python
from graph_utils import load_embeddings, save_embeddings

embeddings = load_embeddings('graphs/item_embeddings_bert.pt')
save_embeddings(new_embeddings, 'graphs/custom_embeddings.pt')
```

### Linear Threshold Model

```python
from linear_threshold_model import LinearThresholdModel, load_ltm_from_graph

ltm = load_ltm_from_graph('graphs/social_graph.pt', seed=42)
rounds = ltm.simulate(seed_nodes, max_iterations=50)
expected = ltm.expected_propagation(seed_nodes, num_simulations=100)
stats = ltm.monte_carlo_propagation(seed_nodes, num_simulations=100)
```

### Métricas

```python
from metrics import (
    mean_reciprocal_rank,
    inter_list_diversity,
    coverage,
    RecommendationMetrics,
    PropagationMetrics
)

mrr = mean_reciprocal_rank(recommendations, ground_truth)
ild = inter_list_diversity(recommendations)
cov = coverage(recommendations, catalog_size)
```

---

## Estadísticas del Dataset

**Grafos:**
- Bipartito: 4,856 users × 2,308 items = 117,988 edges (0.53% densidad)
- Social: 4,856 users, 295,660 edges (1.25% densidad)

**Interacciones:**
- Train: 58,994 interacciones
- Test: 4,856 interacciones (leave-one-out)
- Promedio: 12.1 interacciones/usuario

**Labels:**
- true: 536 (24.6%)
- false: 511 (23.5%)
- unverified: 555 (25.5%)
- non-rumor: 576 (26.4%)

Ver `graphs/graph_stats.txt` para estadísticas detalladas.

---

## Estado del Proyecto

### ✓ Semana 1 (Completada)
- Construcción de grafo bipartito usuario-ítem
- Construcción de grafo social implícito

### ✓ Semana 2 (Infraestructura completada)
- Encoding de nodos con BERT
- Linear Threshold Model implementado
- Métricas MRR, ILD, propagación implementadas
- **Pendiente:** Entrenamiento de GCN baseline (notebook)

### Semana 3 (Por hacer)
- Implementar y entrenar LightGCN
- Implementar Sheaf4Rec (opcional)

### Semana 4 (Por hacer)
- Evaluar todos los modelos (baselines + GNNs)
- Análisis comparativo de propagación
- Redacción de informe Midterm
