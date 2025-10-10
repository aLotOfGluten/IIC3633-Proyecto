# Midterm - Construcción de Grafos para GNNs

Pipeline de construcción de grafos bipartito y social para sistemas de recomendación basados en GNNs, usando datasets Twitter15/16 procesados.

## Contenido

```
midterm/
├── README.md               # Este archivo
├── graph_utils.py          # Utilidades y estadísticas
├── build_graphs.py         # Script principal de construcción
└── graphs/                 # Grafos generados (creado al ejecutar)
    ├── bipartite_graph.pt
    ├── social_graph.pt
    └── graph_stats.txt
```

## Uso Rápido

```bash
python build_graphs.py
```

Esto genera automáticamente los grafos en formato PyTorch Geometric en `graphs/`.

## Grafos Generados

### 1. Grafo Bipartito (User-Item)

Grafo bipartito que conecta usuarios con ítems (tweets) según interacciones del conjunto de entrenamiento.

**Estructura:**
- Usuarios: índices `[0, num_users-1]`
- Ítems: índices `[num_users, num_users+num_items-1]`
- Aristas bidireccionales

**Propiedades:**
```python
bipartite_graph.edge_index        # [2, num_edges]
bipartite_graph.num_users         # Número de usuarios
bipartite_graph.num_items         # Número de ítems
```

### 2. Grafo Social (User-User)

Grafo social implícito construido desde co-interacciones: dos usuarios se conectan si compartieron interacción con al menos `k` ítems en común.

**Estructura:**
- Nodos: usuarios únicamente
- Aristas: conexiones usuario-usuario
- Pesos: número de ítems compartidos

**Propiedades:**
```python
social_graph.edge_index           # [2, num_edges]
social_graph.edge_weight          # [num_edges] - pesos
social_graph.num_nodes            # Número de usuarios
```

**Parámetros:**
- `min_common_items=3` (configurable)

## Configuración

Editar `build_graphs.py` línea 145:

```python
bipartite_graph, social_graph = pipeline.run(min_common_items=3)
```

## Archivos de Entrada

Desde `../data_processing/processed_h1/`:
- `train_interactions_idx.csv`
- `user_map.csv`
- `item_map.csv`

## Uso en Modelos GNN

### Cargar Grafos

```python
import torch

bipartite = torch.load('graphs/bipartite_graph.pt')
social = torch.load('graphs/social_graph.pt')
```

### Embeddings Aprendibles

Los grafos no incluyen matriz de features (`x=None`). Usar embeddings en el modelo:

```python
import torch.nn as nn
from torch_geometric.nn import GCNConv

class RecommenderGNN(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, 32)

    def forward(self, data):
        x = self.embedding(torch.arange(data.num_nodes))
        x = self.conv1(x, data.edge_index)
        return x
```

### Separar Embeddings User-Item

Para modelos como LightGCN:

```python
user_embeddings = embeddings[:num_users]
item_embeddings = embeddings[num_users:]
```

## Estadísticas (dataset procesado h1)

**Grafo Bipartito:**
- Nodos: 7,164 (4,856 usuarios + 2,308 ítems)
- Aristas: 117,988
- Densidad: 0.53%

**Grafo Social:**
- Nodos: 4,856 usuarios
- Aristas: 295,660
- Densidad: 1.25%
- Peso promedio: 4.93 ítems compartidos

Ver `graphs/graph_stats.txt` para métricas detalladas.

## Dependencias

```bash
pip install torch torch-geometric scipy pandas numpy
```

## Próximos Pasos (Semana 2)

- Implementar modelo GCN baseline
- Implementar Linear Threshold Model sobre grafo social
- Calcular métricas: MRR, Inter-List Diversity
