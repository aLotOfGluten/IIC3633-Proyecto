# IIC3633 - Proyecto: Desinformación y Viralidad en Sistemas Recomendadores

Análisis de cómo diferentes algoritmos de recomendación amplifican la propagación de fake news en redes sociales usando el dataset Twitter15/16.

## Integrantes

- Vittorio Salvatore
- Clemente Acevedo
- Cristobal Fuentes

---

## Estructura del Proyecto

```
IIC3633-Proyecto/
├── data_processing/     # Preprocesamiento de Twitter15/16
├── midterm/             # GNN-based Recommender Systems (Semanas 1-4)
└── H1_RecSys.ipynb      # Modelos clásicos (H1)
```

---

## Componentes Principales

### H1: Modelos Clásicos

**Notebook:** [`H1_RecSys.ipynb`](H1_RecSys.ipynb)

Implementa User-KNN, Item-KNN, Most Popular, Random, TF-IDF.

**Métricas:** Precision@K, Recall@K, Coverage, Exposure de labels

---

### Midterm: Graph Neural Networks

**Carpeta:** [`midterm/`](midterm/) | **Documentación:** [`midterm/README.md`](midterm/README.md)

**Notebooks:**
- `Semana2_GCN.ipynb` - GCN-BERT y GCN-Random
- `Semana3_LightGCN.ipynb` - LightGCN (estado del arte)
- `Semana4_Comparative.ipynb` - Comparación completa de 6 modelos

**Modelos implementados:**
- GCN-BERT (GNN + embeddings semánticos)
- GCN-Random (GNN sin features)
- LightGCN (arquitectura simplificada)

**Análisis de propagación:**
- Linear Threshold Model para simular difusión de fake news
- Métricas: Reach, Depth, Speed
- Comparación de amplificación entre modelos

---

## Documentación Detallada

| Sección | README |
|---------|--------|
| Preprocesamiento | [`data_processing/README.md`](data_processing/README.md) |
| Midterm (GNNs) | [`midterm/README.md`](midterm/README.md) |

---

## Dataset: Twitter15/16

**⚠️ Nota importante:** Los valores finales del dataset procesado difieren significativamente de los reportados en los papers originales debido a filtros de calidad, eliminación de datos 2016+ (unverified), y split temporal aplicado.

### Dataset actual (branch `main`, sin filtro temporal)
- **Usuarios:** 4,856
- **Items (tweets):** 2,308
- **Interacciones:** 63,850
- **Labels:** true (24.6%), false (23.5%), unverified (25.5%), non-rumor (26.4%)
- **Split:** Leave-one-out (~92% train, ~8% test)

### Dataset procesado con split temporal (integrado)
- **Total interacciones:** 110,732 (Twitter15 + Twitter16)
- **Usuarios únicos:** 39,958
- **Items únicos:** 1,386
- **Fecha de corte:** Marzo 2015 (para garantizar casos False Rumor en test)
- **Datos 2016+ descartados** (tweets mayormente unverified)
- **Split temporal:** 80% train / 10% val / 10% test
- **Ubicación:** `data_processing/processed_round2/`

**Fuentes:** [Twitter15/16](https://github.com/gszswork/Twitter15_16_dataset) + [Kaggle](https://www.kaggle.com/datasets/syntheticprogrammer/rumor-detection-acl-2017/data)

---

## 🔄 Estado Actual del Proyecto

| Fase | Estado | Descripción |
|------|--------|-------------|
| **H1** | ✅ | Modelos clásicos (User-KNN, Item-KNN, Most Popular, Random, TF-IDF) |
| **Midterm** | ✅ | GNNs (GCN-BERT, GCN-Random, LightGCN) + Linear Threshold Model + Comparación completa |
| **Procesamiento temporal** | ✅ | Datos procesados y unificados en `processed_round2/` |
| **Re-construcción grafo** | ✅ | Grafos temporales construidos con cap de edges (5) y validación correcta |
| **Negative sampling** | ✅ | 382K samples generados (~11/usuario) con ventana temporal y balance 50/50 |
| **Re-entrenamiento GNNs** | ⏳ | Pendiente: re-entrenar modelos (3 capas) con nuevos datos |
| **Parte 2: LTM + Social Graph** | 🔄 | LTM implementado, pendiente integración con grafos temporales |
| **Informe final** | ⏳ | Pendiente consolidación de resultados |

**Detalles técnicos:** Ver [`midterm/README.md`](midterm/README.md)

---

## 📋 Trabajo Técnico Completado

### ✅ 1. Construcción de Grafos Temporales
**Script:** `build_temporal_graphs.py`

**Características:**
- 1 nodo por usuario, 1 nodo por item (validado)
- Cap de 5 en edges duplicados usuario-item
- Datos de Twitter15 + Twitter16 unificados con split temporal

**Output:** `midterm/graphs_temporal/`
- `bipartite_graph.pt` - 39,958 users × 1,386 items = 176,996 edges
- `social_graph.pt` - 39,958 nodos, 311,020 edges (threshold ≥3 items compartidos)
- `train_interactions.csv` - 88,620 interacciones (80%)
- `val_interactions.csv` - 11,534 interacciones (10.4%)
- `test_interactions.csv` - 10,578 interacciones (9.6%)
- `user_map.csv`, `item_map.csv` - Mapeos de índices

### ✅ 2. Negative Sampling Temporal
**Script:** `negative_sampling.py`

**Características:**
- 382,391 samples negativos generados
- 34,618 usuarios únicos (~11 samples por usuario)
- Ventana temporal respetada (first_activity → last_activity)
- Balance 50/50 entre items populares y no populares

**Output:** `data_processing/processed_round2/negative_samples.csv`

---

## 📋 Pendientes Técnicos

### 1. Re-entrenamiento de Modelos GNN (3 capas)
- Re-entrenar GCN-BERT, GCN-Random, LightGCN con:
  - Grafos temporales (`midterm/graphs_temporal/`)
  - Negative sampling implementado
  - 3 capas GNN (actualmente 2)
- Comparar resultados con modelos baseline

### 2. Integración Final con LTM
- Ejecutar Linear Threshold Model con grafos temporales
- Analizar propagación de fake news con nuevos datos
- Comparar reach/depth/speed entre modelos re-entrenados

### 3. Informe y Visualizaciones
- Consolidar resultados finales
- Documentar diferencias con papers originales
- Generar gráficos comparativos para informe
