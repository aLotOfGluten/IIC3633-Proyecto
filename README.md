# IIC3633 - Proyecto: Desinformación y Viralidad en Sistemas Recomendadores

Repositorio para el proyecto semestral del ramo IIC3633: Sistemas Recomendadores.

**Dataset:** Twitter15/16 (rumores y noticias falsas)
**Enfoque:** Análisis de viralidad de desinformación y evaluación de bias en recomendaciones

## Integrantes

- Vittorio Salvatore
- Clemente Acevedo
- Cristobal Fuentes

## Estructura del Proyecto

```
IIC3633-Proyecto/
├── data_processing/           # Preprocesamiento de Twitter15/16
│   ├── preprocess_unified.py  # Pipeline de procesamiento
│   ├── analyze_h1.py           # Análisis exploratorio (EDA)
│   ├── processed_h1/           # Datos procesados para RecSys
│   └── plots_and_reports/      # Visualizaciones y estadísticas
│
├── midterm/                    # GNN-based Recommender Systems
│   ├── build_graphs.py         # Construcción de grafos
│   ├── prepare_features.py     # Generación de embeddings
│   ├── linear_threshold_model.py  # Modelo de propagación
│   ├── metrics.py              # MRR, ILD, propagation metrics
│   ├── demo_ltm.py             # Suite de demos
│   ├── test_pipeline.py        # Tests de verificación
│   └── graphs/                 # Grafos y embeddings generados
│
└── H1_RecSys.ipynb             # Sistemas clásicos (H1)
```

## Inicio Rápido

### 1. Preprocesamiento de Datos

```bash
cd data_processing
python3 preprocess_unified.py
python3 analyze_h1.py
```

**Salidas:**
- `processed_h1/`: Interacciones train/test, mapeos, labels
- `plots_and_reports/`: Análisis de viralidad

### 2. Construcción de Grafos (Midterm)

```bash
cd midterm
python3 build_graphs.py
python3 prepare_features.py --bert --users
python3 test_pipeline.py
```

**Salidas:**
- `graphs/bipartite_graph.pt`: Grafo user-item
- `graphs/social_graph.pt`: Grafo social implícito
- `graphs/item_embeddings_bert.pt`: Node features

### 3. Modelos de Recomendación

**H1 (clásicos):** Ver `H1_RecSys.ipynb`
**Midterm (GNNs):** Ver `midterm/README.md` para guías de implementación

---

## Componentes

### H1: Sistemas de Recomendación Clásicos

**Notebook:** [`H1_RecSys.ipynb`](H1_RecSys.ipynb)

Implementa y compara modelos tradicionales:
- User-KNN, Item-KNN
- Most Popular, Random
- TF-IDF (content-based)

**Métricas:** Precision@K, Recall@K, Coverage, Bias de labels

### Midterm: GNN-based Recommender Systems

**Carpeta:** [`midterm/`](midterm/)

Pipeline completo para sistemas de recomendación basados en GNNs:

**Grafos:**
- Bipartito (user-item): 4,856 users × 2,308 items
- Social (user-user): Grafo implícito de co-interacciones

**Features:**
- BERT embeddings (384-dim) para contenido de tweets
- User embeddings entrenables

**Modelos planeados:**
- GCN (baseline)
- LightGCN (estado del arte)
- Sheaf4Rec (avanzado)

**Análisis de propagación:**
- Linear Threshold Model (LTM)
- Simulación de difusión de desinformación
- Métricas: Reach, Depth, Speed

**Métricas de evaluación:**
- MRR (Mean Reciprocal Rank)
- ILD (Inter-List Diversity)
- Coverage
- Propagation metrics

### Proyección Final: Análisis de Desinformación

**Objetivo:** Medir cómo diferentes algoritmos de recomendación amplifican la propagación de fake news en redes sociales.

**Metodología:**
1. Identificar usuarios expuestos a desinformación (label='false' en top-10)
2. Simular propagación con Linear Threshold Model
3. Comparar alcance entre modelos (Random, Popular, KNN, GCN, LightGCN)
4. Analizar trade-offs entre precisión y amplificación de desinformación

---

## Documentación Detallada

| Componente | README |
|------------|--------|
| Preprocesamiento | [`data_processing/README.md`](data_processing/README.md) |
| GNNs y propagación | [`midterm/README.md`](midterm/README.md) |

---

## Dataset

**Fuentes:**
- Estructura de propagación: [gszswork/Twitter15_16_dataset](https://github.com/gszswork/Twitter15_16_dataset)
- Contenido de tweets: [Kaggle - Rumor Detection ACL 2017](https://www.kaggle.com/datasets/syntheticprogrammer/rumor-detection-acl-2017/data)

**Estadísticas:**
- Usuarios: 4,856
- Items (tweets): 2,308
- Interacciones: 63,850
- Labels: true (24.6%), false (23.5%), unverified (25.5%), non-rumor (26.4%)

**Papers de referencia:**

```bibtex
@inproceedings{ma2017detect,
  title={Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning},
  author={Ma, Jing and Gao, Wei and Wong, Kam-Fai},
  booktitle={ACL},
  year={2017}
}
```

---

## Dependencias

```bash
pip install torch torch-geometric scipy pandas numpy matplotlib sentence-transformers
```

---

## Roadmap del Proyecto

### ✓ H1 (Completado)
- Preprocesamiento de Twitter15/16
- Sistemas clásicos de recomendación
- Análisis de bias y coverage

### ✓ Midterm - Semana 1 (Completado)
- Construcción de grafos bipartito y social
- Infraestructura de análisis

### ✓ Midterm - Semana 2 (Infraestructura completada)
- Node embeddings con BERT
- Linear Threshold Model
- Métricas (MRR, ILD, propagación)
- **Pendiente:** Entrenamiento de GCN baseline

### Midterm - Semana 3 (En progreso)
- Implementar LightGCN
- Implementar Sheaf4Rec (opcional)

### Midterm - Semana 4 (Por hacer)
- Evaluación comparativa de todos los modelos
- Análisis de propagación de desinformación
- Informe Midterm

### Final (Planeado)
- Análisis profundo de amplificación de fake news
- Comparación entre métodos clásicos vs GNNs
- Estudio de trade-offs: precisión vs difusión
- Informe final
