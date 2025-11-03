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

### Dataset procesado con split temporal (branch `development`)
- **Twitter15:** ~100k filas post-procesamiento
- **Fecha de corte:** Marzo 2015 (para garantizar casos False Rumor en test)
- **Datos 2016+ descartados** (tweets mayormente unverified)
- **Split deseado:** 70% train / 15% val / 15% test (temporal)
- **Ubicación:** `data_processing_2/processed_round2/`

**Fuentes:** [Twitter15/16](https://github.com/gszswork/Twitter15_16_dataset) + [Kaggle](https://www.kaggle.com/datasets/syntheticprogrammer/rumor-detection-acl-2017/data)

---

## 🔄 Estado Actual del Proyecto

| Fase | Estado | Descripción |
|------|--------|-------------|
| **H1** | ✅ | Modelos clásicos (User-KNN, Item-KNN, Most Popular, Random, TF-IDF) |
| **Midterm** | ✅ | GNNs (GCN-BERT, GCN-Random, LightGCN) + Linear Threshold Model + Comparación completa |
| **Procesamiento temporal** | 🔄 | Datos procesados en `development`, pendiente integración |
| **Re-construcción grafo** | ⏳ | Pendiente: grafo con datos temporales + validación de colapso correcto |
| **Negative sampling** | ⏳ | Pendiente: 10-15 negativos/usuario, ventana temporal, balance popularidad |
| **Re-entrenamiento GNNs** | ⏳ | Pendiente: re-entrenar modelos (3 capas) con nuevos datos |
| **Parte 2: LTM + Social Graph** | 🔄 | LTM implementado, pendiente integración final |
| **Informe final** | ⏳ | Pendiente consolidación de resultados |

**Detalles técnicos:** Ver [`midterm/README.md`](midterm/README.md)

---

## 📋 Pendientes Técnicos Consolidados

### 1. Construcción de Grafo Definitivo
- Validar colapso correcto: 1 nodo/usuario, 1 nodo/item (tree)
- Implementar cap en edges duplicados entre usuario-item
- Usar datos de `data_processing_2/processed_round2/`

### 2. Negative Sampling Real
- Generar 10-15 negativos por usuario
- Respetar ventana temporal de actividad del usuario
- Balance de popularidad 50/50 (evitar solo items no populares)
- Actualmente: archivo `negative_samples.ipynb` vacío

### 3. Re-entrenamiento de Modelos
- Re-entrenar GCN-BERT, GCN-Random, LightGCN con:
  - Datos temporales (`processed_round2/`)
  - Negative sampling implementado
  - Idealmente 3 capas GNN
- Comparar resultados con modelos actuales

### 4. Parte 2: Grafo Social + LTM
- Grafo social ya implementado (`midterm/social_graph.pt`)
- Linear Threshold Model ya implementado
- Pendiente: ejecutar simulaciones finales con datos actualizados

### 5. Integración Final
- Consolidar pipeline completo: data → graph → negative sampling → models → LTM
- Documentar diferencias con papers originales
- Generar visualizaciones finales para informe
