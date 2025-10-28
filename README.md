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

4,856 usuarios × 2,308 tweets × 63,850 interacciones

**Labels:** true (24.6%), false (23.5%), unverified (25.5%), non-rumor (26.4%)

**Fuentes:** [Twitter15/16](https://github.com/gszswork/Twitter15_16_dataset) + [Kaggle](https://www.kaggle.com/datasets/syntheticprogrammer/rumor-detection-acl-2017/data)

---

## Progreso

| Fase | Estado | Descripción |
|------|--------|-------------|
| **H1** | ✓ | Modelos clásicos (User-KNN, Most Popular, Random, TF-IDF) |
| **Midterm** | ✓ | GNNs (GCN, LightGCN) + Linear Threshold Model + Comparación completa |
| **Final** | 🔄 | Análisis profundo de amplificación de fake news |

**Detalles:** Ver [`midterm/README.md`](midterm/README.md) para progreso por semana

---

## Próximos pasos para el informe

Cabros, para cerrar el midterm falta:

1. **Redactar el informe** (documento aparte)
   - Intro: explicar el problema de desinformación en RecSys
   - Metodología:
     - Construcción de grafos (bipartito + social)
     - Modelos implementados (6 en total: Random, Most Popular, User-KNN, GCN-BERT, GCN-Random, LightGCN)
     - Linear Threshold Model para simular propagación
   - Resultados:
     - Tabla comparativa (ya está en Semana4)
     - Gráfico trade-off precisión vs amplificación
     - Ejemplos de recomendación por usuario (para cumplir feedback de H1)
     - Distribución de labels por modelo
   - Análisis:
     - ¿Por qué LightGCN es el mejor modelo?
     - Trade-off entre MRR y propagación
     - BERT amplifica contenido viral pero peligroso
   - Conclusiones: implicaciones para sistemas reales

2. **Revisar feedback de H1**
   - ✓ Definir bien "desinformación" (label='false' = fake news verificadas)
   - ✓ Evaluar métricas de desinformación en TODOS los modelos (ya está en Semana4)
   - ✓ Agregar ejemplos de recomendación por usuario (ya está en Semana4)

**Nota:** Sheaf4Rec lo dejamos como trabajo futuro por complejidad. Con 6 modelos ya tenemos comparación sólida.
