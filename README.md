# IIC3633 - Proyecto: Desinformación y Viralidad en Sistemas Recomendadores

Repositorio para el proyecto semestral del ramo IIC3633: Sistemas Recomendadores.

**Dataset:** Twitter15/16 (rumores y noticias falsas)
**Enfoque:** Análisis de viralidad de desinformación y evaluación de bias en recomendaciones

## Integrantes

- Vittorio Salvatore
- Clemente Acevedo
- Cristobal Fuentes

## Estructura

```
data_processing/
├── preprocess_unified.py    # Preprocesamiento de Twitter15/16
├── analyze_h1.py             # Análisis exploratorio (EDA)
├── processed_h1/             # Datos procesados para RecSys
└── plots_and_reports/        # Visualizaciones y estadísticas

midterm/
├── build_graphs.py           # Construcción de grafos para GNNs
├── graph_utils.py            # Utilidades y estadísticas
└── graphs/                   # Grafos bipartito y social (generados)

H1_RecSys.ipynb               # Implementación de sistemas de recomendación
```

## Inicio Rápido

### 1. Preprocesamiento de Datos

```bash
cd data_processing
python3 preprocess_unified.py  # Generar datasets
python3 analyze_h1.py          # Análisis exploratorio
```

### 2. Construcción de Grafos (Midterm)

```bash
cd midterm
python3 build_graphs.py        # Generar grafos bipartito y social
```

## Componentes

### H1: Sistemas de Recomendación Clásicos

**Notebook:** [`H1_RecSys.ipynb`](H1_RecSys.ipynb)

Implementa y compara 5 modelos de recomendación (UKNN, IKNN, Most Popular, Random, TF-IDF) con métricas de bias, exposure y coverage.

### Midterm: GNNs para Recomendación

**Carpeta:** [`midterm/`](midterm/)

Pipeline de construcción de grafos bipartito (user-item) y social (user-user) para modelos de recomendación basados en Graph Neural Networks.

**Modelos planeados:**
- GCN (baseline)
- LightGCN
- Sheaf4Rec
- Linear Threshold Model (propagación de desinformación)

**Métricas:** MRR, Inter-List Diversity, alcance de propagación

## Documentación

- [`data_processing/README.md`](data_processing/README.md) - Preprocesamiento y EDA completo
- [`midterm/README.md`](midterm/README.md) - Construcción de grafos y uso en GNNs