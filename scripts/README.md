# Scripts de Procesamiento y Construcción de Grafos

Scripts utilitarios para preprocesar datos, construir grafos y generar negative samples con split temporal.

---

## Scripts Principales

### 1. `build_temporal_graphs.py` - Construcción de grafos temporales

**Descripción:**
Construye grafos bipartito usuario-item y grafo social a partir de datos procesados con split temporal.

**Input:**
- `../data_processing/processed_round2/twitter15_processed.csv`
- `../data_processing/processed_round2/twitter16_processed.csv`

**Output:**
Carpeta `../midterm/graphs_temporal/`:
- `bipartite_graph.pt` - Grafo bipartito (39,958 users × 1,386 items = 176,996 edges)
- `social_graph.pt` - Grafo social (39,958 nodos, 311,020 edges)
- `train_interactions.csv` - 88,620 interacciones (80%)
- `val_interactions.csv` - 11,534 interacciones (10.4%)
- `test_interactions.csv` - 10,578 interacciones (9.6%)
- `user_map.csv`, `item_map.csv` - Mapeos de índices

**Características:**
- Cap de 5 en edges duplicados usuario-item
- Grafo social construido con threshold ≥3 items compartidos
- 1 nodo por usuario, 1 nodo por item (sin duplicados)

**Uso:**
```bash
python scripts/build_temporal_graphs.py
```

---

### 2. `negative_sampling.py` - Negative sampling temporal

**Descripción:**
Genera negative samples para entrenamiento BPR, respetando ventanas temporales de actividad de usuarios.

**Input:**
- `../data_processing/processed_round2/twitter15_user_activity.csv`
- `../data_processing/processed_round2/twitter16_user_activity.csv`
- `../midterm/graphs_temporal/item_map.csv`
- Archivos de interacciones procesados

**Output:**
- `../data_processing/processed_round2/negative_samples.csv` - 382,391 samples

**Características:**
- ~11 negative samples por usuario
- Ventana temporal respetada (`first_activity` → `last_activity`)
- Balance 50/50 entre items populares y no populares
- Exclusión de items ya interactuados por el usuario

**Uso:**
```bash
python scripts/negative_sampling.py
```

---

## Scripts de Split (Experimentales)

Estos scripts generan diferentes tipos de splits para experimentación:

### 3. `create_global_temporal_split.py`
Split temporal global (un único corte temporal para todos los usuarios).

### 4. `create_per_user_temporal_split.py`
Split temporal por usuario (cada usuario tiene su propio corte temporal).

### 5. `create_random_split.py`
Split aleatorio estratificado (baseline sin consideraciones temporales).

### 6. `create_temporal_split.py`
Script general de split temporal (variante del global).

**Nota:** Estos scripts fueron usados para experimentación. El split final usado en el proyecto es el generado por `build_temporal_graphs.py`.

---

## Orden de Ejecución

Para reproducir el pipeline completo desde cero:

```bash
# 1. Procesar datos con split temporal (notebooks en data_processing_temporal/)
cd data_processing_temporal
# Ejecutar: data_processing_t15.ipynb y data_processing_t16.ipynb

# 2. Construir grafos temporales
cd ..
python scripts/build_temporal_graphs.py

# 3. Generar negative samples
python scripts/negative_sampling.py

# 4. Entrenar modelos (notebook principal)
cd midterm
# Ejecutar: GNN_Temporal_Final.ipynb
```

---

## Dependencias

Los scripts requieren:
- `pandas`
- `numpy`
- `torch`
- `scipy`

Instalar con:
```bash
pip install pandas numpy torch scipy
```

O usar el `requirements.txt` del proyecto:
```bash
pip install -r midterm/requirements.txt
```

---

## Integración con el Proyecto

Estos scripts son el **puente** entre el preprocesamiento de datos y el entrenamiento de modelos:

```
data_processing_temporal/
  ↓ (notebooks generan CSVs procesados)
processed_round2/
  ↓ (scripts/ consumen y generan grafos)
midterm/graphs_temporal/
  ↓ (notebooks de midterm/ entrenan modelos)
midterm/GNN_Temporal_Final.ipynb
```

---

## Notas Técnicas

### Cap de edges duplicados
El script `build_temporal_graphs.py` limita a 5 el número máximo de interacciones usuario-item duplicadas. Esto evita que usuarios extremadamente activos dominen el grafo.

### Threshold de grafo social
El grafo social se construye conectando usuarios que comparten ≥3 items. Este threshold balancea densidad del grafo vs. calidad de las conexiones.

### Ventana temporal en negative sampling
Los negative samples se generan solo con items que existían durante el período de actividad del usuario. Esto garantiza que el sampling sea temporalmente coherente.
