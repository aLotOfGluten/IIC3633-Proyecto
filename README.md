# IIC3633 - Proyecto: Desinformación y Viralidad en Sistemas Recomendadores

Análisis de cómo diferentes algoritmos de recomendación amplifican la propagación de fake news en redes sociales usando el dataset Twitter15/16.

## Integrantes

- Vittorio Salvatore
- Clemente Acevedo
- Cristobal Fuentes

---

## 📓 Visualización de Notebooks

**⚠️ Nota importante:** Si un notebook (.ipynb) no se renderiza correctamente en GitHub, puedes abrirlo directamente desde Google Colab:

1. **Opción 1 (Recomendada):** Ve a [Google Colab](https://colab.research.google.com/) → "File" → "Open notebook" → pestaña "GitHub"
2. Pega la URL de este repositorio: `https://github.com/aLotOfGluten/IIC3633-Proyecto`
3. Selecciona el notebook que deseas abrir

**Opción 2:** Agrega `https://colab.research.google.com/github/` antes de la URL del notebook en GitHub.

**Ejemplo:**
```
Original: https://github.com/aLotOfGluten/IIC3633-Proyecto/blob/main/midterm/GNN_Temporal_Final.ipynb
Colab:    https://colab.research.google.com/github/aLotOfGluten/IIC3633-Proyecto/blob/main/midterm/GNN_Temporal_Final.ipynb
```

---

## Estructura del Proyecto

```
IIC3633-Proyecto/
├── README.md
├── notebooks/                    # Notebooks principales
│   └── H1_RecSys.ipynb          # Modelos clásicos (H1)
├── midterm/                     # GNN-based Recommender Systems
│   ├── GNN_Temporal_Final.ipynb    # 🎯 Notebook principal del proyecto
│   ├── Semana2_GCN.ipynb            # GCN baseline + LTM
│   ├── Semana3_LightGCN.ipynb       # LightGCN implementation
│   ├── Semana4_Comparative.ipynb    # Comparación completa
│   ├── Temporal_GNN_Training.ipynb  # Re-entrenamiento con grafos temporales
│   ├── *.py                         # Scripts utilitarios (metrics, LTM, etc.)
│   └── graphs*/                     # Grafos generados
├── scripts/                     # Scripts de procesamiento
│   ├── build_temporal_graphs.py
│   ├── negative_sampling.py
│   └── create_*_split.py
├── data_processing/             # Preprocesamiento básico (H1 + Midterm inicial)
├── data_processing_temporal/    # Procesamiento con split temporal
└── datasets/                    # Datos crudos Twitter15/16
```

---

## Componentes Principales

### H1: Modelos Clásicos

**Notebook:** [`notebooks/H1_RecSys.ipynb`](notebooks/H1_RecSys.ipynb)

Implementa User-KNN, Item-KNN, Most Popular, Random, TF-IDF.

**Métricas:** Precision@K, Recall@K, Coverage, Exposure de labels

---

### Midterm: Graph Neural Networks

**Carpeta:** [`midterm/`](midterm/) | **Documentación:** [`midterm/README.md`](midterm/README.md)

**🎯 Notebook Principal:** [`midterm/GNN_Temporal_Final.ipynb`](midterm/GNN_Temporal_Final.ipynb)
- Implementación completa con grafos temporales, negative sampling y 3 capas GNN
- Modelos GCN-BERT, GCN-Random y LightGCN v2
- Análisis de desinformación y propagación con LTM

**Notebooks del Roadmap (Semanas 1-4):**
- [`Semana2_GCN.ipynb`](midterm/Semana2_GCN.ipynb) - GCN-BERT y GCN-Random baseline
- [`Semana3_LightGCN.ipynb`](midterm/Semana3_LightGCN.ipynb) - LightGCN (estado del arte)
- [`Semana4_Comparative.ipynb`](midterm/Semana4_Comparative.ipynb) - Comparación de 6 modelos
- [`Temporal_GNN_Training.ipynb`](midterm/Temporal_GNN_Training.ipynb) - Re-entrenamiento con grafos temporales

**Modelos implementados:**
- GCN-BERT (GNN + embeddings semánticos BERT)
- GCN-Random (GNN con random embeddings)
- LightGCN (arquitectura simplificada, state-of-the-art)

**Análisis de propagación:**
- Linear Threshold Model para simular difusión de fake news en grafo social
- Métricas: Reach, Depth, Speed
- Comparación de amplificación entre modelos
- Distribución de labels en recomendaciones

---

## Documentación Detallada

| Sección | README |
|---------|--------|
| Preprocesamiento básico | [`data_processing/README.md`](data_processing/README.md) |
| Procesamiento temporal | [`data_processing_temporal/README.md`](data_processing_temporal/README.md) |
| Midterm (GNNs) | [`midterm/README.md`](midterm/README.md) |
| Scripts utilitarios | [`scripts/README.md`](scripts/README.md) |

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
| **H1 - Modelos Clásicos** | ✅ | User-KNN, Item-KNN, Most Popular, Random, TF-IDF implementados |
| **Midterm (Semanas 1-4)** | ✅ | GNNs baseline + LTM + Comparación de 6 modelos |
| **Procesamiento temporal** | ✅ | Datos procesados con split temporal 80/10/10 |
| **Construcción de grafos** | ✅ | Grafos temporales con cap de edges y validación |
| **Negative sampling** | ✅ | 382K samples con ventana temporal (~11/usuario) |
| **Re-entrenamiento GNNs (3 capas)** | ✅ | Implementado en `GNN_Temporal_Final.ipynb` |
| **LTM + Grafo Social** | ✅ | Análisis de propagación implementado |
| **Análisis de desinformación** | ✅ | Distribución de labels y exposición por modelo |
| **Documentación** | 🔄 | READMEs actualizados, pendiente informe final |

**📍 Notebook principal:** [`midterm/GNN_Temporal_Final.ipynb`](midterm/GNN_Temporal_Final.ipynb)

**Detalles técnicos:** Ver [`midterm/README.md`](midterm/README.md)

---

## 📋 Trabajo Técnico Completado

### ✅ 1. Construcción de Grafos Temporales
**Script:** [`scripts/build_temporal_graphs.py`](scripts/build_temporal_graphs.py)

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
**Script:** [`scripts/negative_sampling.py`](scripts/negative_sampling.py)

**Características:**
- 382,391 samples negativos generados
- 34,618 usuarios únicos (~11 samples por usuario)
- Ventana temporal respetada (first_activity → last_activity)
- Balance 50/50 entre items populares y no populares

**Output:** `data_processing/processed_round2/negative_samples.csv`

---

## 📋 Próximos Pasos

### 1. Informe Final
- ✅ Resultados técnicos completos
- 🔄 Consolidar análisis comparativo de todos los modelos
- 🔄 Documentar hallazgos sobre amplificación de desinformación
- 🔄 Generar visualizaciones finales para presentación

### 2. Documentación Adicional
- ✅ READMEs actualizados con nueva estructura
- 🔄 Guía de ejecución end-to-end
- 🔄 Documentar limitaciones y diferencias con papers originales

### 3. Validación Final
- Verificar reproducibilidad de todos los notebooks
- Validar que todas las dependencias estén en `requirements.txt`
- Confirmar que grafos y datos procesados son accesibles

---

## 🔧 Problemas Encontrados y Soluciones Implementadas

Documentación detallada de los desafíos técnicos encontrados durante el desarrollo del proyecto y las soluciones implementadas. Esta sección es crítica para entender la evolución del proyecto desde los notebooks baseline (Semanas 1-4) hasta la implementación final.

---

### Problemas Iniciales (Semanas 1-4)

**Contexto:**

Los notebooks `Semana2_GCN.ipynb`, `Semana3_LightGCN.ipynb` y `Semana4_Comparative.ipynb` implementaron un baseline funcional con:
- Dataset `processed_h1/` (4,856 usuarios, 2,308 items)
- Split leave-one-out (~92% train, ~8% test)
- Grafos bipartito y social construidos
- Modelos GCN y LightGCN con 2 capas
- MRR ~0.048-0.051 (aceptable para baseline)

#### Problema 1: Descarte Masivo de Datos (99.7%)

**Descripción:**
El procesamiento inicial con leave-one-out descartaba casi TODO el dataset Twitter15/16:
- **Dataset completo:** ~607k interacciones (Twitter15)
- **Después de filtros:** Solo 63k interacciones (~10%)
- **Usuarios descartados:** Usuarios con < 8 interacciones eliminados
- **Items descartados:** Solo se mantenían items con suficiente actividad

**Causa raíz:**
- Filtro `MIN_INTERACTIONS = 8` muy restrictivo
- No se consideraba la dimensión temporal del dataset
- Colapsado de cascadas a items no se había implementado correctamente

**Solución:** Unificar Twitter15/16 con filtro temporal (Marzo 2015) → 110k interacciones

---

#### Problema 2: Recomendación Única por Usuario

**Descripción:**
El split leave-one-out asignaba **exactamente 1 item de test por usuario**, lo que generaba:
- Solo 1 caso positivo por usuario en evaluación
- MRR sensible a un solo error
- Imposibilidad de analizar comportamiento temporal de usuarios

**Causa raíz:**
```python
# Split leave-one-out (1 interacción de test por usuario)
test_df = train_df.groupby('user_idx').tail(1)
train_df = train_df.groupby('user_idx').head(-1)
```

**Solución:** Split temporal 80/10/10 sobre interacciones → 10,578 casos de test

---

#### Problema 3: Estructura del Grafo Bipartito

**Descripción:**
El grafo bipartito inicial no colapsaba correctamente las cascadas de propagación:
- **Cascada de propagación:** Múltiples retweets/respuestas al mismo tweet raíz
- **Problema:** Se creaban múltiples nodos para la misma "historia" (source tweet)
- **Edges duplicados:** Un usuario podía tener múltiples edges al mismo item

**Ejemplo:**
```
Usuario A → Tweet 123 (retweet original)
Usuario A → Tweet 123 (retweet 2 horas después)
Usuario A → Tweet 123 (respuesta)

Resultado: 3 edges user_A → item_123 (INCORRECTO)
```

**Solución final:**
```python
# Cap de 5 en edges duplicados usuario-item
grouped = df.groupby(['user_idx', 'item_idx']).size().reset_index(name='count')
grouped['count'] = grouped['count'].clip(upper=MAX_EDGE_CAP)
```

**Justificación del cap:**
- Evita que usuarios extremadamente activos dominen el grafo
- Refleja "interés" sin sobreponderarlo
- Balance entre capturar actividad y evitar bias

---

#### Problema 4: Falta de Decodificación de Timestamps

**Descripción:**
Los IDs de tweets (Snowflake IDs) codifican timestamps, pero no se estaban decodificando:
- **Snowflake ID:** `724703995147751424` → contiene timestamp en bits 22-63
- **Problema:** No se podía ordenar temporalmente las interacciones
- **Impacto:** Imposible hacer split temporal correcto

**Solución:**
```python
def snowflake_to_datetime(snowflake_id: int) -> datetime:
    TWITTER_EPOCH_MS = 1288834974657  # 2010-11-04 01:42:54.657 UTC
    ts_ms = (snowflake_id >> 22) + TWITTER_EPOCH_MS
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
```

Implementado en `data_processing_temporal/data_processing_t15.ipynb` y `t16.ipynb`.

---

#### Problema 5: Cutoff Temporal y Distribución de Labels

**Descripción:**
Datos de 2016+ contenían mayormente tweets `unverified`, diluyendo el análisis de fake news:

| Período | False Rumor (FR) | True Rumor (TR) | Unverified (UR) |
|---------|------------------|-----------------|-----------------|
| 2014-2015 | 49 eventos | 17 eventos | 27 eventos |
| 2016+ | < 5 eventos | < 5 eventos | ~300 eventos |

**Decisión:**
Aplicar cutoff temporal en **Marzo 2015** para garantizar:
- Suficientes casos FR en test set
- Balance razonable entre labels
- Coherencia temporal en la evaluación

---

### 🚨 Problema Crítico: Split Temporal Incorrecto

**Contexto:**

El notebook `Temporal_GNN_Training.ipynb` intentó implementar grafos temporales, pero encontró un **error crítico de diseño** que invalidó completamente el experimento.

#### El Error: Split por Items en vez de Interacciones

**Descripción del error:**

El split se realizó **agrupando por items** y luego separando temporalmente:

```python
# INCORRECTO: Split por items
for item_id in unique_items:
    item_interactions = df[df['item_id'] == item_id].sort_values('timestamp')
    # Dividir las interacciones de ESTE item en train/val/test
    train_items.append(item_id) if es_antes_cutoff(item_id) else val_items.append(item_id)
```

**Resultado:**
- **Train:** Items con timestamps ≤ T1 → 1,110 items únicos
- **Val:** Items con timestamps T1 < t ≤ T2 → 139 items únicos
- **Test:** Items con timestamps > T2 → 137 items únicos
- **Overlap entre splits:** **0 items** compartidos ❌

#### Consecuencias

**1. Cold-Start 100% en Val y Test**

```python
print("Filtrando test set para eliminar items cold-start...")
train_items = set(train_df['item_idx'].unique())
print(f"Items en train: {len(train_items)}")  # 1110
print(f"Test set original: {len(test_df)}")   # 10578

test_df_filtered = test_df[test_df['item_idx'].isin(train_items)].reset_index(drop=True)
print(f"Test set filtrado: {len(test_df_filtered)}")  # 0 ❌
```

**Salida del notebook:**
```
Items en train: 1110
Test set original: 10578
Test set filtrado: 0
Items eliminados (cold-start): 10578
```

**Significado:**
- TODOS los items en test eran completamente nuevos
- El modelo NUNCA había visto esos items en entrenamiento
- Scenario 100% cold-start: **MRR = 0** garantizado

**2. Evaluación Inválida**

Con 0 items en el test set filtrado:
- No se puede calcular MRR, Coverage, ILD
- No se pueden generar recomendaciones significativas
- El experimento completo es inútil

**3. Confusión Conceptual: Recomendación Online vs. Offline**

**Debate interno del equipo:**

> "Espera pero si hacemos eso, ya no sería recomendación online, y estaría sesgado para probar la parte dos del grafo social. Tendríamos que cortar temporalmente por tweet en vez de por evento de tweet, y ponerle mas énfasis al texto para el embedding inicial en vez de para el grafo"

**Análisis:**
- **Recomendación online (cold-start real):** El sistema debe recomendar items NUEVOS que nunca vio
- **Recomendación offline (transductivo):** El sistema recomienda items que vio en train, pero a nuevos usuarios o en nuevos contextos

**Realidad:**
- RecSys papers usan setup **transductivo** (items conocidos) para medir capacidad de generalización
- Cold-start real (items completamente nuevos) requiere content-based features fuertes
- Nuestro análisis de propagación necesita items conocidos para simular difusión en el grafo social

#### Diagrama del Problema

```
INCORRECTO (Split por items):
┌─────────────────────────────────────┐
│ Timeline de Items                   │
├─────────────────────────────────────┤
│ Train Items: [I1, I2, ..., I1110]  │ ≤ Marzo 2015
│ Val Items:   [I1111, ..., I1249]   │ Marzo-Junio 2015
│ Test Items:  [I1250, ..., I1386]   │ > Junio 2015
└─────────────────────────────────────┘
      ↓
  Overlap: 0 items ❌

CORRECTO (Split por interacciones):
┌─────────────────────────────────────┐
│ Timeline de Interacciones           │
├─────────────────────────────────────┤
│ Item I1: [u1→I1, u2→I1, u3→I1, ...│
│           train   val   test       │
│                                     │
│ Item I2: [u4→I2, u5→I2, u6→I2, ...│
│           train   train  val       │
└─────────────────────────────────────┘
      ↓
  Overlap: TODOS los items aparecen
  en train, algunos también en val/test ✅
```

---

### ✅ Solución Final: Grafos Temporales Correctos

**Implementación en `GNN_Temporal_Final.ipynb`**

El notebook final implementa la solución correcta.

#### Cambio Clave: Split por Interacciones Temporales

**Script:** `scripts/build_temporal_graphs.py`

```python
def create_temporal_split(df, train_ratio=0.8, val_ratio=0.1):
    """
    Split temporal sobre INTERACCIONES, no sobre items.
    """
    # Ordenar TODAS las interacciones por timestamp
    df_sorted = df.sort_values('child_datetime')
    
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Dividir por índice de interacción
    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]
    
    return train_df, val_df, test_df
```

**Resultado:**
```
Total interacciones: 110,732
Split:
  Train: 88,620 (80%)
  Val: 11,534 (10.4%)
  Test: 10,578 (9.6%)

Items únicos:
  Train: 1,110
  Val: 872
  Test: 865
  Overlap train-val: 742 (85%)
  Overlap train-test: 721 (83%)
```

✅ **Ahora val y test comparten items con train!**

#### Ventajas del Approach Correcto

**1. Setup Transductivo Estándar**

- Items aparecen en train, luego se observan nuevas interacciones en val/test
- Permite medir capacidad de generalización a nuevos usuarios/contextos
- Es el setup usado en papers de GNN-RecSys (LightGCN, NGCF, etc.)

**2. Evaluación de Propagación Válida**

- Items en test tienen embeddings aprendidos (no random)
- Se puede simular propagación en el grafo social conocido
- Análisis de desinformación es interpretable

**3. Respeta Temporalidad**

- Train solo ve interacciones hasta T1
- Val ve interacciones entre T1 y T2
- Test ve interacciones después de T2
- **Leakage temporal:** ❌ No hay (train nunca ve futuro)

---

### Componentes Adicionales de la Solución

#### 1. Negative Sampling Temporal

**Script:** `scripts/negative_sampling.py`

- 382,391 negative samples generados
- 34,618 usuarios con negatives
- Promedio: 11.05 samples/usuario
- Ventana temporal respetada (`first_activity` → `last_activity`)
- Balance 50/50 entre items populares y no populares

#### 2. Arquitectura con 3 Capas GNN

**Cambio:** De 2 capas (Semanas 2-4) a 3 capas (Final)

```python
class GCNRecommender(nn.Module):
    def __init__(self, ...):
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)       # Nueva capa
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
```

**Justificación:**
- Mayor capacidad de agregación de vecindario
- Captura relaciones a 3-hops en el grafo
- Papers recientes usan 3-4 capas como estándar

#### 3. Construcción Correcta de Grafos

**Estadísticas finales:**
- **Bipartito:** 39,958 users × 1,386 items = 177,240 edges
- **Social:** 39,958 nodos, 311,020 edges (densidad: ~0.02%)
- **Threshold grafo social:** ≥3 items compartidos

---

### Comparación: Baseline vs. Final

| Métrica | Semana2-4 (Baseline) | GNN_Temporal_Final |
|---------|---------------------|---------------------|
| **Dataset** | processed_h1/ | processed_round2/ |
| **Usuarios** | 4,856 | 39,958 |
| **Items** | 2,308 | 1,386 |
| **Split** | Leave-one-out | Temporal 80/10/10 |
| **Capas GNN** | 2 | 3 |
| **Negative Sampling** | Random | Temporal con ventana |
| **Test set válido** | ✅ 4,856 | ✅ 10,578 |
| **MRR evaluable** | ✅ Sí | ✅ Sí |
| **Propagación analizable** | ✅ Sí | ✅ Sí |

---

### Lecciones Aprendidas

#### 1. Split Temporal Correcto ≠ Split por Items

El error crítico fue confundir:
- **Split temporal de items:** Dividir el catálogo de items por fecha de creación
- **Split temporal de interacciones:** Dividir las interacciones usuario-item por timestamp

El segundo es el correcto para RecSys.

#### 2. Cold-Start Real vs. Transductivo

- **Cold-start real** (items nuevos): Requiere content-based features fuertes (BERT, imágenes, metadata)
- **Transductivo** (items conocidos): Permite evaluar calidad de embeddings aprendidos

Para análisis de propagación social, el setup transductivo es más apropiado.

#### 3. Validación Temprana es Crítica

El problema del split incorrecto se detectó tarde (después de entrenar modelos).

**Checklist implementado:**
```python
# Validar SIEMPRE después de crear splits
train_items = set(train_df['item_idx'].unique())
test_items = set(test_df['item_idx'].unique())
overlap = train_items & test_items

print(f"Overlap: {len(overlap)} / {len(test_items)} ({len(overlap)/len(test_items)*100:.1f}%)")
assert len(overlap) > 0.5 * len(test_items), "❌ Más del 50% de items en test son cold-start!"
```

#### 4. Documentar Decisiones de Diseño

Mantener un registro de por qué se toman decisiones (como el cutoff en Marzo 2015) evita confusiones futuras y facilita la reproducibilidad.

---

### Referencias Técnicas

- **LightGCN Paper:** He et al., 2020. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
  - Usa split temporal transductivo
  - 3-4 capas GNN recomendadas

- **NGCF Paper:** Wang et al., 2019. "Neural Graph Collaborative Filtering"
  - Setup transductivo estándar
  - Negative sampling con balance popular/unpopular

- **BPR Loss:** Rendle et al., 2009. "BPR: Bayesian Personalized Ranking from Implicit Feedback"
  - Negative sampling fundamental para implicit feedback

- **Twitter Dataset:** Ma et al., 2017. "Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning"
  - Estructura de cascadas de propagación
  - Necesidad de colapsar cascadas a items únicos
