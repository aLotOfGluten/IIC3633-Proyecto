# Procesamiento Temporal de Datos - Twitter15/16

Pipeline de preprocesamiento con **split temporal** para garantizar evaluación realista de sistemas de recomendación.

**⚠️ Diferencia clave con `data_processing/`:** Esta carpeta implementa filtro temporal y split cronológico, mientras que `data_processing/` usa leave-one-out sin considerar tiempo.

---

## Notebooks

### 1. `data_processing_t15.ipynb` - Twitter15 con split temporal

**Pipeline:**
1. Extracción de timestamps desde Snowflake IDs
2. Filtro temporal: `≤ Marzo 2015`
3. Split temporal 80/10/10 (train/val/test) respetando orden cronológico
4. Cálculo de ventanas de actividad por usuario (`first_activity`, `last_activity`)

**Output:**
- `../data_processing/processed_round2/twitter15_processed.csv` - ~78.5k interacciones
- `../data_processing/processed_round2/twitter15_user_activity.csv` - Ventanas temporales

**Estadísticas (post-filtro):**
- **Usuarios:** ~29,400
- **Items/Trees:** ~750 (de 1,490 originales)
- **Fecha de corte:** Marzo 2015 (para garantizar presencia de FR en test)

### 2. `data_processing_t16.ipynb` - Twitter16 con split temporal

Mismo pipeline que t15, aplicado a Twitter16.

**Output:**
- `../data_processing/processed_round2/twitter16_processed.csv` - ~32k interacciones
- `../data_processing/processed_round2/twitter16_user_activity.csv`

### 3. `negative_samples.ipynb` - Análisis de negative sampling

Notebook exploratorio para analizar los negative samples generados por `../scripts/negative_sampling.py`.

---

## ¿Por qué split temporal?

### Problema con split random:
- **Item leakage:** Items del futuro aparecen en train
- **Cold-start artificial:** No refleja el problema real de recomendar contenido nuevo
- **Distribución no realista:** Labels balanceadas artificialmente

### Ventajas del split temporal:
- ✅ **Realismo:** Predecir el futuro basándose en el pasado
- ✅ **No leakage:** Train solo contiene datos hasta cierta fecha
- ✅ **Cold-start real:** Test contiene items que no estaban en train
- ✅ **Distribución natural:** Refleja la evolución temporal de las noticias

### Decisión de corte (Marzo 2015):
El filtro en Marzo 2015 garantiza que el test set contenga suficientes casos de **False Rumor (FR)** para análisis de desinformación. Datos de 2016+ fueron mayormente clasificados como "unverified", reduciendo su utilidad.

---

## Integración con el proyecto

**Estos datos procesados alimentan:**
1. **Scripts de construcción de grafos:** [`../scripts/build_temporal_graphs.py`](../scripts/build_temporal_graphs.py)
2. **Negative sampling:** [`../scripts/negative_sampling.py`](../scripts/negative_sampling.py)
3. **Notebook principal:** [`../midterm/GNN_Temporal_Final.ipynb`](../midterm/GNN_Temporal_Final.ipynb)

**Flujo completo:**
```
data_processing_temporal (notebooks)
  ↓ genera
processed_round2/ (CSVs)
  ↓ consume
scripts/build_temporal_graphs.py
  ↓ genera
midterm/graphs_temporal/ (grafos .pt)
  ↓ consume
midterm/GNN_Temporal_Final.ipynb
```

---

## Comparación de datasets

| Métrica | `data_processing/` (H1) | `data_processing_temporal/` (Final) |
|---------|-------------------------|-------------------------------------|
| **Filtro temporal** | ❌ No | ✅ Sí (≤ Marzo 2015) |
| **Split** | Leave-one-out | Temporal 80/10/10 |
| **Usuarios** | 4,856 | 39,958 |
| **Items** | 2,308 | 1,386 |
| **Interacciones** | 63,850 | 110,732 |
| **Datos 2016+** | Incluidos | Descartados |
| **Uso** | H1 + Midterm baseline | Versión final con grafos temporales |

---

## Referencias

- **Paper original (Twitter15):** Liu et al., 2015. Real-time Rumor Debunking on Twitter. CIKM.
- **Paper original (Twitter16):** Ma et al., 2016. Detecting Rumors from Microblogs with Recurrent Neural Networks. IJCAI.
- **Dataset combinado:** Ma et al., 2017. Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning. ACL.
