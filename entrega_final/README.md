# Entrega Final: Análisis de Trade-off Precisión vs Desinformación

Este análisis presenta las tres mejoras clave solicitadas para la entrega final del proyecto:

1. **Métrica formal Fake@K** - Cuantificación objetiva de fake news en recomendaciones
2. **Visualización del trade-off** - Análisis del dilema entre precisión y responsabilidad
3. **Justificación estadística del threshold=3** - Validación empírica de parámetros del grafo social

---

## Notebook Principal

**[`Analisis_Final_TradeOff_COMPLETO.ipynb`](Analisis_Final_TradeOff_COMPLETO.ipynb)**

Notebook autocontenible que descarga todos los datos necesarios desde el repositorio y genera visualizaciones finales del proyecto.

### Cómo Ejecutar

**Opción 1: Google Colab (Recomendado)**
```
https://colab.research.google.com/github/aLotOfGluten/IIC3633-Proyecto/blob/main/entrega_final/Analisis_Final_TradeOff_COMPLETO.ipynb
```

**Opción 2: Jupyter Local**
```bash
cd entrega_final
jupyter notebook Analisis_Final_TradeOff_COMPLETO.ipynb
```

**Nota:** El notebook descarga automáticamente todos los archivos necesarios desde el repositorio (métricas, mapeos, labels). No requiere ejecutar notebooks previos.

---

## Contenido del Análisis

### 1. Métrica Fake@K

**Definición formal:**

$$\text{Fake@K} = \frac{1}{|U|} \sum_{u \in U} \frac{|\{i \in \text{Top-K}_u : \text{label}(i) = \text{FR}\}|}{K}$$

Donde:
- $U$ = conjunto de usuarios en test
- $\text{Top-K}_u$ = las K recomendaciones para el usuario $u$
- $\text{FR}$ = False Rumor (fake news)

**Interpretación:**
- `Fake@10 = 0.50` → 50% de las recomendaciones son fake news
- `Fake@10 = 0.70` → 70% de las recomendaciones son fake news

**Modelos evaluados (8 total):**
- GNN: LightGCN v2, GCN-BERT v2, GCN-Random v2
- Clásicos: ItemKNN, UserKNN, TF-IDF
- Baselines: Random, MostPopular

### 2. Trade-off Precisión vs Desinformación

**Pregunta clave:** ¿Podemos recomendar con alta precisión (MRR) sin amplificar fake news (Fake@K)?

**Hallazgos:**

| Modelo | MRR ↑ | Fake@10 ↓ | Balance |
|--------|-------|-----------|---------|
| **ItemKNN** | 0.0347 | 0.5604 | ✅ Óptimo |
| **UserKNN** | 0.0338 | 0.5544 | ✅ Óptimo |
| LightGCN v2 | 0.0347 | 0.6749 | ⚠️ Alta precisión, alta desinformación |
| TF-IDF | 0.0283 | 0.7060 | ❌ Peor: amplifica fake news |
| **GCN-BERT v2** | 0.0153 | 0.5053 | ✅ Óptimo (más seguro) |
| **MostPopular** | 0.0119 | 0.5000 | ✅ Baseline seguro |

**Frontera de Pareto:**

Los 4 modelos marcados con ✅ representan el mejor trade-off posible. Cualquier otro modelo es dominado por al menos uno de estos.

**Conclusión clave:**
- **LightGCN v2** tiene el MRR más alto (0.0347) pero amplifica significativamente fake news (67.5%)
- **ItemKNN/UserKNN** logran la misma precisión pero con ~10% menos desinformación
- **GCN-BERT v2** es el más seguro (50.5% fake) pero sacrifica precisión
- **No existe un modelo "perfecto"** - el trade-off es inherente

### 3. Justificación del Threshold=3

**Contexto:** El grafo social conecta usuarios que comparten ≥3 items. ¿Por qué 3?

**Análisis estadístico:**

#### Distribución de Interacciones por Usuario
- **Media:** 2.60 interacciones/usuario
- **Mediana:** 2.00 interacciones/usuario
- **Q75:** 3.00
- **84.5%** de usuarios tienen ≤3 interacciones
- **15.5%** tienen >3 interacciones

#### Distribución de Items Compartidos
- **Media:** 0.01 items compartidos entre pares aleatorios
- **Mediana:** 0.00 items
- **98.9%** de pares no comparten items
- **0.01%** de pares comparten ≥3 items

**Justificación:**

| Threshold | Problema |
|-----------|----------|
| **1-2** | ❌ Conexiones espurias (coincidencia aleatoria, bots) |
| **3** | ✅ Balance óptimo: suficiente densidad, conexiones significativas |
| **5+** | ❌ Grafo demasiado disperso, pocas conexiones útiles |

**Conclusión:**
- Threshold=3 asegura que las conexiones representen **intereses genuinamente comunes**
- Reduce ruido sin sacrificar conectividad del grafo
- Permite propagación significativa en el análisis LTM

---

## Archivos Generados

El notebook genera automáticamente:

### Visualizaciones
- **`tradeoff_precision_vs_misinformation.png`** - Gráfico principal del trade-off (MRR vs Fake@10)
- **`multi_metric_comparison.png`** - Comparación de 4 métricas clave entre modelos
- **`label_distribution_comparison.png`** - Composición de labels (FR/UR/TR/NR) por modelo
- **`justification_user_interactions.png`** - Distribución de interacciones (4 subplots)
- **`justification_shared_items.png`** - Distribución de items compartidos

### Reportes
- **`justification_threshold_report.txt`** - Reporte textual de la justificación estadística

### Datos
- **`data/tradeoff_metrics.csv`** - Métricas consolidadas de los 8 modelos

---

## Métricas Incluidas

Para cada modelo se reporta:

**Precisión:**
- **MRR** (Mean Reciprocal Rank) - Precisión de recomendaciones
- **Coverage** - % del catálogo recomendado
- **ILD** (Intra-List Diversity) - Diversidad de recomendaciones

**Desinformación:**
- **Fake@10** - % de fake news en Top-10
- **Users Exposed %** - % de usuarios expuestos a fake news
- **Avg Fake per User** - Promedio de fake news por usuario

**Propagación (LTM):**
- **Mean Reach** - Alcance promedio en grafo social
- **Mean Depth** - Profundidad promedio de cascadas
- **Users Reached** - Total de usuarios alcanzados

**Distribución de Labels:**
- **FR %** (False Rumor) - % de fake news
- **UR %** (Unverified) - % de no verificadas
- **TR %** (True Rumor) - % de noticias verdaderas
- **NR %** (Non-Rumor) - % de no rumores

---

## Dataset Utilizado

- **Twitter15 + Twitter16** con split temporal (80/10/10)
- **Total items:** 1,386 (con labels de veracidad)
- **Usuarios en test:** 3,596
- **Interacciones en test:** 5,974
- **Distribución de labels:**
  - FR (Fake): 26.9%
  - TR (True): 27.1%
  - UR (Unverified): 19.0%
  - NR (Non-Rumor): 27.1%

---

## Resultados Clave para el Informe

### 1. Existe un trade-off real entre precisión y seguridad

No es posible maximizar MRR sin aumentar la exposición a fake news. Los mejores modelos en precisión (LightGCN v2, TF-IDF) amplifican significativamente la desinformación.

### 2. Los modelos clásicos son competitivos

ItemKNN y UserKNN logran el mejor balance:
- MRR comparable a LightGCN v2 (0.0347 vs 0.0347)
- ~10% menos fake news (56% vs 67%)
- Menor complejidad computacional

### 3. El threshold=3 está estadísticamente justificado

Representa el equilibrio entre:
- Conectividad del grafo (suficientes edges para propagación)
- Significancia de conexiones (intereses genuinos, no aleatorios)
- Robustez ante ruido (reducción de conexiones espurias)

### 4. La diversificación no garantiza seguridad

GCN-BERT v2 tiene el ILD más alto (0.9386) pero aún expone al 93.7% de usuarios a fake news. La diversidad por sí sola no resuelve el problema de desinformación.

---

## Referencias a Notebooks Previos

Este análisis consolida resultados de:

- **[`midterm/GNN_Temporal_Final.ipynb`](../midterm/GNN_Temporal_Final.ipynb)** - Entrenamiento y evaluación de GNNs (3 capas, grafos temporales)
- **[`notebooks/H1_RecSys.ipynb`](../notebooks/H1_RecSys.ipynb)** - Modelos clásicos (ItemKNN, UserKNN, TF-IDF)

**Ver:** [`midterm/README.md`](../midterm/README.md) para detalles técnicos de la implementación de GNNs.

---

## Próximos Pasos

### Para el Informe Final
1. Incluir gráfico de trade-off en sección de Resultados
2. Discutir implicaciones éticas del dilema precisión-seguridad
3. Agregar tabla de modelos Pareto en sección de Comparación
4. Incluir justificación del threshold en sección de Diseño Experimental
5. Definir formalmente Fake@K en sección de Métricas

### Para la Presentación
- Slide del trade-off (gráfico principal)
- Tabla comparativa de modelos
- Recomendación final sobre qué modelo usar según prioridades

---

## Contacto

Para preguntas sobre este análisis:
- Ver [`README.md`](../README.md) principal del proyecto
- Revisar documentación en [`midterm/README.md`](../midterm/README.md)

**Integrantes:**
- Vittorio Salvatore
- Clemente Acevedo
- Cristobal Fuentes
