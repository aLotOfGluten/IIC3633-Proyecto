# Preprocesamiento Twitter15 + Twitter16

Pipeline de preprocesamiento para datasets Twitter15 y Twitter16, generando estructuras optimizadas para sistemas de recomendación y GNNs.

## Contenido

```
.
├── README.md                      # Este archivo
├── preprocess_unified.py          # Script de preprocesamiento 
├── analyze_h1.py                  # Script de análisis exploratorio (EDA)
├── twitter15/                     # Dataset original Twitter15
│   ├── label.txt
│   └── tree/
├── twitter16/                     # Dataset original Twitter16
│   ├── label.txt
│   └── tree/
├── content/                       # Contenido de tweets
│   ├── twitter15/
│   │   ├── label.txt
│   │   └── source_tweets.txt
│   └── twitter16/
│       ├── label.txt
│       └── source_tweets.txt
├── processed_h1/                  # Datos procesados (generado)
│   ├── train_interactions.csv
│   ├── test_interactions.csv
│   ├── train_interactions_idx.csv
│   ├── test_interactions_idx.csv
│   ├── user_map.csv
│   ├── item_map.csv
│   ├── item_labels.csv
│   ├── item_text_clean.csv
│   └── stats_summary.txt
└── plots_and_reports/             # Análisis y visualizaciones (generado)
    ├── usuarios_hist.png
    ├── items_hist.png
    ├── balance_labels.png
    ├── cascadas_tamano_hist.png
    ├── cascadas_profundidad_hist.png
    ├── viralidad_fake_vs_real.png
    ├── resumen_stats.csv
    ├── top_usuarios.csv
    └── top_items.csv
```

## Uso Rápido

### 1. Preprocesamiento

```bash
python3 preprocess_unified.py
```

Esto generará automáticamente todos los archivos en `processed_h1/`.

### 2. Análisis Exploratorio (EDA)

```bash
python3 analyze_h1.py
```

Genera visualizaciones y estadísticas descriptivas en `plots_and_reports/`.

## Configuración

Editar la clase `Config` en `preprocess_unified.py` (líneas 15-23):

```python
class Config:
    DATASETS = ["twitter15", "twitter16"]
    OUTPUT_DIR = "processed_h1"
    CONTENT_DIR = "content"
    MIN_INTERACTIONS = 8          # Mínimo de interacciones por usuario
    RANDOM_STATE = 42
    LABEL_STRATEGY = '4class'     # Estrategia de etiquetado
```

### Estrategias de Etiquetado

| Estrategia | Descripción | Clases |
|-----------|-------------|--------|
| `4class` | 4 categorías originales (default) | 0=true, 1=false, 2=unverified, 3=non-rumor |
| `3class` | Agrupa ambiguos | 0=true, 1=false, 2=ambiguous |
| `binary_strict` | Solo true vs false | 0=false, 1=true |
| `binary_all` | True vs resto | 0=fake/otros, 1=true |

## Archivos Generados

### Interacciones
- `train_interactions.csv` - Entrenamiento con IDs originales
- `test_interactions.csv` - Test con IDs originales
- `train_interactions_idx.csv` - Entrenamiento con índices numéricos
- `test_interactions_idx.csv` - Test con índices numéricos

**Formato:** `user_id, item_id, interaction` (o `user_idx, item_idx, interaction`)

### Mapeos
- `user_map.csv` - Mapeo `user_id → user_idx` (0 a N)
- `item_map.csv` - Mapeo `item_id → item_idx` (0 a M)

### Etiquetas y Texto
- `item_labels.csv` - `item_id, item_idx, label, binary_label, multiclass_label, dataset`
- `item_text_clean.csv` - `item_id, item_idx, text_clean` (texto procesado de tweets)

### Estadísticas
- `stats_summary.txt` - Métricas completas del procesamiento

## Análisis Exploratorio de Datos (EDA)

El script `analyze_h1.py` genera un análisis completo del dataset procesado, enfocado en **viralidad** y **desinformación**.

### Ejecución

```bash
python3 analyze_h1.py
```

**Requisitos:** `pandas`, `numpy`, `matplotlib`, `scipy` (opcional para test estadístico)

### Análisis Generados

#### 📊 Gráficos (`plots_and_reports/`)

1. **`usuarios_hist.png`** - Distribución de interacciones por usuario
   - Muestra mediana y filtro mínimo (8 interacciones)
   - Justifica decisiones de preprocesamiento

2. **`items_hist.png`** - Distribución de interacciones por item (long-tail)
   - Evidencia la distribución long-tail típica de redes sociales
   - Muestra items altamente virales vs. poco difundidos

3. **`balance_labels.png`** - Balance de clases multiclase
   - 4 categorías: `false`, `true`, `unverified`, `non-rumor`
   - Porcentajes y conteos absolutos
   - Confirma dataset balanceado (~25% cada clase)

4. **`cascadas_tamano_hist.png`** - Distribución de tamaño de cascadas
   - Histograma mostrando número de usuarios por cascada
   - Líneas de promedio y mediana para referencia
   - Visualiza la distribución de alcance de difusión

5. **`cascadas_profundidad_hist.png`** - Distribución de profundidad de cascadas
   - Histograma mostrando niveles de propagación
   - Líneas de promedio y mediana para referencia
   - Muestra cuán profundas son las cascadas de difusión

6. **`viralidad_fake_vs_real.png`** - Viralidad comparativa
   - Gráfico de barras con barras de error (desviación estándar)
   - Compara promedio de interacciones: Noticias Falsas vs Verdaderas
   - **Incluye test estadístico:** Mann-Whitney U test
   - Muestra si las noticias falsas son significativamente más virales

#### 📋 Tablas CSV (`plots_and_reports/`)

1. **`resumen_stats.csv`** - Estadísticas globales del dataset
   - Usuarios, items, interacciones, densidad
   - Distribución multiclase (n y %)

2. **`top_items.csv`** - Top 10 items más virales
   - Identificadores, número de interacciones
   - Tipo (Fake/Real), categoría a anlizar

3. **`top_usuarios.csv`** - Top 10 usuarios más activos
   - Identificadores, número de interacciones

### Hallazgos Principales

Con el dataset procesado (configuración por defecto):

| Métrica | Valor |
|---------|-------|
| **Viralidad Fake** | 32.99 interacciones promedio |
| **Viralidad Real** | 18.07 interacciones promedio |
| **Ratio Fake/Real** | 1.8x más viral |
| **Significancia estadística** | p < 0.001 (altamente significativo) |
| **Top 10 items virales** | 100% Fake |
| **Tamaño cascada promedio** | 411.3 usuarios |
| **Profundidad cascada promedio** | 4.7 niveles |

**Conclusión:** Las noticias falsas son significativamente más virales que las verdaderas en este dataset.

## Pipeline de Procesamiento

El script está estructurado en funciones modulares con type hints completos:

1. **`load_data()`** - Carga labels y contenido de tweets desde ambos datasets
   - Lee `label.txt` para etiquetas de veracidad
   - Lee `source_tweets.txt` y aplica limpieza de texto (URLs → `URL`, mentions → `@USER`, etc.)

2. **`extract_interactions()`** - Extrae interacciones user-item desde árboles de propagación
   - Usuario participa en árbol → `interaction=1`
   - Construye matriz sparse eficientemente usando listas de columnas

3. **`filter_users()`** - Elimina usuarios con < `MIN_INTERACTIONS` (default: 8)

4. **`split_train_test()`** - División train/test usando leave-one-out
   - Última interacción de cada usuario → test (~92% train, ~8% test)
   - Usa `groupby().idxmax()` para máxima eficiencia

5. **`create_mappings()`** - Genera índices numéricos para usuarios e items

6. **`save_interactions()`** - Guarda interacciones con IDs e índices

7. **`save_labels_and_text()`** - Guarda labels (binarias y multiclase) + texto limpio

8. **`print_final_stats()`** - Imprime estadísticas finales del dataset

## Estadísticas del Dataset

Con configuración por defecto:

| Métrica | Valor |
|---------|-------|
| Usuarios | 4,856 |
| Items | 2,178 |
| Interacciones | 63,850 |
| Densidad | 0.60% |
| Interacciones/usuario | 13.15 |
| Interacciones/item | 29.32 |
| Tweets con texto | 2,178 (100%) |

### Distribución de Clases (4-class)

| Clase | Label | Items | % |
|-------|-------|-------|---|
| 0 | true | 536 | 24.6% |
| 1 | false | 511 | 23.5% |
| 2 | unverified | 555 | 25.5% |
| 3 | non-rumor | 576 | 26.4% |

Balance: ~25% por clase ✓

### Contribución
- Twitter15: 64.2% (1,398 items)
- Twitter16: 35.8% (780 items)

## Guía de Uso para Sistemas de Recomendación

### Archivos Necesarios

**Para modelos base (UKNN, Most Popular, Random):**
- `train_interactions_idx.csv` - Matriz user-item (4856 × 2178)
- `test_interactions_idx.csv` - Evaluación
- `user_map.csv`, `item_map.csv` - Mapeos de índices
- `item_labels.csv` - Análisis de bias por clase

### 1. Implementación de Modelos

#### UKNN (User-based KNN)
**Proceso:**
1. Construir matriz sparse user-item con las interacciones de entrenamiento
2. Entrenar KNN con similaridad coseno entre usuarios
3. Para recomendar: encontrar k vecinos más similares y agregar sus items
4. Excluir items ya interactuados por el usuario
5. Sugerencia: k=20 vecinos como punto de partida

#### Most Popular
**Proceso:**
1. Contar interacciones por item en el conjunto de entrenamiento
2. Ordenar items por popularidad (más interacciones primero)
3. Para recomendar: devolver los top-K items más populares
4. Excluir items ya interactuados por el usuario

#### Random
**Proceso:**
1. Obtener lista de todos los items disponibles
2. Para recomendar: seleccionar K items aleatoriamente
3. Excluir items ya interactuados por el usuario
4. Sirve como baseline para comparar con otros métodos

---

### **EXTRA (Solo si hay tiempo - Ideal para sacarnos una buena nota)**

#### IKNN (Item-based KNN)
**Proceso:**
1. Construir matriz sparse user-item con las interacciones de entrenamiento
2. Calcular similaridad coseno entre **items** (transponer matriz user-item)
3. Para recomendar a un usuario:
   - Identificar items con los que el usuario ya interactuó
   - Para cada item del catálogo: calcular score como suma ponderada de similaridades con items del usuario
   - Ordenar por score y devolver top-K
4. Sugerencia: k=20 vecinos como punto de partida

**Diferencia clave con UKNN:**
- UKNN: "recomendar items que usuarios similares a ti consumieron"
- IKNN: "recomendar items similares a los que ya consumiste"

#### TF-IDF + Similaridad de Contenido
**Archivos necesarios:**
- `item_text_clean.csv` - Texto limpio de tweets
- `train_interactions_idx.csv` - Para filtrar items ya vistos

**Proceso:**
1. Cargar texto de tweets desde `item_text_clean.csv`
2. Construir matriz TF-IDF usando `sklearn.TfidfVectorizer`
   - Parámetros sugeridos: `max_features=5000`, `ngram_range=(1,2)`, `min_df=2`
3. Calcular similaridad coseno entre todos los pares de items
4. Para recomendar a un usuario:
   - Identificar items con los que interactuó en train
   - Para cada item del usuario: obtener los K items más similares por contenido
   - Agregar scores y ordenar
   - Excluir items ya interactuados
5. Devolver top-K recomendaciones

**Análisis recomendado:**
- ¿Items de la misma clase (true/false) tienen mayor similaridad textual?
- ¿TF-IDF amplifica o reduce bias comparado con métodos colaborativos?

---

### 2. Análisis de Recomendaciones según Labels

**Métricas de análisis recomendadas:**

| Métrica | Objetivo |
|---------|----------|
| **Bias de labels** | ¿El sistema recomienda más un tipo de rumor que otro? |
| **Precision@K por clase** | Precisión separada para cada label (true, false, etc.) |
| **Exposure por label** | ¿Qué clases de rumores tienen más visibilidad? |
| **Fairness** | Comparar distribución en recomendaciones vs dataset base (~25% por clase) |
| **Coverage por clase** | ¿Qué porcentaje de items de cada label son recomendados? |

**Consideraciones importantes:**
- El dataset está balanceado (~25% por clase), usar como baseline de comparación
- Evaluar si el sistema amplifica ciertos tipos de rumores (ej. más `false` que `true`)
- Comparar los 3 modelos: ¿UKNN amplifica más bias que Random o Most Popular?
- Analizar si usuarios que interactúan con `true` reciben recomendaciones sesgadas
- Considerar el impacto social de recomendar más rumores falsos vs verificados

## Referencias

**Twitter15:**
```bibtex
@inproceedings{liu2015real,
  title={Real-time Rumor Debunking on Twitter},
  author={Liu, Xiaomo and Nourbakhsh, Armineh and Li, Quanzhi and Fang, Rui and Shah, Sameena},
  booktitle={CIKM},
  pages={1867--1870},
  year={2015}
}
```

**Twitter16:**
```bibtex
@inproceedings{ma2016detecting,
  title={Detecting Rumors from Microblogs with Recurrent Neural Networks},
  author={Ma, Jing and Gao, Wei and Mitra, Prasenjit and Kwon, Sejeong and Jansen, Bernard J. and Wong, Kam-Fai and Meeyoung, Cha},
  booktitle={IJCAI},
  year={2016}
}
```

**Dataset:**
```bibtex
@inproceedings{ma2017detect,
  title={Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning},
  author={Ma, Jing and Gao, Wei and Wong, Kam-Fai},
  booktitle={ACL},
  volume={1},
  pages={708--717},
  year={2017}
}
```

---

## Fuentes del Dataset

**Interacciones y estructura de propagación:**
- [gszswork/Twitter15_16_dataset](https://github.com/gszswork/Twitter15_16_dataset)

**Contenido de tweets (source_tweets.txt):**
- [Rumor Detection ACL 2017 - Kaggle](https://www.kaggle.com/datasets/syntheticprogrammer/rumor-detection-acl-2017/data)
