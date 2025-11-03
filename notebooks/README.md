# Notebooks - Modelos Clásicos de Recomendación

Esta carpeta contiene los notebooks de modelos baseline y clásicos del proyecto.

---

## `H1_RecSys.ipynb` - Modelos Clásicos (H1)

**Descripción:**
Implementación de sistemas de recomendación clásicos aplicados al dataset Twitter15/16.

### Modelos Implementados

1. **User-KNN** (User-based Collaborative Filtering)
   - Similaridad coseno entre usuarios
   - Recomienda items de usuarios similares
   - k=20 vecinos por defecto

2. **Item-KNN** (Item-based Collaborative Filtering)
   - Similaridad coseno entre items
   - Recomienda items similares a los consumidos
   - k=20 vecinos por defecto

3. **Most Popular**
   - Baseline simple: recomienda los items más populares
   - Sin personalización

4. **Random**
   - Baseline aleatorio
   - Útil para comparar con otros métodos

5. **TF-IDF + Content-Based**
   - Similaridad semántica basada en texto de tweets
   - TF-IDF vectorization con n-gramas
   - Recomienda items con contenido similar

### Métricas de Evaluación

**Precisión:**
- Precision@K
- Recall@K
- MRR (Mean Reciprocal Rank)

**Diversidad:**
- ILD (Intra-List Diversity)
- Coverage del catálogo

**Fairness/Bias:**
- Exposure por label (true, false, unverified, non-rumor)
- Distribución de labels en recomendaciones
- Comparación con distribución base del dataset

### Análisis de Desinformación

El notebook incluye análisis de:
- ¿Qué modelos amplifican más fake news?
- Distribución de labels (TR, FR, UR, NR) por modelo
- Usuarios expuestos a desinformación en top-10
- Comparación de bias entre métodos colaborativos y content-based

---

## Datos Usados

**Dataset:** `../data_processing/processed_h1/`
- **Usuarios:** 4,856
- **Items:** 2,308
- **Interacciones:** 63,850
- **Split:** Leave-one-out (~92% train, ~8% test)
- **Labels balanceadas:** ~25% por clase (TR, FR, UR, NR)

**⚠️ Nota:** Este notebook usa el dataset SIN filtro temporal. Para la versión con split temporal, ver [`../midterm/GNN_Temporal_Final.ipynb`](../midterm/GNN_Temporal_Final.ipynb).

---

## Resultados Esperados

Los modelos clásicos sirven como **baseline** para comparar con los modelos basados en GNNs.

**Típicamente:**
- User-KNN: MRR ~0.12-0.15 (mejor modelo clásico)
- Item-KNN: Similar a User-KNN
- Most Popular: MRR ~0.007 (bajo, pero alta coverage de populares)
- Random: MRR ~0.001 (baseline mínimo)
- TF-IDF: Variable según calidad del texto

**Hallazgo clave:** Los modelos colaborativos tienden a amplificar contenido viral, lo que incluye tanto fake news como contenido verificado.

---

## Uso

```bash
# 1. Asegurarse de que los datos estén procesados
cd ../data_processing
python preprocess_unified.py  # Si no está hecho ya

# 2. Ejecutar notebook
cd ../notebooks
jupyter notebook H1_RecSys.ipynb
```

---

## Dependencias

- `pandas`, `numpy`
- `scikit-learn` (para KNN, TF-IDF, métricas)
- `matplotlib`, `seaborn` (visualizaciones)
- `scipy` (matrices sparse)

Instalar con:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
```

---

## Comparación con GNNs

| Aspecto | Modelos Clásicos (H1) | GNNs (Midterm) |
|---------|----------------------|----------------|
| **Precisión** | MRR ~0.12-0.15 | MRR ~0.05-0.06 |
| **Complejidad** | Baja (sklearn) | Alta (PyTorch Geometric) |
| **Escalabilidad** | Limitada (KNN) | Mejor (mini-batches) |
| **Features** | Matriz user-item o TF-IDF | Embeddings BERT + estructura grafo |
| **Propagación** | N/A | Simulación con LTM |

**Trade-off:** Los modelos clásicos tienen mejor precisión en este dataset pequeño, pero no capturan la estructura de red social necesaria para analizar propagación de desinformación.

---

## Integración en el Proyecto

Este notebook representa la **Fase H1** del proyecto (modelos baseline). Los resultados se comparan con:
- **Midterm:** GNNs con grafos bipartito y social ([`../midterm/`](../midterm/))
- **Versión Final:** GNNs con grafos temporales ([`../midterm/GNN_Temporal_Final.ipynb`](../midterm/GNN_Temporal_Final.ipynb))

Ver [`../README.md`](../README.md) para el roadmap completo del proyecto.
