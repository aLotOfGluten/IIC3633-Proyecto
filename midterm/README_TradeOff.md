# Análisis Final: Trade-off Precisión vs Desinformación

## Qué es esto

Notebook único con todo el código necesario para las mejoras de la entrega final:

✅ **Métrica formal Fake@K** - Cuantifica exposición a desinformación
✅ **Gráfico de trade-off** - MRR vs Fake@10 con frontera de Pareto
✅ **Justificación estadística** - Del threshold de 3 ítems compartidos

## Cómo usar

1. Abre `Analisis_Final_TradeOff.ipynb`
2. Ajusta las secciones de carga de datos (marcadas con comentarios)
3. Ejecuta todas las celdas
4. Los gráficos se guardan automáticamente como PNG

## Outputs generados

- `tradeoff_precision_vs_misinformation.png` - Gráfico principal del trade-off
- `multi_metric_comparison.png` - Comparación de todas las métricas
- `fake_at_k_sensitivity.png` - Sensibilidad por K
- `justification_user_interactions.png` - Justificación del threshold (parte 1)
- `justification_shared_items.png` - Justificación del threshold (parte 2)
- `justification_threshold_report.txt` - Reporte textual

## Qué incluir en el informe

**Sección "Métricas":** Definición de Fake@K (está en el notebook)
**Sección "Resultados":** Gráfico principal + tabla de modelos Pareto
**Sección "Diseño":** Gráficos de justificación del threshold
**Sección "Análisis":** Discusión del modelo recomendado según balance

---

Todo el código está inline en el notebook - no hay dependencias externas más allá de numpy, pandas, matplotlib y seaborn.
