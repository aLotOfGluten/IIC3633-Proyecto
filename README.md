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
```

## Inicio Rápido

```bash
cd data_processing
python3 preprocess_unified.py  # Generar datasets
python3 analyze_h1.py          # Análisis exploratorio
```

Ver [`data_processing/README.md`](data_processing/README.md) para documentación completa.