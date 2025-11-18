# Entrega Final - IIC3633 Proyecto

## Contenido

Este directorio contiene el análisis final del proyecto de sistemas recomendadores con evaluación de desinformación.

### Archivos

- **Analisis_Final_TradeOff_COMPLETO.ipynb**: Notebook principal con análisis completo del trade-off entre precisión y exposición a desinformación

## Cómo ejecutar el notebook en Google Colab

### Opción 1: Subir el notebook manualmente

1. Ve a [Google Colab](https://colab.research.google.com/)
2. Selecciona "Subir" y elige el archivo `Analisis_Final_TradeOff_COMPLETO.ipynb`
3. Sigue las instrucciones dentro del notebook

### Opción 2: Abrir desde GitHub (recomendado)

1. Sube tu repositorio a GitHub (si aún no lo has hecho)
2. Ve a [Google Colab](https://colab.research.google.com/)
3. Selecciona "GitHub" en el menú de apertura
4. Ingresa tu usuario/repositorio o la URL directa
5. Abre `entrega_final/Analisis_Final_TradeOff_COMPLETO.ipynb`

## Configuración requerida

Antes de ejecutar el notebook, debes configurar las siguientes variables en la segunda celda:

```python
GITHUB_USERNAME = "TU_USUARIO"      # Tu usuario de GitHub
GITHUB_REPO = "IIC3633-Proyecto"    # Nombre de tu repositorio
BRANCH = "main"                      # Rama (main o master)
```

## Archivos de datos necesarios

El notebook descargará automáticamente los siguientes archivos desde tu repositorio de GitHub:

```
midterm/graphs_per_user_temporal/
├── test_interactions.csv      # Interacciones de test
├── train_interactions.csv     # Interacciones de entrenamiento
├── user_map.csv              # Mapeo de usuarios
└── item_map.csv              # Mapeo de ítems

datasets/new_datasets/
├── twitter15/label.txt       # Etiquetas de veracidad Twitter15
└── twitter16/label.txt       # Etiquetas de veracidad Twitter16
```

**Importante**: Estos archivos deben estar en tu repositorio de GitHub en las rutas especificadas.

## Outputs generados

El notebook genera los siguientes archivos de salida:

### Gráficos
- `tradeoff_precision_vs_misinformation.png` - Gráfico principal del trade-off
- `multi_metric_comparison.png` - Comparación de múltiples métricas
- `fake_at_k_sensitivity.png` - Sensibilidad de Fake@K según K
- `justification_user_interactions.png` - Justificación estadística parte 1
- `justification_shared_items.png` - Justificación estadística parte 2

### Reportes
- `justification_threshold_report.txt` - Reporte textual con justificación del threshold

## Contenido del análisis

El notebook implementa:

1. **Métrica formal Fake@K**: Cuantifica exposición a desinformación
2. **Visualización del trade-off**: MRR vs Fake@10 con frontera de Pareto
3. **Justificación estadística**: Análisis del threshold de 3 ítems compartidos
4. **Comparación de modelos**: 8 modelos evaluados (3 GNN + 5 baselines)

## Requisitos

El notebook instala automáticamente las librerías necesarias:
- numpy
- pandas
- matplotlib
- seaborn

## Troubleshooting

### Error: "No se pudo descargar archivo"
- Verifica que tu repositorio sea público
- Verifica que los archivos existan en las rutas correctas
- Verifica que GITHUB_USERNAME, GITHUB_REPO y BRANCH estén correctos

### Error: "ModuleNotFoundError"
- Ejecuta la celda de instalación de dependencias
- Reinicia el runtime de Colab si es necesario

## Contacto

Para problemas o preguntas, consulta el README principal del proyecto.
