# PRUEBA PROMPT 2: Enfoque Especializado

**Objetivo**: Demostrar cómo el análisis previo (EDA) cambia la decisión del modelo.

## Archivos

- **`prompt_2_parte_1.txt`** - Prompt para análisis exploratorio (EDA)
- **`prompt_2_parte_2.txt`** - Prompt especializado (recibe output de Parte 1)
- **`solution_prompt_2.ipynb`** - Jupyter Notebook completo (ejecutable)

## Flujo de ejecución

### OPCIÓN A: Con Claude aquí (más rápido)

```bash
jupyter notebook solution_prompt_2.ipynb
```

Ejecuta todas las celdas. Genera AUC automáticamente.

### OPCIÓN B: Con otras IAs (más realista)

1. **Parte 1 - EDA**:
   - Copia `prompt_2_parte_1.txt` a Claude/Gemini/ChatGPT
   - Pega el JSON desde `../datos/customer_churn_data.json`
   - Obtén el reporte de análisis
   - Guarda en `eda_report.md`

2. **Parte 2 - Especializado**:
   - Copia `prompt_2_parte_2.txt`
   - Pega el reporte de EDA (`eda_report.md`)
   - Pega el JSON nuevamente
   - Obtén la solución completa
   - Guarda código en `solucion_especializada.py` y ejecuta

## Salida esperada

- `eda_report.md` - Reporte de análisis exploratorio
- `solucion_especializada.py` - Código Python con pipeline completo
- `auc_prompt_2.txt` - Solo el número AUC-ROC

## Comparación

Después de completar ambos prompts, ejecuta:

```python
# En solution_prompt_2.ipynb, última celda
# Compara automáticamente auc_prompt_1.txt vs auc_prompt_2.txt
```

## Notas

- Parte 1 explora: nulos, desbalance, outliers, correlaciones
- Parte 2 construye pipeline basado en esos hallazgos
- La diferencia de AUC es lo que queremos demostrar

## Contexto de Negocio

- Presupuesto: Top 10% de contactos (~100 clientes)
- Métrica: **AUC-ROC** (ranking > accuracy)
- Desbalance: ~24% churn
