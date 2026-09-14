# PRUEBA PROMPT 1: Enfoque Genérico

**Objetivo**: Demostrar qué sucede cuando describes un problema sin análisis previo.

## Archivos

- **`prompt_1.txt`** - Prompt genérico (copia a Gemini/ChatGPT)
- **`solution_prompt_1.py`** - Script Python para ejecutar localmente

## Flujo

1. Abre `prompt_1.txt`
2. Copia el contenido y pega en Gemini/ChatGPT
3. Copia el contenido del JSON desde `../datos/customer_churn_data.json`
4. Obtén la respuesta y guarda en `respuesta_prompt_1.txt`
5. O ejecuta: `python solution_prompt_1.py` para obtener AUC automáticamente

## Salida esperada

- `respuesta_prompt_1.txt` - Respuesta de la IA
- `auc_prompt_1.txt` - Solo el número AUC-ROC

## Notas

- Sin análisis de datos previo, la IA propone un modelo "genérico"
- AUC-ROC será la métrica de comparación con Prompt 2
