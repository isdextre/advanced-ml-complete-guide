# Bloque 1: Del Problema a la Elección del Modelo

## 📊 Demostración: Importancia del Análisis Previo

Este bloque demuestra cómo **el análisis exploratorio de datos (EDA) cambia la selección del modelo**.

### 🎯 Objetivo

Mostrar la diferencia entre:
1. **Prompt 1 (Genérico)**: "Dame un modelo para predecir churn" ❌ Sin análisis
2. **Prompt 2 (Especializado)**: Análisis previo → decisión informada ✓ Con EDA

### 📁 Estructura

```
bloque-1-problema-y-seleccion/
├── datos/
│   └── customer_churn_data.json          # Dataset (1,000 clientes)
├── prueba_prompt_1/
│   ├── README.md                         # Instrucciones
│   ├── prompt_1.txt                      # Prompt genérico (copia-pega)
│   └── solution_prompt_1.py              # Script Python ejecutable
├── prueba_prompt_2/
│   ├── README.md                         # Instrucciones
│   ├── prompt_2_parte_1.txt              # EDA
│   ├── prompt_2_parte_2.txt              # Especializado
│   └── solution_prompt_2.ipynb           # Notebook Jupyter completo
├── .gitignore                            # No commitear outputs
└── slides/                               # Diapositivas del taller
```

### 🚀 Quick Start

#### Opción 1: Automático (Jupyter)
```bash
cd prueba_prompt_2
jupyter notebook solution_prompt_2.ipynb
```

#### Opción 2: Con otras IAs (Gemini/ChatGPT)

**Paso 1 - Prompt 1 (Genérico)**:
```bash
cat prueba_prompt_1/prompt_1.txt
# Copia el contenido → Gemini/ChatGPT
# Obtén AUC → guarda en prueba_prompt_1/auc_prompt_1.txt
```

**Paso 2 - Prompt 2 (Especializado)**:
```bash
cat prueba_prompt_2/prompt_2_parte_1.txt
# Copia → IA → obtén EDA report

cat prueba_prompt_2/prompt_2_parte_2.txt
# Copia + EDA report → IA → obtén solución + AUC
# Guarda en prueba_prompt_2/auc_prompt_2.txt
```

#### Paso 3 - Comparar
```python
with open("prueba_prompt_1/auc_prompt_1.txt") as f:
    auc_1 = float(f.read())
with open("prueba_prompt_2/auc_prompt_2.txt") as f:
    auc_2 = float(f.read())

print(f"Prompt 1: {auc_1:.4f}")
print(f"Prompt 2: {auc_2:.4f}")
print(f"Mejora: +{(auc_2 - auc_1):.4f}")
```

### 📊 Dataset

**customer_churn_data.json**:
- **1,000 clientes**
- **10 features**: edad, meses cliente, transacciones, saldo, etc.
- **Target**: churn_90d (1=abandona, 0=activo)
- **Desbalance**: ~24% churn (realista)
- **Problemas reales**: 
  - Valores nulos (~10-15%)
  - Outliers en account_balance
  - Desbalance de clases

### 🎓 Contexto de Negocio

- **Restricción**: Solo contactar TOP 10% de clientes (~100/mes)
- **Métrica**: AUC-ROC (importa el ranking, no solo accuracy)
- **Pregunta**: ¿Quiénes son los que realmente se van?

### 🔍 Diferencia Prompt 1 vs 2

| Aspecto | Prompt 1 (Genérico) | Prompt 2 (Especializado) |
|---------|-------------------|------------------------|
| **Análisis previo** | ❌ No | ✓ Sí (EDA) |
| **Tratamiento de nulos** | Genérico | Informado por EDA |
| **Outliers** | No mencionado | Tratamiento específico |
| **Modelo** | Sugerencia rápida | Justificado en contexto |
| **Métrica** | Accuracy/AUC | AUC-ROC + top 10% |
| **AUC esperado** | ~0.70-0.75 | ~0.78-0.82 |

### 📝 Flujo Completo

```
Dataset JSON
    ↓
Prompt 1: "Dame un modelo" → AUC_1
    ↓
Prompt 2 Parte 1: EDA
    ↓
Prompt 2 Parte 2: Solución especializada → AUC_2
    ↓
Comparar: AUC_2 > AUC_1 ✓
```

### 🛠️ Requisitos

```bash
pip install pandas scikit-learn numpy matplotlib seaborn
jupyter notebook  # si usas Jupyter
```

### 📌 Notas Importantes

- **No commitear outputs** (`.gitignore` ignora `auc_*.txt`, `eda_report.md`, etc.)
- **Reproducibilidad**: Todos usan `random_state=42`
- **Métrica comparación**: `AUC-ROC` en ambos prompts
- **Colaboradores**: Clonen el repo, ejecuten los prompts, comparen resultados

### 🎯 Lecciones Clave

1. **Selección de modelo no es automática** - requiere análisis
2. **El EDA da contexto** - nulos, outliers, desbalance importan
3. **Métricas importan** - AUC ≠ Accuracy en datos desbalanceados
4. **Restricciones de negocio** - top 10% es diferente a accuracy general

---

**Autor**: Angel Samir Pucho Quispe (Taller PUCP - Bloque 1)
