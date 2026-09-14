"""
Modelo de predicción de churn de clientes.
Modelo: Logistic Regression
Métrica: AUC-ROC
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 1. Cargar datos
print("=" * 60)
print("CARGAR DATOS")
print("=" * 60)

with open('datos/customer_churn_data.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")
print(f"\nPrimeras filas:\n{df.head()}")
print(f"\nInfo del dataset:\n{df.info()}")
print(f"\nEstadísticas descriptivas:\n{df.describe()}")

# 2. Preparar datos
print("\n" + "=" * 60)
print("PREPARACIÓN DE DATOS")
print("=" * 60)

# Variable objetivo
y = df['churn_90d']
print(f"Distribución de churn:\n{y.value_counts()}")
print(f"Proporción de churn: {y.mean():.2%}")

# Features: eliminar customer_id y churn_90d
X = df.drop(['customer_id', 'churn_90d'], axis=1).copy()

# Manejar valores nulos
print(f"\nValores nulos por columna:")
print(X.isnull().sum())
# Imputar con la mediana
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

# Codificar product_type (one-hot encoding)
print(f"\nValores únicos en product_type: {X['product_type'].unique()}")
X_encoded = pd.get_dummies(X, columns=['product_type'], drop_first=True)

print(f"\nFeatures después de codificación: {X_encoded.shape[1]}")
print(f"Nombres de features:\n{list(X_encoded.columns)}")

# 3. Dividir en train/test
print("\n" + "=" * 60)
print("DIVIDIR EN TRAIN/TEST")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set size: {X_train.shape[0]} ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"Churn en train: {y_train.mean():.2%}")
print(f"Churn en test: {y_test.mean():.2%}")

# 4. Normalizar features
print("\n" + "=" * 60)
print("NORMALIZACIÓN")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features normalizados (media=0, std=1)")

# 5. Entrenar modelo
print("\n" + "=" * 60)
print("ENTRENAMIENTO")
print("=" * 60)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

print("Modelo Logistic Regression entrenado")
coef_df = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Coeficiente': model.coef_[0]
}).sort_values('Coeficiente', ascending=False)
print(f"Coeficientes:\n{coef_df}")

# 6. Evaluar modelo
print("\n" + "=" * 60)
print("EVALUACIÓN")
print("=" * 60)

# Predicciones
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

# AUC-ROC
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"\n*** AUC-ROC ***")
print(f"Train AUC-ROC: {train_auc:.4f}")
print(f"Test AUC-ROC:  {test_auc:.4f}")

# Matriz de confusión
print(f"\nMatriz de confusión (Test):")
print(confusion_matrix(y_test, y_test_pred))

# Reporte de clasificación
print(f"\nReporte de clasificación (Test):")
print(classification_report(y_test, y_test_pred, target_names=['No Churn', 'Churn']))

# 7. Visualizar ROC Curve
print("\n" + "=" * 60)
print("GRÁFICOS")
print("=" * 60)

fpr, tpr, _ = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC={test_auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Customer Churn Prediction')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
print("ROC Curve guardada en roc_curve.png")

# 8. Resumen final
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"""
Modelo: Logistic Regression
Dataset: {df.shape[0]} clientes, {df.shape[1]-1} features
Split: 80% train ({X_train.shape[0]}), 20% test ({X_test.shape[0]})

RESULTADOS:
  - AUC-ROC (Train): {train_auc:.4f}
  - AUC-ROC (Test):  {test_auc:.4f}

El modelo tiene un desempeño {'EXCELENTE' if test_auc > 0.85 else 'BUENO' if test_auc > 0.75 else 'REGULAR'}
en la predicción de churn de clientes.
""")
