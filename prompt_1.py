import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import warnings
warnings.filterwarnings('ignore')

# Cargar datos
print("Cargando datos...")
with open('/root/.claude/uploads/807b9509-e5bb-5ace-ad85-d08027458da4/95cbeaeb-bank_customer_churn.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Shape: {df.shape}")

# Separar features y target
X = df.drop(['client_id', 'churn_90d'], axis=1)
y = df['churn_90d']

print(f"Features: {X.shape[1]}")
print(f"Target distribution:\n{y.value_counts()}")

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Escalar features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar Random Forest
print("\nEntrenando Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# Predicciones y AUC
print("Evaluando modelo...")
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"\n{'='*50}")
print(f"ROC AUC Score (Test): {auc_score:.4f}")
print(f"{'='*50}")

# Reporte adicional
y_pred = model.predict(X_test)
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Feature importance
print("\nTop 10 Features:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

print(f"\n✓ Modelo entrenado exitosamente")
print(f"✓ AUC: {auc_score:.4f}")
