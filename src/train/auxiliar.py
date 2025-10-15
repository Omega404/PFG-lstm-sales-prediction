import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

# ===============================
# Configuración
# ===============================
DATA_PATH = "data/processed/product_demand.xlsx"
MODEL_DIR = "models"
PRODUCT_ID = "85123A"   # Cambiar para probar otro producto
WINDOW_SIZE = 7         # días usados como contexto

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# Funciones auxiliares
# ===============================
def create_sequences(data, window_size=7):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# ===============================
# Cargar y preparar datos
# ===============================
df = pd.read_excel(DATA_PATH, engine="openpyxl")
df = df[df["StockCode"] == PRODUCT_ID].sort_values("InvoiceDate")

if df.empty:
    raise ValueError(f"No hay datos para el producto {PRODUCT_ID}")

series = df["Quantity"].values

# Escalado
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1,1))

# Secuencias
X, y = create_sequences(series_scaled, window_size=WINDOW_SIZE)
X = X.reshape((X.shape[0], X.shape[1], 1))

# ===============================
# Definir y entrenar modelo
# ===============================
model = Sequential([
    LSTM(64, activation="relu", input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

history = model.fit(
    X, y,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ===============================
# Guardar modelo y scaler
# ===============================
model_path = os.path.join(MODEL_DIR, f"lstm_product_{PRODUCT_ID}.h5")
scaler_path = os.path.join(MODEL_DIR, f"scaler_product_{PRODUCT_ID}.pkl")

model.save(model_path)

import joblib
joblib.dump(scaler, scaler_path)

print(f"✅ Modelo guardado en {model_path}")
print(f"✅ Scaler guardado en {scaler_path}")

# ===============================
# Prueba de predicción
# ===============================
last_window = series_scaled[-WINDOW_SIZE:]
last_window = last_window.reshape((1, WINDOW_SIZE, 1))
pred_scaled = model.predict(last_window)
pred = scaler.inverse_transform(pred_scaled)[0][0]

print(f"📈 Predicción próxima cantidad para {PRODUCT_ID}: {pred:.2f}")

import matplotlib.pyplot as plt

# Predicciones en todo el dataset (excepto la primera ventana)
y_pred_scaled = model.predict(X)
y_pred = scaler.inverse_transform(y_pred_scaled)

# Serie real alineada
real = series[WINDOW_SIZE:]

plt.figure(figsize=(10,5))
plt.plot(real, label="Real")
plt.plot(y_pred.flatten(), label="Predicho")
plt.title(f"Producto {PRODUCT_ID} - Predicción vs Real")
plt.legend()
plt.show()
