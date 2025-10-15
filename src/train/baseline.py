import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ================================
# 1. Crear dataset de ejemplo
# ================================
# Supongamos que son ventas diarias de un producto
np.random.seed(42)
days = pd.date_range(start="2023-01-01", periods=60, freq="D")
sales = np.random.poisson(lam=20, size=60)  # ventas simuladas

df = pd.DataFrame({"fecha": days, "ventas": sales})

# ================================
# 2. Baseline: Media móvil (ventana=7)
# ================================
df["pred_baseline"] = df["ventas"].rolling(window=7).mean().shift(1)

# Eliminar filas sin predicción
df = df.dropna()

# ================================
# 3. Calcular métricas
# ================================
mae = mean_absolute_error(df["ventas"], df["pred_baseline"])
rmse = np.sqrt(mean_squared_error(df["ventas"], df["pred_baseline"]))

print(f"MAE (Media Absoluta del Error): {mae:.2f}")
print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}")

# ================================
# 4. Graficar resultados
# ================================
plt.figure(figsize=(10,5))
plt.plot(df["fecha"], df["ventas"], label="Ventas reales", marker="o")
plt.plot(df["fecha"], df["pred_baseline"], label="Predicción (Media móvil 7d)", linestyle="--")
plt.legend()
plt.title("Baseline de predicción de ventas (Media móvil)")
plt.xlabel("Fecha")
plt.ylabel("Unidades vendidas")
plt.grid(True)
plt.show()
