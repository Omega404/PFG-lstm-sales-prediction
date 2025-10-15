import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ===============================
# Configuración
# ===============================
DATA_PATH = "data/processed/product_demand.xlsx"
MODEL_DIR = "models"
PLOTS_DIR = "results/plots"

# Hiperparámetros optimizados
WINDOW_SIZE = 7         # 1 semana de contexto (mejor para datos volátiles)
FORECAST_HORIZON = 3    # Predecir 3 días adelante (más factible)
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SIZE = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===============================
# Funciones de preparación de datos
# ===============================
def load_and_aggregate_data(file_path, product_id):
    """Carga y agrega por día"""
    print(f"📂 Cargando datos para producto {product_id}...")
    
    df = pd.read_excel(file_path, engine="openpyxl")
    
    # DEBUG: Verificar qué hay en StockCode
    print(f"   Total registros: {len(df)}")
    print(f"   Primeros 5 StockCodes: {df['StockCode'].head().tolist()}")
    print(f"   Tipo de dato: {df['StockCode'].dtype}")
    
    # Convertir StockCode a string
    df['StockCode'] = df['StockCode'].astype(str).str.strip()
    product_id_str = str(product_id).strip()
    
    # Verificar si existe
    existe = product_id_str in df['StockCode'].values
    print(f"   ¿Existe '{product_id_str}'? {existe}")
    
    # Mostrar todos los productos únicos que empiezan con '22'
    productos_22 = df[df['StockCode'].str.startswith('22', na=False)]['StockCode'].unique()
    print(f"   Productos que empiezan con '22': {len(productos_22)}")
    if len(productos_22) > 0:
        print(f"   Primeros 10: {list(productos_22[:10])}")
    
    # Filtrar producto
    df_product = df[df['StockCode'] == product_id_str].copy()
    
    if df_product.empty:
        print(f"\n❌ Producto '{product_id_str}' no encontrado")
        print(f"\n📊 Top 20 productos disponibles:")
        top = df['StockCode'].value_counts().head(20)
        for i, (code, count) in enumerate(top.items(), 1):
            try:
                desc = df[df['StockCode'] == code]['Description'].iloc[0]
            except:
                desc = 'N/A'
            print(f"   {i}. '{code}' - {count} trans - {str(desc)[:45]}")
        
        raise ValueError(f"Producto no encontrado. Elige uno de la lista.")
    
    # Convertir fecha
    df_product['InvoiceDate'] = pd.to_datetime(df_product['InvoiceDate'], errors='coerce')
    
    # Agrupar por día
    daily_sales = df_product.groupby(df_product['InvoiceDate'].dt.date).agg({
        'Quantity': 'sum',
        'UnitPrice': 'mean'
    }).reset_index()
    
    daily_sales.columns = ['Date', 'Quantity', 'AvgPrice']
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    daily_sales = daily_sales.sort_values('Date').reset_index(drop=True)
    
    # Rellenar días faltantes con cero ventas
    date_range = pd.date_range(
        start=daily_sales['Date'].min(),
        end=daily_sales['Date'].max(),
        freq='D'
    )
    full_dates = pd.DataFrame({'Date': date_range})
    daily_sales = full_dates.merge(daily_sales, on='Date', how='left')
    daily_sales['Quantity'] = daily_sales['Quantity'].fillna(0)
    daily_sales['AvgPrice'] = daily_sales['AvgPrice'].ffill().bfill()
    
    # ===== SUAVIZADO CON ROLLING AVERAGE =====
    # Aplicar promedio móvil de 7 días para reducir ruido y picos
    window_smooth = 7
    daily_sales['Quantity_Raw'] = daily_sales['Quantity'].copy()  # Guardar original
    daily_sales['Quantity'] = daily_sales['Quantity'].rolling(
        window=window_smooth, 
        center=True, 
        min_periods=1
    ).mean()
    
    print(f"✅ {len(daily_sales)} días de datos cargados")
    print(f"📊 Suavizado aplicado con ventana de {window_smooth} días")
    print(f"   Rango original: [{daily_sales['Quantity_Raw'].min():.0f}, {daily_sales['Quantity_Raw'].max():.0f}]")
    print(f"   Rango suavizado: [{daily_sales['Quantity'].min():.1f}, {daily_sales['Quantity'].max():.1f}]")
    
    return daily_sales

def create_sequences(data, window_size, forecast_horizon):
    """Crea secuencias X (pasado) -> y (futuro)"""
    X, y = [], []
    
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+forecast_horizon, 0])  # Solo Quantity
    
    return np.array(X), np.array(y)

def prepare_features(daily_sales):
    """Prepara features para el modelo"""
    features = daily_sales[['Quantity', 'AvgPrice']].values
    return features

# ===============================
# Modelo LSTM
# ===============================
def build_lstm_model(input_shape, forecast_horizon):
    """Construye arquitectura LSTM optimizada"""
    model = Sequential([
        # Primera capa LSTM con return_sequences
        LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        
        # Segunda capa LSTM
        LSTM(64, activation='tanh', return_sequences=False),
        Dropout(0.3),
        
        # Capas densas
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(forecast_horizon)  # Salida: predicción de N días
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
    
    return model

# ===============================
# Entrenamiento
# ===============================
def train_model(product_id):
    """Pipeline completo de entrenamiento"""
    print(f"\n{'='*60}")
    print(f"🎯 ENTRENAMIENTO: {product_id}")
    print(f"{'='*60}\n")
    
    # 1. Cargar y preparar datos
    daily_sales = load_and_aggregate_data(DATA_PATH, product_id)
    features = prepare_features(daily_sales)
    
    if len(features) < WINDOW_SIZE + FORECAST_HORIZON + 30:
        print("⚠️  Datos insuficientes para entrenamiento")
        return None
    
    # 2. Normalizar
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    # 3. Crear secuencias
    X, y = create_sequences(features_scaled, WINDOW_SIZE, FORECAST_HORIZON)
    
    # 4. Split train/test (sin shuffle para series temporales)
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📊 Train: {len(X_train)} | Test: {len(X_test)} secuencias")
    
    # 5. Construir modelo
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), FORECAST_HORIZON)
    
    print(f"\n🏗️  Arquitectura del modelo:")
    model.summary()
    
    # 6. Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
    
    # 7. Entrenar
    print(f"\n🚀 Iniciando entrenamiento...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # 8. Evaluar
    print(f"\n📈 Evaluando modelo...")
    y_pred = model.predict(X_test, verbose=0)
    
    # Diagnóstico de datos
    print(f"\n🔍 Diagnóstico de predicciones:")
    print(f"  - Shape y_test: {y_test.shape}")
    print(f"  - Shape y_pred: {y_pred.shape}")
    print(f"  - Rango y_test (escalado): [{y_test.min():.4f}, {y_test.max():.4f}]")
    print(f"  - Rango y_pred (escalado): [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    # Desnormalizar predicciones (reshape para cada día del horizonte)
    y_test_inv = np.zeros_like(y_test)
    y_pred_inv = np.zeros_like(y_pred)
    
    for i in range(FORECAST_HORIZON):
        # Crear array temporal con shape correcto para scaler
        temp_test = np.column_stack([y_test[:, i], np.zeros(len(y_test))])
        temp_pred = np.column_stack([y_pred[:, i], np.zeros(len(y_pred))])
        
        # Desnormalizar
        y_test_inv[:, i] = scaler.inverse_transform(temp_test)[:, 0]
        y_pred_inv[:, i] = scaler.inverse_transform(temp_pred)[:, 0]
    
    print(f"\n  - Rango y_test (original): [{y_test_inv.min():.1f}, {y_test_inv.max():.1f}]")
    print(f"  - Rango y_pred (original): [{y_pred_inv.min():.1f}, {y_pred_inv.max():.1f}]")
    
    # Métricas
    mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
    rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten()))
    r2 = r2_score(y_test_inv.flatten(), y_pred_inv.flatten())
    
    print(f"\n{'='*60}")
    print(f"📊 MÉTRICAS EN TEST SET")
    print(f"{'='*60}")
    print(f"MAE:  {mae:.2f} unidades")
    print(f"RMSE: {rmse:.2f} unidades")
    print(f"R²:   {r2:.3f}")
    print(f"{'='*60}\n")
    
    # 9. Guardar modelo y scaler
    model_path = os.path.join(MODEL_DIR, f"lstm_{product_id}.h5")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{product_id}.pkl")
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"✅ Modelo guardado: {model_path}")
    print(f"✅ Scaler guardado: {scaler_path}")
    
    # 10. Visualización
    plot_results(history, y_test_inv, y_pred_inv, product_id)
    
    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
        'last_sequence': features_scaled[-WINDOW_SIZE:]
    }

# ===============================
# Visualización
# ===============================
def plot_results(history, y_test, y_pred, product_id):
    """Genera gráficas de resultados"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Resultados: {product_id}', fontsize=16, fontweight='bold')
    
    # 1. Pérdida de entrenamiento
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Pérdida durante entrenamiento')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss (Huber)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. MAE durante entrenamiento
    axes[0, 1].plot(history.history['mae'], label='Train MAE')
    axes[0, 1].plot(history.history['val_mae'], label='Val MAE')
    axes[0, 1].set_title('MAE durante entrenamiento')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Predicción día 1 vs Real
    sample_size = min(50, len(y_test))
    axes[1, 0].plot(y_test[:sample_size, 0], label='Real', marker='o', alpha=0.6, markersize=4)
    axes[1, 0].plot(y_pred[:sample_size, 0], label='Predicho', marker='x', alpha=0.6, markersize=4)
    axes[1, 0].set_title('Predicción: Día +1')
    axes[1, 0].set_xlabel('Muestra de test')
    axes[1, 0].set_ylabel('Cantidad')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error absoluto promedio por muestra
    error = np.abs(y_test - y_pred).mean(axis=1)
    axes[1, 1].hist(error, bins=25, edgecolor='black', alpha=0.7, color='coral')
    axes[1, 1].axvline(error.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {error.mean():.1f}')
    axes[1, 1].set_title('Distribución del Error Absoluto')
    axes[1, 1].set_xlabel('Error promedio (unidades)')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, f'results_{product_id}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Gráfica guardada: {plot_path}")
    plt.close()

# ===============================
# Predicción futura
# ===============================
def predict_future(product_id, weeks=2):
    """Predice ventas futuras"""
    model_path = os.path.join(MODEL_DIR, f"lstm_{product_id}.h5")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{product_id}.pkl")
    
    if not os.path.exists(model_path):
        print(f"⚠️  Modelo no encontrado para {product_id}")
        return None
    
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Cargar últimos datos
    daily_sales = load_and_aggregate_data(DATA_PATH, product_id)
    features = prepare_features(daily_sales)
    features_scaled = scaler.transform(features)
    
    # Última secuencia
    last_sequence = features_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, -1)
    
    predictions = []
    for _ in range(weeks):
        pred_scaled = model.predict(last_sequence, verbose=0)
        pred_inv = scaler.inverse_transform(
            np.column_stack([pred_scaled[0], np.zeros(FORECAST_HORIZON)])
        )[:, 0]
        
        predictions.extend(pred_inv)
        
        # Actualizar secuencia (rolling forecast)
        new_rows = np.column_stack([pred_scaled[0], np.zeros(FORECAST_HORIZON)])
        last_sequence = np.append(
            last_sequence[:, FORECAST_HORIZON:, :],
            new_rows[:FORECAST_HORIZON].reshape(1, FORECAST_HORIZON, -1),
            axis=1
        )
    
    return predictions[:weeks*7]

# ===============================
# Ejecución principal
# ===============================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 SISTEMA DE PREDICCIÓN LSTM - VENTAS POR PRODUCTO")
    print("="*60 + "\n")
    
    # Entrenar modelo para un producto más estable
    # Usando producto con muchas transacciones y ventas estables
    PRODUCT_ID = "22423"  # REGENCY CAKESTAND 3 TIER - Top 2, 2203 transacciones
    
    print(f"📦 Producto seleccionado: {PRODUCT_ID}")
    print(f"   (Producto con alto volumen de ventas)")
    print(f"⚙️  Config: WINDOW={WINDOW_SIZE} días, FORECAST={FORECAST_HORIZON} días\n")
    
    result = train_model(PRODUCT_ID)
    
    if result:
        # Hacer predicción de 2 semanas
        print(f"\n{'='*60}")
        print(f"🔮 PREDICCIÓN FUTURA (2 semanas)")
        print(f"{'='*60}\n")
        
        future_pred = predict_future(PRODUCT_ID, weeks=2)
        
        if future_pred is not None:
            for i, pred in enumerate(future_pred, 1):
                print(f"Día {i}: {pred:.1f} unidades")
            
            print(f"\n📊 Promedio semanal: {np.mean(future_pred):.1f} unidades/día")
            print(f"📊 Total 2 semanas: {np.sum(future_pred):.0f} unidades")
    
    print("\n✅ Proceso completado!")