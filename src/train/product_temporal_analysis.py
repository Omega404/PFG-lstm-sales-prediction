import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ===============================
# Configuración
# ===============================
DATA_PATH = "data/processed/product_demand.xlsx"
MODEL_DIR = "models/temporal"
REPORTS_DIR = "results/reports/temporal"
PLOTS_DIR = "results/plots/temporal"

# Configuración multi-temporal
TEMPORAL_CONFIGS = {
    'short': {
        'name': 'Corto Plazo',
        'window_size': 42,      # 6 semanas de histórico
        'forecast_horizon': 14, # Predice 2 semanas
        'epochs': 80,
        'description': 'Decisiones tácticas inmediatas'
    },
    'medium': {
        'name': 'Medio Plazo',
        'window_size': 180,     # 6 meses de histórico
        'forecast_horizon': 30, # Predice 1 mes
        'epochs': 100,
        'description': 'Planificación mensual y compras'
    },
    'long': {
        'name': 'Largo Plazo',
        'window_size': 365,     # 1 año de histórico
        'forecast_horizon': 90, # Predice 3 meses
        'epochs': 120,
        'description': 'Estrategia y presupuesto anual'
    }
}

BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SIZE = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===============================
# Clase Principal
# ===============================
class TemporalProductAnalyzer:
    """
    Analizador temporal de productos con tres horizontes de predicción.
    
    Entrena y usa 3 modelos LSTM independientes por producto:
    - Corto plazo: 6 semanas → 2 semanas
    - Medio plazo: 6 meses → 1 mes  
    - Largo plazo: 1 año → 3 meses
    """
    
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.models = {}  # {product: {short: model, medium: model, long: model}}
        self.scalers = {}
        self.product_info = {}
        
    def load_and_prepare_data(self, stock_code):
        """
        Carga y prepara datos completos de un producto.
        
        Returns:
            DataFrame con ventas diarias completas
        """
        print(f"📂 Cargando datos para {stock_code}...")
        
        df = pd.read_excel(self.data_path, engine="openpyxl")
        df['StockCode'] = df['StockCode'].astype(str).str.strip()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        
        # Filtrar producto
        df_product = df[df['StockCode'] == stock_code].copy()
        
        if df_product.empty:
            raise ValueError(f"No hay datos para {stock_code}")
        
        # Agrupar por día
        daily_sales = df_product.groupby(df_product['InvoiceDate'].dt.date).agg({
            'Quantity': 'sum',
            'UnitPrice': 'mean'
        }).reset_index()
        
        daily_sales.columns = ['Date', 'Quantity', 'AvgPrice']
        daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
        daily_sales = daily_sales.sort_values('Date').reset_index(drop=True)
        
        # Rellenar días faltantes
        date_range = pd.date_range(
            start=daily_sales['Date'].min(),
            end=daily_sales['Date'].max(),
            freq='D'
        )
        full_dates = pd.DataFrame({'Date': date_range})
        daily_sales = full_dates.merge(daily_sales, on='Date', how='left')
        daily_sales['Quantity'] = daily_sales['Quantity'].fillna(0)
        daily_sales['AvgPrice'] = daily_sales['AvgPrice'].ffill().bfill()
        
        # Suavizado adaptativo según plazo
        daily_sales['Quantity_Raw'] = daily_sales['Quantity'].copy()
        daily_sales['Quantity'] = daily_sales['Quantity'].rolling(
            window=7, center=True, min_periods=1
        ).mean()
        
        print(f"✅ {len(daily_sales)} días de datos ({daily_sales['Date'].min().date()} a {daily_sales['Date'].max().date()})")
        
        return daily_sales
    
    def check_data_sufficiency(self, daily_sales, temporal_type):
        """
        Verifica si hay suficientes datos para entrenar el modelo temporal.
        
        Args:
            daily_sales: DataFrame con datos diarios
            temporal_type: 'short', 'medium', o 'long'
            
        Returns:
            (bool, str): (es_suficiente, mensaje)
        """
        config = TEMPORAL_CONFIGS[temporal_type]
        window = config['window_size']
        forecast = config['forecast_horizon']
        
        min_required = window + forecast + 30  # + margen de seguridad
        available = len(daily_sales)
        
        if available < min_required:
            return False, f"Insuficiente ({available} días, necesita {min_required})"
        
        return True, f"Suficiente ({available} días)"
    
    def create_sequences(self, data, window_size, forecast_horizon):
        """Crea secuencias temporales para LSTM"""
        X, y = [], []
        for i in range(len(data) - window_size - forecast_horizon + 1):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size:i+window_size+forecast_horizon, 0])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape, forecast_horizon, temporal_type):
        """
        Construye modelo LSTM adaptado al horizonte temporal.
        
        Arquitectura más profunda para largo plazo (más complejidad).
        """
        if temporal_type == 'long':
            # Modelo más complejo para largo plazo
            model = Sequential([
                LSTM(256, activation='tanh', return_sequences=True, input_shape=input_shape),
                Dropout(0.3),
                LSTM(128, activation='tanh', return_sequences=True),
                Dropout(0.3),
                LSTM(64, activation='tanh'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(forecast_horizon)
            ])
        elif temporal_type == 'medium':
            # Modelo intermedio
            model = Sequential([
                LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape),
                Dropout(0.3),
                LSTM(64, activation='tanh'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(forecast_horizon)
            ])
        else:  # short
            # Modelo más simple para corto plazo
            model = Sequential([
                LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(32, activation='tanh'),
                Dropout(0.2),
                Dense(forecast_horizon)
            ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
        return model
    
    def train_temporal_model(self, stock_code, temporal_type):
        """
        Entrena modelo para un horizonte temporal específico.
        
        Args:
            stock_code: Código del producto
            temporal_type: 'short', 'medium', o 'long'
            
        Returns:
            Diccionario con métricas del modelo o None si falla
        """
        config = TEMPORAL_CONFIGS[temporal_type]
        window_size = config['window_size']
        forecast_horizon = config['forecast_horizon']
        epochs = config['epochs']
        
        print(f"\n{'='*70}")
        print(f"📊 Entrenando modelo {config['name'].upper()}: {stock_code}")
        print(f"   Ventana: {window_size} días → Predicción: {forecast_horizon} días")
        print(f"{'='*70}")
        
        try:
            # Cargar datos
            daily_sales = self.load_and_prepare_data(stock_code)
            
            # Verificar suficiencia
            sufficient, msg = self.check_data_sufficiency(daily_sales, temporal_type)
            if not sufficient:
                print(f"❌ {msg}")
                return None
            
            # Preparar features
            features = daily_sales[['Quantity', 'AvgPrice']].values
            
            # Normalizar
            scaler = MinMaxScaler(feature_range=(0, 1))
            features_scaled = scaler.fit_transform(features)
            
            # Crear secuencias
            X, y = self.create_sequences(features_scaled, window_size, forecast_horizon)
            
            if len(X) < 50:
                print(f"❌ Muy pocas secuencias ({len(X)})")
                return None
            
            # Split temporal
            split_idx = int(len(X) * (1 - TEST_SIZE))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"   Train: {len(X_train)} | Test: {len(X_test)} secuencias")
            
            # Construir modelo
            model = self.build_lstm_model(
                (X_train.shape[1], X_train.shape[2]), 
                forecast_horizon,
                temporal_type
            )
            
            # Callbacks
            early_stop = EarlyStopping(
                monitor='val_loss', 
                patience=20, 
                restore_best_weights=True, 
                verbose=0
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=10, 
                min_lr=1e-6, 
                verbose=0
            )
            
            # Entrenar
            print(f"   🚀 Entrenando ({epochs} épocas máx)...")
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=BATCH_SIZE,
                validation_data=(X_test, y_test),
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            # Evaluar
            y_pred = model.predict(X_test, verbose=0)
            
            # Desnormalizar
            y_test_inv = np.zeros_like(y_test)
            y_pred_inv = np.zeros_like(y_pred)
            
            for i in range(forecast_horizon):
                temp_test = np.column_stack([y_test[:, i], np.zeros(len(y_test))])
                temp_pred = np.column_stack([y_pred[:, i], np.zeros(len(y_pred))])
                y_test_inv[:, i] = scaler.inverse_transform(temp_test)[:, 0]
                y_pred_inv[:, i] = scaler.inverse_transform(temp_pred)[:, 0]
            
            # Métricas
            mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
            rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten()))
            r2 = r2_score(y_test_inv.flatten(), y_pred_inv.flatten())
            
            print(f"   ✅ R²: {r2:.3f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
            
            # Guardar modelo
            model_path = os.path.join(MODEL_DIR, f"lstm_{stock_code}_{temporal_type}.h5")
            scaler_path = os.path.join(MODEL_DIR, f"scaler_{stock_code}_{temporal_type}.pkl")
            
            model.save(model_path)
            joblib.dump(scaler, scaler_path)
            
            # Guardar en diccionario
            if stock_code not in self.models:
                self.models[stock_code] = {}
                self.scalers[stock_code] = {}
            
            self.models[stock_code][temporal_type] = model
            self.scalers[stock_code][temporal_type] = scaler
            
            return {
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'epochs_trained': len(history.history['loss']),
                'temporal_type': temporal_type,
                'window_size': window_size,
                'forecast_horizon': forecast_horizon
            }
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return None
    
    def train_all_temporal_models(self, stock_code):
        """
        Entrena los 3 modelos temporales para un producto.
        
        Returns:
            Diccionario con resultados de cada modelo
        """
        print(f"\n{'='*70}")
        print(f"🎯 ENTRENAMIENTO MULTI-TEMPORAL: {stock_code}")
        print(f"{'='*70}")
        
        results = {}
        
        for temporal_type in ['short', 'medium', 'long']:
            result = self.train_temporal_model(stock_code, temporal_type)
            results[temporal_type] = result
        
        # Resumen
        print(f"\n{'='*70}")
        print(f"📊 RESUMEN MULTI-TEMPORAL: {stock_code}")
        print(f"{'='*70}")
        
        for t_type, result in results.items():
            config = TEMPORAL_CONFIGS[t_type]
            if result:
                print(f"\n✅ {config['name']}:")
                print(f"   R²: {result['r2']:.3f} | MAE: {result['mae']:.2f}")
                print(f"   {config['description']}")
            else:
                print(f"\n❌ {config['name']}: No se pudo entrenar")
        
        return results
    
    def predict_multi_temporal(self, stock_code):
        """
        Genera predicciones para los 3 horizontes temporales.
        
        Returns:
            Diccionario con predicciones de cada plazo
        """
        if stock_code not in self.models:
            print(f"❌ No hay modelos entrenados para {stock_code}")
            return None
        
        print(f"\n{'='*70}")
        print(f"🔮 PREDICCIÓN MULTI-TEMPORAL: {stock_code}")
        print(f"{'='*70}\n")
        
        daily_sales = self.load_and_prepare_data(stock_code)
        predictions = {}
        
        for temporal_type in ['short', 'medium', 'long']:
            if temporal_type not in self.models[stock_code]:
                continue
            
            config = TEMPORAL_CONFIGS[temporal_type]
            model = self.models[stock_code][temporal_type]
            scaler = self.scalers[stock_code][temporal_type]
            window_size = config['window_size']
            forecast_horizon = config['forecast_horizon']
            
            # Preparar última secuencia
            features = daily_sales[['Quantity', 'AvgPrice']].values
            features_scaled = scaler.transform(features)
            last_sequence = features_scaled[-window_size:].reshape(1, window_size, -1)
            
            # Predecir
            pred_scaled = model.predict(last_sequence, verbose=0)
            
            # Desnormalizar
            pred_values = []
            for i in range(forecast_horizon):
                temp = np.column_stack([pred_scaled[0, i], 0])
                pred_value = scaler.inverse_transform(temp.reshape(1, -1))[0, 0]
                pred_values.append(max(0, pred_value))
            
            predictions[temporal_type] = {
                'name': config['name'],
                'horizon_days': forecast_horizon,
                'predictions': pred_values,
                'total': sum(pred_values),
                'average_daily': np.mean(pred_values),
                'trend': 'CRECIENTE' if pred_values[-1] > pred_values[0] else 'DECRECIENTE'
            }
            
            print(f"📊 {config['name']} ({forecast_horizon} días):")
            print(f"   Total proyectado: {sum(pred_values):.1f} unidades")
            print(f"   Promedio diario: {np.mean(pred_values):.1f} unidades")
            print(f"   Tendencia: {predictions[temporal_type]['trend']}")
            print()
        
        return predictions
    
    def generate_temporal_report(self, stock_codes):
        """
        Genera reporte completo multi-temporal para múltiples productos.
        """
        print(f"\n{'='*70}")
        print(f"📈 REPORTE MULTI-TEMPORAL DE PRODUCTOS")
        print(f"{'='*70}\n")
        print(f"Productos a analizar: {len(stock_codes)}\n")
        
        all_results = []
        
        for i, stock_code in enumerate(stock_codes, 1):
            print(f"[{i}/{len(stock_codes)}] Procesando {stock_code}...")
            
            # Entrenar modelos
            training_results = self.train_all_temporal_models(stock_code)
            
            # Generar predicciones
            predictions = self.predict_multi_temporal(stock_code)
            
            if predictions:
                # Compilar resultado
                product_result = {
                    'StockCode': stock_code,
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Agregar métricas de entrenamiento
                for t_type, metrics in training_results.items():
                    prefix = f"{t_type}_"
                    if metrics:
                        product_result[f"{prefix}r2"] = metrics['r2']
                        product_result[f"{prefix}mae"] = metrics['mae']
                        product_result[f"{prefix}rmse"] = metrics['rmse']
                    else:
                        product_result[f"{prefix}r2"] = None
                        product_result[f"{prefix}mae"] = None
                        product_result[f"{prefix}rmse"] = None
                
                # Agregar predicciones
                for t_type, pred in predictions.items():
                    prefix = f"{t_type}_pred_"
                    product_result[f"{prefix}total"] = pred['total']
                    product_result[f"{prefix}avg_daily"] = pred['average_daily']
                    product_result[f"{prefix}trend"] = pred['trend']
                
                all_results.append(product_result)
        
        # Guardar reporte
        if all_results:
            df_results = pd.DataFrame(all_results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            excel_path = os.path.join(REPORTS_DIR, f'analisis_temporal_{timestamp}.xlsx')
            df_results.to_excel(excel_path, index=False)
            
            json_path = os.path.join(REPORTS_DIR, f'analisis_temporal_{timestamp}.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n{'='*70}")
            print(f"✅ REPORTE GENERADO")
            print(f"{'='*70}")
            print(f"📁 Excel: {excel_path}")
            print(f"📁 JSON: {json_path}")
            print(f"\nProductos procesados: {len(all_results)}")
        
        return df_results if all_results else None

# ===============================
# Ejecución principal
# ===============================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 ANÁLISIS TEMPORAL MULTI-PLAZO DE PRODUCTOS")
    print("="*70 + "\n")
    
    # Leer productos disponibles de entrenamientos anteriores
    results_path = os.path.join("results/reports", 'resultados_entrenamiento.xlsx')
    
    if os.path.exists(results_path):
        df_hist = pd.read_excel(results_path)
        good_products = df_hist[
            (df_hist['Status'] == 'Éxito') & 
            (df_hist['R2'] >= 0.3)
        ]['StockCode'].unique()
        
        # Usar top 5 productos para demostración
        products_to_analyze = list(good_products[:5])
        
        print(f"✅ Productos seleccionados (top 5 con R²≥0.3):")
        for p in products_to_analyze:
            print(f"   - {p}")
    else:
        # Productos de ejemplo
        products_to_analyze = ['20727', '20723', '20728']
        print(f"⚠️  Usando productos de ejemplo: {products_to_analyze}")
    
    # Inicializar analizador
    analyzer = TemporalProductAnalyzer()
    
    # Generar reporte multi-temporal
    df_report = analyzer.generate_temporal_report(products_to_analyze)
    
    print("\n✅ Proceso completado!")
