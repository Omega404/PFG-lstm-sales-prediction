import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ===============================
# Configuración
# ===============================
DATA_PATH = "data/processed/product_demand.xlsx"
MODEL_DIR = "models"
REPORTS_DIR = "reports"
PLOTS_DIR = "plots"
WINDOW_SIZE = 21  # 3 semanas de contexto
MIN_DATA_POINTS = 50  # mínimo de registros para entrenar

for dir_path in [MODEL_DIR, REPORTS_DIR, PLOTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ===============================
# Clase de Feature Engineering
# ===============================
class FeatureEngineer:
    """Clase para crear features avanzadas de series temporales"""
    
    @staticmethod
    def add_temporal_features(df):
        """Agrega features temporales básicas"""
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        df['Month'] = df['InvoiceDate'].dt.month
        df['Quarter'] = df['InvoiceDate'].dt.quarter
        df['DayOfMonth'] = df['InvoiceDate'].dt.day
        df['WeekOfYear'] = df['InvoiceDate'].dt.isocalendar().week
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['IsMonthStart'] = df['InvoiceDate'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['InvoiceDate'].dt.is_month_end.astype(int)
        return df
    
    @staticmethod
    def add_lag_features(series, lags=[1, 7, 14, 21]):
        """Agrega features de lag (valores pasados)"""
        features = pd.DataFrame({'value': series})
        for lag in lags:
            features[f'lag_{lag}'] = features['value'].shift(lag)
        return features
    
    @staticmethod
    def add_rolling_features(series, windows=[7, 14, 21]):
        """Agrega features de ventanas móviles"""
        features = pd.DataFrame({'value': series})
        for window in windows:
            features[f'rolling_mean_{window}'] = features['value'].rolling(window).mean()
            features[f'rolling_std_{window}'] = features['value'].rolling(window).std()
            features[f'rolling_min_{window}'] = features['value'].rolling(window).min()
            features[f'rolling_max_{window}'] = features['value'].rolling(window).max()
        return features
    
    @staticmethod
    def add_trend_features(series):
        """Agrega features de tendencia"""
        features = pd.DataFrame({'value': series})
        
        # Diferencias
        features['diff_1'] = features['value'].diff(1)
        features['diff_7'] = features['value'].diff(7)
        
        # Tasa de cambio
        features['pct_change_1'] = features['value'].pct_change(1)
        features['pct_change_7'] = features['value'].pct_change(7)
        
        return features
    
    @staticmethod
    def add_cyclical_features(df):
        """Convierte features cíclicas a sin/cos"""
        df['day_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        return df


# ===============================
# Clase principal del sistema
# ===============================
class ProductDemandLSTM:
    def __init__(self, data_path, window_size=21):
        self.data_path = data_path
        self.window_size = window_size
        self.models = {}
        self.product_analytics = {}
        self.feature_engineer = FeatureEngineer()
        
    def load_and_prepare_data(self):
        """Carga y prepara los datos"""
        print("📂 Cargando datos...")
        df = pd.read_excel(self.data_path, engine="openpyxl")
        
        # Convertir fecha
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Eliminar valores negativos (devoluciones)
        df = df[df['Quantity'] > 0].copy()
        
        # Features temporales
        df = self.feature_engineer.add_temporal_features(df)
        df = self.feature_engineer.add_cyclical_features(df)
        
        # Calcular precio total si existe
        if 'UnitPrice' in df.columns:
            df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        self.df = df
        print(f"✅ Datos cargados: {len(df)} registros")
        print(f"📅 Rango: {df['InvoiceDate'].min()} a {df['InvoiceDate'].max()}")
        print(f"📦 Productos únicos: {df['StockCode'].nunique()}")
        
        return df
    
    def analyze_product(self, product_id):
        """Realiza análisis exploratorio de un producto"""
        df_product = self.df[self.df['StockCode'] == product_id].copy()
        
        if len(df_product) < MIN_DATA_POINTS:
            return None
        
        # Agrupar por día
        daily = df_product.groupby(df_product['InvoiceDate'].dt.date).agg({
            'Quantity': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        daily.columns = ['Date', 'Quantity', 'NumOrders']
        
        # Estadísticas básicas
        analytics = {
            'ProductID': product_id,
            'TotalSales': daily['Quantity'].sum(),
            'AvgDailySales': daily['Quantity'].mean(),
            'StdDailySales': daily['Quantity'].std(),
            'MedianDailySales': daily['Quantity'].median(),
            'MaxDailySales': daily['Quantity'].max(),
            'MinDailySales': daily['Quantity'].min(),
            'CV': daily['Quantity'].std() / daily['Quantity'].mean() if daily['Quantity'].mean() > 0 else 0,
            'TotalDays': len(daily),
            'AvgOrdersPerDay': daily['NumOrders'].mean(),
            'FirstSale': daily['Date'].min(),
            'LastSale': daily['Date'].max()
        }
        
        # Volatilidad
        analytics['Volatility'] = 'Alta' if analytics['CV'] > 1 else ('Media' if analytics['CV'] > 0.5 else 'Baja')
        
        # Tendencia (regresión lineal simple)
        x = np.arange(len(daily))
        y = daily['Quantity'].values
        slope, _, r_value, _, _ = stats.linregress(x, y)
        analytics['Trend'] = 'Creciente' if slope > 0.5 else ('Decreciente' if slope < -0.5 else 'Estable')
        analytics['TrendStrength'] = abs(r_value)
        
        # Estacionalidad (simplificado)
        if 'DayOfWeek' in df_product.columns:
            day_avg = df_product.groupby('DayOfWeek')['Quantity'].mean()
            analytics['BestDayOfWeek'] = int(day_avg.idxmax())
            analytics['WorstDayOfWeek'] = int(day_avg.idxmin())
            analytics['SeasonalityScore'] = day_avg.std() / day_avg.mean() if day_avg.mean() > 0 else 0
        
        self.product_analytics[product_id] = analytics
        return analytics
    
    def create_advanced_features(self, product_id):
        """Crea dataset con features avanzadas para un producto"""
        df_product = self.df[self.df['StockCode'] == product_id].copy()
        
        # Agrupar por día
        daily = df_product.groupby(df_product['InvoiceDate'].dt.date).agg({
            'Quantity': 'sum',
            'DayOfWeek': 'first',
            'Month': 'first',
            'IsWeekend': 'first',
            'day_sin': 'first',
            'day_cos': 'first',
            'month_sin': 'first',
            'month_cos': 'first'
        }).reset_index()
        daily.columns = ['Date', 'Quantity', 'DayOfWeek', 'Month', 'IsWeekend', 
                        'day_sin', 'day_cos', 'month_sin', 'month_cos']
        
        # Rellenar fechas faltantes
        date_range = pd.date_range(start=daily['Date'].min(), end=daily['Date'].max())
        daily = daily.set_index('Date').reindex(date_range, fill_value=0).reset_index()
        daily.columns = ['Date', 'Quantity', 'DayOfWeek', 'Month', 'IsWeekend',
                        'day_sin', 'day_cos', 'month_sin', 'month_cos']
        
        # Features de lag
        lag_features = self.feature_engineer.add_lag_features(daily['Quantity'], lags=[1, 7, 14])
        
        # Features de rolling
        rolling_features = self.feature_engineer.add_rolling_features(daily['Quantity'], windows=[7, 14, 21])
        
        # Features de tendencia
        trend_features = self.feature_engineer.add_trend_features(daily['Quantity'])
        
        # Combinar todo
        features_df = pd.concat([
            daily[['Date', 'Quantity', 'DayOfWeek', 'IsWeekend', 'day_sin', 'day_cos', 'month_sin', 'month_cos']],
            lag_features.drop('value', axis=1),
            rolling_features.drop('value', axis=1),
            trend_features.drop('value', axis=1)
        ], axis=1)
        
        # Eliminar NaN
        features_df = features_df.dropna()
        
        return features_df
    
    def create_sequences(self, data, window_size):
        """Crea secuencias para el LSTM con múltiples features"""
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size, 0])  # Predecir Quantity
        return np.array(X), np.array(y)
    
    def build_advanced_model(self, input_shape):
        """Construye modelo LSTM bidireccional avanzado"""
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',  # Más robusto a outliers
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_product_model(self, product_id, test_size=0.2, validation_split=0.15):
        """Entrena modelo avanzado para un producto"""
        print(f"\n{'='*60}")
        print(f"🎯 ENTRENANDO MODELO: {product_id}")
        print(f"{'='*60}")
        
        # Análisis previo
        analytics = self.analyze_product(product_id)
        if analytics is None:
            print(f"⚠️  Datos insuficientes para {product_id}")
            return None
        
        print(f"📊 Ventas totales: {analytics['TotalSales']:.0f} unidades")
        print(f"📈 Promedio diario: {analytics['AvgDailySales']:.1f} ± {analytics['StdDailySales']:.1f}")
        print(f"🎲 Volatilidad: {analytics['Volatility']} (CV: {analytics['CV']:.2f})")
        print(f"📉 Tendencia: {analytics['Trend']} (R²: {analytics['TrendStrength']:.2f})")
        
        # Crear features avanzadas
        features_df = self.create_advanced_features(product_id)
        
        if len(features_df) < self.window_size + 20:
            print(f"⚠️  Datos insuficientes después del feature engineering")
            return None
        
        # Preparar datos
        feature_cols = [col for col in features_df.columns if col not in ['Date', 'Quantity']]
        feature_cols = ['Quantity'] + feature_cols  # Quantity primero
        
        data = features_df[feature_cols].values
        
        # Escalar
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Crear secuencias
        X, y = self.create_sequences(data_scaled, self.window_size)
        
        # Split temporal (sin shuffle)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\n📦 Datos preparados:")
        print(f"  - Training: {len(X_train)} secuencias")
        print(f"  - Testing: {len(X_test)} secuencias")
        print(f"  - Features: {X_train.shape[2]}")
        print(f"  - Window size: {self.window_size}")
        
        # Construir modelo
        model = self.build_advanced_model((X_train.shape[1], X_train.shape[2]))
        
        print(f"\n🏗️  Arquitectura del modelo:")
        model.summary()
        
        # Callbacks
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
            min_lr=0.00001,
            verbose=1
        )
        
        # Entrenar
        print(f"\n🚀 Iniciando entrenamiento...")
        history = model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Evaluar
        print(f"\n📊 EVALUACIÓN EN TEST SET:")
        y_pred = model.predict(X_test, verbose=0).flatten()
        
        # Desnormalizar
        y_test_full = np.column_stack([y_test] + [np.zeros(len(y_test)) for _ in range(data.shape[1]-1)])
        y_pred_full = np.column_stack([y_pred] + [np.zeros(len(y_pred)) for _ in range(data.shape[1]-1)])
        
        y_test_original = scaler.inverse_transform(y_test_full)[:, 0]
        y_pred_original = scaler.inverse_transform(y_pred_full)[:, 0]
        
        # Métricas
        mae = mean_absolute_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        mape = np.mean(np.abs((y_test_original - y_pred_original) / (y_test_original + 1))) * 100
        r2 = r2_score(y_test_original, y_pred_original)
        
        print(f"  • MAE:  {mae:.2f} unidades")
        print(f"  • RMSE: {rmse:.2f} unidades")
        print(f"  • MAPE: {mape:.2f}%")
        print(f"  • R²:   {r2:.3f}")
        
        # Guardar información del modelo
        self.models[product_id] = {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'history': history.history,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2
            },
            'test_data': {
                'y_true': y_test_original,
                'y_pred': y_pred_original,
                'dates': features_df['Date'].iloc[-len(y_test):].values
            },
            'last_sequence': data_scaled[-self.window_size:],
            'analytics': analytics
        }
        
        # Guardar modelo
        model_path = os.path.join(MODEL_DIR, f'lstm_{product_id}.h5')
        scaler_path = os.path.join(MODEL_DIR, f'scaler_{product_id}.pkl')
        metadata_path = os.path.join(MODEL_DIR, f'metadata_{product_id}.pkl')
        
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump({
            'feature_cols': feature_cols,
            'metrics': self.models[product_id]['metrics'],
            'analytics': analytics
        }, metadata_path)
        
        print(f"\n💾 Modelo guardado exitosamente")
        
        # Generar visualizaciones
        self.plot_training_history(product_id)
        self.plot_predictions(product_id)
        
        return model
    
    def predict_demand(self, product_id, days_ahead=30):
        """Predice demanda futura con intervalos de confianza"""
        if product_id not in self.models:
            print(f"⚠️  Modelo no encontrado para {product_id}")
            return None
        
        model_info = self.models[product_id]
        model = model_info['model']
        scaler = model_info['scaler']
        last_seq = model_info['last_sequence'].copy()
        
        predictions = []
        current_seq = last_seq.reshape(1, self.window_size, -1)
        
        for _ in range(days_ahead):
            pred_scaled = model.predict(current_seq, verbose=0)
            
            # Desnormalizar
            pred_full = np.zeros((1, len(model_info['feature_cols'])))
            pred_full[0, 0] = pred_scaled[0, 0]
            pred_original = scaler.inverse_transform(pred_full)[0, 0]
            
            predictions.append(max(0, pred_original))
            
            # Actualizar secuencia
            new_row = current_seq[0, -1:, :].copy()
            new_row[0, 0] = pred_scaled[0, 0]
            current_seq = np.append(current_seq[:, 1:, :], new_row.reshape(1, 1, -1), axis=1)
        
        # Calcular intervalo de confianza basado en error histórico
        mae = model_info['metrics']['mae']
        predictions = np.array(predictions)
        confidence_lower = predictions - (1.96 * mae)
        confidence_upper = predictions + (1.96 * mae)
        
        return {
            'predictions': predictions,
            'confidence_lower': np.maximum(0, confidence_lower),
            'confidence_upper': confidence_upper,
            'dates': pd.date_range(
                start=datetime.now(),
                periods=days_ahead,
                freq='D'
            )
        }
    
    def plot_training_history(self, product_id):
        """Visualiza el historial de entrenamiento"""
        if product_id not in self.models:
            return
        
        history = self.models[product_id]['history']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'Training History - {product_id}', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # MAE
        axes[1].plot(history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(history['val_mae'], label='Val MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title(f'MAE Evolution - {product_id}', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'training_history_{product_id}.png'), dpi=150)
        plt.close()
        
    def plot_predictions(self, product_id):
        """Visualiza predicciones vs valores reales"""
        if product_id not in self.models:
            return
        
        test_data = self.models[product_id]['test_data']
        y_true = test_data['y_true']
        y_pred = test_data['y_pred']
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Serie temporal
        axes[0].plot(y_true, label='Real', linewidth=2, alpha=0.7)
        axes[0].plot(y_pred, label='Predicción', linewidth=2, alpha=0.7)
        axes[0].fill_between(range(len(y_true)), y_true, y_pred, alpha=0.2)
        axes[0].set_xlabel('Días', fontsize=12)
        axes[0].set_ylabel('Cantidad', fontsize=12)
        axes[0].set_title(f'Predicción vs Real - {product_id}', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.5, s=50)
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_xlabel('Valor Real', fontsize=12)
        axes[1].set_ylabel('Valor Predicho', fontsize=12)
        axes[1].set_title(f'Correlation Plot - R²: {self.models[product_id]["metrics"]["r2"]:.3f}', 
                         fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'predictions_{product_id}.png'), dpi=150)
        plt.close()
    
    def generate_product_report(self):
        """Genera reporte completo de todos los productos"""
        print("\n📊 Generando reporte de productos...")
        
        report_data = []
        for product_id, model_info in self.models.items():
            analytics = model_info['analytics']
            metrics = model_info['metrics']
            
            report_data.append({
                'ProductID': product_id,
                'TotalSales': analytics['TotalSales'],
                'AvgDailySales': analytics['AvgDailySales'],
                'Volatility': analytics['Volatility'],
                'Trend': analytics['Trend'],
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'MAPE': metrics['mape'],
                'R2': metrics['r2'],
                'ModelQuality': 'Excelente' if metrics['r2'] > 0.8 else ('Bueno' if metrics['r2'] > 0.6 else 'Mejorable')
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('R2', ascending=False)
        
        report_path = os.path.join(REPORTS_DIR, 'product_models_report.xlsx')
        report_df.to_excel(report_path, index=False)
        
        print(f"✅ Reporte guardado en: {report_path}")
        print(f"\n📈 Resumen:")
        print(report_df.to_string(index=False))
        
        return report_df


# ===============================
# Ejecución principal
# ===============================
if __name__ == "__main__":
    print("="*70)
    print("🚀 SISTEMA AVANZADO DE PREDICCIÓN DE DEMANDA DE PRODUCTOS")
    print("="*70)
    
    # Inicializar sistema
    system = ProductDemandLSTM(DATA_PATH, window_size=WINDOW_SIZE)
    
    # Cargar datos
    df = system.load_and_prepare_data()
    
    # Seleccionar top productos
    print(f"\n🔍 Identificando productos principales...")
    product_counts = df['StockCode'].value_counts()
    top_products = product_counts[product_counts >= MIN_DATA_POINTS].head(10).index.tolist()
    
    print(f"✅ {len(top_products)} productos seleccionados para entrenamiento")
    
    # Entrenar modelos
    for i, product_id in enumerate(top_products, 1):
        print(f"\n[{i}/{len(top_products)}] Procesando producto {product_id}...")
        system.train_product_model(product_id)
    
    # Generar reporte
    report = system.generate_product_report()
    
    # Predicciones de ejemplo
    print("\n" + "="*70)
    print("🔮 PREDICCIONES A 30 DÍAS")
    print("="*70)
    
    for product_id in list(system.models.keys())[:3]:
        print(f"\n📦 Producto: {product_id}")
        pred_data = system.predict_demand(product_id, days_ahead=30)
        
        if pred_data:
            preds = pred_data['predictions']
            print(f"  • Promedio próximos 7 días: {preds[:7].mean():.1f} unidades/día")
            print(f"  • Promedio próximos 30 días: {preds.mean():.1f} unidades/día")
            print(f"  • Total estimado 30 días: {preds.sum():.0f} unidades")
            print(f"  • Rango: {preds.min():.1f} - {preds.max():.1f}")
    
    print("\n" + "="*70)
    print("✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"📁 Modelos: {MODEL_DIR}/")
    print(f"📊 Reportes: {REPORTS_DIR}/")
    print(f"📈 Gráficos: {PLOTS_DIR}/")
    print("="*70)