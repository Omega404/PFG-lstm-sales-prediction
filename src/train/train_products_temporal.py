"""
═══════════════════════════════════════════════════════════════════════════════
PRODUCT TEMPORAL ANALYZER - Entrenamiento Unificado SHORT/MEDIUM/LONG
═══════════════════════════════════════════════════════════════════════════════

Sistema unificado de entrenamiento de modelos LSTM para predicción de demanda
de productos con tres horizontes temporales.

Consistente con CustomerTemporalAnalyzer V3:
- SHORT:  30 días → 7 días (predicción semanal)
- MEDIUM: 120 días → 7 días (con más contexto)
- LONG:   240 días → 7 días (contexto completo)

Author: Sistema PFG LSTM
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow para experiment tracking
try:
    import mlflow
    import mlflow.keras
    MLFLOW_AVAILABLE = True
    print("✅ MLflow disponible")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow no instalado - tracking deshabilitado (pip install mlflow)")

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

class ProductTemporalConfig:
    """Configuración unificada para productos - forecast uniforme de 7 días"""

    SHORT = {
        'name': 'short',
        'window_days': 30,      # 1 mes de historial
        'forecast_days': 7,     # Predecir próximos 7 días (consistente con clientes)
        'lstm_units': [64, 32],
        'epochs': 20,
        'batch_size': 32
    }

    MEDIUM = {
        'name': 'medium',
        'window_days': 120,     # 4 meses de historial
        'forecast_days': 7,     # Mismo forecast (comparación justa)
        'lstm_units': [128, 64],
        'epochs': 30,
        'batch_size': 64
    }

    LONG = {
        'name': 'long',
        'window_days': 240,     # 8 meses de historial
        'forecast_days': 7,     # Mismo forecast (comparación justa)
        'lstm_units': [128, 64, 32],
        'epochs': 40,
        'batch_size': 64
    }

    # Features por timestep (Quantity, AvgPrice)
    N_FEATURES = 2

    # Callbacks
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 8
    REDUCE_LR_FACTOR = 0.5
    MIN_LEARNING_RATE = 1e-7


# ═══════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL: PRODUCT TEMPORAL ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class ProductTemporalAnalyzer:
    """
    Análisis temporal completo de productos con LSTM
    Arquitectura análoga a CustomerTemporalAnalyzer
    """

    def __init__(self, data_path='data/processed/online_retail_2.xlsx',
                 output_dir='models/temporal/products'):
        """
        Inicializa el analizador temporal de productos

        Args:
            data_path: ruta al dataset Online Retail II
            output_dir: carpeta para guardar modelos y resultados
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.products = []

        # Crear estructura de directorios
        os.makedirs(output_dir, exist_ok=True)
        for horizon in ['short', 'medium', 'long']:
            os.makedirs(f'{output_dir}/{horizon}', exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  PRODUCT TEMPORAL ANALYZER - Inicializado")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}")

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 1: CARGA Y PREPROCESAMIENTO
    # ═══════════════════════════════════════════════════════════════════════

    def load_and_preprocess_data(self):
        """Carga y preprocesa el dataset"""
        print(f"\n{'─'*70}")
        print("FASE 1: Carga y Preprocesamiento de Datos")
        print(f"{'─'*70}")

        print(f"\n📊 Cargando dataset: {self.data_path}")
        self.df = pd.read_excel(self.data_path, engine='openpyxl')
        print(f"   Registros cargados: {len(self.df):,}")

        # Limpieza
        initial_count = len(self.df)
        self.df = self.df[self.df['Quantity'] > 0]
        self.df = self.df[self.df['UnitPrice'] > 0]

        # Convertir fecha
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.df['Date'] = self.df['InvoiceDate'].dt.date

        # Limpiar StockCode
        self.df['StockCode'] = self.df['StockCode'].astype(str).str.strip()

        # Ordenar
        self.df = self.df.sort_values(['StockCode', 'InvoiceDate'])

        print(f"\n✅ Limpieza completada:")
        print(f"   Registros eliminados: {initial_count - len(self.df):,}")
        print(f"   Registros válidos: {len(self.df):,}")
        print(f"   Productos únicos: {self.df['StockCode'].nunique():,}")
        print(f"   Rango: {self.df['InvoiceDate'].min()} → {self.df['InvoiceDate'].max()}")

        return self.df

    def select_products(self, min_transactions=50, top_n=100):
        """
        Selecciona productos para entrenar

        Args:
            min_transactions: mínimo de transacciones
            top_n: top N productos por volumen
        """
        print(f"\n{'─'*70}")
        print("FASE 2: Selección de Productos")
        print(f"{'─'*70}")

        # Contar transacciones por producto
        product_counts = self.df.groupby('StockCode').agg({
            'InvoiceNo': 'nunique',
            'Quantity': 'sum',
            'UnitPrice': 'mean'
        }).reset_index()

        product_counts.columns = ['StockCode', 'TotalTransactions', 'TotalQuantity', 'AvgPrice']

        # Filtrar por mínimo
        product_counts = product_counts[product_counts['TotalTransactions'] >= min_transactions]

        # Ordenar por volumen
        product_counts = product_counts.sort_values('TotalQuantity', ascending=False)

        # Tomar top N
        self.products = product_counts.head(top_n)['StockCode'].tolist()

        print(f"\n📦 Productos seleccionados:")
        print(f"   Total candidatos: {len(product_counts):,}")
        print(f"   Mín transacciones: {min_transactions}")
        print(f"   Top seleccionados: {len(self.products)}")
        print(f"\n   Top 10 productos:")
        for i, row in product_counts.head(10).iterrows():
            print(f"      {row['StockCode']}: {row['TotalQuantity']:,.0f} unidades ({row['TotalTransactions']} trans)")

        return self.products

    def prepare_product_sequences(self, horizon_config):
        """
        Prepara secuencias temporales para un producto

        Args:
            horizon_config: dict con configuración (window_days, forecast_days, etc.)

        Returns:
            X, y: arrays de secuencias y targets
        """
        window_days = horizon_config['window_days']
        forecast_days = horizon_config['forecast_days']

        all_X = []
        all_y = []

        print(f"\n🔄 Preparando secuencias ({window_days}→{forecast_days} días)...")

        for product_code in self.products:
            # Filtrar producto
            df_product = self.df[self.df['StockCode'] == product_code].copy()

            if len(df_product) == 0:
                continue

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

            # Suavizado
            daily_sales['Quantity'] = daily_sales['Quantity'].rolling(
                window=7, center=True, min_periods=1
            ).mean()

            # Extraer features
            features = daily_sales[['Quantity', 'AvgPrice']].values

            # Crear secuencias
            for i in range(len(features) - window_days - forecast_days + 1):
                X_seq = features[i:i+window_days]
                y_seq = features[i+window_days:i+window_days+forecast_days, 0]  # Solo Quantity

                all_X.append(X_seq)
                all_y.append(y_seq)

        X = np.array(all_X)
        y = np.array(all_y)

        print(f"   Secuencias generadas: {len(X):,}")
        print(f"   Shape X: {X.shape}")
        print(f"   Shape y: {y.shape}")

        return X, y

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 3: CONSTRUCCIÓN DEL MODELO
    # ═══════════════════════════════════════════════════════════════════════

    def build_lstm_model(self, horizon_config):
        """Construye modelo LSTM para un horizonte"""
        window_days = horizon_config['window_days']
        forecast_days = horizon_config['forecast_days']
        lstm_units = horizon_config['lstm_units']

        print(f"\n🏗️  Construyendo modelo LSTM...")
        print(f"   Window: {window_days} días")
        print(f"   Forecast: {forecast_days} días")
        print(f"   LSTM units: {lstm_units}")

        # Input
        inputs = keras.Input(shape=(window_days, ProductTemporalConfig.N_FEATURES))

        # LSTM layers
        x = inputs
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1)
            x = layers.LSTM(units, return_sequences=return_sequences, name=f'lstm_{i+1}')(x)
            x = layers.Dropout(0.2, name=f'dropout_{i+1}')(x)

        # Output layer
        outputs = layers.Dense(forecast_days, activation='relu', name='forecast')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name=f'product_lstm_{horizon_config["name"]}')

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        print("\n" + "="*70)
        model.summary()
        print("="*70)

        return model

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 4: ENTRENAMIENTO
    # ═══════════════════════════════════════════════════════════════════════

    def train_horizon_model(self, horizon_config, platform="local", script_version="v1"):
        """Entrena modelo para un horizonte temporal con MLflow tracking"""
        horizon_name = horizon_config['name']
        output_path = f'{self.output_dir}/{horizon_name}'

        print(f"\n{'='*70}")
        print(f"  ENTRENANDO MODELO: {horizon_name.upper()}")
        print(f"{'='*70}")

        # Iniciar MLflow run si está disponible
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment("products_temporal")
            run_name = f"products_{horizon_name}_{platform}"
            mlflow.start_run(run_name=run_name)
            print(f"📊 MLflow tracking: {mlflow.active_run().info.run_id}")

            # Log de hiperparámetros
            mlflow.log_params({
                "model_type": "products",
                "horizon": horizon_name,
                "window_days": horizon_config['window_days'],
                "forecast_days": horizon_config['forecast_days'],
                "lstm_units": str(horizon_config['lstm_units']),
                "epochs_config": horizon_config['epochs'],
                "batch_size": horizon_config['batch_size'],
                "early_stopping_patience": ProductTemporalConfig.EARLY_STOPPING_PATIENCE,
                "reduce_lr_patience": ProductTemporalConfig.REDUCE_LR_PATIENCE,
                "reduce_lr_factor": ProductTemporalConfig.REDUCE_LR_FACTOR,
                "platform": platform,
                "script_version": script_version,
                "n_features": ProductTemporalConfig.N_FEATURES
            })

        # Preparar datos
        X, y = self.prepare_product_sequences(horizon_config)

        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        print(f"\n📊 Split de datos:")
        print(f"   Train: {len(X_train):,} secuencias")
        print(f"   Val:   {len(X_val):,} secuencias")

        # Normalizar
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_flat = X_train.reshape(-1, ProductTemporalConfig.N_FEATURES)
        X_train_scaled = scaler_X.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled.reshape(X_train.shape)

        X_val_flat = X_val.reshape(-1, ProductTemporalConfig.N_FEATURES)
        X_val_scaled = scaler_X.transform(X_val_flat)
        X_val_scaled = X_val_scaled.reshape(X_val.shape)

        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)

        # Construir modelo
        model = self.build_lstm_model(horizon_config)

        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=ProductTemporalConfig.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=ProductTemporalConfig.REDUCE_LR_FACTOR,
            patience=ProductTemporalConfig.REDUCE_LR_PATIENCE,
            min_lr=ProductTemporalConfig.MIN_LEARNING_RATE,
            verbose=1
        )

        # Entrenar
        print(f"\n🚀 Iniciando entrenamiento...")
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=horizon_config['epochs'],
            batch_size=horizon_config['batch_size'],
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # Evaluar
        y_pred_scaled = model.predict(X_val_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        mae = np.mean(np.abs(y_val - y_pred))
        rmse = np.sqrt(np.mean((y_val - y_pred)**2))

        print(f"\n{'='*70}")
        print(f"  MÉTRICAS FINALES - {horizon_name.upper()}")
        print(f"{'='*70}")
        print(f"   MAE:  {mae:.2f} unidades")
        print(f"   RMSE: {rmse:.2f} unidades")
        print(f"{'='*70}")

        # Guardar
        print(f"\n💾 Guardando modelo...")
        model.save(f'{output_path}/model_best.keras')

        with open(f'{output_path}/scaler_X.pkl', 'wb') as f:
            pickle.dump(scaler_X, f)

        with open(f'{output_path}/scaler_y.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)

        with open(f'{output_path}/training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        # Guardar métricas
        metrics = {
            'horizon': horizon_name,
            'window_days': horizon_config['window_days'],
            'forecast_days': horizon_config['forecast_days'],
            'mae': float(mae),
            'rmse': float(rmse),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        }

        with open(f'{output_path}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # MLflow: Log métricas y artefactos
        if MLFLOW_AVAILABLE:
            # Métricas principales
            mlflow.log_metrics({
                "mae": float(mae),
                "rmse": float(rmse),
                "final_train_loss": float(history.history['loss'][-1]),
                "final_val_loss": float(history.history['val_loss'][-1]),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "epochs_trained": len(history.history['loss']),
                "best_epoch": len(history.history['loss']) - ProductTemporalConfig.EARLY_STOPPING_PATIENCE
                    if len(history.history['loss']) < horizon_config['epochs'] else len(history.history['loss'])
            })

            # Log curvas de entrenamiento por época
            for epoch in range(len(history.history['loss'])):
                mlflow.log_metrics({
                    "epoch_train_loss": history.history['loss'][epoch],
                    "epoch_val_loss": history.history['val_loss'][epoch]
                }, step=epoch)

            # Log artefactos
            mlflow.log_artifact(f'{output_path}/metrics.json', 'metrics')
            mlflow.log_artifact(f'{output_path}/training_history.pkl', 'history')

            # Log modelo (requiere formato específico)
            try:
                mlflow.keras.log_model(model, "model", registered_model_name=f"products_{horizon_name}")
                print(f"   ✅ Modelo registrado en MLflow Model Registry")
            except Exception as e:
                print(f"   ⚠️ No se pudo registrar modelo en MLflow: {e}")

            # Tags adicionales
            mlflow.set_tags({
                "team": "PFG_LSTM",
                "model_family": "products_temporal",
                "converged": "yes" if len(history.history['loss']) < horizon_config['epochs'] else "no"
            })

            # Cerrar run
            mlflow.end_run()
            print(f"   📊 MLflow run finalizado")

        print(f"   ✅ Guardado en: {output_path}/")

        return model, history, metrics


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Función principal de entrenamiento"""
    import argparse

    parser = argparse.ArgumentParser(description='Entrenar modelos LSTM de productos')
    parser.add_argument('--data', default='data/processed/online_retail_2.xlsx',
                        help='Ruta al dataset')
    parser.add_argument('--output', default='models/temporal/products',
                        help='Directorio de salida')
    parser.add_argument('--horizon', choices=['short', 'medium', 'long', 'all'],
                        default='all', help='Horizonte a entrenar')
    parser.add_argument('--min-trans', type=int, default=50,
                        help='Mínimo de transacciones por producto')
    parser.add_argument('--top-n', type=int, default=100,
                        help='Top N productos a entrenar')
    parser.add_argument('--platform', default='local',
                        choices=['local', 'kaggle', 'colab'],
                        help='Plataforma de entrenamiento (para MLflow tracking)')
    parser.add_argument('--script-version', default='v1',
                        help='Versión del script (para MLflow tracking)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("  PRODUCT TEMPORAL ANALYZER - Entrenamiento")
    print("="*70)
    print(f"\n📝 Configuración:")
    print(f"   Dataset: {args.data}")
    print(f"   Output: {args.output}")
    print(f"   Horizonte: {args.horizon}")
    print(f"   Mín transacciones: {args.min_trans}")
    print(f"   Top productos: {args.top_n}")

    # Inicializar
    analyzer = ProductTemporalAnalyzer(
        data_path=args.data,
        output_dir=args.output
    )

    # Cargar datos
    analyzer.load_and_preprocess_data()
    analyzer.select_products(min_transactions=args.min_trans, top_n=args.top_n)

    # Entrenar según horizonte
    horizons = []
    if args.horizon == 'all':
        horizons = [
            ProductTemporalConfig.SHORT,
            ProductTemporalConfig.MEDIUM,
            ProductTemporalConfig.LONG
        ]
    else:
        config_map = {
            'short': ProductTemporalConfig.SHORT,
            'medium': ProductTemporalConfig.MEDIUM,
            'long': ProductTemporalConfig.LONG
        }
        horizons = [config_map[args.horizon]]

    # Entrenar cada horizonte
    results = {}
    for config in horizons:
        model, history, metrics = analyzer.train_horizon_model(
            config,
            platform=args.platform,
            script_version=args.script_version
        )
        results[config['name']] = metrics

        # Limpiar memoria
        del model
        import gc
        gc.collect()

    # Resumen final
    print("\n" + "="*70)
    print("  RESUMEN FINAL")
    print("="*70)
    for horizon_name, metrics in results.items():
        print(f"\n{horizon_name.upper()}:")
        print(f"   MAE:  {metrics['mae']:.2f} unidades")
        print(f"   RMSE: {metrics['rmse']:.2f} unidades")
        print(f"   Samples: {metrics['train_samples']:,} train / {metrics['val_samples']:,} val")

    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)


if __name__ == '__main__':
    main()
