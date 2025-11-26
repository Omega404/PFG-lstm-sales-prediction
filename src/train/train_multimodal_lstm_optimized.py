"""
═══════════════════════════════════════════════════════════════════════════════
MULTIMODAL LSTM OPTIMIZADO - Basado en 209 Experimentos
═══════════════════════════════════════════════════════════════════════════════

Configuración óptima derivada del análisis experimental exhaustivo:

RESULTADOS CLAVE DE EXPERIMENTACIÓN:
✓ 209 experimentos ejecutados
✓ 9 de 10 mejores configuraciones usan window=60 días
✓ Mejor configuración: batch_32_w60 (MAE: 4,311£)
✓ Arquitectura óptima: medium_large_shared
✓ Optimizer óptimo: RMSprop 0.001
✓ Loss weights óptimos: valor_agresivo (customer_value=3.0x)

CONFIGURACIÓN IMPLEMENTADA:
- Window: 60 días (óptimo para dataset de 375 días)
- Forecast: 7 días
- Batch size: 64 (balance precisión/velocidad para producción)
- Arquitectura: Shared LSTM [128] → Customer LSTM [32] + Product LSTM [32]
- Optimizer: RMSprop con learning rate 0.001
- Loss weights: customer_value=3.0, customer_count=0.3, customer_invoices=0.3, product_quantity=1.0

MÉTRICAS ESPERADAS:
- Customer Value MAE: ~4,490£
- Weighted MAE: ~3,283
- Overfitting Ratio: ~0.95 (excelente generalización)

Author: Sistema PFG LSTM - Update 15
Date: Noviembre 2025
Version: Optimized (experimental-based)
═══════════════════════════════════════════════════════════════════════════════
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# MLflow tracking
try:
    from mlflow_tracker import MLflowTracker, log_training_history
    MLFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ MLflow no disponible - tracking deshabilitado")
    MLFLOW_AVAILABLE = False

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
print(f"MLflow tracking: {'✓ Habilitado' if MLFLOW_AVAILABLE else '✗ Deshabilitado'}")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN OPTIMIZADA (basada en experimentación)
# ═══════════════════════════════════════════════════════════════════════════

class OptimizedConfig:
    """
    Configuración optimizada basada en 209 experimentos

    Hallazgos clave:
    - Window=60 días es óptimo (9/10 mejores configuraciones)
    - Batch=64 es mejor balance producción (batch=32 es más preciso pero 76% más lento)
    - Arquitectura medium_large_shared es óptima
    - RMSprop 0.001 ligeramente mejor que Adam
    - Loss weight agresivo en customer_value funciona mejor
    """

    # Configuración temporal óptima
    WINDOW_DAYS = 60  # Óptimo según experimentación
    FORECAST_DAYS = 7

    # Arquitectura óptima: medium_large_shared
    SHARED_LSTM_UNITS = [128]  # Una capa shared más grande
    CUSTOMER_LSTM_UNITS = [32]  # Customer branch pequeña
    PRODUCT_LSTM_UNITS = [32]   # Product branch pequeña

    # Hiperparámetros óptimos
    BATCH_SIZE = 64  # Balance precisión/velocidad (batch=32 es 3% mejor pero 76% más lento)
    EPOCHS = 100
    LEARNING_RATE = 0.001  # Óptimo para RMSprop
    OPTIMIZER = 'rmsprop'  # Ligeramente mejor que Adam

    # Loss weights óptimos: valor_agresivo
    LOSS_WEIGHTS = {
        'customer_value': 3.0,      # Prioridad máxima (3.0x)
        'customer_count': 0.3,       # Prioridad baja
        'customer_invoices': 0.3,    # Prioridad baja
        'product_quantity': 1.0      # Prioridad media
    }

    # Features
    CUSTOMER_FEATURES = 8
    PRODUCT_FEATURES = 2
    TOTAL_FEATURES = 10

    # Callbacks
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 8
    REDUCE_LR_FACTOR = 0.5
    MIN_LEARNING_RATE = 1e-7

    # Regularización
    DROPOUT_RATE = 0.2


# ═══════════════════════════════════════════════════════════════════════════
# MULTIMODAL LSTM OPTIMIZADO
# ═══════════════════════════════════════════════════════════════════════════

class OptimizedMultimodalLSTM:
    """
    Modelo multimodal LSTM con configuración óptima derivada experimentalmente
    """

    def __init__(self, data_path='data/processed/online_retail_2.xlsx',
                 output_dir='models/multimodal_optimized',
                 enable_mlflow=True):
        """
        Inicializa el modelo optimizado

        Args:
            data_path: ruta al dataset
            output_dir: carpeta para guardar modelo
            enable_mlflow: habilitar tracking con MLflow
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.model = None

        # MLflow tracker
        if MLFLOW_AVAILABLE and enable_mlflow:
            self.tracker = MLflowTracker(
                experiment_name="multimodal_lstm_optimized",
                enabled=True
            )
        else:
            self.tracker = None

        # Crear directorio
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"  MULTIMODAL LSTM OPTIMIZADO - Inicializado")
        print(f"{'='*80}")
        print(f"📊 Configuración basada en 209 experimentos")
        print(f"   Window: {OptimizedConfig.WINDOW_DAYS} días (óptimo)")
        print(f"   Batch size: {OptimizedConfig.BATCH_SIZE} (balance)")
        print(f"   Arquitectura: medium_large_shared")
        print(f"   Optimizer: {OptimizedConfig.OPTIMIZER.upper()}")
        print(f"   Loss weights: valor_agresivo (customer_value=3.0x)")
        print(f"\n💾 Output directory: {output_dir}")
        print(f"📡 MLflow tracking: {'✓ Habilitado' if self.tracker else '✗ Deshabilitado'}")

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 1: CARGA Y PREPROCESAMIENTO
    # ═══════════════════════════════════════════════════════════════════════

    def load_and_preprocess_data(self):
        """Carga y preprocesa el dataset"""
        print(f"\n{'─'*80}")
        print("FASE 1: Carga y Preprocesamiento")
        print(f"{'─'*80}")

        print(f"\n📊 Cargando: {self.data_path}")
        self.df = pd.read_excel(self.data_path, engine='openpyxl')
        print(f"   Registros: {len(self.df):,}")

        # Limpieza
        initial = len(self.df)
        self.df = self.df[self.df['CustomerID'].notna()]
        self.df = self.df[self.df['Quantity'] > 0]
        self.df = self.df[self.df['UnitPrice'] > 0]
        self.df = self.df[self.df['Description'].notna()]

        # Fechas y valor
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.df['Date'] = self.df['InvoiceDate'].dt.date
        self.df['TotalValue'] = self.df['Quantity'] * self.df['UnitPrice']
        self.df = self.df.sort_values(['CustomerID', 'InvoiceDate'])

        print(f"\n✅ Limpieza:")
        print(f"   Eliminados: {initial - len(self.df):,}")
        print(f"   Válidos: {len(self.df):,}")
        print(f"   Clientes: {self.df['CustomerID'].nunique():,}")
        print(f"   Productos: {self.df['StockCode'].nunique():,}")
        print(f"   Rango: {self.df['InvoiceDate'].min()} → {self.df['InvoiceDate'].max()}")

        return self.df

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 2: PREPARACIÓN DE SECUENCIAS
    # ═══════════════════════════════════════════════════════════════════════

    def prepare_sequences(self):
        """
        Prepara secuencias multimodales con configuración óptima
        Window=60 días, Forecast=7 días
        """
        print(f"\n{'─'*80}")
        print(f"FASE 2: Preparación de Secuencias Multimodales")
        print(f"{'─'*80}")
        print(f"⚙️  Window: {OptimizedConfig.WINDOW_DAYS} días")
        print(f"⚙️  Forecast: {OptimizedConfig.FORECAST_DAYS} días")

        # Crear series temporales
        print(f"\n🔨 Creando series temporales agregadas...")

        min_date = self.df['InvoiceDate'].min()
        max_date = self.df['InvoiceDate'].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        daily_data = pd.DataFrame({'Date': date_range})
        daily_data['Date_only'] = pd.to_datetime(daily_data['Date'].dt.date)

        # Métricas de clientes
        customer_daily = self.df.groupby('Date').agg({
            'TotalValue': ['sum', 'mean', 'std'],
            'CustomerID': 'nunique',
            'InvoiceNo': 'nunique',
            'Quantity': 'sum'
        }).reset_index()

        customer_daily.columns = ['Date', 'CustomerValue_sum', 'CustomerValue_mean',
                                 'CustomerValue_std', 'UniqueCustomers',
                                 'UniqueInvoices', 'TotalQuantity']
        customer_daily['Date'] = pd.to_datetime(customer_daily['Date'])

        # Métricas de productos (top 100)
        top_products = self.df.groupby('StockCode')['Quantity'].sum().nlargest(100).index
        product_df = self.df[self.df['StockCode'].isin(top_products)]

        product_daily = product_df.groupby('Date').agg({
            'Quantity': ['sum', 'mean'],
            'UnitPrice': 'mean',
            'StockCode': 'nunique'
        }).reset_index()

        product_daily.columns = ['Date', 'ProductQty_sum', 'ProductQty_mean',
                                'AvgPrice', 'UniqueProducts']
        product_daily['Date'] = pd.to_datetime(product_daily['Date'])

        # Merge
        daily_merged = daily_data.merge(customer_daily, left_on='Date_only',
                                       right_on='Date', how='left')
        daily_merged = daily_merged.merge(product_daily, left_on='Date_only',
                                         right_on='Date', how='left',
                                         suffixes=('', '_prod'))

        # Features finales (10)
        feature_cols = [
            'CustomerValue_sum', 'CustomerValue_mean', 'CustomerValue_std',
            'UniqueCustomers', 'UniqueInvoices', 'TotalQuantity',
            'ProductQty_sum', 'ProductQty_mean', 'AvgPrice', 'UniqueProducts'
        ]

        for col in feature_cols:
            daily_merged[col] = daily_merged[col].fillna(0)

        features = daily_merged[feature_cols].values

        print(f"   Features shape: {features.shape}")
        print(f"   Features: {len(feature_cols)} columnas")

        # Crear secuencias
        print(f"\n🔨 Generando secuencias...")

        X_sequences = []
        y_customer_value = []
        y_customer_count = []
        y_customer_invoices = []
        y_product_qty = []

        window = OptimizedConfig.WINDOW_DAYS
        forecast = OptimizedConfig.FORECAST_DAYS
        total_days = len(features)
        n_sequences = total_days - window - forecast + 1

        for i in range(n_sequences):
            X = features[i:i+window]
            future = features[i+window:i+window+forecast]

            # Targets
            y_customer_value.append(np.mean(future[:, 0]))
            y_customer_count.append(np.mean(future[:, 3]))
            y_customer_invoices.append(np.mean(future[:, 4]))
            y_product_qty.append(np.sum(future[:, 6]))

            X_sequences.append(X)

        # Arrays
        X = np.array(X_sequences)
        y_val = np.array(y_customer_value).reshape(-1, 1)
        y_cnt = np.array(y_customer_count).reshape(-1, 1)
        y_inv = np.array(y_customer_invoices).reshape(-1, 1)
        y_qty = np.array(y_product_qty).reshape(-1, 1)

        print(f"\n✅ Secuencias creadas:")
        print(f"   Total: {len(X):,}")
        print(f"   X shape: {X.shape}")
        print(f"   Y shapes: {y_val.shape}, {y_cnt.shape}, {y_inv.shape}, {y_qty.shape}")

        return X, y_val, y_cnt, y_inv, y_qty

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 3: CONSTRUCCIÓN DEL MODELO OPTIMIZADO
    # ═══════════════════════════════════════════════════════════════════════

    def build_optimized_model(self):
        """
        Construye modelo con arquitectura óptima: medium_large_shared

        Arquitectura:
        Input (60, 10) → Shared LSTM [128] → Branches:
          ├─ Customer LSTM [32] → 3 outputs (value, count, invoices)
          └─ Product LSTM [32] → 1 output (quantity)
        """
        print(f"\n{'─'*80}")
        print(f"FASE 3: Construcción del Modelo Optimizado")
        print(f"{'─'*80}")
        print(f"🏗️  Arquitectura: medium_large_shared")
        print(f"   Input: ({OptimizedConfig.WINDOW_DAYS}, {OptimizedConfig.TOTAL_FEATURES})")
        print(f"   Shared LSTM: {OptimizedConfig.SHARED_LSTM_UNITS}")
        print(f"   Customer LSTM: {OptimizedConfig.CUSTOMER_LSTM_UNITS}")
        print(f"   Product LSTM: {OptimizedConfig.PRODUCT_LSTM_UNITS}")
        print(f"   Optimizer: {OptimizedConfig.OPTIMIZER.upper()} (lr={OptimizedConfig.LEARNING_RATE})")
        print(f"   Loss weights: {OptimizedConfig.LOSS_WEIGHTS}")

        # Input
        inputs = keras.Input(
            shape=(OptimizedConfig.WINDOW_DAYS, OptimizedConfig.TOTAL_FEATURES),
            name='input'
        )

        # Shared LSTM layers
        x = inputs
        for i, units in enumerate(OptimizedConfig.SHARED_LSTM_UNITS):
            x = layers.LSTM(
                units,
                return_sequences=True,
                name=f'shared_lstm_{i+1}'
            )(x)
            x = layers.Dropout(OptimizedConfig.DROPOUT_RATE, name=f'shared_dropout_{i+1}')(x)

        shared_output = x

        # Customer Branch
        customer_x = shared_output
        for i, units in enumerate(OptimizedConfig.CUSTOMER_LSTM_UNITS):
            customer_x = layers.LSTM(
                units,
                return_sequences=False,
                name=f'customer_lstm_{i+1}'
            )(customer_x)
            customer_x = layers.Dropout(OptimizedConfig.DROPOUT_RATE, name=f'customer_dropout_{i+1}')(customer_x)

        customer_value_out = layers.Dense(1, activation='relu', name='customer_value')(customer_x)
        customer_count_out = layers.Dense(1, activation='relu', name='customer_count')(customer_x)
        customer_invoices_out = layers.Dense(1, activation='relu', name='customer_invoices')(customer_x)

        # Product Branch
        product_x = shared_output
        for i, units in enumerate(OptimizedConfig.PRODUCT_LSTM_UNITS):
            product_x = layers.LSTM(
                units,
                return_sequences=False,
                name=f'product_lstm_{i+1}'
            )(product_x)
            product_x = layers.Dropout(OptimizedConfig.DROPOUT_RATE, name=f'product_dropout_{i+1}')(product_x)

        product_qty_out = layers.Dense(1, activation='relu', name='product_quantity')(product_x)

        # Modelo
        model = Model(
            inputs=inputs,
            outputs=[customer_value_out, customer_count_out, customer_invoices_out, product_qty_out],
            name='multimodal_lstm_optimized'
        )

        # Optimizer óptimo: RMSprop
        if OptimizedConfig.OPTIMIZER == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=OptimizedConfig.LEARNING_RATE)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=OptimizedConfig.LEARNING_RATE)

        # Compile con loss weights óptimos
        model.compile(
            optimizer=optimizer,
            loss={
                'customer_value': 'mse',
                'customer_count': 'mse',
                'customer_invoices': 'mse',
                'product_quantity': 'mse'
            },
            loss_weights=OptimizedConfig.LOSS_WEIGHTS,
            metrics={
                'customer_value': ['mae'],
                'customer_count': ['mae'],
                'customer_invoices': ['mae'],
                'product_quantity': ['mae']
            }
        )

        print(f"\n✅ Modelo construido:")
        model.summary()

        self.model = model
        return model

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 4: ENTRENAMIENTO
    # ═══════════════════════════════════════════════════════════════════════

    def train(self):
        """Entrena el modelo optimizado"""
        print(f"\n{'='*80}")
        print(f"  ENTRENAMIENTO - Configuración Optimizada")
        print(f"{'='*80}")

        # Iniciar MLflow run
        if self.tracker:
            run_name = f"optimized_w60_batch64_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            params = {
                'model_type': 'multimodal_optimized',
                'window_days': OptimizedConfig.WINDOW_DAYS,
                'forecast_days': OptimizedConfig.FORECAST_DAYS,
                'batch_size': OptimizedConfig.BATCH_SIZE,
                'epochs': OptimizedConfig.EPOCHS,
                'learning_rate': OptimizedConfig.LEARNING_RATE,
                'optimizer': OptimizedConfig.OPTIMIZER,
                'shared_lstm': str(OptimizedConfig.SHARED_LSTM_UNITS),
                'customer_lstm': str(OptimizedConfig.CUSTOMER_LSTM_UNITS),
                'product_lstm': str(OptimizedConfig.PRODUCT_LSTM_UNITS),
                'loss_weight_customer_value': OptimizedConfig.LOSS_WEIGHTS['customer_value'],
                'loss_weight_customer_count': OptimizedConfig.LOSS_WEIGHTS['customer_count'],
                'loss_weight_customer_invoices': OptimizedConfig.LOSS_WEIGHTS['customer_invoices'],
                'loss_weight_product_quantity': OptimizedConfig.LOSS_WEIGHTS['product_quantity'],
                'based_on_experiments': 209
            }
            tags = {
                'model_family': 'multimodal_lstm_optimized',
                'experiment_based': 'true',
                'config': 'medium_large_shared',
                'team': 'pfg'
            }
            self.tracker.start_run(run_name=run_name, params=params, tags=tags)
            print(f"   MLflow run: {run_name}")

        # Preparar datos
        X, y_val, y_cnt, y_inv, y_qty = self.prepare_sequences()

        # Normalizar
        print(f"\n🔧 Normalizando...")
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        scaler_y_val = StandardScaler()
        y_val_scaled = scaler_y_val.fit_transform(y_val)

        scaler_y_cnt = StandardScaler()
        y_cnt_scaled = scaler_y_cnt.fit_transform(y_cnt)

        scaler_y_inv = StandardScaler()
        y_inv_scaled = scaler_y_inv.fit_transform(y_inv)

        scaler_y_qty = StandardScaler()
        y_qty_scaled = scaler_y_qty.fit_transform(y_qty)

        # Split train/val (80/20)
        indices = np.arange(len(X_scaled))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

        X_train = X_scaled[train_idx]
        X_val = X_scaled[val_idx]

        y_train = {
            'customer_value': y_val_scaled[train_idx],
            'customer_count': y_cnt_scaled[train_idx],
            'customer_invoices': y_inv_scaled[train_idx],
            'product_quantity': y_qty_scaled[train_idx]
        }

        y_val_dict = {
            'customer_value': y_val_scaled[val_idx],
            'customer_count': y_cnt_scaled[val_idx],
            'customer_invoices': y_inv_scaled[val_idx],
            'product_quantity': y_qty_scaled[val_idx]
        }

        print(f"   Train: {len(X_train):,}")
        print(f"   Val: {len(X_val):,}")

        # Construir modelo
        self.build_optimized_model()

        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=OptimizedConfig.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=OptimizedConfig.REDUCE_LR_FACTOR,
                patience=OptimizedConfig.REDUCE_LR_PATIENCE,
                min_lr=OptimizedConfig.MIN_LEARNING_RATE,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=f'{self.output_dir}/model_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Entrenar
        print(f"\n{'─'*80}")
        print(f"🚀 Iniciando entrenamiento...")
        print(f"{'─'*80}")

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val_dict),
            epochs=OptimizedConfig.EPOCHS,
            batch_size=OptimizedConfig.BATCH_SIZE,
            callbacks=callbacks_list,
            verbose=1
        )

        # Guardar
        print(f"\n💾 Guardando modelo y scalers...")
        self.model.save(f'{self.output_dir}/model_final.keras')

        with open(f'{self.output_dir}/scaler_X.pkl', 'wb') as f:
            pickle.dump(scaler_X, f)
        with open(f'{self.output_dir}/scaler_y_value.pkl', 'wb') as f:
            pickle.dump(scaler_y_val, f)
        with open(f'{self.output_dir}/scaler_y_count.pkl', 'wb') as f:
            pickle.dump(scaler_y_cnt, f)
        with open(f'{self.output_dir}/scaler_y_invoices.pkl', 'wb') as f:
            pickle.dump(scaler_y_inv, f)
        with open(f'{self.output_dir}/scaler_y_quantity.pkl', 'wb') as f:
            pickle.dump(scaler_y_qty, f)

        # Evaluar
        print(f"\n{'─'*80}")
        print(f"📊 Evaluación Final")
        print(f"{'─'*80}")

        predictions = self.model.predict(X_val)

        # Desnormalizar
        pred_val = scaler_y_val.inverse_transform(predictions[0])
        true_val = scaler_y_val.inverse_transform(y_val_dict['customer_value'])

        pred_cnt = scaler_y_cnt.inverse_transform(predictions[1])
        true_cnt = scaler_y_cnt.inverse_transform(y_val_dict['customer_count'])

        pred_inv = scaler_y_inv.inverse_transform(predictions[2])
        true_inv = scaler_y_inv.inverse_transform(y_val_dict['customer_invoices'])

        pred_qty = scaler_y_qty.inverse_transform(predictions[3])
        true_qty = scaler_y_qty.inverse_transform(y_val_dict['product_quantity'])

        # Métricas
        mae_val = mean_absolute_error(true_val, pred_val)
        mae_cnt = mean_absolute_error(true_cnt, pred_cnt)
        mae_inv = mean_absolute_error(true_inv, pred_inv)
        mae_qty = mean_absolute_error(true_qty, pred_qty)

        rmse_val = np.sqrt(mean_squared_error(true_val, pred_val))
        rmse_cnt = np.sqrt(mean_squared_error(true_cnt, pred_cnt))
        rmse_inv = np.sqrt(mean_squared_error(true_inv, pred_inv))
        rmse_qty = np.sqrt(mean_squared_error(true_qty, pred_qty))

        # Calcular weighted MAE (50% value, 20% count, 15% invoices, 15% quantity)
        weighted_mae = (0.50 * mae_val + 0.20 * mae_cnt +
                       0.15 * mae_inv + 0.15 * mae_qty)

        # Calcular overfitting ratio
        train_predictions = self.model.predict(X_train)
        train_mae_val = mean_absolute_error(
            scaler_y_val.inverse_transform(y_train['customer_value']),
            scaler_y_val.inverse_transform(train_predictions[0])
        )
        overfitting_ratio = mae_val / train_mae_val if train_mae_val > 0 else 1.0

        metrics = {
            'config': 'optimized_based_on_209_experiments',
            'window_days': OptimizedConfig.WINDOW_DAYS,
            'forecast_days': OptimizedConfig.FORECAST_DAYS,
            'batch_size': OptimizedConfig.BATCH_SIZE,
            'optimizer': OptimizedConfig.OPTIMIZER,
            'epochs_trained': len(history.history['loss']),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'customer_value_mae': float(mae_val),
            'customer_value_rmse': float(rmse_val),
            'customer_count_mae': float(mae_cnt),
            'customer_count_rmse': float(rmse_cnt),
            'customer_invoices_mae': float(mae_inv),
            'customer_invoices_rmse': float(rmse_inv),
            'product_quantity_mae': float(mae_qty),
            'product_quantity_rmse': float(rmse_qty),
            'weighted_mae': float(weighted_mae),
            'overfitting_ratio': float(overfitting_ratio),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'expected_customer_value_mae': '~4490',
            'expected_weighted_mae': '~3283'
        }

        print(f"\n✅ MÉTRICAS FINALES:")
        print(f"{'─'*80}")
        print(f"   Customer Value MAE:  {mae_val:,.2f} £  (esperado: ~4,490£)")
        print(f"   Customer Count MAE:  {mae_cnt:.2f}")
        print(f"   Customer Invoices MAE: {mae_inv:.2f}")
        print(f"   Product Quantity MAE: {mae_qty:.2f}")
        print(f"{'─'*80}")
        print(f"   Weighted MAE:        {weighted_mae:.2f}  (esperado: ~3,283)")
        print(f"   Overfitting Ratio:   {overfitting_ratio:.3f}  (esperado: ~0.95)")
        print(f"{'─'*80}")

        # Guardar métricas
        with open(f'{self.output_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n💾 Métricas guardadas en: {self.output_dir}/metrics.json")

        # Log a MLflow
        if self.tracker:
            numeric_metrics = {
                'window_days': OptimizedConfig.WINDOW_DAYS,
                'forecast_days': OptimizedConfig.FORECAST_DAYS,
                'batch_size': OptimizedConfig.BATCH_SIZE,
                'epochs_trained': len(history.history['loss']),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'customer_value_mae': float(mae_val),
                'customer_value_rmse': float(rmse_val),
                'customer_count_mae': float(mae_cnt),
                'customer_count_rmse': float(rmse_cnt),
                'customer_invoices_mae': float(mae_inv),
                'customer_invoices_rmse': float(rmse_inv),
                'product_quantity_mae': float(mae_qty),
                'product_quantity_rmse': float(rmse_qty),
                'weighted_mae': float(weighted_mae),
                'overfitting_ratio': float(overfitting_ratio),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1])
            }

            self.tracker.log_metrics(numeric_metrics)
            log_training_history(self.tracker, history.history)
            self.tracker.log_artifact(f'{self.output_dir}/metrics.json', 'metrics')
            self.tracker.log_model(
                self.model,
                artifact_path='model',
                registered_model_name='multimodal_lstm_optimized'
            )
            self.tracker.end_run()
            print(f"   ✓ Registrado en MLflow")

        return self.model, history, metrics


# ═══════════════════════════════════════════════════════════════════════════
# MAIN - EJECUCIÓN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*80)
    print("  MULTIMODAL LSTM OPTIMIZADO")
    print("  Basado en 209 Experimentos - Configuración Óptima")
    print("="*80)
    print("\n📊 CONFIGURACIÓN:")
    print(f"   Window: {OptimizedConfig.WINDOW_DAYS} días")
    print(f"   Batch size: {OptimizedConfig.BATCH_SIZE}")
    print(f"   Arquitectura: medium_large_shared")
    print(f"   Optimizer: {OptimizedConfig.OPTIMIZER.upper()}")
    print(f"   Loss weights: valor_agresivo (customer_value=3.0x)")

    # Inicializar
    model_trainer = OptimizedMultimodalLSTM(
        data_path='data/processed/online_retail_2.xlsx',
        output_dir='models/multimodal_optimized'
    )

    # Cargar datos
    model_trainer.load_and_preprocess_data()

    # Entrenar
    model, history, metrics = model_trainer.train()

    print("\n" + "="*80)
    print("  ✅ ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"\n📁 Modelo guardado en: models/multimodal_optimized/")
    print(f"   ├─ model_final.keras")
    print(f"   ├─ model_best.keras")
    print(f"   ├─ scaler_*.pkl (5 scalers)")
    print(f"   └─ metrics.json")
    print("\n🎉 ¡Modelo optimizado listo para producción!")
