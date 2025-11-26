"""
═══════════════════════════════════════════════════════════════════════════════
GRID SEARCH EXPERIMENT - Multimodal LSTM (Multi-Dimensional Analysis)
═══════════════════════════════════════════════════════════════════════════════

Script para realizar experimentación sistemática con diferentes configuraciones
del modelo multimodal LSTM y registrar todos los resultados en MLflow.

NUEVO: Análisis Multi-dimensional
Cada parámetro (arquitectura, optimizer, batch_size, loss_weights) se prueba
en MÚLTIPLES ventanas temporales (15, 30, 60, 120, 240, 360 días) para
identificar interacciones entre hiperparámetros y contexto temporal.

Experimentos:
1. Arquitecturas LSTM (8 configs × 6 windows = 48 experimentos)
2. Loss Weights (6 configs × 6 windows = 36 experimentos)
3. Ventanas Temporales (11 configs = 11 experimentos)
4. Optimizers (9 configs × 6 windows = 54 experimentos)
5. Batch Sizes (5 configs × 6 windows = 30 experimentos)

Total: 179 experimentos (~3-4 horas)

Uso:
    python experiment_multimodal_grid_search.py --experiment arquitecturas
    python experiment_multimodal_grid_search.py --experiment loss_weights
    python experiment_multimodal_grid_search.py --experiment temporal
    python experiment_multimodal_grid_search.py --experiment optimizers
    python experiment_multimodal_grid_search.py --experiment batch_sizes
    python experiment_multimodal_grid_search.py --experiment all

Author: Sistema PFG LSTM
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
import argparse
from datetime import datetime
from itertools import product
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
    print("[!] MLflow no disponible - tracking deshabilitado")
    MLFLOW_AVAILABLE = False
    MLflowTracker = None

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
print(f"MLflow tracking: {'[OK] Habilitado' if MLFLOW_AVAILABLE else '[X] Deshabilitado'}")


# ═══════════════════════════════════════════════════════════════════════════
# GRIDS DE EXPERIMENTACIÓN
# ═══════════════════════════════════════════════════════════════════════════

# Window sizes para análisis multi-dimensional
WINDOW_SIZES = [15, 30, 60, 120, 240, 360]

EXPERIMENT_GRIDS = {
    # 1. Experimentar con diferentes arquitecturas LSTM
    'arquitecturas': {
        'name': 'Arquitecturas LSTM',
        'test_multiple_windows': True,  # NUEVO: Testear en múltiples ventanas
        'base_config': {
            'forecast_days': 7,
            'epochs': 30,  # Menos epochs para iterar más rápido
            'batch_size': 128,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss_weights': {'customer_value': 1.0, 'customer_count': 0.5,
                            'customer_invoices': 0.5, 'product_quantity': 1.0}
        },
        'variations': [
            # Shallow networks
            {
                'name': 'shallow_small',
                'shared_lstm_units': [64],
                'customer_lstm_units': [32],
                'product_lstm_units': [32]
            },
            {
                'name': 'shallow_medium',
                'shared_lstm_units': [128],
                'customer_lstm_units': [32],
                'product_lstm_units': [32]
            },
            # Medium networks (baseline)
            {
                'name': 'medium_baseline',
                'shared_lstm_units': [128, 64],
                'customer_lstm_units': [32],
                'product_lstm_units': [32]
            },
            {
                'name': 'medium_large_shared',
                'shared_lstm_units': [256, 128],
                'customer_lstm_units': [32],
                'product_lstm_units': [32]
            },
            # Deep networks
            {
                'name': 'deep_small',
                'shared_lstm_units': [128, 64, 32],
                'customer_lstm_units': [32],
                'product_lstm_units': [32]
            },
            {
                'name': 'deep_large',
                'shared_lstm_units': [256, 128, 64],
                'customer_lstm_units': [64, 32],
                'product_lstm_units': [64, 32]
            },
            # Very deep
            {
                'name': 'very_deep',
                'shared_lstm_units': [256, 128, 64, 32],
                'customer_lstm_units': [64, 32],
                'product_lstm_units': [64, 32]
            },
            # Asymmetric (más capacidad en customer branch)
            {
                'name': 'asymmetric_customer',
                'shared_lstm_units': [128, 64],
                'customer_lstm_units': [64, 32],
                'product_lstm_units': [16]
            }
        ]
    },

    # 2. Experimentar con loss weights
    'loss_weights': {
        'name': 'Loss Weights',
        'test_multiple_windows': True,  # NUEVO: Testear en múltiples ventanas
        'base_config': {
            'forecast_days': 7,
            'shared_lstm_units': [128, 64],
            'customer_lstm_units': [32],
            'product_lstm_units': [32],
            'epochs': 30,
            'batch_size': 128,
            'optimizer': 'adam',
            'learning_rate': 0.001
        },
        'variations': [
            # Balanceado (baseline)
            {
                'name': 'balanceado',
                'loss_weights': {
                    'customer_value': 1.0,
                    'customer_count': 0.5,
                    'customer_invoices': 0.5,
                    'product_quantity': 1.0
                }
            },
            # Enfocado en cliente
            {
                'name': 'prioridad_cliente',
                'loss_weights': {
                    'customer_value': 2.0,
                    'customer_count': 1.0,
                    'customer_invoices': 1.0,
                    'product_quantity': 0.5
                }
            },
            # Enfocado en producto
            {
                'name': 'prioridad_producto',
                'loss_weights': {
                    'customer_value': 0.5,
                    'customer_count': 0.3,
                    'customer_invoices': 0.3,
                    'product_quantity': 2.0
                }
            },
            # Pesos iguales
            {
                'name': 'equitativo',
                'loss_weights': {
                    'customer_value': 1.0,
                    'customer_count': 1.0,
                    'customer_invoices': 1.0,
                    'product_quantity': 1.0
                }
            },
            # Solo KPIs principales
            {
                'name': 'solo_kpis_principales',
                'loss_weights': {
                    'customer_value': 1.0,
                    'customer_count': 0.1,
                    'customer_invoices': 0.1,
                    'product_quantity': 1.0
                }
            },
            # Enfoque agresivo en valor
            {
                'name': 'valor_agresivo',
                'loss_weights': {
                    'customer_value': 3.0,
                    'customer_count': 0.3,
                    'customer_invoices': 0.3,
                    'product_quantity': 1.0
                }
            }
        ]
    },

    # 3. Experimentar con ventanas temporales
    'temporal': {
        'name': 'Ventanas Temporales',
        'test_multiple_windows': False,  # Ya varía ventanas explícitamente
        'base_config': {
            'shared_lstm_units': [128, 64],
            'customer_lstm_units': [32],
            'product_lstm_units': [32],
            'epochs': 30,
            'batch_size': 128,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss_weights': {'customer_value': 1.0, 'customer_count': 0.5,
                            'customer_invoices': 0.5, 'product_quantity': 1.0}
        },
        'variations': [
            # Ventanas temporales con forecast de 7 días
            {'name': 'w15d', 'window_days': 15, 'forecast_days': 7},   # 15 días de contexto
            {'name': 'w30d', 'window_days': 30, 'forecast_days': 7},   # 30 días (1 mes)
            {'name': 'w120d', 'window_days': 120, 'forecast_days': 7}, # 120 días (4 meses)
            {'name': 'w240d', 'window_days': 240, 'forecast_days': 7}, # 240 días (8 meses)
            {'name': 'w360d', 'window_days': 360, 'forecast_days': 7}  # 360 días (ciclo anual completo)
        ]
    },

    # 4. Experimentar con optimizers
    'optimizers': {
        'name': 'Optimizers',
        'test_multiple_windows': True,  # NUEVO: Testear en múltiples ventanas
        'base_config': {
            'forecast_days': 7,
            'shared_lstm_units': [128, 64],
            'customer_lstm_units': [32],
            'product_lstm_units': [32],
            'epochs': 30,
            'batch_size': 128,
            'loss_weights': {'customer_value': 1.0, 'customer_count': 0.5,
                            'customer_invoices': 0.5, 'product_quantity': 1.0}
        },
        'variations': [
            # Adam con diferentes learning rates
            {'name': 'adam_0.001', 'optimizer': 'adam', 'learning_rate': 0.001},
            {'name': 'adam_0.0001', 'optimizer': 'adam', 'learning_rate': 0.0001},
            {'name': 'adam_0.01', 'optimizer': 'adam', 'learning_rate': 0.01},

            # RMSprop
            {'name': 'rmsprop_0.001', 'optimizer': 'rmsprop', 'learning_rate': 0.001},
            {'name': 'rmsprop_0.0001', 'optimizer': 'rmsprop', 'learning_rate': 0.0001},

            # SGD con momentum
            {'name': 'sgd_momentum_0.01', 'optimizer': 'sgd', 'learning_rate': 0.01, 'momentum': 0.9},
            {'name': 'sgd_momentum_0.001', 'optimizer': 'sgd', 'learning_rate': 0.001, 'momentum': 0.9},

            # Nadam (Adam + Nesterov)
            {'name': 'nadam_0.001', 'optimizer': 'nadam', 'learning_rate': 0.001},

            # AdamW (Adam with weight decay)
            {'name': 'adamw_0.001', 'optimizer': 'adamw', 'learning_rate': 0.001}
        ]
    },

    # 5. Experimentar con batch sizes
    'batch_sizes': {
        'name': 'Batch Sizes',
        'test_multiple_windows': True,  # NUEVO: Testear en múltiples ventanas
        'base_config': {
            'forecast_days': 7,
            'shared_lstm_units': [128, 64],
            'customer_lstm_units': [32],
            'product_lstm_units': [32],
            'epochs': 30,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss_weights': {'customer_value': 1.0, 'customer_count': 0.5,
                            'customer_invoices': 0.5, 'product_quantity': 1.0}
        },
        'variations': [
            {'name': 'batch_32', 'batch_size': 32},
            {'name': 'batch_64', 'batch_size': 64},
            {'name': 'batch_128', 'batch_size': 128},
            {'name': 'batch_256', 'batch_size': 256},
            {'name': 'batch_512', 'batch_size': 512}
        ]
    }
}


# ═══════════════════════════════════════════════════════════════════════════
# CLASE DE EXPERIMENTACIÓN
# ═══════════════════════════════════════════════════════════════════════════

class MultimodalExperimenter:
    """
    Gestiona la experimentación sistemática del modelo multimodal
    """

    def __init__(self, data_path='data/processed/online_retail_2.xlsx',
                 output_dir='models/experiments/multimodal'):
        """
        Inicializa el experimentador

        Args:
            data_path: ruta al dataset
            output_dir: carpeta para guardar resultados
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None

        # Crear directorios
        os.makedirs(output_dir, exist_ok=True)

        # MLflow tracker (se creara dinamicamente por categoria)
        self.tracker = None
        self.current_experiment_type = None

        print(f"\n{'='*80}")
        print(f"  MULTIMODAL EXPERIMENTER - Inicializado")
        print(f"{'='*80}")
        print(f"Data: {data_path}")
        print(f"Output: {output_dir}")
        print(f"MLflow: {'[OK] Habilitado' if self.tracker else '[X] Deshabilitado'}")

    def load_and_preprocess_data(self):
        """Carga y preprocesa el dataset (una sola vez)"""
        print(f"\n{'-'*80}")
        print("Cargando y preprocesando datos...")
        print(f"{'-'*80}")

        self.df = pd.read_excel(self.data_path, engine='openpyxl')

        # Limpieza
        self.df = self.df[self.df['CustomerID'].notna()]
        self.df = self.df[self.df['Quantity'] > 0]
        self.df = self.df[self.df['UnitPrice'] > 0]
        self.df = self.df[self.df['Description'].notna()]

        # Convertir fecha
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.df['Date'] = self.df['InvoiceDate'].dt.date
        self.df['TotalValue'] = self.df['Quantity'] * self.df['UnitPrice']

        # Ordenar
        self.df = self.df.sort_values(['CustomerID', 'InvoiceDate'])

        print(f"[OK] Datos cargados: {len(self.df):,} registros")
        print(f"   Clientes: {self.df['CustomerID'].nunique():,}")
        print(f"   Productos: {self.df['StockCode'].nunique():,}")

    def prepare_sequences(self, window_days, forecast_days):
        """Prepara secuencias para un horizonte específico"""

        # Rango completo de fechas
        min_date = self.df['InvoiceDate'].min()
        max_date = self.df['InvoiceDate'].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        # DataFrame con todas las fechas
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

        # Métricas de productos
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

        # Features finales
        feature_cols = [
            'CustomerValue_sum', 'CustomerValue_mean', 'CustomerValue_std',
            'UniqueCustomers', 'UniqueInvoices', 'TotalQuantity',
            'ProductQty_sum', 'ProductQty_mean', 'AvgPrice', 'UniqueProducts'
        ]

        for col in feature_cols:
            daily_merged[col] = daily_merged[col].fillna(0)

        features = daily_merged[feature_cols].values

        # Crear secuencias
        X_sequences = []
        y_customer_value = []
        y_customer_count = []
        y_customer_invoices = []
        y_product_qty = []

        total_days = len(features)
        n_sequences = total_days - window_days - forecast_days + 1

        for i in range(n_sequences):
            X = features[i:i+window_days]
            future = features[i+window_days:i+window_days+forecast_days]

            X_sequences.append(X)
            y_customer_value.append(np.mean(future[:, 0]))
            y_customer_count.append(np.mean(future[:, 3]))
            y_customer_invoices.append(np.mean(future[:, 4]))
            y_product_qty.append(np.sum(future[:, 6]))

        X = np.array(X_sequences)
        y_val = np.array(y_customer_value).reshape(-1, 1)
        y_cnt = np.array(y_customer_count).reshape(-1, 1)
        y_inv = np.array(y_customer_invoices).reshape(-1, 1)
        y_qty = np.array(y_product_qty).reshape(-1, 1)

        return X, y_val, y_cnt, y_inv, y_qty

    def build_model(self, config):
        """Construye modelo con configuración específica"""

        window_days = config['window_days']
        shared_lstm = config['shared_lstm_units']
        customer_lstm = config['customer_lstm_units']
        product_lstm = config['product_lstm_units']

        # Input layer
        inputs = keras.Input(shape=(window_days, 10), name='multimodal_input')

        # Shared LSTM layers
        x = inputs
        for i, units in enumerate(shared_lstm):
            x = layers.LSTM(units, return_sequences=True, name=f'shared_lstm_{i+1}')(x)
            x = layers.Dropout(0.2, name=f'shared_dropout_{i+1}')(x)

        # Customer branch
        customer_x = x
        for i, units in enumerate(customer_lstm):
            # Only last LSTM layer should have return_sequences=False
            return_seq = (i < len(customer_lstm) - 1)
            customer_x = layers.LSTM(units, return_sequences=return_seq,
                                    name=f'customer_lstm_{i+1}')(customer_x)
            customer_x = layers.Dropout(0.2, name=f'customer_dropout_{i+1}')(customer_x)

        customer_value_out = layers.Dense(1, activation='relu', name='customer_value')(customer_x)
        customer_count_out = layers.Dense(1, activation='relu', name='customer_count')(customer_x)
        customer_invoices_out = layers.Dense(1, activation='relu', name='customer_invoices')(customer_x)

        # Product branch
        product_x = x
        for i, units in enumerate(product_lstm):
            # Only last LSTM layer should have return_sequences=False
            return_seq = (i < len(product_lstm) - 1)
            product_x = layers.LSTM(units, return_sequences=return_seq,
                                   name=f'product_lstm_{i+1}')(product_x)
            product_x = layers.Dropout(0.2, name=f'product_dropout_{i+1}')(product_x)

        product_qty_out = layers.Dense(1, activation='relu', name='product_quantity')(product_x)

        # Model
        model = Model(
            inputs=inputs,
            outputs=[customer_value_out, customer_count_out,
                    customer_invoices_out, product_qty_out]
        )

        # Optimizer
        optimizer_name = config.get('optimizer', 'adam')
        lr = config.get('learning_rate', 0.001)

        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_name == 'adamw':
            optimizer = keras.optimizers.AdamW(learning_rate=lr)
        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        elif optimizer_name == 'sgd':
            momentum = config.get('momentum', 0.0)
            optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
        elif optimizer_name == 'nadam':
            optimizer = keras.optimizers.Nadam(learning_rate=lr)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=lr)

        # Compile
        loss_weights = config.get('loss_weights', {
            'customer_value': 1.0, 'customer_count': 0.5,
            'customer_invoices': 0.5, 'product_quantity': 1.0
        })

        model.compile(
            optimizer=optimizer,
            loss={'customer_value': 'mse', 'customer_count': 'mse',
                  'customer_invoices': 'mse', 'product_quantity': 'mse'},
            loss_weights=loss_weights,
            metrics={'customer_value': ['mae'], 'customer_count': ['mae'],
                    'customer_invoices': ['mae'], 'product_quantity': ['mae']}
        )

        return model

    def run_single_experiment(self, config, experiment_name):
        """Ejecuta un experimento individual"""

        print(f"\n{'-'*80}")
        print(f"Experimento: {experiment_name}")
        print(f"{'-'*80}")

        # Iniciar MLflow run
        if self.tracker:
            run_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            params = {
                'experiment_name': experiment_name,
                'window_days': config['window_days'],
                'forecast_days': config['forecast_days'],
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
                'optimizer': config.get('optimizer', 'adam'),
                'learning_rate': config.get('learning_rate', 0.001),
                'shared_lstm_units': str(config['shared_lstm_units']),
                'customer_lstm_units': str(config['customer_lstm_units']),
                'product_lstm_units': str(config['product_lstm_units']),
                'loss_weight_customer_value': config['loss_weights']['customer_value'],
                'loss_weight_customer_count': config['loss_weights']['customer_count'],
                'loss_weight_customer_invoices': config['loss_weights']['customer_invoices'],
                'loss_weight_product_quantity': config['loss_weights']['product_quantity']
            }
            tags = {'experiment_type': config.get('experiment_type', 'custom')}
            self.tracker.start_run(run_name=run_name, params=params, tags=tags)

        try:
            # Preparar datos
            X, y_val, y_cnt, y_inv, y_qty = self.prepare_sequences(
                config['window_days'], config['forecast_days']
            )

            # Normalizar
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

            # Split
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

            # Construir modelo
            model = self.build_model(config)

            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                       restore_best_weights=True, verbose=0),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=5, min_lr=1e-7, verbose=0)
            ]

            # Entrenar
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val_dict),
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                callbacks=callbacks_list,
                verbose=0
            )

            # Evaluar
            predictions = model.predict(X_val, verbose=0)

            pred_val = scaler_y_val.inverse_transform(predictions[0])
            true_val = scaler_y_val.inverse_transform(y_val_dict['customer_value'])

            pred_cnt = scaler_y_cnt.inverse_transform(predictions[1])
            true_cnt = scaler_y_cnt.inverse_transform(y_val_dict['customer_count'])

            pred_inv = scaler_y_inv.inverse_transform(predictions[2])
            true_inv = scaler_y_inv.inverse_transform(y_val_dict['customer_invoices'])

            pred_qty = scaler_y_qty.inverse_transform(predictions[3])
            true_qty = scaler_y_qty.inverse_transform(y_val_dict['product_quantity'])

            # Métricas
            metrics = {
                'customer_value_mae': float(mean_absolute_error(true_val, pred_val)),
                'customer_value_rmse': float(np.sqrt(mean_squared_error(true_val, pred_val))),
                'customer_count_mae': float(mean_absolute_error(true_cnt, pred_cnt)),
                'customer_count_rmse': float(np.sqrt(mean_squared_error(true_cnt, pred_cnt))),
                'customer_invoices_mae': float(mean_absolute_error(true_inv, pred_inv)),
                'customer_invoices_rmse': float(np.sqrt(mean_squared_error(true_inv, pred_inv))),
                'product_quantity_mae': float(mean_absolute_error(true_qty, pred_qty)),
                'product_quantity_rmse': float(np.sqrt(mean_squared_error(true_qty, pred_qty))),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'epochs_trained': len(history.history['loss']),
                'total_params': model.count_params()
            }

            print(f"[OK] MAE - Value: {metrics['customer_value_mae']:.2f} | "
                  f"Count: {metrics['customer_count_mae']:.2f} | "
                  f"Qty: {metrics['product_quantity_mae']:.2f}")

            # Log a MLflow
            if self.tracker:
                self.tracker.log_metrics(metrics)
                log_training_history(self.tracker, history.history)

            return metrics

        except Exception as e:
            print(f"[ERROR] Error en experimento {experiment_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            # Asegurar que el run de MLflow se cierra siempre
            if self.tracker:
                try:
                    self.tracker.end_run()
                except:
                    pass

    def set_experiment_category(self, experiment_type):
        """Configura el experimento de MLflow según la categoría"""
        if MLFLOW_AVAILABLE and MLflowTracker is not None:
            # Cerrar tracker anterior si existe
            if self.tracker:
                try:
                    self.tracker.end_run()
                except:
                    pass

            # Crear nuevo tracker con nombre de experimento por categoría
            experiment_name = f"LSTM_multimodal_exp_{experiment_type}"
            self.tracker = MLflowTracker(
                experiment_name=experiment_name,
                enabled=True
            )
            self.current_experiment_type = experiment_type
            print(f"[OK] MLflow experiment: {experiment_name}")
        else:
            self.tracker = None

    def run_experiment_grid(self, experiment_type):
        """Ejecuta todos los experimentos de un grid"""

        if experiment_type not in EXPERIMENT_GRIDS:
            print(f"[ERROR] Tipo de experimento '{experiment_type}' no existe")
            return

        # Configurar experimento de MLflow para esta categoría
        self.set_experiment_category(experiment_type)

        grid = EXPERIMENT_GRIDS[experiment_type]
        test_multiple_windows = grid.get('test_multiple_windows', False)

        # Calcular total de experimentos
        if test_multiple_windows:
            total_experiments = len(grid['variations']) * len(WINDOW_SIZES)
            print(f"\n{'='*80}")
            print(f"  GRID SEARCH: {grid['name']} (Multi-dimensional)")
            print(f"{'='*80}")
            print(f"Variaciones: {len(grid['variations'])}")
            print(f"Window sizes: {len(WINDOW_SIZES)} ({WINDOW_SIZES})")
            print(f"Total experimentos: {total_experiments}")
        else:
            total_experiments = len(grid['variations'])
            print(f"\n{'='*80}")
            print(f"  GRID SEARCH: {grid['name']}")
            print(f"{'='*80}")
            print(f"Total experimentos: {total_experiments}")

        results = []
        exp_counter = 0

        for variation in grid['variations']:
            if test_multiple_windows:
                # Testear esta variación en cada window size
                for window_size in WINDOW_SIZES:
                    exp_counter += 1

                    # Combinar base config con variation y window_size
                    config = {**grid['base_config'], **variation}
                    config['window_days'] = window_size
                    config['experiment_type'] = experiment_type

                    # Nombre único que incluye window size
                    exp_name = f"{variation['name']}_w{window_size}"

                    print(f"\n[{exp_counter}/{total_experiments}] {exp_name}")

                    try:
                        metrics = self.run_single_experiment(config, exp_name)
                        results.append({
                            'name': exp_name,
                            'config': config,
                            'metrics': metrics
                        })
                    except Exception as e:
                        print(f"[ERROR] Error: {e}")
                        continue
            else:
                # Testear solo una vez (comportamiento original)
                exp_counter += 1

                # Combinar base config con variation
                config = {**grid['base_config'], **variation}
                config['experiment_type'] = experiment_type

                print(f"\n[{exp_counter}/{total_experiments}] {variation['name']}")

                try:
                    metrics = self.run_single_experiment(config, variation['name'])
                    results.append({
                        'name': variation['name'],
                        'config': config,
                        'metrics': metrics
                    })
                except Exception as e:
                    print(f"[ERROR] Error: {e}")
                    continue

        # Resumen
        print(f"\n{'='*80}")
        print(f"  RESUMEN - {grid['name']}")
        print(f"{'='*80}")

        # Ordenar por customer_value_mae
        results.sort(key=lambda x: x['metrics']['customer_value_mae'])

        print(f"\n🏆 Top 5 Mejores (por Customer Value MAE):")
        print(f"{'-'*80}")
        for i, result in enumerate(results[:5], 1):
            m = result['metrics']
            print(f"{i}. {result['name']}")
            print(f"   Customer Value MAE: {m['customer_value_mae']:.2f}")
            print(f"   Customer Count MAE: {m['customer_count_mae']:.2f}")
            print(f"   Product Qty MAE: {m['product_quantity_mae']:.2f}")
            print(f"   Val Loss: {m['final_val_loss']:.6f}")

        # Guardar resultados
        results_file = f"{self.output_dir}/results_{experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Resultados guardados en: {results_file}")

        return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN - EJECUCIÓN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Grid Search para Multimodal LSTM')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['arquitecturas', 'loss_weights', 'temporal',
                               'optimizers', 'batch_sizes', 'all'],
                       help='Tipo de experimento a ejecutar')
    parser.add_argument('--data', type=str,
                       default='data/processed/online_retail_2.xlsx',
                       help='Ruta al dataset')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("  MULTIMODAL LSTM - GRID SEARCH EXPERIMENT")
    print("="*80)

    # Inicializar experimenter
    experimenter = MultimodalExperimenter(data_path=args.data)

    # Cargar datos
    experimenter.load_and_preprocess_data()

    # Ejecutar experimentos
    if args.experiment == 'all':
        for exp_type in ['arquitecturas', 'loss_weights', 'temporal',
                        'optimizers', 'batch_sizes']:
            experimenter.run_experiment_grid(exp_type)
    else:
        experimenter.run_experiment_grid(args.experiment)

    print("\n" + "="*80)
    print("  [COMPLETE] EXPERIMENTACION COMPLETADA")
    print("="*80)
    print("\n[INFO] Para ver resultados en MLflow:")
    print("   mlflow ui --port 5000")
    print("   http://localhost:5000")


if __name__ == '__main__':
    main()
