"""
===============================================================================
ANALIZADOR MULTIMODAL OPTIMIZADO
===============================================================================

Utiliza el modelo multimodal LSTM optimizado basado en 209 experimentos
para generar predicciones AGREGADAS de negocio:

- Customer Value: Valor total de ventas esperado
- Customer Count: Cantidad de clientes únicos esperados
- Customer Invoices: Cantidad de facturas esperadas
- Product Quantity: Cantidad total de productos esperados

Author: Sistema PFG LSTM
Date: Noviembre 2025
Version: Optimized Dashboard
===============================================================================
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Fix encoding para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# TensorFlow
import tensorflow as tf
from tensorflow import keras

print("=" * 80)
print("ANALIZADOR MULTIMODAL OPTIMIZADO")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")


# ===========================================================================
# CONFIGURACIÓN ÓPTIMA (basada en 209 experimentos)
# ===========================================================================

class OptimizedModelConfig:
    """
    Configuración óptima derivada de 209 experimentos de grid search.

    RESULTADOS CLAVE DE EXPERIMENTACIÓN:
    - 209 experimentos ejecutados
    - 9 de 10 mejores configuraciones usan window=60 días
    - Mejor configuración: batch_32_w60 (MAE: 4,311£)
    - Arquitectura óptima: medium_large_shared
    - Optimizer óptimo: RMSprop 0.001
    - Loss weights óptimos: valor_agresivo (customer_value=3.0x)
    """

    # Paths
    MODEL_PATH = 'models/multimodal_optimized'
    DATA_PATH = 'data/processed/online_retail_2.xlsx'

    # Configuración temporal óptima
    WINDOW_DAYS = 60  # Óptimo según experimentación (9/10 mejores)
    FORECAST_DAYS = 7

    # Arquitectura óptima: medium_large_shared
    ARCHITECTURE = {
        'name': 'medium_large_shared',
        'shared_lstm': [128],
        'customer_lstm': [32],
        'product_lstm': [32]
    }

    # Hiperparámetros óptimos
    HYPERPARAMETERS = {
        'batch_size': 64,  # Balance precisión/velocidad
        'learning_rate': 0.001,
        'optimizer': 'rmsprop',
        'epochs': 100,
        'early_stopping_patience': 15
    }

    # Loss weights óptimos: valor_agresivo
    LOSS_WEIGHTS = {
        'customer_value': 3.0,      # Prioridad máxima
        'customer_count': 0.3,
        'customer_invoices': 0.3,
        'product_quantity': 1.0
    }

    # Features (10 total)
    FEATURES = [
        'CustomerValue_sum', 'CustomerValue_mean', 'CustomerValue_std',
        'UniqueCustomers', 'UniqueInvoices', 'TotalQuantity',
        'ProductQty_sum', 'ProductQty_mean', 'AvgPrice', 'UniqueProducts'
    ]

    # Métricas esperadas del modelo
    EXPECTED_METRICS = {
        'customer_value_mae': '~4,490 £',
        'weighted_mae': '~3,283',
        'overfitting_ratio': '~0.95'
    }

    @classmethod
    def get_config_summary(cls):
        """Retorna resumen de configuración para mostrar en dashboard"""
        return {
            'experiments_count': 209,
            'window_days': cls.WINDOW_DAYS,
            'forecast_days': cls.FORECAST_DAYS,
            'architecture': cls.ARCHITECTURE['name'],
            'shared_lstm_units': cls.ARCHITECTURE['shared_lstm'],
            'customer_lstm_units': cls.ARCHITECTURE['customer_lstm'],
            'product_lstm_units': cls.ARCHITECTURE['product_lstm'],
            'optimizer': cls.HYPERPARAMETERS['optimizer'],
            'learning_rate': cls.HYPERPARAMETERS['learning_rate'],
            'batch_size': cls.HYPERPARAMETERS['batch_size'],
            'loss_weights': cls.LOSS_WEIGHTS,
            'expected_metrics': cls.EXPECTED_METRICS,
            'key_findings': [
                '9/10 mejores configuraciones usan window=60 días',
                'RMSprop ligeramente mejor que Adam',
                'Loss weight agresivo en customer_value mejora resultados',
                'Arquitectura medium_large_shared es óptima'
            ]
        }


# ===========================================================================
# ANALIZADOR PRINCIPAL
# ===========================================================================

class OptimizedMultimodalAnalyzer:
    """
    Analizador que usa el modelo multimodal optimizado para predicciones agregadas.
    """

    def __init__(self, config=None):
        self.config = config or OptimizedModelConfig()
        self.df = None
        self.model = None
        self.scalers = {}
        self.daily_features = None
        self.predictions = None
        self.model_metrics = None

    def load_data(self):
        """Cargar y preprocesar dataset"""
        print("\n" + "-" * 80)
        print("[DATA] Cargando dataset...")
        print("-" * 80)

        self.df = pd.read_excel(self.config.DATA_PATH, engine='openpyxl')

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

        print(f"[OK] Dataset cargado: {len(self.df):,} transacciones válidas")
        print(f"   Eliminados: {initial - len(self.df):,}")
        print(f"   Clientes únicos: {self.df['CustomerID'].nunique():,}")
        print(f"   Productos únicos: {self.df['StockCode'].nunique():,}")
        print(f"   Período: {self.df['InvoiceDate'].min().date()} a {self.df['InvoiceDate'].max().date()}")

        return self.df

    def load_model(self):
        """Cargar modelo multimodal optimizado"""
        print("\n" + "-" * 80)
        print("[MODEL] Cargando modelo multimodal optimizado...")
        print("-" * 80)

        model_path = self.config.MODEL_PATH

        # Verificar si existe el modelo
        model_file = f'{model_path}/model_best.keras'
        if not os.path.exists(model_file):
            model_file = f'{model_path}/model_final.keras'

        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Modelo no encontrado en {model_path}. "
                f"Ejecuta train_multimodal_lstm_optimized.py primero."
            )

        # Cargar modelo
        self.model = keras.models.load_model(model_file)

        # Cargar scalers
        scaler_files = {
            'X': 'scaler_X.pkl',
            'y_value': 'scaler_y_value.pkl',
            'y_count': 'scaler_y_count.pkl',
            'y_invoices': 'scaler_y_invoices.pkl',
            'y_quantity': 'scaler_y_quantity.pkl'
        }

        for name, filename in scaler_files.items():
            filepath = f'{model_path}/{filename}'
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.scalers[name] = pickle.load(f)

        # Cargar métricas del modelo
        metrics_file = f'{model_path}/metrics.json'
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                self.model_metrics = json.load(f)

        print(f"[OK] Modelo cargado: {model_file}")
        print(f"   Scalers: {len(self.scalers)}")

        if self.model_metrics:
            print(f"\n[METRICS] Métricas del modelo entrenado:")
            print(f"   Customer Value MAE: {self.model_metrics.get('customer_value_mae', 'N/A'):,.2f} £")
            print(f"   Customer Count MAE: {self.model_metrics.get('customer_count_mae', 'N/A'):.2f}")
            print(f"   Product Quantity MAE: {self.model_metrics.get('product_quantity_mae', 'N/A'):.2f}")
            print(f"   Weighted MAE: {self.model_metrics.get('weighted_mae', 'N/A'):,.2f}")
            print(f"   Overfitting Ratio: {self.model_metrics.get('overfitting_ratio', 'N/A'):.3f}")

        return self.model

    def prepare_daily_features(self):
        """Preparar features diarias agregadas"""
        print("\n" + "-" * 80)
        print("[FEATURES] Preparando features diarias...")
        print("-" * 80)

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

        # Rellenar NaN
        feature_cols = self.config.FEATURES
        for col in feature_cols:
            if col in daily_merged.columns:
                daily_merged[col] = daily_merged[col].fillna(0)

        self.daily_features = daily_merged

        print(f"[OK] Features diarias preparadas: {len(daily_merged)} días")
        print(f"   Features: {len(feature_cols)}")

        return self.daily_features

    def predict_next_period(self):
        """Generar predicciones para el próximo período"""
        print("\n" + "-" * 80)
        print(f"[PREDICT] Prediciendo próximos {self.config.FORECAST_DAYS} días...")
        print("-" * 80)

        if self.daily_features is None:
            self.prepare_daily_features()

        # Obtener últimos WINDOW_DAYS días como input
        feature_cols = self.config.FEATURES
        features = self.daily_features[feature_cols].values

        # Tomar últimos 60 días
        window = self.config.WINDOW_DAYS
        if len(features) < window:
            raise ValueError(f"No hay suficientes datos. Se necesitan {window} días, hay {len(features)}")

        X_last = features[-window:]

        # Normalizar
        if 'X' in self.scalers:
            X_scaled = self.scalers['X'].transform(X_last)
        else:
            X_scaled = X_last

        X_input = X_scaled.reshape(1, window, len(feature_cols))

        # Predecir
        predictions_raw = self.model.predict(X_input, verbose=0)

        # Desnormalizar
        pred_value = predictions_raw[0][0, 0]
        pred_count = predictions_raw[1][0, 0]
        pred_invoices = predictions_raw[2][0, 0]
        pred_quantity = predictions_raw[3][0, 0]

        if 'y_value' in self.scalers:
            pred_value = self.scalers['y_value'].inverse_transform([[pred_value]])[0, 0]
        if 'y_count' in self.scalers:
            pred_count = self.scalers['y_count'].inverse_transform([[pred_count]])[0, 0]
        if 'y_invoices' in self.scalers:
            pred_invoices = self.scalers['y_invoices'].inverse_transform([[pred_invoices]])[0, 0]
        if 'y_quantity' in self.scalers:
            pred_quantity = self.scalers['y_quantity'].inverse_transform([[pred_quantity]])[0, 0]

        # Calcular estadísticas históricas para comparación
        last_7_days = self.daily_features.tail(7)
        historical = {
            'avg_daily_value': last_7_days['CustomerValue_sum'].mean(),
            'avg_daily_customers': last_7_days['UniqueCustomers'].mean(),
            'avg_daily_invoices': last_7_days['UniqueInvoices'].mean(),
            'avg_daily_quantity': last_7_days['TotalQuantity'].mean()
        }

        # Guardar predicciones
        self.predictions = {
            'forecast_period': f'Próximos {self.config.FORECAST_DAYS} días',
            'generated_at': datetime.now().isoformat(),
            'predictions': {
                'customer_value': {
                    'predicted': float(max(0, pred_value)),
                    'historical_avg': float(historical['avg_daily_value']),
                    'unit': '£',
                    'description': 'Valor total de ventas esperado (promedio diario)'
                },
                'customer_count': {
                    'predicted': float(max(0, pred_count)),
                    'historical_avg': float(historical['avg_daily_customers']),
                    'unit': 'clientes',
                    'description': 'Clientes únicos esperados (promedio diario)'
                },
                'customer_invoices': {
                    'predicted': float(max(0, pred_invoices)),
                    'historical_avg': float(historical['avg_daily_invoices']),
                    'unit': 'facturas',
                    'description': 'Facturas esperadas (promedio diario)'
                },
                'product_quantity': {
                    'predicted': float(max(0, pred_quantity)),
                    'historical_avg': float(historical['avg_daily_quantity']),
                    'unit': 'unidades',
                    'description': 'Productos vendidos esperados (total período)'
                }
            },
            'totals_7_days': {
                'revenue': float(max(0, pred_value) * self.config.FORECAST_DAYS),
                'customers': float(max(0, pred_count) * self.config.FORECAST_DAYS),
                'invoices': float(max(0, pred_invoices) * self.config.FORECAST_DAYS),
                'products': float(max(0, pred_quantity))
            }
        }

        print(f"\n[OK] Predicciones generadas:")
        print(f"{'─'*60}")
        print(f"   {'Métrica':<25} {'Predicción':>15} {'Histórico':>15}")
        print(f"{'─'*60}")
        for key, data in self.predictions['predictions'].items():
            print(f"   {key:<25} {data['predicted']:>12,.1f} {data['historical_avg']:>12,.1f} {data['unit']}")
        print(f"{'─'*60}")
        print(f"\n   TOTALES 7 DÍAS:")
        print(f"   Revenue esperado: £{self.predictions['totals_7_days']['revenue']:,.2f}")
        print(f"   Clientes esperados: {self.predictions['totals_7_days']['customers']:,.0f}")
        print(f"   Productos esperados: {self.predictions['totals_7_days']['products']:,.0f}")

        return self.predictions

    def get_historical_trends(self, days=30):
        """Obtener tendencias históricas para gráficos"""
        if self.daily_features is None:
            self.prepare_daily_features()

        last_n = self.daily_features.tail(days).copy()
        last_n['Date_str'] = last_n['Date_only'].dt.strftime('%Y-%m-%d')

        trends = {
            'dates': last_n['Date_str'].tolist(),
            'customer_value': last_n['CustomerValue_sum'].tolist(),
            'customer_count': last_n['UniqueCustomers'].tolist(),
            'invoices': last_n['UniqueInvoices'].tolist(),
            'quantity': last_n['TotalQuantity'].tolist()
        }

        return trends

    def get_model_info(self):
        """Obtener información completa del modelo para mostrar en dashboard"""
        config_summary = self.config.get_config_summary()

        info = {
            'model_type': 'Multimodal LSTM Optimizado',
            'based_on': '209 experimentos de grid search',
            'config': config_summary,
            'trained_metrics': self.model_metrics or {},
            'status': 'loaded' if self.model is not None else 'not_loaded'
        }

        return info

    def export_predictions(self, output_dir='output/optimized_predictions'):
        """Exportar predicciones"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Exportar predicciones
        if self.predictions:
            filepath = f'{output_dir}/predictions_{timestamp}.json'
            with open(filepath, 'w') as f:
                json.dump(self.predictions, f, indent=2)
            print(f"[OK] Predicciones exportadas: {filepath}")

        # Exportar info del modelo
        model_info = self.get_model_info()
        filepath = f'{output_dir}/model_info_{timestamp}.json'
        with open(filepath, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        print(f"[OK] Info del modelo exportada: {filepath}")


# ===========================================================================
# FUNCIÓN PRINCIPAL
# ===========================================================================

def main():
    """Ejecutar análisis con modelo optimizado"""
    print("\n" + "=" * 80)
    print("ANÁLISIS CON MODELO MULTIMODAL OPTIMIZADO")
    print("=" * 80)

    analyzer = OptimizedMultimodalAnalyzer()

    # 1. Cargar datos
    analyzer.load_data()

    # 2. Cargar modelo
    analyzer.load_model()

    # 3. Generar predicciones
    analyzer.predict_next_period()

    # 4. Mostrar info del modelo
    print("\n" + "=" * 80)
    print("CONFIGURACIÓN ÓPTIMA UTILIZADA")
    print("=" * 80)
    config = OptimizedModelConfig.get_config_summary()
    print(f"\n   Experimentos: {config['experiments_count']}")
    print(f"   Window: {config['window_days']} días")
    print(f"   Arquitectura: {config['architecture']}")
    print(f"   Optimizer: {config['optimizer']} (lr={config['learning_rate']})")
    print(f"   Loss weights: {config['loss_weights']}")
    print(f"\n   Hallazgos clave:")
    for finding in config['key_findings']:
        print(f"   • {finding}")

    # 5. Exportar
    analyzer.export_predictions()

    print("\n" + "=" * 80)
    print("[OK] ANÁLISIS COMPLETADO")
    print("=" * 80)

    return analyzer


if __name__ == '__main__':
    analyzer = main()
