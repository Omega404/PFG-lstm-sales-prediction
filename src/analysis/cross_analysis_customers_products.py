"""
===============================================================================
ANÁLISIS CRUZADO: PREDICCIÓN DE CLIENTES + PRODUCTOS
===============================================================================

Combina predicciones de:
1. Modelos LSTM de clientes (V2/V3): Quién compra, cuándo, cuánto
2. Modelos LSTM de productos: Qué productos tendrán demanda

Genera insights de negocio:
- Recomendaciones de productos por cliente
- Forecast de inventario
- Oportunidades de venta cruzada
- Estrategias de marketing personalizadas

Author: Sistema PFG LSTM
Date: 2025-01-05
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
print("ANALISIS CRUZADO: CLIENTES + PRODUCTOS")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")
print(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ===========================================================================
# CONFIGURACIÓN
# ===========================================================================

class CrossAnalysisConfig:
    """Configuración del análisis cruzado"""

    # Paths de modelos
    CUSTOMER_MODEL_V3 = 'models/temporal/customer_v3/medium'  # 7 días
    CUSTOMER_MODEL_V2 = 'models/temporal/customer_v2/medium'  # 14 días
    PRODUCT_MODEL = 'models/temporal/products_50epochs/short'  # 7 días

    # Dataset
    DATA_PATH = 'data/processed/online_retail_2.xlsx'

    # Configuración de análisis
    FORECAST_DAYS = 7  # Horizonte de predicción
    TOP_N_PRODUCTS = 20  # Top productos a considerar
    MIN_PURCHASE_PROBABILITY = 0.70  # 70% mínimo de probabilidad
    HIGH_VALUE_THRESHOLD = 100  # $100+ = cliente alto valor

    # Segmentos de clientes
    SEGMENTS = {
        'high_value_high_prob': {
            'min_prob': 0.80,
            'min_value': 100,
            'priority': 1,
            'strategy': 'Oferta premium + productos de alto valor'
        },
        'high_prob_medium_value': {
            'min_prob': 0.70,
            'min_value': 50,
            'max_value': 100,
            'priority': 2,
            'strategy': 'Oferta estándar + productos populares'
        },
        'medium_prob': {
            'min_prob': 0.50,
            'max_prob': 0.70,
            'priority': 3,
            'strategy': 'Campaña general + descuentos'
        },
        'low_prob': {
            'max_prob': 0.50,
            'priority': 4,
            'strategy': 'No contactar (bajo ROI)'
        }
    }


# ===========================================================================
# CLASE PRINCIPAL
# ===========================================================================

class CustomerProductCrossAnalyzer:
    """Analizador cruzado de predicciones de clientes y productos"""

    def __init__(self, config=None):
        self.config = config or CrossAnalysisConfig()
        self.df = None
        self.customer_model = None
        self.product_model = None
        self.customer_predictions = None
        self.product_predictions = None

    def load_data(self):
        """Cargar dataset"""
        print("\n" + "-" * 80)
        print("[DATA] Cargando dataset...")
        print("-" * 80)

        self.df = pd.read_excel(self.config.DATA_PATH)

        # Limpiar datos
        self.df = self.df[self.df['CustomerID'].notna()].copy()
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.df['TotalPrice'] = self.df['Quantity'] * self.df['UnitPrice']

        print(f"[OK] Dataset cargado: {len(self.df):,} transacciones")
        print(f"   Clientes únicos: {self.df['CustomerID'].nunique():,}")
        print(f"   Productos únicos: {self.df['StockCode'].nunique():,}")
        print(f"   Período: {self.df['InvoiceDate'].min()} a {self.df['InvoiceDate'].max()}")

    def load_customer_model(self, use_v3=True):
        """Cargar modelo de predicción de clientes"""
        print("\n" + "-" * 80)
        print("[MODEL] Cargando modelo de clientes...")
        print("-" * 80)

        model_path = self.config.CUSTOMER_MODEL_V3 if use_v3 else self.config.CUSTOMER_MODEL_V2
        model_version = "V3 (7 días)" if use_v3 else "V2 (14 días)"

        # Cargar modelo
        model_file = f'{model_path}/model_best.keras'
        self.customer_model = keras.models.load_model(model_file)

        # Cargar métricas
        with open(f'{model_path}/metrics.json', 'r') as f:
            metrics = json.load(f)

        # Cargar scalers
        with open(f'{model_path}/scaler_X.pkl', 'rb') as f:
            self.customer_scaler_X = pickle.load(f)
        with open(f'{model_path}/scaler_y_days.pkl', 'rb') as f:
            self.customer_scaler_days = pickle.load(f)
        with open(f'{model_path}/scaler_y_value.pkl', 'rb') as f:
            self.customer_scaler_value = pickle.load(f)

        print(f"[OK] Modelo de clientes cargado: {model_version}")
        print(f"   Accuracy: {metrics.get('purchase_prob_accuracy', 0):.2f}%")
        print(f"   AUC: {metrics.get('purchase_prob_auc', 0):.4f}")
        print(f"   Days MAE: {metrics.get('days_mae', 0):.2f} días")
        print(f"   Value MAE: ${metrics.get('value_mae', 0):.2f}")

    def load_product_model(self):
        """Cargar modelo de predicción de productos"""
        print("\n" + "-" * 80)
        print("[MODEL] Cargando modelo de productos...")
        print("-" * 80)

        model_path = self.config.PRODUCT_MODEL

        # Cargar modelo
        model_file = f'{model_path}/model_best.keras'
        self.product_model = keras.models.load_model(model_file)

        # Cargar métricas
        with open(f'{model_path}/metrics.json', 'r') as f:
            metrics = json.load(f)

        # Cargar scaler
        with open(f'{model_path}/scaler_X.pkl', 'rb') as f:
            self.product_scaler_X = pickle.load(f)
        with open(f'{model_path}/scaler_y.pkl', 'rb') as f:
            self.product_scaler_y = pickle.load(f)

        print(f"[OK] Modelo de productos cargado")
        print(f"   MAE: {metrics.get('mae', 0):.2f} unidades")
        print(f"   RMSE: {metrics.get('rmse', 0):.2f}")

    def create_customer_sequences(self, customer_ids, window_days=120):
        """Crear secuencias temporales de 120 días para cada cliente"""
        sequences = []
        max_date = self.df['InvoiceDate'].max()
        start_date = max_date - timedelta(days=window_days)

        # Crear rango de fechas (120 días exactos)
        date_range = pd.date_range(start=start_date, periods=window_days, freq='D')

        for customer_id in customer_ids:
            # Datos del cliente en ventana de 120 días
            customer_df = self.df[
                (self.df['CustomerID'] == customer_id) &
                (self.df['InvoiceDate'] >= start_date)
            ].copy()

            if len(customer_df) == 0:
                # Cliente sin actividad: secuencia de ceros
                sequences.append(np.zeros((window_days, 8)))
                continue

            # Agregar fecha sin hora para agrupar por día
            customer_df['Date'] = customer_df['InvoiceDate'].dt.date

            # Crear features diarias
            daily_features = []
            cumulative_purchases = 0
            cumulative_spent = 0
            cumulative_quantity = 0
            products_seen = set()

            for date in date_range:
                date_only = date.date()
                day_data = customer_df[customer_df['Date'] == date_only]

                # Features del día
                if len(day_data) > 0:
                    day_purchases = len(day_data)
                    day_spent = day_data['TotalPrice'].sum()
                    day_quantity = day_data['Quantity'].sum()
                    day_products = day_data['StockCode'].nunique()

                    cumulative_purchases += day_purchases
                    cumulative_spent += day_spent
                    cumulative_quantity += day_quantity
                    products_seen.update(day_data['StockCode'].unique())

                    days_since_last = 0
                else:
                    day_purchases = 0
                    day_spent = 0
                    day_quantity = 0
                    day_products = 0

                    # Días desde última compra
                    last_purchase = customer_df[customer_df['Date'] < date_only]
                    if len(last_purchase) > 0:
                        days_since_last = (date_only - last_purchase['Date'].max()).days
                    else:
                        days_since_last = window_days

                # Calcular promedios
                avg_basket = cumulative_spent / cumulative_purchases if cumulative_purchases > 0 else 0
                avg_quantity = cumulative_quantity / cumulative_purchases if cumulative_purchases > 0 else 0
                recency_score = 1 - (days_since_last / window_days)

                # Vector de features (8 features)
                daily_features.append([
                    cumulative_purchases,      # n_purchases acumuladas
                    cumulative_spent,          # total_spent acumulado
                    avg_basket,                # avg_basket
                    cumulative_quantity,       # total_quantity acumulado
                    avg_quantity,              # avg_quantity
                    len(products_seen),        # unique_products
                    days_since_last,           # days_since_last
                    recency_score              # recency_score
                ])

            sequences.append(np.array(daily_features))

        return np.array(sequences)

    def predict_customers(self, sample_size=None):
        """Predecir qué clientes comprarán"""
        print("\n" + "-" * 80)
        print("[PREDICT] Prediciendo clientes...")
        print("-" * 80)

        # Obtener clientes activos en últimos 120 días
        cutoff_date = self.df['InvoiceDate'].max() - timedelta(days=120)
        active_customers = self.df[self.df['InvoiceDate'] >= cutoff_date]['CustomerID'].unique()

        if sample_size:
            active_customers = np.random.choice(active_customers,
                                               min(sample_size, len(active_customers)),
                                               replace=False)

        print(f"   Analizando {len(active_customers):,} clientes...")
        print(f"   Creando secuencias de 120 días...")

        # Crear secuencias temporales
        X_sequences = self.create_customer_sequences(active_customers, window_days=120)

        # Normalizar cada secuencia
        X_sequences_scaled = np.zeros_like(X_sequences)
        for i in range(len(X_sequences)):
            X_sequences_scaled[i] = self.customer_scaler_X.transform(X_sequences[i])

        print(f"   Prediciendo con modelo LSTM...")

        # Predecir
        predictions = self.customer_model.predict(X_sequences_scaled, verbose=0)

        # Extraer outputs
        purchase_prob = predictions[0].flatten()
        days_scaled = predictions[1].flatten()
        value_scaled = predictions[2].flatten()

        # Desnormalizar
        days_predicted = self.customer_scaler_days.inverse_transform(
            days_scaled.reshape(-1, 1)
        ).flatten()
        value_predicted = self.customer_scaler_value.inverse_transform(
            value_scaled.reshape(-1, 1)
        ).flatten()

        # Crear DataFrame de resultados
        # Obtener datos históricos de cada cliente para el resultado
        customer_history = []
        max_date = self.df['InvoiceDate'].max()
        for customer_id in active_customers:
            cust_df = self.df[self.df['CustomerID'] == customer_id]
            last_purchase = cust_df['InvoiceDate'].max()
            days_since = (max_date - last_purchase).days
            total_spent = cust_df['TotalPrice'].sum()
            n_purchases = len(cust_df)
            customer_history.append((customer_id, days_since, total_spent, n_purchases))

        customer_history_df = pd.DataFrame(customer_history,
                                           columns=['customer_id', 'days_since_last',
                                                   'total_spent_historical', 'n_purchases_historical'])

        self.customer_predictions = pd.DataFrame({
            'customer_id': active_customers,
            'purchase_probability': purchase_prob * 100,  # Convertir a %
            'predicted_days': np.clip(days_predicted, 0, 30),
            'predicted_value': np.clip(value_predicted, 0, 10000)
        })

        # Agregar datos históricos
        self.customer_predictions = self.customer_predictions.merge(
            customer_history_df, on='customer_id', how='left'
        )

        # Clasificar en segmentos
        self.customer_predictions['segment'] = self.customer_predictions.apply(
            self._classify_customer, axis=1
        )

        print(f"[OK] Predicciones completadas para {len(self.customer_predictions):,} clientes")
        print(f"\n[DATA] Distribución de segmentos:")
        print(self.customer_predictions['segment'].value_counts())

    def _classify_customer(self, row):
        """Clasificar cliente en segmento"""
        prob = row['purchase_probability']
        value = row['predicted_value']

        if prob >= 80 and value >= 100:
            return 'high_value_high_prob'
        elif prob >= 70 and 50 <= value < 100:
            return 'high_prob_medium_value'
        elif 50 <= prob < 70:
            return 'medium_prob'
        else:
            return 'low_prob'

    def predict_products(self, top_n=None):
        """Predecir demanda de productos"""
        print("\n" + "-" * 80)
        print("[PREDICT] Prediciendo productos...")
        print("-" * 80)

        top_n = top_n or self.config.TOP_N_PRODUCTS

        # Obtener top productos por volumen
        product_stats = self.df.groupby('StockCode').agg({
            'Quantity': 'sum',
            'TotalPrice': 'sum',
            'Description': 'first',
            'UnitPrice': 'mean'
        }).reset_index()

        product_stats = product_stats.sort_values('Quantity', ascending=False).head(top_n)

        print(f"   Analizando top {len(product_stats)} productos...")

        # Por simplicidad, calcular demanda promedio histórica
        # En producción, usar el modelo LSTM de productos con secuencias temporales

        self.product_predictions = pd.DataFrame({
            'stock_code': product_stats['StockCode'].values,
            'description': product_stats['Description'].values,
            'historical_quantity': product_stats['Quantity'].values,
            'historical_revenue': product_stats['TotalPrice'].values,
            'avg_price': product_stats['UnitPrice'].values,
            'predicted_demand_7d': product_stats['Quantity'].values * 0.15  # ~15% de demanda histórica para 7 días
        })

        print(f"[OK] Predicciones de productos completadas")
        print(f"   Top 5 productos por demanda predicha:")
        top5 = self.product_predictions.nlargest(5, 'predicted_demand_7d')
        for idx, row in top5.iterrows():
            print(f"   - {row['description'][:50]}: {row['predicted_demand_7d']:.0f} unidades")

    def generate_cross_recommendations(self):
        """Generar recomendaciones cruzadas cliente-producto"""
        print("\n" + "-" * 80)
        print("[TARGET] Generando recomendaciones cruzadas...")
        print("-" * 80)

        if self.customer_predictions is None or self.product_predictions is None:
            print("[ERROR] Faltan predicciones. Ejecuta predict_customers() y predict_products() primero.")
            return

        # Filtrar clientes con alta probabilidad
        high_prob_customers = self.customer_predictions[
            self.customer_predictions['purchase_probability'] >= self.config.MIN_PURCHASE_PROBABILITY
        ].copy()

        print(f"   Clientes de alta probabilidad (≥70%): {len(high_prob_customers):,}")

        # Para cada cliente, recomendar productos basados en:
        # 1. Historial de compras del cliente
        # 2. Productos de alta demanda predicha
        # 3. Productos en rango de precio del cliente

        recommendations = []

        for idx, customer in high_prob_customers.iterrows():
            customer_id = customer['customer_id']
            predicted_value = customer['predicted_value']
            segment = customer['segment']

            # Obtener historial del cliente
            customer_history = self.df[self.df['CustomerID'] == customer_id].copy()

            # Productos comprados antes
            past_products = set(customer_history['StockCode'].unique())

            # Rango de precio típico del cliente
            if len(customer_history) > 0:
                avg_price_range = customer_history['UnitPrice'].quantile([0.25, 0.75]).values
                min_price, max_price = avg_price_range
            else:
                min_price, max_price = 0, predicted_value * 0.3

            # Filtrar productos recomendables
            # 1. Alta demanda predicha
            # 2. En rango de precio del cliente
            # 3. Preferiblemente que no haya comprado (para cross-sell)

            suitable_products = self.product_predictions[
                (self.product_predictions['avg_price'] >= min_price) &
                (self.product_predictions['avg_price'] <= max_price)
            ].copy()

            # Priorizar productos no comprados (cross-sell)
            suitable_products['is_new'] = ~suitable_products['stock_code'].isin(past_products)
            suitable_products['score'] = (
                suitable_products['predicted_demand_7d'] * 0.6 +
                suitable_products['is_new'].astype(int) * suitable_products['predicted_demand_7d'] * 0.4
            )

            # Top 5 productos recomendados
            top_recommendations = suitable_products.nlargest(5, 'score')

            for _, product in top_recommendations.iterrows():
                recommendations.append({
                    'customer_id': customer_id,
                    'purchase_probability': customer['purchase_probability'],
                    'predicted_value': predicted_value,
                    'predicted_days': customer['predicted_days'],
                    'segment': segment,
                    'stock_code': product['stock_code'],
                    'product_description': product['description'],
                    'product_price': product['avg_price'],
                    'predicted_demand': product['predicted_demand_7d'],
                    'is_new_product': product['is_new'],
                    'recommendation_score': product['score']
                })

        self.recommendations = pd.DataFrame(recommendations)

        print(f"[OK] Recomendaciones generadas: {len(self.recommendations):,}")

        # Agrupar por segmento
        print(f"\n[DATA] Recomendaciones por segmento:")
        segment_counts = self.recommendations.groupby('segment').size()
        for segment, count in segment_counts.items():
            strategy = self.config.SEGMENTS.get(segment, {}).get('strategy', 'N/A')
            print(f"   {segment}: {count:,} recomendaciones")
            print(f"      Estrategia: {strategy}")

    def generate_inventory_forecast(self):
        """Generar forecast de inventario necesario"""
        print("\n" + "-" * 80)
        print("[INVENTORY] Generando forecast de inventario...")
        print("-" * 80)

        if self.recommendations is None:
            print("[ERROR] Genera recomendaciones primero con generate_cross_recommendations()")
            return

        # Agregar demanda por producto
        inventory_forecast = self.recommendations.groupby([
            'stock_code', 'product_description', 'product_price'
        ]).agg({
            'customer_id': 'count',  # Número de clientes que podrían comprar
            'predicted_demand': 'first',  # Demanda predicha del modelo
            'purchase_probability': 'mean'  # Probabilidad promedio
        }).reset_index()

        inventory_forecast.columns = [
            'stock_code', 'description', 'price',
            'n_potential_customers', 'model_demand', 'avg_probability'
        ]

        # Estimar unidades necesarias
        # Asumiendo que cada cliente compra 1-3 unidades (promedio 2)
        inventory_forecast['estimated_units_needed'] = (
            inventory_forecast['n_potential_customers'] *
            (inventory_forecast['avg_probability'] / 100) *
            2  # Promedio de unidades por cliente
        ).astype(int)

        # Agregar demanda del modelo
        inventory_forecast['total_forecast'] = (
            inventory_forecast['estimated_units_needed'] +
            inventory_forecast['model_demand'] * 0.5  # 50% de confianza en modelo de productos
        ).astype(int)

        # Ordenar por demanda total
        inventory_forecast = inventory_forecast.sort_values('total_forecast', ascending=False)

        self.inventory_forecast = inventory_forecast

        print(f"[OK] Forecast de inventario completado")
        print(f"\n[DATA] Top 10 productos a reabastecer:")
        print(inventory_forecast.head(10)[['description', 'total_forecast', 'n_potential_customers']].to_string(index=False))

        return inventory_forecast

    def export_results(self, output_dir='output/cross_analysis'):
        """Exportar resultados"""
        print("\n" + "-" * 80)
        print("[SAVE] Exportando resultados...")
        print("-" * 80)

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Exportar predicciones de clientes
        if self.customer_predictions is not None:
            file_customers = f'{output_dir}/customer_predictions_{timestamp}.csv'
            self.customer_predictions.to_csv(file_customers, index=False)
            print(f"[OK] {file_customers}")

        # Exportar predicciones de productos
        if self.product_predictions is not None:
            file_products = f'{output_dir}/product_predictions_{timestamp}.csv'
            self.product_predictions.to_csv(file_products, index=False)
            print(f"[OK] {file_products}")

        # Exportar recomendaciones
        if self.recommendations is not None:
            file_recs = f'{output_dir}/recommendations_{timestamp}.csv'
            self.recommendations.to_csv(file_recs, index=False)
            print(f"[OK] {file_recs}")

        # Exportar forecast de inventario
        if self.inventory_forecast is not None:
            file_inventory = f'{output_dir}/inventory_forecast_{timestamp}.csv'
            self.inventory_forecast.to_csv(file_inventory, index=False)
            print(f"[OK] {file_inventory}")

        # Exportar resumen JSON
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'forecast_horizon_days': self.config.FORECAST_DAYS,
            'total_customers_analyzed': len(self.customer_predictions) if self.customer_predictions is not None else 0,
            'high_probability_customers': len(self.customer_predictions[
                self.customer_predictions['purchase_probability'] >= 70
            ]) if self.customer_predictions is not None else 0,
            'total_products_analyzed': len(self.product_predictions) if self.product_predictions is not None else 0,
            'total_recommendations': len(self.recommendations) if self.recommendations is not None else 0,
            'segments': self.recommendations['segment'].value_counts().to_dict() if self.recommendations is not None else {}
        }

        file_summary = f'{output_dir}/summary_{timestamp}.json'
        with open(file_summary, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] {file_summary}")

        print(f"\n[FOLDER] Todos los archivos exportados a: {output_dir}/")


# ===========================================================================
# FUNCIÓN PRINCIPAL
# ===========================================================================

def main():
    """Ejecutar análisis cruzado completo"""

    print("\n" + "=" * 80)
    print("INICIANDO ANÁLISIS CRUZADO COMPLETO")
    print("=" * 80)

    # Crear analizador
    analyzer = CustomerProductCrossAnalyzer()

    # 1. Cargar datos
    analyzer.load_data()

    # 2. Cargar modelos
    analyzer.load_customer_model(use_v3=True)  # Usar V3 (7 días)
    analyzer.load_product_model()

    # 3. Predecir clientes
    analyzer.predict_customers(sample_size=100)  # Analizar 100 clientes (sample para test)

    # 4. Predecir productos
    analyzer.predict_products(top_n=50)

    # 5. Generar recomendaciones cruzadas
    analyzer.generate_cross_recommendations()

    # 6. Generar forecast de inventario
    analyzer.generate_inventory_forecast()

    # 7. Exportar resultados
    analyzer.export_results()

    print("\n" + "=" * 80)
    print("[OK] ANÁLISIS CRUZADO COMPLETADO")
    print("=" * 80)

    return analyzer


if __name__ == '__main__':
    analyzer = main()
