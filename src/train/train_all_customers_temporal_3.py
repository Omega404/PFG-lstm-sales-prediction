"""
═══════════════════════════════════════════════════════════════════════════════
CUSTOMER TEMPORAL ANALYZER V3 - Ventana de predicción UNIFORME
═══════════════════════════════════════════════════════════════════════════════

VERSIÓN 3: Misma ventana de predicción para todos (7 días)
- SHORT: 30 días → 7 días
- MEDIUM: 120 días → 7 días (igual forecast que SHORT)
- LONG: 240 días → 7 días (igual forecast que SHORT)

Objetivo: Evaluar el impacto de mayor contexto histórico (ventana más larga)
manteniendo la misma dificultad de predicción. Permite comparación justa.

Author: Sistema PFG LSTM
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow para experiment tracking
try:
    from mlflow_tracker import MLflowTracker, log_training_history
    MLFLOW_AVAILABLE = True
    print("✅ MLflow disponible")
except ImportError:
    MLFLOW_AVAILABLE = False
    MLflowTracker = None
    log_training_history = None
    print("⚠️ MLflow no instalado - tracking deshabilitado (pip install mlflow)")

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

class TemporalConfig:
    """Configuración V3: Ventana de predicción UNIFORME (7/7/7 días)"""

    # Horizontes de análisis - MISMO forecast para comparación justa
    SHORT = {
        'name': 'short',
        'window_days': 30,      # 1 mes de historial
        'forecast_days': 7,     # Predecir próxima semana
        'lstm_units': [64, 32],
        'epochs': 20,
        'batch_size': 32
    }

    MEDIUM = {
        'name': 'medium',
        'window_days': 120,     # 4 meses de historial
        'forecast_days': 7,     # ⭐ REDUCIDO: 7 días (era 30)
        'lstm_units': [128, 64],
        'epochs': 30,
        'batch_size': 64
    }

    LONG = {
        'name': 'long',
        'window_days': 240,     # 8 meses de historial
        'forecast_days': 7,     # ⭐ REDUCIDO: 7 días (era 60)
        'lstm_units': [128, 64, 32],
        'epochs': 40,
        'batch_size': 64
    }
    
    # Features por timestep
    N_FEATURES = 8
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 20
    REDUCE_LR_PATIENCE = 10
    REDUCE_LR_FACTOR = 0.5
    MIN_LEARNING_RATE = 1e-7


# ═══════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL: CUSTOMER TEMPORAL ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class CustomerTemporalAnalyzer:
    """
    Análisis temporal completo de clientes con LSTM
    Estructura análoga a ProductTemporalAnalyzer
    """
    
    def __init__(self, data_path='data/online_retail.csv', output_dir='models/temporal/customer_v3'):
        """
        Inicializa el analizador temporal de clientes V3

        Args:
            data_path: ruta al dataset Online Retail II
            output_dir: carpeta para guardar modelos y resultados (V3)
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.customers = []
        self.rfm_data = None
        self.segment_labels = {}

        # Crear estructura de directorios V3
        os.makedirs(output_dir, exist_ok=True)
        for horizon in ['short', 'medium', 'long']:
            os.makedirs(f'{output_dir}/{horizon}', exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"  CUSTOMER TEMPORAL ANALYZER - Inicializado")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # FASE 1: CARGA Y PREPROCESAMIENTO
    # ═══════════════════════════════════════════════════════════════════════
    
    def load_and_preprocess_data(self):
        """
        Carga y preprocesa el dataset (análogo a load_data en productos)
        """
        print(f"\n{'─'*70}")
        print("FASE 1: Carga y Preprocesamiento de Datos")
        print(f"{'─'*70}")
        
        # Cargar dataset
        print(f"\n📊 Cargando dataset: {self.data_path}")
        self.df = pd.read_excel(self.data_path, engine='openpyxl')
        print(f"   Registros cargados: {len(self.df):,}")
        
        # Limpieza
        initial_count = len(self.df)
        self.df = self.df[self.df['CustomerID'].notna()]
        self.df = self.df[self.df['Quantity'] > 0]
        self.df = self.df[self.df['UnitPrice'] > 0]
        
        # Convertir fecha
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.df['Date'] = self.df['InvoiceDate'].dt.date
        
        # Calcular valor total
        self.df['TotalValue'] = self.df['Quantity'] * self.df['UnitPrice']
        
        # Ordenar
        self.df = self.df.sort_values(['CustomerID', 'InvoiceDate'])
        
        print(f"\n✅ Limpieza completada:")
        print(f"   Registros eliminados: {initial_count - len(self.df):,}")
        print(f"   Registros válidos: {len(self.df):,}")
        print(f"   Clientes únicos: {self.df['CustomerID'].nunique():,}")
        print(f"   Rango: {self.df['InvoiceDate'].min()} → {self.df['InvoiceDate'].max()}")
        print(f"   Días totales: {(self.df['InvoiceDate'].max() - self.df['InvoiceDate'].min()).days}")
        
        return self.df
    
    def calculate_rfm_metrics(self):
        """
        Calcula métricas RFM para segmentación (análogo a product features)
        """
        print(f"\n{'─'*70}")
        print("FASE 2: Cálculo de Métricas RFM")
        print(f"{'─'*70}")
        
        reference_date = self.df['InvoiceDate'].max() + timedelta(days=1)
        
        # Agrupar por cliente
        rfm = self.df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'TotalValue': 'sum'  # Monetary
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Calcular scores (1-5)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop').astype(int)
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
        rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
        
        # Segmentación K-Means
        scaler = RobustScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        rfm['Segment'] = kmeans.fit_predict(rfm_scaled)
        
        # Etiquetar segmentos
        segment_stats = rfm.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        })
        
        segment_stats['Score'] = (
            segment_stats['Monetary'] * 0.4 + 
            segment_stats['Frequency'] * 0.4 - 
            segment_stats['Recency'] * 0.2
        )
        
        sorted_segments = segment_stats.sort_values('Score', ascending=False)
        segment_names = ['VIP', 'Loyal', 'Potential', 'At_Risk', 'Hibernating']
        
        self.segment_labels = {
            seg_id: segment_names[min(i, len(segment_names)-1)]
            for i, (seg_id, _) in enumerate(sorted_segments.iterrows())
        }
        
        rfm['Segment_Label'] = rfm['Segment'].map(self.segment_labels)
        
        self.rfm_data = rfm
        
        print(f"\n✅ RFM calculado para {len(rfm)} clientes")
        print(f"\nDistribución de segmentos:")
        print(rfm['Segment_Label'].value_counts())
        
        # Guardar RFM
        rfm.to_csv(f'{self.output_dir}/customer_rfm_segments.csv')
        print(f"\n💾 RFM guardado: {self.output_dir}/customer_rfm_segments.csv")
        
        return rfm
    
    # ═══════════════════════════════════════════════════════════════════════
    # FASE 3: GENERACIÓN DE SECUENCIAS TEMPORALES
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_customer_sequences(self, min_transactions=5):
        """
        Genera secuencias temporales para cada cliente
        Similar a generate_sequences en productos
        """
        print(f"\n{'─'*70}")
        print("FASE 3: Generación de Secuencias Temporales")
        print(f"{'─'*70}")
        
        print(f"\nFiltrando clientes con ≥{min_transactions} transacciones...")
        
        valid_customers = []
        
        for customer_id in self.df['CustomerID'].unique():
            customer_data = self.df[self.df['CustomerID'] == customer_id].copy()
            
            # Filtrar por número mínimo de transacciones
            n_transactions = customer_data['InvoiceNo'].nunique()
            if n_transactions < min_transactions:
                continue
            
            # Agregar por día
            daily_data = customer_data.groupby('Date').agg({
                'TotalValue': 'sum',
                'Quantity': 'sum',
                'InvoiceNo': 'nunique',
                'StockCode': 'nunique'
            }).reset_index()
            
            daily_data.columns = ['Date', 'DailyValue', 'DailyQuantity', 'DailyTransactions', 'DailyUniqueProducts']
            
            # Crear secuencia completa (rellenar días sin compras)
            date_range = pd.date_range(
                start=customer_data['InvoiceDate'].min().date(),
                end=customer_data['InvoiceDate'].max().date(),
                freq='D'
            )
            
            full_sequence = pd.DataFrame({'Date': date_range.date})
            full_sequence = full_sequence.merge(daily_data, on='Date', how='left')
            full_sequence = full_sequence.fillna(0)
            
            # Calcular features temporales
            full_sequence['Date'] = pd.to_datetime(full_sequence['Date'])
            full_sequence['DayOfWeek'] = full_sequence['Date'].dt.dayofweek
            full_sequence['IsWeekend'] = (full_sequence['DayOfWeek'] >= 5).astype(int)
            full_sequence['DayOfMonth'] = full_sequence['Date'].dt.day
            full_sequence['Month'] = full_sequence['Date'].dt.month
            
            # Días desde última compra
            purchase_indices = full_sequence[full_sequence['DailyValue'] > 0].index
            full_sequence['DaysSinceLastPurchase'] = 0
            
            for i in range(len(full_sequence)):
                prev_purchases = purchase_indices[purchase_indices < i]
                if len(prev_purchases) > 0:
                    full_sequence.loc[i, 'DaysSinceLastPurchase'] = i - prev_purchases[-1]
                else:
                    full_sequence.loc[i, 'DaysSinceLastPurchase'] = i
            
            # Info RFM
            if customer_id in self.rfm_data.index:
                rfm_info = self.rfm_data.loc[customer_id]
                segment_label = rfm_info['Segment_Label']
                rfm_score = rfm_info['RFM_Score']
            else:
                segment_label = 'Unknown'
                rfm_score = 0
            
            # Features finales (8 features)
            feature_cols = [
                'DailyValue', 'DailyQuantity', 'DailyTransactions', 'DailyUniqueProducts',
                'DayOfWeek', 'IsWeekend', 'DaysSinceLastPurchase', 'Month'
            ]
            
            sequence_array = full_sequence[feature_cols].values
            
            valid_customers.append({
                'CustomerID': customer_id,
                'Sequence': sequence_array,
                'Dates': full_sequence['Date'].values,
                'TotalDays': len(full_sequence),
                'TotalPurchases': n_transactions,
                'AvgDaysBetweenPurchases': len(full_sequence) / n_transactions if n_transactions > 0 else 0,
                'Segment': segment_label,
                'RFM_Score': rfm_score,
                'TotalRevenue': customer_data['TotalValue'].sum()
            })
        
        self.customers = valid_customers
        
        print(f"\n✅ Secuencias generadas:")
        print(f"   Clientes válidos: {len(valid_customers):,}")
        print(f"   Promedio días/cliente: {np.mean([c['TotalDays'] for c in valid_customers]):.1f}")
        print(f"   Promedio compras/cliente: {np.mean([c['TotalPurchases'] for c in valid_customers]):.1f}")
        
        # Guardar info de clientes
        customer_summary = pd.DataFrame([{
            'CustomerID': c['CustomerID'],
            'TotalDays': c['TotalDays'],
            'TotalPurchases': c['TotalPurchases'],
            'AvgDaysBetweenPurchases': c['AvgDaysBetweenPurchases'],
            'Segment': c['Segment'],
            'RFM_Score': c['RFM_Score'],
            'TotalRevenue': c['TotalRevenue']
        } for c in valid_customers])
        
        customer_summary.to_csv(f'{self.output_dir}/customer_summary.csv', index=False)
        print(f"\n💾 Summary guardado: {self.output_dir}/customer_summary.csv")
        
        return valid_customers
    
    # ═══════════════════════════════════════════════════════════════════════
    # FASE 4: PREPARACIÓN DE DATOS PARA LSTM
    # ═══════════════════════════════════════════════════════════════════════
    
    def prepare_lstm_data(self, horizon_config):
        """
        Prepara datos para entrenamiento LSTM en un horizonte específico
        Análogo a prepare_training_data en productos
        """
        window_days = horizon_config['window_days']
        forecast_days = horizon_config['forecast_days']
        horizon_name = horizon_config['name']
        
        print(f"\n{'─'*70}")
        print(f"Preparando datos para horizonte: {horizon_name.upper()}")
        print(f"{'─'*70}")
        print(f"Window: {window_days} días | Forecast: {forecast_days} días")
        
        X_sequences = []
        y_purchase_prob = []
        y_days_until = []
        y_purchase_value = []
        customer_ids = []
        
        samples_created = 0
        customers_with_samples = 0
        
        for customer in self.customers:
            sequence = customer['Sequence']
            customer_id = customer['CustomerID']
            
            # Verificar longitud suficiente
            if len(sequence) < window_days + forecast_days:
                continue
            
            # Ventanas deslizantes
            customer_samples = 0
            for i in range(len(sequence) - window_days - forecast_days):
                # Input: window_days anteriores
                X = sequence[i:i+window_days]
                
                # Target: forecast_days siguientes
                future = sequence[i+window_days:i+window_days+forecast_days]
                
                # Output 1: ¿Habrá compra?
                will_purchase = int(np.sum(future[:, 0]) > 0)  # Column 0 = DailyValue
                
                # Output 2: Días hasta próxima compra
                if will_purchase:
                    purchase_days = np.where(future[:, 0] > 0)[0]
                    days_until = purchase_days[0] if len(purchase_days) > 0 else forecast_days
                else:
                    days_until = forecast_days
                
                # Output 3: Valor total de compras
                total_value = np.sum(future[:, 0])
                
                X_sequences.append(X)
                y_purchase_prob.append(will_purchase)
                y_days_until.append(days_until)
                y_purchase_value.append(total_value)
                customer_ids.append(customer_id)
                
                customer_samples += 1
                samples_created += 1
            
            if customer_samples > 0:
                customers_with_samples += 1
        
        # Convertir a arrays
        X = np.array(X_sequences)
        y_prob = np.array(y_purchase_prob)
        y_days = np.array(y_days_until).reshape(-1, 1)
        y_value = np.array(y_purchase_value).reshape(-1, 1)
        
        print(f"\n✅ Dataset creado:")
        print(f"   Muestras totales: {len(X):,}")
        print(f"   Clientes con muestras: {customers_with_samples:,}")
        print(f"   Shape X: {X.shape}")
        print(f"   Muestras con compra: {np.sum(y_prob):,} ({np.mean(y_prob)*100:.1f}%)")
        print(f"   Días promedio hasta compra: {np.mean(y_days):.1f}")
        print(f"   Valor promedio: ${np.mean(y_value):.2f}")
        
        return X, y_prob, y_days, y_value, customer_ids
    
    # ═══════════════════════════════════════════════════════════════════════
    # FASE 5: CONSTRUCCIÓN Y ENTRENAMIENTO DE MODELOS LSTM
    # ═══════════════════════════════════════════════════════════════════════
    
    def build_lstm_model(self, horizon_config):
        """
        Construye modelo LSTM multi-output
        Análogo a build_model en productos
        """
        window_days = horizon_config['window_days']
        lstm_units = horizon_config['lstm_units']
        dropout_rate = 0.3
        
        print(f"\n🏗️  Construyendo modelo LSTM...")
        print(f"   Window: {window_days} días")
        print(f"   Features: {TemporalConfig.N_FEATURES}")
        print(f"   LSTM units: {lstm_units}")
        
        # Input
        input_layer = layers.Input(shape=(window_days, TemporalConfig.N_FEATURES), name='input_sequence')
        
        # LSTM layers
        x = input_layer
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1)
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                name=f'lstm_{i+1}'
            )(x)
            x = layers.BatchNormalization()(x)
        
        # Output 1: Probabilidad de compra
        out_prob = layers.Dense(32, activation='relu', name='dense_prob')(x)
        out_prob = layers.Dropout(dropout_rate)(out_prob)
        output_prob = layers.Dense(1, activation='sigmoid', name='purchase_probability')(out_prob)
        
        # Output 2: Días hasta compra
        out_days = layers.Dense(32, activation='relu', name='dense_days')(x)
        out_days = layers.Dropout(dropout_rate)(out_days)
        output_days = layers.Dense(1, activation='relu', name='days_until_purchase')(out_days)
        
        # Output 3: Valor de compra
        out_value = layers.Dense(32, activation='relu', name='dense_value')(x)
        out_value = layers.Dropout(dropout_rate)(out_value)
        output_value = layers.Dense(1, activation='relu', name='purchase_value')(out_value)
        
        # Modelo
        model = Model(
            inputs=input_layer,
            outputs=[output_prob, output_days, output_value]
        )
        
        # Compilar
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'purchase_probability': 'binary_crossentropy',
                'days_until_purchase': 'mse',
                'purchase_value': 'mse'
            },
            loss_weights={
                'purchase_probability': 1.0,
                'days_until_purchase': 0.5,
                'purchase_value': 0.5
            },
            metrics={
                'purchase_probability': ['accuracy', tf.keras.metrics.AUC(name='auc')],
                'days_until_purchase': ['mae'],
                'purchase_value': ['mae']
            }
        )
        
        print(f"✅ Modelo construido")
        return model
    
    def train_horizon_model(self, horizon_config, platform="local", script_version="v3"):
        """
        Entrena modelo para un horizonte temporal específico con MLflow tracking
        V3: Forecast uniforme de 7 días para comparación justa
        """
        horizon_name = horizon_config['name']
        epochs = horizon_config['epochs']
        batch_size = horizon_config['batch_size']

        print(f"\n{'═'*70}")
        print(f"ENTRENANDO MODELO: {horizon_name.upper()} - V3")
        print(f"{'═'*70}")

        # Iniciar MLflow tracking
        tracker = None
        if MLFLOW_AVAILABLE and MLflowTracker:
            tracker = MLflowTracker(experiment_name="customers_temporal_v3", enabled=True)
            tracker.start_run(
                run_name=f"customers_v3_{horizon_name}_{platform}",
                params={
                    "model_type": "customers",
                    "version": "v3",
                    "horizon": horizon_name,
                    "window_days": horizon_config['window_days'],
                    "forecast_days": horizon_config['forecast_days'],
                    "lstm_units": str(horizon_config['lstm_units']),
                    "epochs_config": epochs,
                    "batch_size": batch_size,
                    "platform": platform,
                    "script_version": script_version,
                    "n_features": TemporalConfig.N_FEATURES
                },
                tags={"team": "PFG_LSTM", "model_family": "customers_temporal", "version": "v3", "forecast_uniform": "7_days"}
            )
            print(f"📊 MLflow tracking: {tracker.run_id}")
        
        # Preparar datos
        X, y_prob, y_days, y_value, customer_ids = self.prepare_lstm_data(horizon_config)
        
        # Normalizar
        scaler_X = RobustScaler()
        X_scaled = X.reshape(-1, TemporalConfig.N_FEATURES)
        X_scaled = scaler_X.fit_transform(X_scaled)
        X_scaled = X_scaled.reshape(-1, horizon_config['window_days'], TemporalConfig.N_FEATURES)
        
        scaler_y_days = RobustScaler()
        scaler_y_value = RobustScaler()
        y_days_scaled = scaler_y_days.fit_transform(y_days)
        y_value_scaled = scaler_y_value.fit_transform(y_value)
        
        # Split train/val
        indices = np.arange(len(X_scaled))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        
        y_train = {
            'purchase_probability': y_prob[train_idx],
            'days_until_purchase': y_days_scaled[train_idx],
            'purchase_value': y_value_scaled[train_idx]
        }
        
        y_val = {
            'purchase_probability': y_prob[val_idx],
            'days_until_purchase': y_days_scaled[val_idx],
            'purchase_value': y_value_scaled[val_idx]
        }
        
        print(f"\n📊 Split de datos:")
        print(f"   Train: {len(X_train):,} muestras")
        print(f"   Val: {len(X_val):,} muestras")
        
        # Construir modelo
        model = self.build_lstm_model(horizon_config)
        
        # Callbacks
        horizon_dir = f'{self.output_dir}/{horizon_name}'
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=TemporalConfig.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=TemporalConfig.REDUCE_LR_FACTOR,
            patience=TemporalConfig.REDUCE_LR_PATIENCE,
            min_lr=TemporalConfig.MIN_LEARNING_RATE,
            verbose=1
        )
        
        checkpoint = callbacks.ModelCheckpoint(
            filepath=f'{horizon_dir}/model_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Entrenar
        print(f"\n🚀 Iniciando entrenamiento ({epochs} epochs)...")
        
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Guardar modelo final
        model.save(f'{horizon_dir}/model_final.keras')
        
        # Guardar scalers
        with open(f'{horizon_dir}/scaler_X.pkl', 'wb') as f:
            pickle.dump(scaler_X, f)
        with open(f'{horizon_dir}/scaler_y_days.pkl', 'wb') as f:
            pickle.dump(scaler_y_days, f)
        with open(f'{horizon_dir}/scaler_y_value.pkl', 'wb') as f:
            pickle.dump(scaler_y_value, f)
        
        # Guardar historial
        with open(f'{horizon_dir}/training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        # ═══════════════════════════════════════════════════════════════════
        # FIX PERMANENTE: Extraer métricas del TRAINING HISTORY (best epoch)
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n📊 Extrayendo métricas del training history (best epoch)...")

        # Encontrar mejor epoch basado en val_loss
        val_losses = history.history['val_loss']
        best_epoch_idx = val_losses.index(min(val_losses))

        print(f"   Best epoch: {best_epoch_idx + 1}/{len(val_losses)}")
        print(f"   Val loss: {val_losses[best_epoch_idx]:.2f}")

        # Extraer métricas CORRECTAS del best epoch
        val_accuracy_raw = history.history['val_purchase_probability_accuracy'][best_epoch_idx]
        val_auc = history.history['val_purchase_probability_auc'][best_epoch_idx]
        total_loss = history.history['val_loss'][best_epoch_idx]
        purchase_prob_loss = history.history['val_purchase_probability_loss'][best_epoch_idx]
        days_loss = history.history['val_days_until_purchase_loss'][best_epoch_idx]
        value_loss = history.history['val_purchase_value_loss'][best_epoch_idx]

        # Keras SIEMPRE devuelve accuracy en escala 0-1 en training history
        accuracy_percent = val_accuracy_raw * 100
        keras_scale = "0-1"

        print(f"\n✅ MÉTRICAS DEL TRAINING HISTORY:")
        print(f"   Accuracy raw: {val_accuracy_raw:.4f}")
        print(f"   Accuracy %: {accuracy_percent:.2f}%")
        print(f"   AUC: {val_auc:.4f}")

        # ═══════════════════════════════════════════════════════════════════
        # Calcular MAE en escala REAL (desnormalizado)
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n📊 Calculando MAE en escala real...")

        # Hacer predicciones
        y_pred = model.predict(X_val, verbose=0)
        y_pred_prob, y_pred_days_scaled, y_pred_value_scaled = y_pred

        # Desnormalizar predicciones y targets
        y_pred_days_real = scaler_y_days.inverse_transform(y_pred_days_scaled)
        y_pred_value_real = scaler_y_value.inverse_transform(y_pred_value_scaled)

        y_true_days_real = scaler_y_days.inverse_transform(y_days_scaled[val_idx])
        y_true_value_real = scaler_y_value.inverse_transform(y_value_scaled[val_idx])

        # Calcular MAE en escala real
        days_mae_real = mean_absolute_error(y_true_days_real, y_pred_days_real)
        value_mae_real = mean_absolute_error(y_true_value_real, y_pred_value_real)

        metrics = {
            'horizon': horizon_name,
            'total_loss': float(total_loss),
            'purchase_prob_loss': float(purchase_prob_loss),
            'days_loss': float(days_loss),
            'value_loss': float(value_loss),
            'purchase_prob_accuracy': float(accuracy_percent),  # ✅ Del training history
            'purchase_prob_accuracy_raw': float(val_accuracy_raw),  # DEBUG: valor original
            'keras_scale_detected': keras_scale,  # Siempre "0-1" en training history
            'purchase_prob_auc': float(val_auc),
            'days_mae': float(days_mae_real),  # ✅ MAE en escala REAL (días)
            'value_mae': float(value_mae_real),  # ✅ MAE en escala REAL (dólares)
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'epochs_trained': len(history.history['loss']),
            'best_epoch': best_epoch_idx + 1
        }

        # Guardar métricas
        with open(f'{horizon_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✅ Entrenamiento completado:")
        print(f"   Epochs: {metrics['epochs_trained']}")
        print(f"   Val Loss: {metrics['total_loss']:.4f}")
        print(f"   Accuracy: {metrics['purchase_prob_accuracy']:.2f}%")  # ✅ Auto-corregido
        print(f"   AUC: {metrics['purchase_prob_auc']:.4f}")
        print(f"   Days MAE: {metrics['days_mae']:.2f} días")  # ✅ Escala real
        print(f"   Value MAE: ${metrics['value_mae']:.2f}")  # ✅ Escala real

        # MLflow: Log métricas
        if tracker:
            tracker.log_metrics({
                "total_loss": float(metrics['total_loss']),
                "purchase_prob_accuracy": float(metrics['purchase_prob_accuracy']),
                "purchase_prob_auc": float(metrics['purchase_prob_auc']),
                "days_mae": float(metrics['days_mae']),
                "value_mae": float(metrics['value_mae']),
                "train_samples": metrics['train_samples'],
                "val_samples": metrics['val_samples'],
                "epochs_trained": metrics['epochs_trained']
            })
            if log_training_history:
                log_training_history(tracker, history.history)
            tracker.log_artifact(f'{horizon_dir}/metrics.json', 'metrics')
            tracker.log_artifact(f'{horizon_dir}/training_history.pkl', 'history')
            if tracker.log_model(model, "model", f"customers_v3_{horizon_name}"):
                print(f"   ✅ Modelo registrado en MLflow")
            tracker.end_run()
            print(f"   📊 MLflow run finalizado")

        # Visualizar
        self._plot_training_history(history.history, horizon_dir)

        print(f"\n💾 Guardado en: {horizon_dir}/")

        return model, history, metrics
    
    def _plot_training_history(self, history, output_dir):
        """Visualiza curvas de entrenamiento"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Loss total
        axes[0, 0].plot(history['loss'], label='Train')
        axes[0, 0].plot(history['val_loss'], label='Val')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Purchase prob - accuracy
        axes[0, 1].plot(history['purchase_probability_accuracy'], label='Train')
        axes[0, 1].plot(history['val_purchase_probability_accuracy'], label='Val')
        axes[0, 1].set_title('Purchase Probability Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Purchase prob - AUC
        axes[0, 2].plot(history['purchase_probability_auc'], label='Train')
        axes[0, 2].plot(history['val_purchase_probability_auc'], label='Val')
        axes[0, 2].set_title('Purchase Probability AUC')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Days MAE
        axes[1, 0].plot(history['days_until_purchase_mae'], label='Train')
        axes[1, 0].plot(history['val_days_until_purchase_mae'], label='Val')
        axes[1, 0].set_title('Days Until Purchase MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Value MAE
        axes[1, 1].plot(history['purchase_value_mae'], label='Train')
        axes[1, 1].plot(history['val_purchase_value_mae'], label='Val')
        axes[1, 1].set_title('Purchase Value MAE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Learning rate
        if 'lr' in history:
            axes[1, 2].plot(history['lr'])
            axes[1, 2].set_title('Learning Rate')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # ═══════════════════════════════════════════════════════════════════════
    # FASE 6: ENTRENAMIENTO DE TODOS LOS HORIZONTES
    # ═══════════════════════════════════════════════════════════════════════
    
    def train_all_horizons(self):
        """
        Entrena modelos para los 3 horizontes temporales
        Análogo a train_all_products
        """
        print(f"\n{'═'*70}")
        print("ENTRENAMIENTO DE TODOS LOS HORIZONTES TEMPORALES")
        print(f"{'═'*70}")
        
        horizons = [
            TemporalConfig.SHORT,
            TemporalConfig.MEDIUM,
            TemporalConfig.LONG
        ]
        
        results = {}
        
        for i, horizon_config in enumerate(horizons, 1):
            print(f"\n\n{'█'*70}")
            print(f"HORIZONTE {i}/3: {horizon_config['name'].upper()}")
            print(f"{'█'*70}")
            
            try:
                model, history, metrics = self.train_horizon_model(horizon_config)
                results[horizon_config['name']] = {
                    'status': 'SUCCESS',
                    'metrics': metrics
                }
            except Exception as e:
                print(f"\n❌ Error entrenando {horizon_config['name']}: {e}")
                results[horizon_config['name']] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
        
        # Guardar resumen final
        with open(f'{self.output_dir}/training_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Imprimir resumen
        print(f"\n\n{'═'*70}")
        print("RESUMEN FINAL DE ENTRENAMIENTO")
        print(f"{'═'*70}\n")
        
        for horizon, result in results.items():
            status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
            print(f"{status_icon} {horizon.upper()}: {result['status']}")

            if result['status'] == 'SUCCESS':
                metrics = result['metrics']
                print(f"   Accuracy: {metrics['purchase_prob_accuracy']:.2f}%")
                print(f"   AUC: {metrics['purchase_prob_auc']:.4f}")
                print(f"   Days MAE: {metrics['days_mae']:.2f} días")
                print(f"   Value MAE: ${metrics['value_mae']:.2f}")
            print()
        
        print(f"💾 Resultados guardados en: {self.output_dir}/")
        print(f"\n{'═'*70}\n")
        
        return results


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Pipeline completo de análisis temporal de clientes V3"""
    import argparse

    parser = argparse.ArgumentParser(description='Entrenar modelos LSTM de clientes V3')
    parser.add_argument('--horizon', choices=['short', 'medium', 'long', 'all'],
                        help='Horizonte a entrenar')
    parser.add_argument('--platform', default='local', choices=['local', 'kaggle', 'colab'])
    parser.add_argument('--script-version', default='v3_fixed')
    args = parser.parse_args()

    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  CUSTOMER TEMPORAL ANALYZER V3 - Pipeline Completo".center(68) + "║")
    print("║" + "  Forecast Uniforme 7 días para Comparación Justa".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")

    # Determinar qué horizontes entrenar
    horizons_to_train = []

    if args.horizon:
        # Modo línea de comandos
        if args.horizon == 'all':
            horizons_to_train = [TemporalConfig.SHORT, TemporalConfig.MEDIUM, TemporalConfig.LONG]
            print("\n✅ Entrenando: SHORT + MEDIUM + LONG")
        elif args.horizon == 'short':
            horizons_to_train = [TemporalConfig.SHORT]
            print("\n✅ Entrenando: SHORT")
        elif args.horizon == 'medium':
            horizons_to_train = [TemporalConfig.MEDIUM]
            print("\n✅ Entrenando: MEDIUM")
        elif args.horizon == 'long':
            horizons_to_train = [TemporalConfig.LONG]
            print("\n✅ Entrenando: LONG")
    else:
        # Modo interactivo
        print("\n📋 Selecciona modo de entrenamiento:")
        print("   1. Entrenar TODOS los horizontes (SHORT + MEDIUM + LONG)")
        print("   2. Entrenar solo SHORT")
        print("   3. Entrenar solo MEDIUM")
        print("   4. Entrenar solo LONG")
        print("   5. Entrenar SHORT + MEDIUM")
        print("   6. Entrenar MEDIUM + LONG")

        try:
            opcion = input("\n👉 Ingresa el número (1-6): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n❌ Cancelado por el usuario")
            return

        if opcion == '1':
            horizons_to_train = [TemporalConfig.SHORT, TemporalConfig.MEDIUM, TemporalConfig.LONG]
            print("\n✅ Entrenando: SHORT + MEDIUM + LONG")
        elif opcion == '2':
            horizons_to_train = [TemporalConfig.SHORT]
            print("\n✅ Entrenando: SHORT")
        elif opcion == '3':
            horizons_to_train = [TemporalConfig.MEDIUM]
            print("\n✅ Entrenando: MEDIUM")
        elif opcion == '4':
            horizons_to_train = [TemporalConfig.LONG]
            print("\n✅ Entrenando: LONG")
        elif opcion == '5':
            horizons_to_train = [TemporalConfig.SHORT, TemporalConfig.MEDIUM]
            print("\n✅ Entrenando: SHORT + MEDIUM")
        elif opcion == '6':
            horizons_to_train = [TemporalConfig.MEDIUM, TemporalConfig.LONG]
            print("\n✅ Entrenando: MEDIUM + LONG")
        else:
            print("\n❌ Opción inválida")
            return

    start_time = datetime.now()
    print(f"\n⏰ Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Inicializar
    analyzer = CustomerTemporalAnalyzer(
        data_path='data/processed/online_retail_2.xlsx',
        output_dir='models/temporal/customer'
    )

    # Pipeline de preparación (común para todos)
    print("="*70)
    print("FASE 1: Preparación de datos")
    print("="*70)
    analyzer.load_and_preprocess_data()
    analyzer.calculate_rfm_metrics()
    analyzer.generate_customer_sequences(min_transactions=5)

    # Entrenar horizontes seleccionados
    print(f"\n{'='*70}")
    print(f"FASE 2: Entrenamiento de {len(horizons_to_train)} horizonte(s)")
    print(f"{'='*70}")

    results = {}

    for i, horizon_config in enumerate(horizons_to_train, 1):
        print(f"\n\n{'█'*70}")
        print(f"HORIZONTE {i}/{len(horizons_to_train)}: {horizon_config['name'].upper()}")
        print(f"{'█'*70}")

        try:
            model, history, metrics = analyzer.train_horizon_model(
                horizon_config,
                platform=args.platform,
                script_version=args.script_version
            )
            results[horizon_config['name']] = {
                'status': 'SUCCESS',
                'metrics': metrics
            }
        except Exception as e:
            print(f"\n❌ Error entrenando {horizon_config['name']}: {e}")
            results[horizon_config['name']] = {
                'status': 'FAILED',
                'error': str(e)
            }

    # Tiempo total
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Resumen final
    print(f"\n\n{'═'*70}")
    print("RESUMEN FINAL DE ENTRENAMIENTO")
    print(f"{'═'*70}\n")

    for horizon, result in results.items():
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_icon} {horizon.upper()}: {result['status']}")

        if result['status'] == 'SUCCESS':
            metrics = result['metrics']
            print(f"   Accuracy: {metrics['purchase_prob_accuracy']:.2f}%")
            print(f"   AUC: {metrics['purchase_prob_auc']:.4f}")
            print(f"   Days MAE: {metrics['days_mae']:.2f} días")
            print(f"   Value MAE: ${metrics['value_mae']:.2f}")
        print()

    print(f"⏰ Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Duración total: {duration/60:.1f} minutos ({duration:.0f} segundos)")

    print(f"\n💾 Modelos guardados en: {analyzer.output_dir}/")
    print("\n" + "═"*70)
    print("✅ PIPELINE COMPLETADO")
    print("═"*70 + "\n")


if __name__ == '__main__':
    main()
