"""
═══════════════════════════════════════════════════════════════════════════════
CUSTOMER PREDICTOR & EVALUATOR - Sistema de Inferencia Masiva
═══════════════════════════════════════════════════════════════════════════════

Archivo complementario a train_all_customers_temporal.py
Realiza predicciones masivas sobre todos los clientes y genera reportes

Funciones principales:
1. Cargar modelos entrenados de los 3 horizontes
2. Predecir comportamiento futuro de TODOS los clientes
3. Evaluar métricas de performance
4. Generar rankings y segmentaciones predictivas
5. Identificar oportunidades de negocio automáticamente

Análogo a: evaluate_products.py / predict_products.py

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

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

print(f"TensorFlow version: {tf.__version__}")


# ═══════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL: CUSTOMER PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

class CustomerPredictor:
    """
    Sistema de inferencia masiva para predicción de comportamiento de clientes
    Carga modelos entrenados y realiza predicciones en batch
    """
    
    def __init__(self, models_dir='models/temporal/customer'):
        """
        Inicializa el predictor con modelos pre-entrenados
        
        Args:
            models_dir: directorio base con modelos de 3 horizontes
        """
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        self.horizons = ['short', 'medium', 'long']
        
        print(f"\n{'='*70}")
        print(f"  CUSTOMER PREDICTOR - Sistema de Inferencia Masiva")
        print(f"{'='*70}")
        print(f"Models directory: {models_dir}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # CARGA DE MODELOS
    # ═══════════════════════════════════════════════════════════════════════
    
    def load_models(self):
        """
        Carga los 3 modelos LSTM entrenados (short, medium, long)
        """
        print(f"\n{'─'*70}")
        print("Cargando Modelos Entrenados")
        print(f"{'─'*70}\n")
        
        for horizon in self.horizons:
            horizon_dir = f'{self.models_dir}/{horizon}'
            
            try:
                # Cargar modelo
                model_path = f'{horizon_dir}/model_best.keras'
                if not os.path.exists(model_path):
                    model_path = f'{horizon_dir}/model_final.keras'
                
                print(f"📦 Cargando {horizon.upper()}...")
                self.models[horizon] = keras.models.load_model(model_path)
                
                # Cargar scalers
                with open(f'{horizon_dir}/scaler_X.pkl', 'rb') as f:
                    scaler_X = pickle.load(f)
                
                with open(f'{horizon_dir}/scaler_y_days.pkl', 'rb') as f:
                    scaler_y_days = pickle.load(f)
                
                with open(f'{horizon_dir}/scaler_y_value.pkl', 'rb') as f:
                    scaler_y_value = pickle.load(f)
                
                self.scalers[horizon] = {
                    'X': scaler_X,
                    'y_days': scaler_y_days,
                    'y_value': scaler_y_value
                }
                
                # Cargar métricas de entrenamiento
                with open(f'{horizon_dir}/metrics.json', 'r') as f:
                    metrics = json.load(f)
                
                print(f"   ✅ Modelo cargado: {model_path}")
                print(f"   📊 Accuracy: {metrics['purchase_prob_accuracy']*100:.2f}%")
                print(f"   📊 AUC: {metrics['purchase_prob_auc']:.4f}")
                print()
                
            except Exception as e:
                print(f"   ❌ Error cargando {horizon}: {e}\n")
        
        print(f"✅ {len(self.models)}/3 modelos cargados correctamente")
        return self.models
    
    def load_customer_data(self, data_path='data/online_retail.csv'):
        """
        Carga y preprocesa datos de clientes (análogo a train)
        """
        print(f"\n{'─'*70}")
        print("Cargando Datos de Clientes")
        print(f"{'─'*70}\n")
        
        print(f"📊 Cargando dataset: {data_path}")
        df = pd.read_csv(data_path, encoding='ISO-8859-1')
        
        # Limpieza
        df = df[df['CustomerID'].notna()]
        df = df[df['Quantity'] > 0]
        df = df[df['UnitPrice'] > 0]
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Date'] = df['InvoiceDate'].dt.date
        df['TotalValue'] = df['Quantity'] * df['UnitPrice']
        df = df.sort_values(['CustomerID', 'InvoiceDate'])
        
        print(f"✅ Dataset cargado:")
        print(f"   Registros válidos: {len(df):,}")
        print(f"   Clientes únicos: {df['CustomerID'].nunique():,}")
        print(f"   Rango: {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}")
        
        self.df = df
        return df
    
    def load_customer_segments(self):
        """
        Carga segmentación RFM de clientes
        """
        segments_path = f'{self.models_dir}/customer_rfm_segments.csv'
        
        if os.path.exists(segments_path):
            self.rfm_data = pd.read_csv(segments_path, index_col='CustomerID')
            print(f"\n✅ Segmentos RFM cargados: {len(self.rfm_data)} clientes")
            print(f"   Distribución:")
            print(self.rfm_data['Segment_Label'].value_counts())
        else:
            print(f"\n⚠️  Segmentos RFM no encontrados en {segments_path}")
            self.rfm_data = None
        
        return self.rfm_data
    
    # ═══════════════════════════════════════════════════════════════════════
    # GENERACIÓN DE SECUENCIAS PARA PREDICCIÓN
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_prediction_sequences(self, horizon_config):
        """
        Genera secuencias de los últimos N días para cada cliente
        (para hacer predicción hacia el futuro)
        """
        window_days = horizon_config['window_days']
        horizon_name = horizon_config['name']
        
        print(f"\n{'─'*70}")
        print(f"Generando Secuencias para Predicción: {horizon_name.upper()}")
        print(f"{'─'*70}")
        print(f"Window: {window_days} días (últimos datos disponibles)")
        
        sequences = []
        customer_ids = []
        
        for customer_id in self.df['CustomerID'].unique():
            customer_data = self.df[self.df['CustomerID'] == customer_id].copy()
            
            # Necesitamos al menos window_days de historia
            date_range_days = (customer_data['InvoiceDate'].max() - customer_data['InvoiceDate'].min()).days
            if date_range_days < window_days:
                continue
            
            # Agregar por día
            daily_data = customer_data.groupby('Date').agg({
                'TotalValue': 'sum',
                'Quantity': 'sum',
                'InvoiceNo': 'nunique',
                'StockCode': 'nunique'
            }).reset_index()
            
            daily_data.columns = ['Date', 'DailyValue', 'DailyQuantity', 'DailyTransactions', 'DailyUniqueProducts']
            
            # Secuencia completa
            date_range = pd.date_range(
                start=customer_data['InvoiceDate'].min().date(),
                end=customer_data['InvoiceDate'].max().date(),
                freq='D'
            )
            
            full_sequence = pd.DataFrame({'Date': date_range.date})
            full_sequence = full_sequence.merge(daily_data, on='Date', how='left')
            full_sequence = full_sequence.fillna(0)
            
            # Features temporales
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
            
            # Tomar últimos window_days
            if len(full_sequence) >= window_days:
                last_sequence = full_sequence.iloc[-window_days:]
                
                feature_cols = [
                    'DailyValue', 'DailyQuantity', 'DailyTransactions', 'DailyUniqueProducts',
                    'DayOfWeek', 'IsWeekend', 'DaysSinceLastPurchase', 'Month'
                ]
                
                sequence_array = last_sequence[feature_cols].values
                
                sequences.append(sequence_array)
                customer_ids.append(customer_id)
        
        print(f"\n✅ Secuencias generadas:")
        print(f"   Clientes con datos suficientes: {len(sequences):,}")
        print(f"   Shape por secuencia: ({window_days}, 8)")
        
        return np.array(sequences), customer_ids
    
    # ═══════════════════════════════════════════════════════════════════════
    # PREDICCIÓN MASIVA
    # ═══════════════════════════════════════════════════════════════════════
    
    def predict_horizon(self, horizon):
        """
        Realiza predicciones para todos los clientes en un horizonte
        """
        if horizon not in self.models:
            print(f"❌ Modelo {horizon} no cargado")
            return None
        
        # Configuración del horizonte
        horizon_configs = {
            'short': {'name': 'short', 'window_days': 28, 'forecast_days': 14},
            'medium': {'name': 'medium', 'window_days': 90, 'forecast_days': 30},
            'long': {'name': 'long', 'window_days': 180, 'forecast_days': 60}
        }
        
        config = horizon_configs[horizon]
        
        print(f"\n{'═'*70}")
        print(f"PREDICIENDO: {horizon.upper()}")
        print(f"{'═'*70}")
        
        # Generar secuencias
        X, customer_ids = self.generate_prediction_sequences(config)
        
        if len(X) == 0:
            print("❌ No hay clientes con datos suficientes")
            return None
        
        # Normalizar
        print(f"\n🔄 Normalizando datos...")
        X_scaled = X.reshape(-1, 8)
        X_scaled = self.scalers[horizon]['X'].transform(X_scaled)
        X_scaled = X_scaled.reshape(-1, config['window_days'], 8)
        
        # Predecir
        print(f"🧠 Ejecutando predicción LSTM...")
        predictions = self.models[horizon].predict(X_scaled, batch_size=128, verbose=1)
        
        # Desnormalizar outputs de regresión
        pred_prob = predictions[0].flatten()
        pred_days = self.scalers[horizon]['y_days'].inverse_transform(predictions[1]).flatten()
        pred_value = self.scalers[horizon]['y_value'].inverse_transform(predictions[2]).flatten()
        
        # Crear DataFrame de resultados
        results_df = pd.DataFrame({
            'CustomerID': customer_ids,
            'Horizon': horizon,
            'PurchaseProbability': pred_prob,
            'DaysUntilPurchase': np.maximum(0, pred_days),  # No negativos
            'ExpectedValue': np.maximum(0, pred_value)  # No negativos
        })
        
        # Añadir segmento RFM si está disponible
        if self.rfm_data is not None:
            results_df = results_df.merge(
                self.rfm_data[['Segment_Label', 'RFM_Score', 'Recency', 'Frequency', 'Monetary']],
                left_on='CustomerID',
                right_index=True,
                how='left'
            )
        
        # Clasificación por probabilidad
        results_df['ProbabilityCategory'] = pd.cut(
            results_df['PurchaseProbability'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Revenue esperado
        results_df['ExpectedRevenue'] = results_df['PurchaseProbability'] * results_df['ExpectedValue']
        
        print(f"\n✅ Predicción completada:")
        print(f"   Clientes predichos: {len(results_df):,}")
        print(f"\n   Distribución de probabilidad:")
        print(results_df['ProbabilityCategory'].value_counts())
        print(f"\n   Métricas promedio:")
        print(f"   • Probabilidad: {results_df['PurchaseProbability'].mean()*100:.1f}%")
        print(f"   • Días hasta compra: {results_df['DaysUntilPurchase'].mean():.1f}")
        print(f"   • Valor esperado: ${results_df['ExpectedValue'].mean():.2f}")
        print(f"   • Revenue esperado total: ${results_df['ExpectedRevenue'].sum():,.2f}")
        
        self.predictions[horizon] = results_df
        
        # Guardar
        output_dir = f'{self.models_dir}/{horizon}/predictions'
        os.makedirs(output_dir, exist_ok=True)
        
        results_df.to_csv(f'{output_dir}/customer_predictions.csv', index=False)
        print(f"\n💾 Predicciones guardadas: {output_dir}/customer_predictions.csv")
        
        return results_df
    
    def predict_all_horizons(self):
        """
        Realiza predicciones para los 3 horizontes temporales
        """
        print(f"\n{'═'*70}")
        print("PREDICCIÓN MASIVA - TODOS LOS HORIZONTES")
        print(f"{'═'*70}")
        
        all_predictions = {}
        
        for horizon in self.horizons:
            if horizon in self.models:
                try:
                    predictions = self.predict_horizon(horizon)
                    all_predictions[horizon] = predictions
                except Exception as e:
                    print(f"\n❌ Error prediciendo {horizon}: {e}")
        
        return all_predictions
    
    # ═══════════════════════════════════════════════════════════════════════
    # ANÁLISIS Y REPORTES
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_business_insights(self):
        """
        Genera insights de negocio a partir de las predicciones
        """
        print(f"\n{'═'*70}")
        print("GENERANDO INSIGHTS DE NEGOCIO")
        print(f"{'═'*70}")
        
        if not self.predictions:
            print("❌ No hay predicciones disponibles. Ejecuta predict_all_horizons() primero.")
            return None
        
        insights = {}
        
        for horizon, df in self.predictions.items():
            print(f"\n{'─'*70}")
            print(f"Insights: {horizon.upper()}")
            print(f"{'─'*70}")
            
            # 1. Top clientes por probabilidad
            top_prob = df.nlargest(100, 'PurchaseProbability')
            
            # 2. Top clientes por valor esperado
            top_value = df.nlargest(100, 'ExpectedRevenue')
            
            # 3. Clientes VIP en riesgo
            if 'Segment_Label' in df.columns:
                vip_risk = df[
                    (df['Segment_Label'].isin(['VIP', 'Loyal'])) &
                    (df['PurchaseProbability'] < 0.3)
                ].sort_values('Monetary', ascending=False)
            else:
                vip_risk = pd.DataFrame()
            
            # 4. Quick wins (alta prob, bajo histórico)
            if 'Monetary' in df.columns:
                median_monetary = df['Monetary'].median()
                quick_wins = df[
                    (df['PurchaseProbability'] > 0.7) &
                    (df['Monetary'] < median_monetary)
                ]
            else:
                quick_wins = pd.DataFrame()
            
            # Guardar
            insights_dir = f'{self.models_dir}/{horizon}/insights'
            os.makedirs(insights_dir, exist_ok=True)
            
            top_prob.to_csv(f'{insights_dir}/top_100_probability.csv', index=False)
            top_value.to_csv(f'{insights_dir}/top_100_revenue.csv', index=False)
            
            if len(vip_risk) > 0:
                vip_risk.to_csv(f'{insights_dir}/vip_at_risk.csv', index=False)
            
            if len(quick_wins) > 0:
                quick_wins.to_csv(f'{insights_dir}/quick_wins.csv', index=False)
            
            # Métricas
            horizon_insights = {
                'horizon': horizon,
                'total_customers': len(df),
                'high_probability_count': len(df[df['PurchaseProbability'] > 0.7]),
                'total_expected_revenue': df['ExpectedRevenue'].sum(),
                'avg_purchase_probability': df['PurchaseProbability'].mean(),
                'avg_expected_value': df['ExpectedValue'].mean(),
                'vip_at_risk_count': len(vip_risk),
                'vip_at_risk_value': vip_risk['Monetary'].sum() if len(vip_risk) > 0 else 0,
                'quick_wins_count': len(quick_wins),
                'quick_wins_potential': quick_wins['ExpectedRevenue'].sum() if len(quick_wins) > 0 else 0
            }
            
            insights[horizon] = horizon_insights
            
            # Imprimir
            print(f"\n📊 Resumen {horizon.upper()}:")
            print(f"   Total clientes: {horizon_insights['total_customers']:,}")
            print(f"   Alta probabilidad (>70%): {horizon_insights['high_probability_count']:,}")
            print(f"   Revenue esperado: ${horizon_insights['total_expected_revenue']:,.2f}")
            print(f"   VIPs en riesgo: {horizon_insights['vip_at_risk_count']} (${horizon_insights['vip_at_risk_value']:,.2f})")
            print(f"   Quick wins: {horizon_insights['quick_wins_count']} (${horizon_insights['quick_wins_potential']:,.2f})")
            
            print(f"\n💾 Guardado en: {insights_dir}/")
        
        # Guardar resumen JSON
        with open(f'{self.models_dir}/business_insights_summary.json', 'w') as f:
            json.dump(insights, f, indent=2)
        
        print(f"\n✅ Insights generados para {len(insights)} horizontes")
        
        return insights
    
    def generate_comparison_report(self):
        """
        Compara predicciones entre los 3 horizontes
        """
        print(f"\n{'═'*70}")
        print("REPORTE COMPARATIVO ENTRE HORIZONTES")
        print(f"{'═'*70}")
        
        if len(self.predictions) < 2:
            print("⚠️  Se necesitan al menos 2 horizontes para comparación")
            return None
        
        # Clientes comunes en todos los horizontes
        common_customers = set(self.predictions['short']['CustomerID'])
        for horizon in ['medium', 'long']:
            if horizon in self.predictions:
                common_customers &= set(self.predictions[horizon]['CustomerID'])
        
        print(f"\n📊 Clientes comunes en todos los horizontes: {len(common_customers):,}")
        
        # Comparar predicciones
        comparison_data = []
        
        for customer_id in list(common_customers)[:1000]:  # Primeros 1000 para eficiencia
            row = {'CustomerID': customer_id}
            
            for horizon in self.horizons:
                if horizon in self.predictions:
                    df = self.predictions[horizon]
                    customer_pred = df[df['CustomerID'] == customer_id].iloc[0]
                    
                    row[f'{horizon}_prob'] = customer_pred['PurchaseProbability']
                    row[f'{horizon}_days'] = customer_pred['DaysUntilPurchase']
                    row[f'{horizon}_value'] = customer_pred['ExpectedValue']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Correlaciones
        print(f"\n🔍 Correlaciones entre horizontes:")
        
        for i, h1 in enumerate(self.horizons[:-1]):
            for h2 in self.horizons[i+1:]:
                if f'{h1}_prob' in comparison_df.columns and f'{h2}_prob' in comparison_df.columns:
                    corr = comparison_df[f'{h1}_prob'].corr(comparison_df[f'{h2}_prob'])
                    print(f"   {h1.upper()} vs {h2.upper()}: {corr:.3f}")
        
        # Guardar
        comparison_df.to_csv(f'{self.models_dir}/horizon_comparison.csv', index=False)
        print(f"\n💾 Comparación guardada: {self.models_dir}/horizon_comparison.csv")
        
        return comparison_df
    
    def create_visualizations(self):
        """
        Crea visualizaciones de las predicciones
        """
        print(f"\n{'─'*70}")
        print("Generando Visualizaciones")
        print(f"{'─'*70}")
        
        for horizon, df in self.predictions.items():
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Predicciones de Clientes - Horizonte {horizon.upper()}', fontsize=16)
            
            # 1. Distribución de probabilidad
            axes[0, 0].hist(df['PurchaseProbability'], bins=50, edgecolor='black')
            axes[0, 0].set_title('Distribución de Probabilidad de Compra')
            axes[0, 0].set_xlabel('Probabilidad')
            axes[0, 0].set_ylabel('Frecuencia')
            axes[0, 0].axvline(df['PurchaseProbability'].mean(), color='red', linestyle='--', label='Media')
            axes[0, 0].legend()
            
            # 2. Distribución de días hasta compra
            axes[0, 1].hist(df['DaysUntilPurchase'], bins=50, edgecolor='black')
            axes[0, 1].set_title('Distribución de Días hasta Compra')
            axes[0, 1].set_xlabel('Días')
            axes[0, 1].set_ylabel('Frecuencia')
            
            # 3. Distribución de valor esperado
            axes[0, 2].hist(df['ExpectedValue'].clip(upper=df['ExpectedValue'].quantile(0.95)), 
                           bins=50, edgecolor='black')
            axes[0, 2].set_title('Distribución de Valor Esperado (95% percentil)')
            axes[0, 2].set_xlabel('Valor ($)')
            axes[0, 2].set_ylabel('Frecuencia')
            
            # 4. Scatter: Probabilidad vs Valor
            axes[1, 0].scatter(df['PurchaseProbability'], df['ExpectedValue'], 
                              alpha=0.5, s=10)
            axes[1, 0].set_title('Probabilidad vs Valor Esperado')
            axes[1, 0].set_xlabel('Probabilidad de Compra')
            axes[1, 0].set_ylabel('Valor Esperado ($)')
            
            # 5. Categorías de probabilidad
            if 'ProbabilityCategory' in df.columns:
                cat_counts = df['ProbabilityCategory'].value_counts()
                axes[1, 1].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%')
                axes[1, 1].set_title('Categorías de Probabilidad')
            
            # 6. Por segmento (si existe)
            if 'Segment_Label' in df.columns:
                segment_avg = df.groupby('Segment_Label')['PurchaseProbability'].mean().sort_values()
                segment_avg.plot(kind='barh', ax=axes[1, 2])
                axes[1, 2].set_title('Probabilidad Promedio por Segmento')
                axes[1, 2].set_xlabel('Probabilidad')
            
            plt.tight_layout()
            
            viz_dir = f'{self.models_dir}/{horizon}/visualizations'
            os.makedirs(viz_dir, exist_ok=True)
            plt.savefig(f'{viz_dir}/predictions_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Visualización guardada: {viz_dir}/predictions_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Pipeline completo de predicción e inferencia
    """
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  CUSTOMER PREDICTOR - Inferencia Masiva".center(68) + "║")
    print("║" + "  Predicción de comportamiento futuro de clientes".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    
    start_time = datetime.now()
    print(f"\n⏰ Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Inicializar predictor
    predictor = CustomerPredictor(models_dir='models/temporal/customer')
    
    # Pipeline
    try:
        # 1. Cargar modelos
        predictor.load_models()
        
        # 2. Cargar datos
        predictor.load_customer_data('data/online_retail.csv')
        predictor.load_customer_segments()
        
        # 3. Predecir en todos los horizontes
        predictions = predictor.predict_all_horizons()
        
        # 4. Generar insights
        insights = predictor.generate_business_insights()
        
        # 5. Comparar horizontes
        comparison = predictor.generate_comparison_report()
        
        # 6. Visualizaciones
        predictor.create_visualizations()
        
    except Exception as e:
        print(f"\n❌ Error en el pipeline: {e}")
        import traceback
        traceback.print_exc()
    
    # Tiempo total
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n⏰ Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Duración total: {duration/60:.1f} minutos ({duration:.0f} segundos)")
    
    print("\n" + "═"*70)
    print("✅ PREDICCIÓN MASIVA COMPLETADA")
    print("═"*70)
    print("\n📁 Revisa los resultados en:")
    print("   models/temporal/customer/[horizon]/predictions/")
    print("   models/temporal/customer/[horizon]/insights/")
    print("\n")


if __name__ == '__main__':
    main()
