import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ===============================
# Configuración
# ===============================
class PredictConfig:
    MODEL_DIR = "models"
    PLOTS_DIR = "plots/predictions"
    REPORTS_DIR = "reports"
    LOGS_DIR = "logs"
    DATA_PATH = "data/processed/product_demand.xlsx"
    
    # Parámetros de predicción
    FORECAST_DAYS = 30  # Días a predecir
    CONFIDENCE_INTERVAL = 0.95
    
    def __init__(self):
        for directory in [self.PLOTS_DIR, self.REPORTS_DIR]:
            os.makedirs(directory, exist_ok=True)

pred_config = PredictConfig()


# ===============================
# Clase de Predicción
# ===============================
class ProductPredictor:
    def __init__(self, product_id):
        self.product_id = product_id
        self.model = None
        self.scaler = None
        self.metrics = None
        self.predictions = None
        
    def load_model(self):
        """Carga el modelo entrenado y sus componentes"""
        try:
            model_path = os.path.join(pred_config.MODEL_DIR, f'{self.product_id}_model.h5')
            scaler_path = os.path.join(pred_config.MODEL_DIR, f'{self.product_id}_scaler.pkl')
            metrics_path = os.path.join(pred_config.LOGS_DIR, f'{self.product_id}_metrics.json')
            
            self.model = load_model(model_path, compile=False)
            self.scaler = joblib.load(scaler_path)
            
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
            
            print(f"✅ Modelo cargado: {self.product_id}")
            if self.metrics:
                print(f"   MAE: {self.metrics['MAE']:.2f} | RMSE: {self.metrics['RMSE']:.2f} | R²: {self.metrics['R2']:.4f}")
            
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo {self.product_id}: {str(e)}")
            return False
    
    def prepare_forecast_features(self, last_window, forecast_date):
        """Prepara features para una fecha futura"""
        # Features temporales
        day_of_week = forecast_date.weekday()
        month = forecast_date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Season
        if month in [12, 1, 2]:
            season = 0
        elif month in [3, 4, 5]:
            season = 1
        elif month in [6, 7, 8]:
            season = 2
        else:
            season = 3
        
        # Para features que dependen de datos históricos, usar últimos valores
        last_values = last_window[-1].copy()
        
        # Actualizar features temporales
        feature_vector = last_values.copy()
        feature_vector[3] = day_of_week  # DayOfWeek
        feature_vector[4] = month  # Month
        feature_vector[5] = is_weekend  # IsWeekend
        feature_vector[6] = season  # Season
        
        return feature_vector
    
    def forecast(self, df_product, days_ahead=30):
        """Genera predicciones futuras"""
        print(f"\n🔮 Generando predicciones para {self.product_id}...")
        print(f"   Horizonte: {days_ahead} días")
        
        # Preparar datos históricos
        feature_cols = [
            'Quantity', 'TotalPrice', 'NumTransactions',
            'DayOfWeek', 'Month', 'IsWeekend', 'Season',
            'MA_7', 'MA_14', 'STD_7',
            'Lag_1', 'Lag_7'
        ]
        
        # Asegurar que df_product tiene las columnas necesarias
        if not all(col in df_product.columns for col in feature_cols):
            print(f"   ⚠️  Faltan columnas necesarias. Recalculando features...")
            df_product = self._recalculate_features(df_product)
        
        # Escalar datos históricos
        historical_scaled = self.scaler.transform(df_product[feature_cols].values)
        
        # Tomar última ventana
        window_size = self.model.input_shape[1]
        last_window = historical_scaled[-window_size:].copy()
        
        # Predecir día por día
        predictions = []
        confidence_intervals = []
        dates = []
        
        last_date = df_product['Date'].max()
        current_window = last_window.reshape(1, window_size, -1)
        
        for day in range(days_ahead):
            # Fecha de predicción
            forecast_date = last_date + timedelta(days=day+1)
            dates.append(forecast_date)
            
            # Predicción
            pred_scaled = self.model.predict(current_window, verbose=0)[0, 0]
            
            # Invertir escalado (solo para Quantity)
            # Crear array con todas las features, pero solo nos interesa Quantity
            dummy_features = np.zeros((1, len(feature_cols)))
            dummy_features[0, 0] = pred_scaled
            pred_original = self.scaler.inverse_transform(dummy_features)[0, 0]
            
            # Asegurar no negativos
            pred_original = max(0, pred_original)
            predictions.append(pred_original)
            
            # Intervalo de confianza (basado en MAE del modelo)
            if self.metrics:
                mae = self.metrics['MAE']
                ci_lower = max(0, pred_original - 1.96 * mae)
                ci_upper = pred_original + 1.96 * mae
                confidence_intervals.append((ci_lower, ci_upper))
            else:
                confidence_intervals.append((pred_original * 0.8, pred_original * 1.2))
            
            # Actualizar ventana para siguiente predicción
            next_features = self.prepare_forecast_features(current_window[0], forecast_date)
            next_features[0] = pred_scaled  # Actualizar Quantity con predicción
            
            # Actualizar medias móviles (simplificado)
            if day > 0:
                recent_preds = [predictions[max(0, i)] for i in range(day-6, day+1)]
                next_features[7] = np.mean(recent_preds)  # MA_7
                if day >= 13:
                    recent_preds_14 = [predictions[max(0, i)] for i in range(day-13, day+1)]
                    next_features[8] = np.mean(recent_preds_14)  # MA_14
            
            # Shift window
            current_window = np.append(
                current_window[:, 1:, :],
                next_features.reshape(1, 1, -1),
                axis=1
            )
        
        # Guardar resultados
        self.predictions = pd.DataFrame({
            'Date': dates,
            'Predicted_Quantity': predictions,
            'CI_Lower': [ci[0] for ci in confidence_intervals],
            'CI_Upper': [ci[1] for ci in confidence_intervals]
        })
        
        print(f"   ✓ Predicciones generadas")
        print(f"   📊 Demanda promedio predicha: {np.mean(predictions):.1f} unidades/día")
        print(f"   📈 Demanda total predicha: {np.sum(predictions):.0f} unidades")
        
        return self.predictions
    
    def _recalculate_features(self, df_product):
        """Recalcula features si faltan"""
        # Features temporales básicas
        if 'DayOfWeek' not in df_product.columns:
            df_product['DayOfWeek'] = df_product['Date'].dt.dayofweek
        if 'Month' not in df_product.columns:
            df_product['Month'] = df_product['Date'].dt.month
        if 'IsWeekend' not in df_product.columns:
            df_product['IsWeekend'] = (df_product['DayOfWeek'] >= 5).astype(int)
        if 'Season' not in df_product.columns:
            df_product['Season'] = df_product['Month'].apply(
                lambda m: 0 if m in [12,1,2] else 1 if m in [3,4,5] else 2 if m in [6,7,8] else 3
            )
        
        # Features derivadas
        if 'MA_7' not in df_product.columns:
            df_product['MA_7'] = df_product['Quantity'].rolling(window=7, min_periods=1).mean()
        if 'MA_14' not in df_product.columns:
            df_product['MA_14'] = df_product['Quantity'].rolling(window=14, min_periods=1).mean()
        if 'STD_7' not in df_product.columns:
            df_product['STD_7'] = df_product['Quantity'].rolling(window=7, min_periods=1).std().fillna(0)
        if 'Lag_1' not in df_product.columns:
            df_product['Lag_1'] = df_product['Quantity'].shift(1).fillna(0)
        if 'Lag_7' not in df_product.columns:
            df_product['Lag_7'] = df_product['Quantity'].shift(7).fillna(0)
        
        # Asegurar otras columnas
        if 'TotalPrice' not in df_product.columns:
            df_product['TotalPrice'] = 0
        if 'NumTransactions' not in df_product.columns:
            df_product['NumTransactions'] = 0
        
        return df_product
    
    def plot_forecast(self, df_historical, save=True):
        """Visualiza predicciones con histórico"""
        if self.predictions is None:
            print("⚠️  No hay predicciones generadas")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Gráfico 1: Serie completa con predicción
        ax1 = axes[0]
        
        # Histórico
        ax1.plot(df_historical['Date'], df_historical['Quantity'], 
                label='Histórico', linewidth=2, color='#2E86AB', alpha=0.8)
        
        # Predicción
        ax1.plot(self.predictions['Date'], self.predictions['Predicted_Quantity'],
                label='Predicción', linewidth=2, color='#A23B72', linestyle='--')
        
        # Intervalo de confianza
        ax1.fill_between(
            self.predictions['Date'],
            self.predictions['CI_Lower'],
            self.predictions['CI_Upper'],
            alpha=0.2, color='#A23B72', label='IC 95%'
        )
        
        ax1.axvline(x=df_historical['Date'].max(), color='red', 
                   linestyle=':', linewidth=2, label='Inicio Predicción')
        
        ax1.set_title(f'Predicción de Demanda - {self.product_id}', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Fecha', fontsize=12)
        ax1.set_ylabel('Cantidad', fontsize=12)
        ax1.legend(fontsize=10, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Solo predicción detallada
        ax2 = axes[1]
        
        ax2.plot(self.predictions['Date'], self.predictions['Predicted_Quantity'],
                marker='o', linewidth=2, markersize=4, color='#A23B72')
        ax2.fill_between(
            self.predictions['Date'],
            self.predictions['CI_Lower'],
            self.predictions['CI_Upper'],
            alpha=0.3, color='#A23B72'
        )
        
        # Añadir líneas de referencia
        mean_pred = self.predictions['Predicted_Quantity'].mean()
        ax2.axhline(y=mean_pred, color='green', linestyle='--', 
                   linewidth=1.5, label=f'Promedio: {mean_pred:.1f}')
        
        ax2.set_title('Detalle de Predicción (30 días)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Fecha', fontsize=12)
        ax2.set_ylabel('Cantidad Predicha', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(pred_config.PLOTS_DIR, f'{self.product_id}_forecast.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"   💾 Gráfico guardado: {plot_path}")
        
        plt.close()
    
    def generate_insights(self):
        """Genera insights sobre las predicciones"""
        if self.predictions is None:
            return None
        
        insights = {
            'product_id': self.product_id,
            'forecast_period': f"{self.predictions['Date'].min().strftime('%Y-%m-%d')} a {self.predictions['Date'].max().strftime('%Y-%m-%d')}",
            'total_predicted': float(self.predictions['Predicted_Quantity'].sum()),
            'daily_average': float(self.predictions['Predicted_Quantity'].mean()),
            'daily_std': float(self.predictions['Predicted_Quantity'].std()),
            'min_day': float(self.predictions['Predicted_Quantity'].min()),
            'max_day': float(self.predictions['Predicted_Quantity'].max()),
            'peak_date': self.predictions.loc[self.predictions['Predicted_Quantity'].idxmax(), 'Date'].strftime('%Y-%m-%d'),
            'low_date': self.predictions.loc[self.predictions['Predicted_Quantity'].idxmin(), 'Date'].strftime('%Y-%m-%d'),
        }
        
        # Análisis de tendencia
        first_week = self.predictions['Predicted_Quantity'].iloc[:7].mean()
        last_week = self.predictions['Predicted_Quantity'].iloc[-7:].mean()
        trend_change = ((last_week - first_week) / first_week) * 100
        
        insights['trend'] = 'creciente' if trend_change > 5 else 'decreciente' if trend_change < -5 else 'estable'
        insights['trend_change_pct'] = float(trend_change)
        
        # Recomendaciones de inventario
        safety_stock = insights['daily_average'] * 7  # 1 semana de stock de seguridad
        reorder_point = insights['daily_average'] * 14  # 2 semanas
        
        insights['recommendations'] = {
            'safety_stock': float(safety_stock),
            'reorder_point': float(reorder_point),
            'suggested_order_quantity': float(insights['total_predicted'] * 1.1)  # +10% buffer
        }
        
        return insights
    
    def save_predictions(self):
        """Guarda predicciones en Excel"""
        if self.predictions is None:
            return
        
        report_path = os.path.join(pred_config.REPORTS_DIR, f'{self.product_id}_forecast.xlsx')
        
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            # Predicciones
            self.predictions.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Insights
            insights = self.generate_insights()
            if insights:
                insights_df = pd.DataFrame([insights])
                insights_df.to_excel(writer, sheet_name='Insights', index=False)
            
            # Métricas del modelo
            if self.metrics:
                metrics_df = pd.DataFrame([self.metrics])
                metrics_df.to_excel(writer, sheet_name='Model_Metrics', index=False)
        
        print(f"   💾 Reporte guardado: {report_path}")


# ===============================
# Sistema de Predicción Batch
# ===============================
class BatchPredictor:
    def __init__(self):
        self.predictors = {}
        self.results = []
    
    def load_available_models(self):
        """Carga todos los modelos disponibles"""
        print("📂 Buscando modelos entrenados...")
        
        model_files = [f for f in os.listdir(pred_config.MODEL_DIR) if f.endswith('_model.h5')]
        product_ids = [f.replace('_model.h5', '') for f in model_files]
        
        print(f"   Modelos encontrados: {len(product_ids)}")
        
        for product_id in product_ids:
            predictor = ProductPredictor(product_id)
            if predictor.load_model():
                self.predictors[product_id] = predictor
        
        print(f"✅ Modelos cargados: {len(self.predictors)}")
        return list(self.predictors.keys())
    
    def predict_all(self, df_original, days_ahead=30):
        """Genera predicciones para todos los productos"""
        print("\n" + "="*70)
        print(" 🔮 GENERANDO PREDICCIONES PARA TODOS LOS PRODUCTOS")
        print("="*70)
        
        for product_id, predictor in self.predictors.items():
            print(f"\n{'─'*70}")
            try:
                # Filtrar y preparar datos del producto
                df_product = self._prepare_product_data(df_original, product_id)
                
                if df_product is None or len(df_product) < 21:
                    print(f"   ⚠️  Datos insuficientes para {product_id}")
                    continue
                
                # Generar predicciones
                predictions = predictor.forecast(df_product, days_ahead=days_ahead)
                
                # Visualizar
                predictor.plot_forecast(df_product, save=True)
                
                # Guardar
                predictor.save_predictions()
                
                # Insights
                insights = predictor.generate_insights()
                self.results.append(insights)
                
                print(f"   ✓ Completado: {product_id}")
                
            except Exception as e:
                print(f"   ❌ Error en {product_id}: {str(e)}")
                continue
        
        self._generate_summary_report()
    
    def _prepare_product_data(self, df_original, product_id):
        """Prepara datos de un producto para predicción"""
        df_product = df_original[df_original['StockCode'] == product_id].copy()
        
        if len(df_product) == 0:
            return None
        
        # Asegurar fecha
        df_product['InvoiceDate'] = pd.to_datetime(df_product['InvoiceDate'])
        
        # Agregar por día
        daily_agg = df_product.groupby(df_product['InvoiceDate'].dt.date).agg({
            'Quantity': 'sum',
            'UnitPrice': 'mean',
        }).reset_index()
        
        daily_agg.rename(columns={'InvoiceDate': 'Date'}, inplace=True)
        daily_agg['Date'] = pd.to_datetime(daily_agg['Date'])
        
        # Calcular TotalPrice
        daily_agg['TotalPrice'] = daily_agg['Quantity'] * daily_agg['UnitPrice']
        daily_agg['NumTransactions'] = 1  # Simplificado
        
        # Completar serie temporal
        date_range = pd.date_range(
            start=daily_agg['Date'].min(),
            end=daily_agg['Date'].max(),
            freq='D'
        )
        
        complete_df = pd.DataFrame({'Date': date_range})
        daily_agg = complete_df.merge(daily_agg, on='Date', how='left')
        
        # Rellenar valores faltantes
        daily_agg['Quantity'] = daily_agg['Quantity'].fillna(0)
        daily_agg['TotalPrice'] = daily_agg['TotalPrice'].fillna(0)
        daily_agg['NumTransactions'] = daily_agg['NumTransactions'].fillna(0)
        
        # Features temporales
        daily_agg['DayOfWeek'] = daily_agg['Date'].dt.dayofweek
        daily_agg['Month'] = daily_agg['Date'].dt.month
        daily_agg['IsWeekend'] = (daily_agg['DayOfWeek'] >= 5).astype(int)
        daily_agg['Season'] = daily_agg['Month'].apply(
            lambda m: 0 if m in [12,1,2] else 1 if m in [3,4,5] else 2 if m in [6,7,8] else 3
        )
        
        # Medias móviles
        daily_agg['MA_7'] = daily_agg['Quantity'].rolling(window=7, min_periods=1).mean()
        daily_agg['MA_14'] = daily_agg['Quantity'].rolling(window=14, min_periods=1).mean()
        daily_agg['STD_7'] = daily_agg['Quantity'].rolling(window=7, min_periods=1).std().fillna(0)
        
        # Lags
        daily_agg['Lag_1'] = daily_agg['Quantity'].shift(1).fillna(0)
        daily_agg['Lag_7'] = daily_agg['Quantity'].shift(7).fillna(0)
        
        return daily_agg
    
    def _generate_summary_report(self):
        """Genera reporte resumen de todas las predicciones"""
        print("\n" + "="*70)
        print(" 📊 RESUMEN DE PREDICCIONES")
        print("="*70)
        
        if not self.results:
            print("   ⚠️  No hay resultados para mostrar")
            return
        
        summary_df = pd.DataFrame(self.results)
        
        # Mostrar tabla
        display_cols = ['product_id', 'total_predicted', 'daily_average', 'trend', 'trend_change_pct']
        print("\n" + summary_df[display_cols].to_string(index=False))
        
        # Estadísticas
        print(f"\n📈 Estadísticas Generales:")
        print(f"   Productos analizados: {len(summary_df)}")
        print(f"   Demanda total predicha: {summary_df['total_predicted'].sum():,.0f} unidades")
        print(f"   Demanda promedio diaria: {summary_df['daily_average'].mean():.1f} unidades")
        
        # Top productos
        print(f"\n🏆 Top 5 productos con mayor demanda:")
        top5 = summary_df.nlargest(5, 'total_predicted')[['product_id', 'total_predicted', 'daily_average']]
        for idx, row in top5.iterrows():
            print(f"   {row['product_id']:15s} | Total: {row['total_predicted']:8.0f} | Promedio: {row['daily_average']:6.1f}")
        
        # Guardar resumen
        summary_path = os.path.join(pred_config.REPORTS_DIR, 'predictions_summary.xlsx')
        summary_df.to_excel(summary_path, index=False)
        print(f"\n💾 Resumen guardado: {summary_path}")
        print("\n" + "="*70)
        print(" ✅ PREDICCIONES COMPLETADAS")
        print("="*70)

# ===============================
# Ejecución Principal
# ===============================
if __name__ == "__main__":
    # Configurar visualización
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    print("="*70)
    print(" 🚀 SISTEMA DE PREDICCIÓN DE DEMANDA")
    print("="*70)
    
    # Inicializar batch predictor
    batch = BatchPredictor()
    
    # Cargar modelos
    available_models = batch.load_available_models()
    
    if not available_models:
        print("\n❌ No se encontraron modelos entrenados")
        print("   Ejecuta primero: python train_products.py")
        exit(1)
    
    # Cargar datos originales
    print(f"\n📂 Cargando datos desde: {pred_config.DATA_PATH}")
    df = pd.read_excel(pred_config.DATA_PATH, engine='openpyxl')
    print(f"   Registros cargados: {len(df)}")
    
    # Generar predicciones
    batch.predict_all(df, days_ahead=pred_config.FORECAST_DAYS)
    print(f"\n📁 Archivos generados:")
    print(f"   - Gráficos: {pred_config.PLOTS_DIR}/")
    print(f"   - Reportes: {pred_config.REPORTS_DIR}/")