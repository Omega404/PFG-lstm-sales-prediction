import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ===============================
# Configuración
# ===============================
DATA_PATH = "data/processed/product_demand.xlsx"
MODEL_DIR = "models/trained"
REPORTS_DIR = "results/reports"
PLOTS_DIR = "results/plots"

# Hiperparámetros
WINDOW_SIZE = 7
FORECAST_HORIZON = 3
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SIZE = 0.2

# Criterios de filtrado OPTIMIZADOS (más estrictos para mejor calidad)
FILTER_CONFIG = {
    'min_transactions': 650,        # ↑ Más datos = mejor modelo
    'min_days_with_sales': 130,     # ↑ Buena cobertura temporal  
    'max_cv': 200,                  # ↓ Solo productos estables ⭐ CLAVE
    'max_negative_pct': 3,          # ↓ Pocas devoluciones
    'min_avg_quantity': 4,          # ↑ Productos con volumen
    'min_sales_density': 30         # ↑ Ventas más frecuentes
}

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===============================
# Funciones de análisis de productos
# ===============================
def analyze_all_products(file_path):
    """Analiza todos los productos y calcula métricas de calidad"""
    print("📊 Analizando todos los productos del dataset...")
    
    df = pd.read_excel(file_path, engine="openpyxl")
    df['StockCode'] = df['StockCode'].astype(str).str.strip()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    
    print(f"   Total registros: {len(df):,}")
    print(f"   Productos únicos: {df['StockCode'].nunique():,}")
    
    # Calcular métricas por producto
    products_analysis = []
    
    for stock_code in df['StockCode'].unique():
        product_data = df[df['StockCode'] == stock_code]
        quantities = product_data['Quantity'].values
        dates = product_data['InvoiceDate']
        
        # Métricas básicas
        total_transactions = len(product_data)
        mean_qty = np.mean(quantities)
        std_qty = np.std(quantities)
        cv = (std_qty / abs(mean_qty) * 100) if mean_qty != 0 else 9999
        
        # Métricas temporales
        date_range = (dates.max() - dates.min()).days
        unique_days = dates.dt.date.nunique()
        sales_density = (unique_days / date_range * 100) if date_range > 0 else 0
        
        # Devoluciones
        negative_count = (quantities < 0).sum()
        negative_pct = (negative_count / total_transactions * 100)
        
        # Descripción
        description = product_data['Description'].iloc[0] if 'Description' in product_data.columns else 'N/A'
        
        products_analysis.append({
            'StockCode': stock_code,
            'Description': str(description)[:50],
            'Transactions': total_transactions,
            'UniqueDays': unique_days,
            'DateRange': date_range,
            'SalesDensity': sales_density,
            'MeanQty': mean_qty,
            'StdQty': std_qty,
            'CV': cv,
            'NegativeCount': negative_count,
            'NegativePct': negative_pct,
            'MinQty': quantities.min(),
            'MaxQty': quantities.max()
        })
    
    df_analysis = pd.DataFrame(products_analysis)
    
    print(f"\n✅ Análisis completo de {len(df_analysis)} productos")
    return df_analysis

def filter_products(df_analysis, config):
    """Filtra productos según criterios de calidad"""
    print(f"\n🔍 Aplicando filtros de calidad...")
    print(f"   Criterios:")
    for key, value in config.items():
        print(f"   - {key}: {value}")
    
    initial_count = len(df_analysis)
    
    # Aplicar filtros
    filtered = df_analysis[
        (df_analysis['Transactions'] >= config['min_transactions']) &
        (df_analysis['UniqueDays'] >= config['min_days_with_sales']) &
        (df_analysis['CV'] < config['max_cv']) &
        (df_analysis['NegativePct'] < config['max_negative_pct']) &
        (df_analysis['MeanQty'] >= config['min_avg_quantity']) &
        (df_analysis['SalesDensity'] >= config['min_sales_density'])
    ].copy()
    
    # Ordenar por calidad (CV más bajo = mejor)
    filtered = filtered.sort_values('CV').reset_index(drop=True)
    
    print(f"\n📉 Resultados del filtrado:")
    print(f"   Total inicial: {initial_count}")
    print(f"   Después de filtros: {len(filtered)} ({len(filtered)/initial_count*100:.1f}%)")
    print(f"   Productos rechazados: {initial_count - len(filtered)}")
    
    return filtered

def save_analysis_report(df_analysis, filtered_products, output_path):
    """Guarda reporte de análisis"""
    report = {
        'fecha_analisis': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_productos': len(df_analysis),
        'productos_aptos': len(filtered_products),
        'tasa_aprobacion': f"{len(filtered_products)/len(df_analysis)*100:.1f}%",
        'filtros_aplicados': FILTER_CONFIG,
        'estadisticas_generales': {
            'cv_promedio': float(df_analysis['CV'].mean()),
            'cv_mediana': float(df_analysis['CV'].median()),
            'transacciones_promedio': float(df_analysis['Transactions'].mean()),
            'densidad_ventas_promedio': float(df_analysis['SalesDensity'].mean())
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Guardar lista de productos aptos
    filtered_products.to_excel(
        output_path.replace('.json', '_productos_aptos.xlsx'),
        index=False
    )
    
    print(f"\n📄 Reporte guardado en: {output_path}")

# ===============================
# Funciones de entrenamiento
# ===============================
def create_sequences(data, window_size, forecast_horizon):
    """Crea secuencias X (pasado) -> y (futuro)"""
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+forecast_horizon, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, forecast_horizon):
    """Construye arquitectura LSTM"""
    model = Sequential([
        LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, activation='tanh', return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(forecast_horizon)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
    return model

def prepare_product_data(df, stock_code):
    """Prepara datos de un producto para entrenamiento"""
    df['StockCode'] = df['StockCode'].astype(str).str.strip()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    
    # Filtrar producto
    df_product = df[df['StockCode'] == stock_code].copy()
    
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
    window_smooth = 7
    daily_sales['Quantity'] = daily_sales['Quantity'].rolling(
        window=window_smooth, center=True, min_periods=1
    ).mean()
    
    return daily_sales[['Quantity', 'AvgPrice']].values

def train_single_product(df, stock_code, description):
    """Entrena modelo para un producto"""
    try:
        # Preparar datos
        features = prepare_product_data(df, stock_code)
        
        if len(features) < WINDOW_SIZE + FORECAST_HORIZON + 30:
            return None, "Datos insuficientes"
        
        # Normalizar
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled = scaler.fit_transform(features)
        
        # Crear secuencias
        X, y = create_sequences(features_scaled, WINDOW_SIZE, FORECAST_HORIZON)
        
        # Split
        split_idx = int(len(X) * (1 - TEST_SIZE))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        if len(X_train) < 50:
            return None, "Datos de entrenamiento insuficientes"
        
        # Construir modelo
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]), FORECAST_HORIZON)
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0)
        
        # Entrenar
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
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
        
        for i in range(FORECAST_HORIZON):
            temp_test = np.column_stack([y_test[:, i], np.zeros(len(y_test))])
            temp_pred = np.column_stack([y_pred[:, i], np.zeros(len(y_pred))])
            y_test_inv[:, i] = scaler.inverse_transform(temp_test)[:, 0]
            y_pred_inv[:, i] = scaler.inverse_transform(temp_pred)[:, 0]
        
        # Métricas
        mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
        rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten()))
        r2 = r2_score(y_test_inv.flatten(), y_pred_inv.flatten())
        
        # Guardar modelo
        model_path = os.path.join(MODEL_DIR, f"lstm_{stock_code}.h5")
        scaler_path = os.path.join(MODEL_DIR, f"scaler_{stock_code}.pkl")
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'epochs_trained': len(history.history['loss']),
            'final_loss': history.history['val_loss'][-1]
        }, "Éxito"
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# ===============================
# Entrenamiento masivo
# ===============================
def train_multiple_products(top_n=20):
    """Entrena modelos para múltiples productos"""
    print("\n" + "="*70)
    print("🚀 ENTRENAMIENTO MASIVO DE PRODUCTOS")
    print("="*70 + "\n")
    
    # Analizar productos
    df_analysis = analyze_all_products(DATA_PATH)
    
    # Filtrar productos aptos
    filtered = filter_products(df_analysis, FILTER_CONFIG)
    
    # Guardar reporte de análisis
    analysis_report_path = os.path.join(REPORTS_DIR, 'analisis_productos.json')
    save_analysis_report(df_analysis, filtered, analysis_report_path)
    
    # Seleccionar top N productos
    products_to_train = filtered.head(top_n)
    
    print(f"\n🎯 Productos seleccionados para entrenamiento: {len(products_to_train)}")
    print("\nTop 10 candidatos:")
    for i, row in products_to_train.head(10).iterrows():
        print(f"   {i+1}. {row['StockCode']} - CV: {row['CV']:.1f}% - {row['Description']}")
    
    # Cargar datos completos
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    
    # Entrenar cada producto
    results = []
    print(f"\n{'='*70}")
    print("⚙️  INICIANDO ENTRENAMIENTO")
    print(f"{'='*70}\n")
    
    for idx, row in products_to_train.iterrows():
        stock_code = row['StockCode']
        description = row['Description']
        
        print(f"[{idx+1}/{len(products_to_train)}] Entrenando {stock_code}... ", end='')
        
        metrics, status = train_single_product(df, stock_code, description)
        
        if metrics:
            print(f"✅ R²={metrics['r2']:.3f} | MAE={metrics['mae']:.2f}")
            results.append({
                'StockCode': stock_code,
                'Description': description,
                'Status': status,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'R2': metrics['r2'],
                'Epochs': metrics['epochs_trained'],
                'CV': row['CV'],
                'Transactions': row['Transactions']
            })
        else:
            print(f"❌ {status}")
            results.append({
                'StockCode': stock_code,
                'Description': description,
                'Status': status,
                'MAE': None,
                'RMSE': None,
                'R2': None,
                'Epochs': None,
                'CV': row['CV'],
                'Transactions': row['Transactions']
            })
    
    # Guardar resultados con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_results = pd.DataFrame(results)
    
    # Agregar información de la iteración
    df_results['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_results['Iteracion'] = timestamp
    df_results['FiltroCV'] = FILTER_CONFIG['max_cv']
    df_results['FiltroMinTrans'] = FILTER_CONFIG['min_transactions']
    
    # ===== SISTEMA DE TRACKING HISTÓRICO =====
    results_path = os.path.join(REPORTS_DIR, 'resultados_entrenamiento.xlsx')
    history_path = os.path.join(REPORTS_DIR, f'entrenamiento_{timestamp}.xlsx')
    
    # Guardar resultado actual con timestamp
    df_results.to_excel(history_path, index=False)
    print(f"\n📄 Resultado guardado: {history_path}")
    
    # Actualizar archivo histórico acumulativo
    if os.path.exists(results_path):
        # Leer resultados anteriores
        df_historico = pd.read_excel(results_path)
        
        # Agregar nuevos resultados
        df_combined = pd.concat([df_historico, df_results], ignore_index=True)
        
        # Guardar archivo combinado
        df_combined.to_excel(results_path, index=False)
        print(f"📄 Historial actualizado: {results_path}")
        print(f"   Total registros históricos: {len(df_combined)}")
    else:
        # Primera ejecución: crear archivo histórico
        df_results.to_excel(results_path, index=False)
        print(f"📄 Historial creado: {results_path}")
    
    # Resumen con comparación histórica
    successful = df_results[df_results['Status'] == 'Éxito']
    
    print(f"\n{'='*70}")
    print("📊 RESUMEN DE ENTRENAMIENTO")
    print(f"{'='*70}")
    print(f"Total productos entrenados: {len(products_to_train)}")
    print(f"Exitosos: {len(successful)} ({len(successful)/len(products_to_train)*100:.1f}%)")
    print(f"Fallidos: {len(products_to_train) - len(successful)}")
    
    if len(successful) > 0:
        print(f"\n📈 Estadísticas de modelos exitosos:")
        print(f"   MAE promedio: {successful['MAE'].mean():.2f} ± {successful['MAE'].std():.2f}")
        print(f"   RMSE promedio: {successful['RMSE'].mean():.2f} ± {successful['RMSE'].std():.2f}")
        print(f"   R² promedio: {successful['R2'].mean():.3f} ± {successful['R2'].std():.3f}")
        print(f"   R² mínimo: {successful['R2'].min():.3f}")
        print(f"   R² máximo: {successful['R2'].max():.3f}")
        print(f"   Mejor modelo: {successful.loc[successful['R2'].idxmax(), 'StockCode']} (R²={successful['R2'].max():.3f})")
        
        # Distribución por calidad
        excelentes = (successful['R2'] >= 0.6).sum()
        buenos = ((successful['R2'] >= 0.4) & (successful['R2'] < 0.6)).sum()
        aceptables = ((successful['R2'] >= 0.3) & (successful['R2'] < 0.4)).sum()
        malos = (successful['R2'] < 0.3).sum()
        
        print(f"\n📊 Distribución por calidad:")
        print(f"   Excelentes (R²≥0.6): {excelentes} ({excelentes/len(successful)*100:.1f}%)")
        print(f"   Buenos (0.4≤R²<0.6): {buenos} ({buenos/len(successful)*100:.1f}%)")
        print(f"   Aceptables (0.3≤R²<0.4): {aceptables} ({aceptables/len(successful)*100:.1f}%)")
        print(f"   Malos (R²<0.3): {malos} ({malos/len(successful)*100:.1f}%)")
    
    # Comparación con historial si existe
    if os.path.exists(results_path):
        df_historico = pd.read_excel(results_path)
        
        # Contar iteraciones únicas
        iteraciones_anteriores = df_historico['Iteracion'].nunique() - 1  # -1 para excluir la actual
        
        if iteraciones_anteriores > 0:
            print(f"\n📈 COMPARACIÓN CON ITERACIONES ANTERIORES:")
            print(f"   Total iteraciones realizadas: {iteraciones_anteriores + 1}")
            
            # Comparar con última iteración
            iteraciones = df_historico['Iteracion'].unique()
            if len(iteraciones) >= 2:
                iter_anterior = iteraciones[-2]
                iter_actual = iteraciones[-1]
                
                df_anterior = df_historico[df_historico['Iteracion'] == iter_anterior]
                df_actual = df_historico[df_historico['Iteracion'] == iter_actual]
                
                exit_anterior = df_anterior[df_anterior['Status'] == 'Éxito']
                exit_actual = df_actual[df_actual['Status'] == 'Éxito']
                
                if len(exit_anterior) > 0 and len(exit_actual) > 0:
                    r2_anterior = exit_anterior['R2'].mean()
                    r2_actual = exit_actual['R2'].mean()
                    mejora = ((r2_actual - r2_anterior) / abs(r2_anterior)) * 100
                    
                    print(f"\n   Iteración anterior:")
                    print(f"     R² promedio: {r2_anterior:.3f}")
                    print(f"     Filtro CV: {df_anterior['FiltroCV'].iloc[0]}")
                    
                    print(f"\n   Iteración actual:")
                    print(f"     R² promedio: {r2_actual:.3f}")
                    print(f"     Filtro CV: {df_actual['FiltroCV'].iloc[0]}")
                    print(f"     Mejora: {mejora:+.1f}%")
    
    print(f"\n📁 Archivos generados:")
    print(f"   - {analysis_report_path}")
    print(f"   - {history_path}")
    print(f"   - {results_path} (historial acumulativo)")
    print(f"\n✅ Proceso completado!")

# ===============================
# Análisis histórico
# ===============================
def analyze_training_history():
    """Analiza el historial completo de entrenamientos"""
    results_path = os.path.join(REPORTS_DIR, 'resultados_entrenamiento.xlsx')
    
    if not os.path.exists(results_path):
        print("❌ No hay historial de entrenamientos disponible")
        return
    
    df = pd.read_excel(results_path)
    
    print("\n" + "="*70)
    print("📊 ANÁLISIS HISTÓRICO DE ENTRENAMIENTOS")
    print("="*70 + "\n")
    
    # Información general
    iteraciones = df['Iteracion'].unique()
    print(f"Total de iteraciones: {len(iteraciones)}")
    print(f"Total de modelos entrenados: {len(df)}")
    print(f"Total de modelos exitosos: {len(df[df['Status'] == 'Éxito'])}")
    
    # Por iteración
    print(f"\n{'='*70}")
    print("RESULTADOS POR ITERACIÓN")
    print(f"{'='*70}\n")
    
    for i, iter_id in enumerate(iteraciones, 1):
        df_iter = df[df['Iteracion'] == iter_id]
        exitosos = df_iter[df_iter['Status'] == 'Éxito']
        
        print(f"Iteración {i} ({iter_id}):")
        print(f"  Timestamp: {df_iter['Timestamp'].iloc[0]}")
        print(f"  Filtros: CV<{df_iter['FiltroCV'].iloc[0]}, Trans≥{df_iter['FiltroMinTrans'].iloc[0]}")
        print(f"  Modelos: {len(exitosos)}/{len(df_iter)} exitosos")
        
        if len(exitosos) > 0:
            print(f"  R² promedio: {exitosos['R2'].mean():.3f}")
            print(f"  R² rango: [{exitosos['R2'].min():.3f}, {exitosos['R2'].max():.3f}]")
            print(f"  MAE promedio: {exitosos['MAE'].mean():.2f}")
            
            # Contar por calidad
            excelentes = (exitosos['R2'] >= 0.6).sum()
            buenos = ((exitosos['R2'] >= 0.4) & (exitosos['R2'] < 0.6)).sum()
            print(f"  Calidad: {excelentes} excelentes, {buenos} buenos")
        
        print()
    
    # Mejor modelo de todos los tiempos
    exitosos_total = df[df['Status'] == 'Éxito']
    if len(exitosos_total) > 0:
        mejor = exitosos_total.loc[exitosos_total['R2'].idxmax()]
        print(f"🏆 MEJOR MODELO DE TODOS LOS TIEMPOS:")
        print(f"   Producto: {mejor['StockCode']}")
        print(f"   R²: {mejor['R2']:.3f}")
        print(f"   MAE: {mejor['MAE']:.2f}")
        print(f"   Iteración: {mejor['Iteracion']}")
    
    # Top 10 modelos históricos
    print(f"\n📈 TOP 10 MODELOS HISTÓRICOS (por R²):")
    top10 = exitosos_total.nlargest(10, 'R2')
    for i, (idx, row) in enumerate(top10.iterrows(), 1):
        print(f"   {i}. {row['StockCode']} - R²: {row['R2']:.3f} | MAE: {row['MAE']:.2f} (Iter: {row['Iteracion'][:8]})")

# ===============================
# Ejecución principal
# ===============================
if __name__ == "__main__":
    import sys
    
    # Permitir ver historial con argumento
    if len(sys.argv) > 1 and sys.argv[1] == '--history':
        analyze_training_history()
    else:
        # Entrenar Top 20 productos más estables
        train_multiple_products(top_n=20)