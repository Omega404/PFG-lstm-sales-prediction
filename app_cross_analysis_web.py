"""
Sistema Web de Análisis Cruzado - Clientes + Productos
Incluye TODAS las funcionalidades (útiles actuales + mejorables a futuro)
"""
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime, timedelta
import sys
import urllib.request

# Fix encoding para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ===========================================================================
# DESCARGA DE MODELOS DESDE CLOUD STORAGE
# ===========================================================================

def download_from_cloud_storage():
    """Descargar modelos y datos desde Google Cloud Storage"""
    print("\n" + "=" * 80)
    print("DESCARGANDO MODELOS Y DATOS DESDE CLOUD STORAGE")
    print("=" * 80)

    bucket_url = 'https://storage.googleapis.com/lstm-models-pfg'

    files_to_download = [
        # Customer V3 models
        ('customer_v3/medium/model_best.keras', 'models/temporal/customer_v3/medium/model_best.keras'),
        ('customer_v3/medium/scaler_X.pkl', 'models/temporal/customer_v3/medium/scaler_X.pkl'),
        ('customer_v3/medium/scaler_y_days.pkl', 'models/temporal/customer_v3/medium/scaler_y_days.pkl'),
        ('customer_v3/medium/scaler_y_value.pkl', 'models/temporal/customer_v3/medium/scaler_y_value.pkl'),
        ('customer_v3/medium/metrics.json', 'models/temporal/customer_v3/medium/metrics.json'),
        # Product models
        ('products/short/model_best.keras', 'models/temporal/products_50epochs/short/model_best.keras'),
        ('products/short/scaler_X.pkl', 'models/temporal/products_50epochs/short/scaler_X.pkl'),
        ('products/short/scaler_y.pkl', 'models/temporal/products_50epochs/short/scaler_y.pkl'),
        ('products/short/metrics.json', 'models/temporal/products_50epochs/short/metrics.json'),
        # Data
        ('data/online_retail_2.xlsx', 'data/processed/online_retail_2.xlsx'),
    ]

    for cloud_path, local_path in files_to_download:
        if os.path.exists(local_path):
            print(f"✓ {local_path} ya existe (skip)")
            continue

        url = f'{bucket_url}/{cloud_path}'
        print(f"⬇ Descargando {cloud_path}...")

        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(url, local_path)
            file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
            print(f"  ✓ Descargado ({file_size:.1f} MB)")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            raise

    print("\n✓ TODOS LOS ARCHIVOS DESCARGADOS")
    print("=" * 80 + "\n")

# Descargar modelos al inicio (solo una vez)
download_from_cloud_storage()

# Importar el analizador cruzado
sys.path.append('src/analysis')
from cross_analysis_customers_products import CustomerProductCrossAnalyzer, CrossAnalysisConfig

app = Flask(__name__)
CORS(app)

# Variable global para el analizador
analyzer = None
last_analysis_date = None

# ===========================================================================
# INICIALIZACIÓN
# ===========================================================================

def initialize_analyzer():
    """Inicializar el analizador al inicio"""
    global analyzer, last_analysis_date

    print("=" * 80)
    print("INICIALIZANDO SISTEMA WEB DE ANÁLISIS CRUZADO")
    print("=" * 80)

    analyzer = CustomerProductCrossAnalyzer()

    # Cargar datos y modelos
    print("\n[1/3] Cargando dataset...")
    analyzer.load_data()

    print("\n[2/3] Cargando modelos LSTM...")
    analyzer.load_customer_model(use_v3=True)
    analyzer.load_product_model()

    print("\n[3/3] Ejecutando predicciones iniciales...")
    analyzer.predict_customers(sample_size=100)
    analyzer.predict_products(top_n=50)
    analyzer.generate_cross_recommendations()

    last_analysis_date = datetime.now()

    print("\n✅ Sistema inicializado correctamente")
    print("=" * 80)

# ===========================================================================
# RUTAS - PÁGINAS HTML
# ===========================================================================

@app.route('/')
def index():
    """Página principal - Dashboard"""
    return render_template('cross_analysis_dashboard.html')

@app.route('/customers')
def customers_page():
    """Página de clientes"""
    return render_template('customers_detail.html')

@app.route('/products')
def products_page():
    """Página de productos"""
    return render_template('products_detail.html')

@app.route('/customer/<int:customer_id>')
def customer_detail(customer_id):
    """Detalle de un cliente específico"""
    return render_template('customer_detail.html', customer_id=customer_id)

# ===========================================================================
# API ENDPOINTS - FUNCIONALIDADES ÚTILES (ALTA CONFIABILIDAD)
# ===========================================================================

@app.route('/api/customers/ranking', methods=['GET'])
def get_customer_ranking():
    """
    ✅ ÚTIL: Ranking de clientes por probabilidad
    Confiabilidad: ALTA (87% accuracy)
    """
    if analyzer is None or analyzer.customer_predictions is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500

    # Obtener parámetros
    top_n = request.args.get('top_n', default=10, type=int)
    segment = request.args.get('segment', default=None, type=str)

    # Filtrar por segmento si se especifica
    df = analyzer.customer_predictions.copy()
    if segment:
        df = df[df['segment'] == segment]

    # Ordenar por probabilidad y tomar top N
    df = df.nlargest(top_n, 'purchase_probability')

    # Preparar respuesta
    result = {
        'total_customers': len(analyzer.customer_predictions),
        'filtered_customers': len(df),
        'ranking': df.to_dict(orient='records'),
        'metadata': {
            'last_update': last_analysis_date.isoformat() if last_analysis_date else None,
            'model_accuracy': 87.59,
            'confidence': 'ALTA'
        }
    }

    return jsonify(result)

@app.route('/api/customers/segments', methods=['GET'])
def get_customer_segments():
    """
    ✅ ÚTIL: Segmentación de clientes
    Confiabilidad: ALTA
    """
    if analyzer is None or analyzer.customer_predictions is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500

    # Contar por segmento
    segments_count = analyzer.customer_predictions['segment'].value_counts().to_dict()
    total = len(analyzer.customer_predictions)

    # Formato esperado por el frontend
    result = {
        'total_customers': total,
        'segments': {
            'high_value_high_prob': segments_count.get('high_value_high_prob', 0),
            'medium_prob': segments_count.get('medium_prob', 0),
            'low_prob': segments_count.get('low_prob', 0)
        },
        'strategies': {
            'high_value_high_prob': 'Contacto VIP inmediato - Oferta premium',
            'medium_prob': 'Campaña general con descuentos',
            'low_prob': 'No contactar (bajo ROI)'
        },
        'metadata': {
            'confidence': 'ALTA',
            'model_accuracy': 87.59
        }
    }

    return jsonify(result)

@app.route('/api/customers/<int:customer_id>/details', methods=['GET'])
def get_customer_details(customer_id):
    """
    ✅ ÚTIL: Detalle de cliente + recomendaciones
    Confiabilidad: ALTA para probabilidad y recomendaciones
    """
    if analyzer is None or analyzer.customer_predictions is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500

    # Buscar cliente
    customer = analyzer.customer_predictions[
        analyzer.customer_predictions['customer_id'] == customer_id
    ]

    if len(customer) == 0:
        return jsonify({'error': 'Customer not found'}), 404

    customer = customer.iloc[0].to_dict()

    # Obtener recomendaciones para este cliente
    recommendations = []
    if analyzer.recommendations is not None:
        recs = analyzer.recommendations[
            analyzer.recommendations['customer_id'] == customer_id
        ].nlargest(5, 'recommendation_score')
        recommendations = recs.to_dict(orient='records')

    # Calcular timing (ventana estimada)
    days = customer.get('predicted_days', 7)
    if days <= 7:
        timing = 'ESTA SEMANA'
        timing_class = 'urgent'
    elif days <= 14:
        timing = 'PRÓXIMA SEMANA'
        timing_class = 'high'
    else:
        timing = 'LARGO PLAZO'
        timing_class = 'medium'

    result = {
        'customer': customer,
        'recommendations': recommendations,
        'timing': {
            'window': f'{int(days)}-{int(days)+5} días',
            'label': timing,
            'class': timing_class
        },
        'metadata': {
            'confidence_probability': 'ALTA (87%)',
            'confidence_recommendations': 'ALTA',
            'confidence_value': 'BAJA (solo referencial)',  # ⚠️ Advertencia
        }
    }

    return jsonify(result)

@app.route('/api/products/ranking', methods=['GET'])
def get_product_ranking():
    """
    ✅ ÚTIL: Ranking RELATIVO de productos
    Confiabilidad: ALTA para comparación relativa
    """
    if analyzer is None or analyzer.product_predictions is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500

    top_n = request.args.get('top_n', default=20, type=int)

    # Obtener productos ordenados por demanda relativa
    df = analyzer.product_predictions.copy()
    df = df.nlargest(top_n, 'predicted_demand_7d')

    # Agregar ranking relativo
    df['ranking'] = range(1, len(df) + 1)

    # Calcular clientes potenciales
    if analyzer.recommendations is not None:
        customer_counts = analyzer.recommendations.groupby('stock_code')['customer_id'].count()
        df['n_potential_customers'] = df['stock_code'].map(customer_counts).fillna(0).astype(int)
    else:
        df['n_potential_customers'] = 0

    result = {
        'total_products': len(analyzer.product_predictions),
        'ranking': df.to_dict(orient='records'),
        'metadata': {
            'note': 'Rankings son RELATIVOS. Producto #1 tiene más demanda que #2.',
            'confidence': 'ALTA para comparación',
            'warning': 'Cantidades exactas NO confiables (MAE alto)'
        }
    }

    return jsonify(result)

@app.route('/api/products/<stock_code>/cross-sell', methods=['GET'])
def get_product_cross_sell(stock_code):
    """
    ✅ ÚTIL: Productos relacionados / cross-sell
    Confiabilidad: ALTA
    """
    if analyzer is None or analyzer.df is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500

    # Buscar clientes que compraron este producto
    customers_bought = set(analyzer.df[
        analyzer.df['StockCode'] == stock_code
    ]['CustomerID'].unique())

    if len(customers_bought) == 0:
        return jsonify({'error': 'Product not found or no purchases'}), 404

    # Buscar otros productos que compraron esos clientes
    other_products = analyzer.df[
        (analyzer.df['CustomerID'].isin(customers_bought)) &
        (analyzer.df['StockCode'] != stock_code)
    ].groupby('StockCode').agg({
        'CustomerID': 'nunique',
        'Description': 'first',
        'UnitPrice': 'mean'
    }).reset_index()

    other_products.columns = ['stock_code', 'common_customers', 'description', 'avg_price']
    other_products['cross_sell_rate'] = (
        other_products['common_customers'] / len(customers_bought) * 100
    ).round(2)

    # Top 10 productos relacionados
    other_products = other_products.nlargest(10, 'cross_sell_rate')

    result = {
        'product': stock_code,
        'total_customers': len(customers_bought),
        'related_products': other_products.to_dict(orient='records'),
        'metadata': {
            'confidence': 'ALTA',
            'note': 'Basado en patrones históricos reales'
        }
    }

    return jsonify(result)

# ===========================================================================
# API ENDPOINTS - FUNCIONALIDADES MEJORABLES (BAJA CONFIABILIDAD ACTUAL)
# ===========================================================================

@app.route('/api/forecast/revenue', methods=['GET'])
def get_revenue_forecast():
    """
    ⚠️ MEJORABLE: Forecast de ingresos
    Confiabilidad actual: BAJA (MAE $123)
    TODO: Mejorar con más datos/mejor entrenamiento
    """
    if analyzer is None or analyzer.customer_predictions is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500

    # Calcular forecast por segmento (con advertencia de baja confiabilidad)
    df = analyzer.customer_predictions.copy()

    by_segment = df.groupby('segment')['predicted_value'].sum().to_dict()

    total_revenue = df['predicted_value'].sum()
    high_prob_customers = len(df[df['purchase_probability'] >= 70])

    result = {
        'forecast': {
            'total_revenue_7d': round(total_revenue, 2),
            'revenue_by_segment': {
                'high_value_high_prob': round(by_segment.get('high_value_high_prob', 0), 2),
                'medium_prob': round(by_segment.get('medium_prob', 0), 2),
                'low_prob': round(by_segment.get('low_prob', 0), 2)
            },
            'high_prob_customers': high_prob_customers
        },
        'metadata': {
            'confidence': 'BAJA',
            'mae': 123.72,
            'warning': '⚠️ Valores monetarios tienen ~40% error. Usar solo como estimación aproximada.',
            'status': 'MEJORABLE - Requiere más datos de entrenamiento'
        }
    }

    return jsonify(result)

@app.route('/api/forecast/inventory', methods=['GET'])
def get_inventory_forecast():
    """
    ⚠️ MEJORABLE: Forecast de inventario en unidades
    Confiabilidad actual: BAJA (MAE > mean)
    TODO: Mejorar modelo de productos
    """
    if analyzer is None or analyzer.product_predictions is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500

    top_n = request.args.get('top_n', default=5, type=int)

    # Usar predicciones de productos
    df = analyzer.product_predictions.copy()
    df = df.nlargest(top_n, 'predicted_demand_7d')

    # Renombrar columna para el frontend
    df['total_forecast'] = df['predicted_demand_7d']

    result = {
        'forecast': df.to_dict(orient='records'),
        'metadata': {
            'confidence': 'BAJA',
            'mae': 19.17,
            'warning': '⚠️ Cantidades exactas NO confiables. Usar ranking relativo solamente.',
            'recommendation': 'NO usar para planificación de inventario precisa',
            'status': 'MEJORABLE - Requiere mejor modelo de productos'
        }
    }

    return jsonify(result)

@app.route('/api/campaign/roi', methods=['POST'])
def calculate_campaign_roi():
    """
    ⚠️ MEJORABLE: Cálculo de ROI de campaña
    Confiabilidad actual: BAJA (basado en valores predichos imprecisos)
    TODO: Validar con conversiones reales
    """
    data = request.json
    cost_per_contact = data.get('cost_per_contact', 5)

    if analyzer is None or analyzer.customer_predictions is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500

    # Clientes a contactar (alta + media probabilidad)
    contactable = analyzer.customer_predictions[
        analyzer.customer_predictions['purchase_probability'] >= 50
    ]

    campaign_cost = len(contactable) * cost_per_contact
    expected_revenue = contactable['predicted_value'].sum()
    roi = ((expected_revenue - campaign_cost) / campaign_cost * 100) if campaign_cost > 0 else 0

    result = {
        'campaign': {
            'customers_to_contact': len(contactable),
            'cost_per_contact': cost_per_contact,
            'total_cost': round(campaign_cost, 2),
            'expected_revenue': round(expected_revenue, 2),
            'roi_percentage': round(roi, 2)
        },
        'metadata': {
            'confidence': 'BAJA',
            'warning': '⚠️ ROI calculado con valores predichos imprecisos (~40% error)',
            'recommendation': 'Usar solo como guía aproximada. Validar con conversiones reales.',
            'status': 'MEJORABLE - Requiere tracking de conversión real'
        }
    }

    return jsonify(result)

# ===========================================================================
# API ENDPOINTS - UTILIDADES
# ===========================================================================

@app.route('/api/status', methods=['GET'])
def get_status():
    """Estado del sistema"""
    if analyzer is None:
        return jsonify({
            'status': 'not_initialized',
            'message': 'Analyzer not initialized'
        }), 500

    result = {
        'status': 'operational',
        'last_analysis': last_analysis_date.isoformat() if last_analysis_date else None,
        'total_customers_analyzed': len(analyzer.customer_predictions) if analyzer.customer_predictions is not None else 0,
        'total_products_analyzed': len(analyzer.product_predictions) if analyzer.product_predictions is not None else 0,
        'total_recommendations': len(analyzer.recommendations) if analyzer.recommendations is not None else 0,
        'models': {
            'customer_model': 'V3 (7 días)',
            'customer_accuracy': 87.59,
            'customer_auc': 0.66,
            'product_model': 'SHORT (7 días)',
            'product_mae': 19.17
        }
    }

    return jsonify(result)

@app.route('/api/refresh', methods=['POST'])
def refresh_predictions():
    """Ejecutar nuevas predicciones"""
    global last_analysis_date

    try:
        sample_size = request.json.get('sample_size', 100) if request.json else 100

        print(f"\n🔄 Ejecutando nuevas predicciones (sample={sample_size})...")

        analyzer.predict_customers(sample_size=sample_size)
        analyzer.predict_products(top_n=50)
        analyzer.generate_cross_recommendations()

        last_analysis_date = datetime.now()

        return jsonify({
            'status': 'success',
            'message': f'Predicciones actualizadas con {sample_size} clientes',
            'timestamp': last_analysis_date.isoformat()
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/export/<format>', methods=['GET'])
def export_results(format):
    """Exportar resultados"""
    if analyzer is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500

    try:
        if format == 'csv':
            analyzer.export_results()
            return jsonify({
                'status': 'success',
                'message': 'Resultados exportados a output/cross_analysis/'
            })
        elif format == 'json':
            # Exportar como JSON
            result = {
                'customers': analyzer.customer_predictions.to_dict(orient='records') if analyzer.customer_predictions is not None else [],
                'products': analyzer.product_predictions.to_dict(orient='records') if analyzer.product_predictions is not None else [],
                'recommendations': analyzer.recommendations.to_dict(orient='records') if analyzer.recommendations is not None else []
            }
            return jsonify(result)
        else:
            return jsonify({'error': 'Invalid format'}), 400

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    # Inicializar el analizador al arrancar
    initialize_analyzer()

    # Iniciar servidor
    print("\n🚀 Iniciando servidor web...")
    print("📊 Dashboard disponible en: http://localhost:5001")
    print("\nEndpoints API disponibles:")
    print("  ✅ ALTA CONFIABILIDAD:")
    print("     GET  /api/customers/ranking")
    print("     GET  /api/customers/segments")
    print("     GET  /api/customers/<id>/details")
    print("     GET  /api/products/ranking")
    print("     GET  /api/products/<code>/cross-sell")
    print("\n  ⚠️  MEJORABLE (baja confiabilidad actual):")
    print("     GET  /api/forecast/revenue")
    print("     GET  /api/forecast/inventory")
    print("     POST /api/campaign/roi")
    print("\n  🔧 UTILIDADES:")
    print("     GET  /api/status")
    print("     POST /api/refresh")
    print("     GET  /api/export/<format>")
    print("\nPresiona Ctrl+C para detener el servidor\n")

    # Puerto configurable para Cloud Run (usa PORT env var, default 5001)
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
