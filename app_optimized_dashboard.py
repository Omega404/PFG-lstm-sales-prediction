"""
===============================================================================
DASHBOARD WEB - MODELO MULTIMODAL OPTIMIZADO
===============================================================================

Dashboard que muestra predicciones agregadas de negocio usando el modelo
LSTM multimodal optimizado basado en 209 experimentos.

Características:
- Predicciones agregadas (revenue, clientes, productos)
- Visualización de parámetros óptimos del modelo
- Tendencias históricas
- Métricas del modelo entrenado

Author: Sistema PFG LSTM
Date: Noviembre 2025
===============================================================================
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

# Fix encoding para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Importar el analizador optimizado
sys.path.append('src/analysis')
from optimized_multimodal_analyzer import OptimizedMultimodalAnalyzer, OptimizedModelConfig

app = Flask(__name__)
CORS(app)

# Variables globales
analyzer = None
last_prediction_date = None


# ===========================================================================
# INICIALIZACIÓN
# ===========================================================================

def initialize_analyzer():
    """Inicializar el analizador con modelo optimizado"""
    global analyzer, last_prediction_date

    print("=" * 80)
    print("INICIALIZANDO DASHBOARD - MODELO OPTIMIZADO")
    print("=" * 80)

    try:
        analyzer = OptimizedMultimodalAnalyzer()

        # 1. Cargar datos
        print("\n[1/3] Cargando dataset...")
        analyzer.load_data()

        # 2. Cargar modelo optimizado
        print("\n[2/3] Cargando modelo multimodal optimizado...")
        analyzer.load_model()

        # 3. Generar predicciones iniciales
        print("\n[3/3] Generando predicciones...")
        analyzer.predict_next_period()

        last_prediction_date = datetime.now()

        print("\n" + "=" * 80)
        print("[OK] DASHBOARD INICIALIZADO CORRECTAMENTE")
        print("=" * 80)

        return True

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPara usar este dashboard, primero entrena el modelo optimizado:")
        print("  python src/train/train_multimodal_lstm_optimized.py")
        return False
    except Exception as e:
        print(f"\n[ERROR] Error inicializando: {e}")
        return False


# ===========================================================================
# RUTAS - PÁGINAS HTML
# ===========================================================================

@app.route('/')
def index():
    """Página principal - Dashboard optimizado"""
    return render_template('optimized_dashboard.html')


# ===========================================================================
# API - PREDICCIONES
# ===========================================================================

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Obtener predicciones del modelo optimizado"""
    if analyzer is None or analyzer.predictions is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500

    return jsonify({
        'status': 'success',
        'predictions': analyzer.predictions,
        'last_update': last_prediction_date.isoformat() if last_prediction_date else None
    })


@app.route('/api/predictions/refresh', methods=['POST'])
def refresh_predictions():
    """Regenerar predicciones"""
    global last_prediction_date

    if analyzer is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500

    try:
        analyzer.predict_next_period()
        last_prediction_date = datetime.now()

        return jsonify({
            'status': 'success',
            'message': 'Predicciones actualizadas',
            'timestamp': last_prediction_date.isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===========================================================================
# API - MODELO Y CONFIGURACIÓN ÓPTIMA
# ===========================================================================

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Obtener información del modelo y parámetros óptimos"""
    if analyzer is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500

    return jsonify({
        'status': 'success',
        'model_info': analyzer.get_model_info()
    })


@app.route('/api/model/config', methods=['GET'])
def get_optimal_config():
    """Obtener configuración óptima derivada de experimentación"""
    config = OptimizedModelConfig.get_config_summary()

    return jsonify({
        'status': 'success',
        'optimal_config': config
    })


@app.route('/api/model/metrics', methods=['GET'])
def get_model_metrics():
    """Obtener métricas del modelo entrenado"""
    if analyzer is None or analyzer.model_metrics is None:
        return jsonify({'error': 'Model metrics not available'}), 500

    return jsonify({
        'status': 'success',
        'metrics': analyzer.model_metrics
    })


# ===========================================================================
# API - TENDENCIAS HISTÓRICAS
# ===========================================================================

@app.route('/api/trends', methods=['GET'])
def get_trends():
    """Obtener tendencias históricas para gráficos"""
    if analyzer is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500

    days = request.args.get('days', default=30, type=int)
    trends = analyzer.get_historical_trends(days=days)

    return jsonify({
        'status': 'success',
        'trends': trends
    })


# ===========================================================================
# API - ESTADO DEL SISTEMA
# ===========================================================================

@app.route('/api/status', methods=['GET'])
def get_status():
    """Estado del sistema"""
    if analyzer is None:
        return jsonify({
            'status': 'not_initialized',
            'message': 'Analyzer not initialized. Run train_multimodal_lstm_optimized.py first.'
        }), 500

    config = OptimizedModelConfig.get_config_summary()

    return jsonify({
        'status': 'operational',
        'model_type': 'Multimodal LSTM Optimizado',
        'based_on_experiments': config['experiments_count'],
        'last_prediction': last_prediction_date.isoformat() if last_prediction_date else None,
        'optimal_config': {
            'window_days': config['window_days'],
            'architecture': config['architecture'],
            'optimizer': config['optimizer'],
            'learning_rate': config['learning_rate']
        },
        'model_loaded': analyzer.model is not None,
        'data_loaded': analyzer.df is not None
    })


# ===========================================================================
# API - COMPARACIÓN CON HISTÓRICO
# ===========================================================================

@app.route('/api/comparison', methods=['GET'])
def get_comparison():
    """Comparar predicciones con datos históricos"""
    if analyzer is None or analyzer.predictions is None:
        return jsonify({'error': 'Predictions not available'}), 500

    predictions = analyzer.predictions['predictions']

    comparison = []
    for metric, data in predictions.items():
        change = ((data['predicted'] - data['historical_avg']) / data['historical_avg'] * 100) \
                 if data['historical_avg'] > 0 else 0

        comparison.append({
            'metric': metric,
            'predicted': data['predicted'],
            'historical': data['historical_avg'],
            'change_percent': round(change, 1),
            'trend': 'up' if change > 5 else ('down' if change < -5 else 'stable'),
            'unit': data['unit'],
            'description': data['description']
        })

    return jsonify({
        'status': 'success',
        'comparison': comparison
    })


# ===========================================================================
# API - EXPORTAR DATOS
# ===========================================================================

@app.route('/api/export', methods=['GET'])
def export_data():
    """Exportar todas las predicciones y configuración"""
    if analyzer is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500

    export_data = {
        'exported_at': datetime.now().isoformat(),
        'model_info': analyzer.get_model_info(),
        'predictions': analyzer.predictions,
        'optimal_config': OptimizedModelConfig.get_config_summary()
    }

    return jsonify(export_data)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    # Inicializar
    if initialize_analyzer():
        # Mostrar información del modelo
        config = OptimizedModelConfig.get_config_summary()

        print("\n" + "=" * 80)
        print("CONFIGURACIÓN ÓPTIMA DEL MODELO")
        print("=" * 80)
        print(f"\n   Basado en: {config['experiments_count']} experimentos")
        print(f"   Window: {config['window_days']} días")
        print(f"   Forecast: {config['forecast_days']} días")
        print(f"   Arquitectura: {config['architecture']}")
        print(f"   Optimizer: {config['optimizer']} (lr={config['learning_rate']})")
        print(f"   Batch size: {config['batch_size']}")
        print(f"\n   Loss Weights:")
        for key, value in config['loss_weights'].items():
            print(f"      {key}: {value}")
        print(f"\n   Hallazgos clave:")
        for finding in config['key_findings']:
            print(f"   - {finding}")

        # Iniciar servidor
        print("\n" + "=" * 80)
        print("SERVIDOR WEB")
        print("=" * 80)
        print("\n   Dashboard disponible en: http://localhost:5002")
        print("\n   API Endpoints:")
        print("   GET  /api/predictions      - Predicciones actuales")
        print("   GET  /api/model/info       - Info del modelo")
        print("   GET  /api/model/config     - Parámetros óptimos")
        print("   GET  /api/trends           - Tendencias históricas")
        print("   GET  /api/comparison       - Comparación pred vs histórico")
        print("   GET  /api/status           - Estado del sistema")
        print("   POST /api/predictions/refresh - Actualizar predicciones")
        print("\n   Presiona Ctrl+C para detener\n")

        port = int(os.environ.get('PORT', 5002))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("\n[ERROR] No se pudo inicializar el dashboard.")
        print("Asegúrate de haber entrenado el modelo primero:")
        print("  python src/train/train_multimodal_lstm_optimized.py")
