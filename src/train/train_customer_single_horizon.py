"""
═══════════════════════════════════════════════════════════════════════════════
CUSTOMER TEMPORAL ANALYZER - ENTRENAMIENTO POR HORIZONTE INDIVIDUAL
═══════════════════════════════════════════════════════════════════════════════

Permite entrenar SHORT, MEDIUM o LONG de forma independiente.

Uso:
    python src/train/train_customer_single_horizon.py --horizon short
    python src/train/train_customer_single_horizon.py --horizon medium
    python src/train/train_customer_single_horizon.py --horizon long

Author: Sistema PFG LSTM
Date: 2025
"""

import argparse
import sys
from datetime import datetime
from train_all_customers_temporal import CustomerTemporalAnalyzer, TemporalConfig


def main():
    """Entrena un solo horizonte temporal"""

    # Parser de argumentos
    parser = argparse.ArgumentParser(description='Entrenar un horizonte temporal específico')
    parser.add_argument(
        '--horizon',
        type=str,
        required=True,
        choices=['short', 'medium', 'long'],
        help='Horizonte a entrenar: short, medium o long'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/online_retail_2.xlsx',
        help='Ruta al archivo de datos (default: data/processed/online_retail_2.xlsx)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/temporal/customer',
        help='Directorio de salida (default: models/temporal/customer)'
    )

    args = parser.parse_args()

    # Configuración del horizonte
    horizon_configs = {
        'short': TemporalConfig.SHORT,
        'medium': TemporalConfig.MEDIUM,
        'long': TemporalConfig.LONG
    }

    horizon_config = horizon_configs[args.horizon]

    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + f"  ENTRENAMIENTO DE HORIZONTE: {args.horizon.upper()}".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")

    print(f"\n📊 Configuración:")
    print(f"   Horizonte: {args.horizon.upper()}")
    print(f"   Ventana: {horizon_config['window_days']} días")
    print(f"   Pronóstico: {horizon_config['forecast_days']} días")
    print(f"   Epochs: {horizon_config['epochs']}")
    print(f"   Batch size: {horizon_config['batch_size']}")
    print(f"   LSTM units: {horizon_config['lstm_units']}")

    start_time = datetime.now()
    print(f"\n⏰ Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Inicializar analyzer
    print("="*70)
    analyzer = CustomerTemporalAnalyzer(
        data_path=args.data,
        output_dir=args.output
    )

    # Pipeline de preparación
    print("\n🔄 FASE 1: Carga y preprocesamiento...")
    analyzer.load_and_preprocess_data()

    print("\n🔄 FASE 2: Cálculo de métricas RFM...")
    analyzer.calculate_rfm_metrics()

    print("\n🔄 FASE 3: Generación de secuencias temporales...")
    analyzer.generate_customer_sequences(min_transactions=5)

    # Entrenar solo el horizonte seleccionado
    print(f"\n🚀 FASE 4: Entrenamiento de {args.horizon.upper()}...")
    print("="*70)

    try:
        model, history, metrics = analyzer.train_horizon_model(horizon_config)

        print(f"\n✅ ENTRENAMIENTO EXITOSO - {args.horizon.upper()}")
        print("="*70)
        print(f"   Epochs ejecutados: {metrics['epochs_trained']}")
        print(f"   Val Loss: {metrics['total_loss']:.4f}")
        print(f"   Accuracy: {metrics['purchase_prob_accuracy']*100:.2f}%")
        print(f"   AUC: {metrics['purchase_prob_auc']:.4f}")
        print(f"   Days MAE: {metrics['days_mae']:.2f}")
        print(f"   Value MAE: ${metrics['value_mae']:.2f}")

        status = 'SUCCESS'

    except Exception as e:
        print(f"\n❌ ERROR EN ENTRENAMIENTO: {e}")
        status = 'FAILED'

    # Tiempo total
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n⏰ Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Duración total: {duration/60:.1f} minutos ({duration:.0f} segundos)")

    print("\n" + "═"*70)
    print(f"✅ HORIZONTE {args.horizon.upper()}: {status}")
    print("═"*70 + "\n")


if __name__ == '__main__':
    main()
