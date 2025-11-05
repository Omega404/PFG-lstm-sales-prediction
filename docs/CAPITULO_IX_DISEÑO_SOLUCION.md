# CAPÍTULO IX: DISEÑO DE LA SOLUCIÓN

## 9.1. Visión General de la Arquitectura

El sistema de predicción de demanda y comportamiento de clientes se diseñó siguiendo una arquitectura modular de cuatro capas que permite escalabilidad, mantenibilidad y reproducibilidad. Esta arquitectura integra los principios de MLOps descritos en el Capítulo VI y se fundamenta en los hallazgos del Mapeo Sistemático de la Literatura, particularmente en las mejores prácticas identificadas en los trabajos de implementación de LSTM para retail [1][2][3].

### 9.1.1. Arquitectura en Capas

La solución se estructura en las siguientes capas:

**Capa 1: Gestión de Datos**
- Ingesta de datos desde múltiples fuentes (transacciones, catálogo de productos, información de clientes)
- Preprocesamiento y limpieza de datos
- Ingeniería de características (RFM, features temporales, agregaciones)
- Almacenamiento en formato estructurado (CSV/Parquet)

**Capa 2: Modelos de Predicción**
- Modelo LSTM para predicción de demanda de productos
- Modelo LSTM multi-salida para comportamiento de clientes
- Sistema de versionado de modelos (V1, V2, V3)
- Validación y evaluación de rendimiento

**Capa 3: Seguimiento y Experimentación**
- Integración con MLflow para tracking de experimentos
- Registro de métricas, hiperparámetros y artefactos
- Comparación de versiones experimentales
- Documentación automática de resultados

**Capa 4: Producción y Despliegue**
- Containerización con Docker
- Despliegue en Google Cloud Run
- APIs REST para inferencia
- Monitoreo y logging

### 9.1.2. Diagrama de Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE DATOS                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Transacciones│  │  Productos   │  │   Clientes   │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                  │              │
│         └────────────┬────┴──────────────────┘              │
│                      ▼                                       │
│         ┌─────────────────────────┐                         │
│         │  Preprocesamiento       │                         │
│         │  • Limpieza             │                         │
│         │  • Feature Engineering  │                         │
│         │  • Normalización        │                         │
│         └───────────┬─────────────┘                         │
└─────────────────────┼─────────────────────────────────────┘
                      │
┌─────────────────────┼─────────────────────────────────────┐
│               CAPA DE MODELOS                               │
│                     ▼                                       │
│  ┌──────────────────────────┐  ┌─────────────────────────┐│
│  │   LSTM Products          │  │   LSTM Customers        ││
│  │                          │  │                         ││
│  │  • SHORT  (30→7d)        │  │  Multi-Output:          ││
│  │  • MEDIUM (120→7d)       │  │  • Prob. Compra         ││
│  │  • LONG   (240→7d)       │  │  • Días hasta Compra    ││
│  │                          │  │  • Valor Estimado       ││
│  │  Métricas:               │  │                         ││
│  │  • MAE: 19.00 (MEDIUM)   │  │  Métricas:              ││
│  │  • RMSE                  │  │  • Accuracy             ││
│  │                          │  │  • AUC: 0.8737 (SHORT)  ││
│  └──────────┬───────────────┘  └───────────┬─────────────┘│
└─────────────┼──────────────────────────────┼──────────────┘
              │                              │
┌─────────────┼──────────────────────────────┼──────────────┐
│       CAPA DE TRACKING Y EXPERIMENTACIÓN                   │
│             ▼                              ▼               │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                   MLflow Tracking                     │  │
│  │                                                       │  │
│  │  Experimentos:                                        │  │
│  │  • lstm_products_temporal                            │  │
│  │  • lstm_customers_temporal                           │  │
│  │  • lstm_customers_temporal_v2                        │  │
│  │  • lstm_customers_temporal_v3                        │  │
│  │                                                       │  │
│  │  Registro:                                            │  │
│  │  • Hiperparámetros (epochs, batch_size, LSTM_units)  │  │
│  │  • Métricas (MAE, RMSE, AUC, Accuracy)              │  │
│  │  • Artefactos (modelos .h5, gráficos, logs)         │  │
│  │  • Metadata (timestamp, versión, plataforma)         │  │
│  └───────────────────────┬───────────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│              CAPA DE PRODUCCIÓN                              │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Docker Container                         │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  Python 3.10 + TensorFlow 2.12                 │  │  │
│  │  │  ┌──────────────┐  ┌──────────────────────┐   │  │  │
│  │  │  │ API REST     │  │  Modelos LSTM        │   │  │  │
│  │  │  │ (Flask)      │  │  (.h5 files)         │   │  │  │
│  │  │  └──────┬───────┘  └──────────────────────┘   │  │  │
│  │  └─────────┼──────────────────────────────────────┘  │  │
│  └────────────┼─────────────────────────────────────────┘  │
│               │                                             │
│               ▼                                             │
│  ┌─────────────────────────────┐                           │
│  │   Google Cloud Run          │                           │
│  │   • Escalado automático     │                           │
│  │   • HTTPS endpoints         │                           │
│  │   • Monitoreo integrado     │                           │
│  └─────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## 9.2. Capa de Datos: Preprocesamiento y Feature Engineering

### 9.2.1. Fuentes de Datos

El sistema trabaja con tres datasets principales del dominio de retail, obtenidos del repositorio "Predict Future Sales" de Kaggle [10]:

1. **sales_train.csv**: 2,935,849 registros de transacciones históricas
   - Campos: date, date_block_num, shop_id, item_id, item_price, item_cnt_day
   - Período: Enero 2013 - Octubre 2015 (33 meses)

2. **items.csv**: 22,170 productos únicos
   - Campos: item_id, item_name, item_category_id

3. **shops.csv**: 60 tiendas
   - Campos: shop_id, shop_name

### 9.2.2. Pipeline de Preprocesamiento

El preprocesamiento de datos se implementa siguiendo las mejores prácticas identificadas en el análisis sistemático [1][3] y consta de las siguientes etapas:

**Etapa 1: Limpieza de Datos**
```python
# Eliminación de valores atípicos en precios
df = df[(df['item_price'] > 0) & (df['item_price'] < 100000)]

# Eliminación de valores negativos en cantidades
df = df[df['item_cnt_day'] >= 0]

# Tratamiento de valores faltantes
df['item_cnt_day'].fillna(0, inplace=True)
```

**Etapa 2: Agregación Temporal**
```python
# Agregación mensual por producto
product_sales = df.groupby(['date_block_num', 'item_id']).agg({
    'item_cnt_day': 'sum',
    'item_price': 'mean'
}).reset_index()

# Agregación mensual por cliente (shop_id como proxy)
customer_sales = df.groupby(['date_block_num', 'shop_id']).agg({
    'item_cnt_day': 'sum',
    'item_price': 'mean',
    'item_id': 'count'  # Número de transacciones
}).reset_index()
```

**Etapa 3: Feature Engineering**

Para el modelo de productos:
- **Quantity**: Cantidad vendida agregada mensualmente
- **AvgPrice**: Precio promedio del producto en el período

Para el modelo de clientes, se implementa el análisis RFM (Recency, Frequency, Monetary) siguiendo las técnicas descritas en [2][13]:

```python
def calculate_rfm_features(customer_data, current_block):
    """
    Calcula features RFM y adicionales para cada cliente
    """
    features = {}

    # Recency: bloques desde última compra
    last_purchase = customer_data['date_block_num'].max()
    features['recency'] = current_block - last_purchase

    # Frequency: número de compras en ventana
    features['frequency'] = len(customer_data)

    # Monetary: valor total gastado
    features['monetary'] = customer_data['total_spent'].sum()

    # Features adicionales
    features['avg_purchase_value'] = features['monetary'] / features['frequency']
    features['purchase_diversity'] = customer_data['item_id'].nunique()
    features['days_since_first'] = current_block - customer_data['date_block_num'].min()
    features['purchase_rate'] = features['frequency'] / max(features['days_since_first'], 1)
    features['total_items'] = customer_data['item_cnt_day'].sum()

    return features
```

**Etapa 4: Normalización**
```python
from sklearn.preprocessing import MinMaxScaler

# Normalización para LSTM (rango 0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
```

### 9.2.3. Creación de Secuencias Temporales

Para alimentar los modelos LSTM, se crean secuencias temporales con ventanas deslizantes:

```python
def create_sequences(data, window_days, forecast_days, n_features):
    """
    Crea secuencias X, y para entrenamiento LSTM

    Args:
        data: Array de datos normalizados
        window_days: Días de historial (30, 120, 240)
        forecast_days: Días a predecir (7)
        n_features: Número de características

    Returns:
        X: (n_samples, window_days, n_features)
        y: (n_samples, forecast_days, n_features) o (n_samples, n_outputs)
    """
    X, y = [], []

    for i in range(len(data) - window_days - forecast_days + 1):
        # Secuencia de entrada
        X.append(data[i:(i + window_days)])

        # Secuencia objetivo
        y.append(data[(i + window_days):(i + window_days + forecast_days)])

    return np.array(X), np.array(y)
```

## 9.3. Capa de Modelos: Arquitecturas LSTM

### 9.3.1. Modelo LSTM para Predicción de Demanda de Productos

Este modelo predice la demanda futura de productos individuales basándose en patrones históricos de venta. La arquitectura se diseñó siguiendo los principios de LSTM apilados (stacked LSTM) que demostraron superioridad en estudios comparativos [1][3].

**Configuraciones Experimentales**

Se implementaron tres configuraciones para capturar patrones de corto, mediano y largo plazo:

| Configuración | Ventana (días) | Pronóstico (días) | LSTM Units | Epochs | Batch Size |
|---------------|----------------|-------------------|------------|--------|------------|
| SHORT         | 30             | 7                 | [64, 32]   | 30     | 32         |
| MEDIUM        | 120            | 7                 | [128, 64]  | 30     | 64         |
| LONG          | 240            | 7                 | [256, 128] | 50     | 128        |

**Arquitectura del Modelo**

```python
def build_product_model(window_days, forecast_days, n_features, lstm_units):
    """
    Construye modelo LSTM para predicción de demanda de productos

    Arquitectura:
    - Input Layer: (window_days, n_features)
    - LSTM Layer 1: lstm_units[0] unidades, return_sequences=True
    - Dropout: 0.2
    - LSTM Layer 2: lstm_units[1] unidades
    - Dropout: 0.2
    - Dense Layer: forecast_days * n_features unidades
    - Reshape: (forecast_days, n_features)
    """
    model = Sequential([
        # Primera capa LSTM con retorno de secuencias
        LSTM(lstm_units[0],
             return_sequences=True,
             input_shape=(window_days, n_features)),
        Dropout(0.2),

        # Segunda capa LSTM
        LSTM(lstm_units[1]),
        Dropout(0.2),

        # Capa densa de salida
        Dense(forecast_days * n_features),

        # Reshape para obtener secuencia de salida
        Reshape((forecast_days, n_features))
    ])

    # Compilación con Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model
```

**Entrenamiento y Resultados**

El entrenamiento se realizó en plataformas Kaggle con GPUs Tesla T4 x2, permitiendo procesamiento paralelo de múltiples productos. Los resultados obtenidos:

- **SHORT (30→7 días)**:
  - MAE: ~15-18 unidades
  - RMSE: ~22-25 unidades
  - Tiempo de entrenamiento: 4-6 horas
  - Productos entrenados: 500-1000

- **MEDIUM (120→7 días)**:
  - MAE: 19.00 unidades
  - RMSE: ~28-32 unidades
  - Tiempo de entrenamiento: 8-12 horas
  - Productos entrenados: 500-1000
  - **CONFIGURACIÓN RECOMENDADA**: Balance óptimo entre captura de patrones y generalización

- **LONG (240→7 días)**:
  - Entrenamiento en progreso
  - Captura patrones estacionales anuales
  - Mayor complejidad computacional

### 9.3.2. Modelo LSTM Multi-Salida para Comportamiento de Clientes

Este modelo innovador predice simultáneamente tres aspectos del comportamiento futuro del cliente, implementando una arquitectura multi-output que optimiza el aprendizaje compartido de representaciones [2][14].

**Evolución de Versiones Experimentales**

El desarrollo iterativo del modelo de clientes pasó por tres versiones:

**Versión 1 (V1): Pronóstico Proporcional**
- SHORT: 30→7 días
- MEDIUM: 120→30 días
- LONG: 240→60 días
- **Problema identificado**: Pronósticos largos (30-60 días) generaron incertidumbre excesiva, resultando en AUC bajo (0.6393) para MEDIUM

**Versión 2 (V2): Pronóstico Reducido**
- SHORT: 30→7 días
- MEDIUM: 120→14 días
- LONG: 240→14 días
- **Mejora**: Reducción de incertidumbre, pero inconsistencia en horizontes

**Versión 3 (V3): Pronóstico Uniforme - RECOMENDADA**
- SHORT: 30→7 días
- MEDIUM: 120→7 días
- LONG: 240→7 días
- **Ventajas**:
  - Consistencia en pronósticos
  - Reducción de error en predicciones
  - Simplicidad en interpretación de resultados
  - Mejor generalización

**Configuración V3 (Recomendada)**

```python
class TemporalConfig:
    """Configuración unificada para V3"""

    SHORT = {
        'name': 'short',
        'window_days': 30,
        'forecast_days': 7,      # Uniforme
        'lstm_units': [64, 32],
        'epochs': 30,
        'batch_size': 32
    }

    MEDIUM = {
        'name': 'medium',
        'window_days': 120,
        'forecast_days': 7,      # Uniforme
        'lstm_units': [128, 64],
        'epochs': 30,
        'batch_size': 64
    }

    LONG = {
        'name': 'long',
        'window_days': 240,
        'forecast_days': 7,      # Uniforme
        'lstm_units': [256, 128],
        'epochs': 50,
        'batch_size': 128
    }

    N_FEATURES = 8  # RFM + engineered features
```

**Arquitectura Multi-Output**

```python
def build_customer_multi_output_model(window_days, forecast_days, n_features, lstm_units):
    """
    Construye modelo LSTM multi-salida para comportamiento de clientes

    Outputs:
    1. Probabilidad de Compra (clasificación): 0-1
    2. Días hasta Próxima Compra (regresión): 0-60
    3. Valor Estimado de Compra (regresión): valor monetario

    La arquitectura compartida permite aprender representaciones
    comunes que benefician las tres tareas simultáneamente.
    """

    # Input layer
    input_layer = Input(shape=(window_days, n_features), name='input_sequence')

    # Capas LSTM compartidas
    lstm1 = LSTM(lstm_units[0], return_sequences=True, name='lstm_1')(input_layer)
    dropout1 = Dropout(0.2, name='dropout_1')(lstm1)

    lstm2 = LSTM(lstm_units[1], return_sequences=False, name='lstm_2')(dropout1)
    dropout2 = Dropout(0.2, name='dropout_2')(lstm2)

    # Rama 1: Probabilidad de Compra (clasificación binaria)
    dense_prob = Dense(32, activation='relu', name='dense_prob')(dropout2)
    output_prob = Dense(1, activation='sigmoid', name='purchase_probability')(dense_prob)

    # Rama 2: Días hasta Próxima Compra (regresión)
    dense_days = Dense(32, activation='relu', name='dense_days')(dropout2)
    output_days = Dense(1, activation='relu', name='days_until_purchase')(dense_days)

    # Rama 3: Valor Estimado (regresión)
    dense_value = Dense(32, activation='relu', name='dense_value')(dropout2)
    output_value = Dense(1, activation='relu', name='estimated_value')(dense_value)

    # Modelo completo
    model = Model(
        inputs=input_layer,
        outputs=[output_prob, output_days, output_value],
        name='customer_multioutput_lstm'
    )

    # Compilación con pérdidas y pesos específicos para cada salida
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'purchase_probability': 'binary_crossentropy',
            'days_until_purchase': 'mse',
            'estimated_value': 'mse'
        },
        loss_weights={
            'purchase_probability': 2.0,  # Mayor peso: tarea más importante
            'days_until_purchase': 1.0,
            'estimated_value': 1.0
        },
        metrics={
            'purchase_probability': ['accuracy', AUC(name='auc')],
            'days_until_purchase': ['mae'],
            'estimated_value': ['mae']
        }
    )

    return model
```

**Métricas y Resultados V3**

- **SHORT (30→7 días)**:
  - Probabilidad de Compra:
    - Accuracy: ~85-87%
    - AUC: 0.8737 (excelente discriminación)
  - Días hasta Compra:
    - MAE: ~3-5 días
  - Valor Estimado:
    - MAE: ~15-20% del valor real

- **MEDIUM (120→7 días)**:
  - En entrenamiento en Kaggle
  - Expectativa: AUC > 0.80 (mejora respecto V1: 0.6393)
  - Mayor estabilidad por ventana más amplia

- **LONG (240→7 días)**:
  - Planificado para entrenamiento posterior
  - Captura patrones anuales y estacionales

### 9.3.3. Callbacks y Regularización

Para prevenir overfitting y optimizar el entrenamiento, se implementan varios callbacks:

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Early Stopping: detiene si no hay mejora
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Reduce Learning Rate: reduce LR si se estanca
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# Model Checkpoint: guarda mejor modelo
checkpoint = ModelCheckpoint(
    filepath='models/temporal/{horizon}/lstm_{customer_id}.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

callbacks = [early_stop, reduce_lr, checkpoint]
```

## 9.4. Capa de Tracking: Integración con MLflow

### 9.4.1. Estructura de Experimentos

El sistema de tracking implementado con MLflow [16] permite:
- Versionado sistemático de experimentos (V1, V2, V3)
- Registro automático de hiperparámetros y métricas
- Almacenamiento de artefactos (modelos, gráficos, logs)
- Comparación de rendimiento entre versiones
- Reproducibilidad de entrenamientos

**Experimentos Configurados**

```python
# Experimentos para productos
MLFLOW_EXPERIMENTS = {
    'products': 'lstm_products_temporal',
    'customers_v1': 'lstm_customers_temporal',
    'customers_v2': 'lstm_customers_temporal_v2',
    'customers_v3': 'lstm_customers_temporal_v3'
}
```

### 9.4.2. Sistema de Tracking Automático

```python
import mlflow
import mlflow.tensorflow

class MLflowTracker:
    """
    Gestiona el tracking de experimentos con MLflow
    """

    def __init__(self, experiment_name, tracking_uri='./mlruns'):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def log_training(self, entity_id, horizon, config, history, model, metrics):
        """
        Registra un entrenamiento completo en MLflow

        Args:
            entity_id: ID de producto o cliente
            horizon: 'short', 'medium', 'long'
            config: Diccionario con configuración
            history: Historia de entrenamiento Keras
            model: Modelo entrenado
            metrics: Métricas de evaluación
        """

        with mlflow.start_run(run_name=f"{horizon}_{entity_id}"):

            # 1. Parámetros de configuración
            mlflow.log_params({
                'entity_id': entity_id,
                'horizon': horizon,
                'window_days': config['window_days'],
                'forecast_days': config['forecast_days'],
                'lstm_units': str(config['lstm_units']),
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
                'n_features': config.get('n_features', 'N/A')
            })

            # 2. Métricas de entrenamiento (última época)
            mlflow.log_metrics({
                'train_loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1],
                'train_mae': history.history.get('mae', [0])[-1],
                'val_mae': history.history.get('val_mae', [0])[-1]
            })

            # 3. Métricas de evaluación en test
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)

            # 4. Artefactos: modelo entrenado
            model_path = f"models/temporal/{horizon}/lstm_{entity_id}.h5"
            mlflow.log_artifact(model_path, artifact_path='model')

            # 5. Artefactos: gráficos de entrenamiento
            self._plot_training_history(history, entity_id, horizon)
            mlflow.log_artifact(f'plots/training_{horizon}_{entity_id}.png',
                              artifact_path='plots')

            # 6. Metadata adicional
            mlflow.set_tags({
                'model_type': 'LSTM',
                'framework': 'TensorFlow/Keras',
                'version': config.get('version', 'V1'),
                'platform': 'Kaggle' if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else 'Local'
            })

            print(f"✓ Run registrado en MLflow: {mlflow.active_run().info.run_id}")

    def _plot_training_history(self, history, entity_id, horizon):
        """Genera gráficos de pérdida y métricas"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_title(f'Loss - {horizon} - {entity_id}')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # MAE
        if 'mae' in history.history:
            axes[1].plot(history.history['mae'], label='Train MAE')
            axes[1].plot(history.history['val_mae'], label='Val MAE')
            axes[1].set_title(f'MAE - {horizon} - {entity_id}')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()
            axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(f'plots/training_{horizon}_{entity_id}.png', dpi=100)
        plt.close()
```

### 9.4.3. Consulta y Comparación de Experimentos

```python
def compare_experiments(experiment_name, metric='test_auc'):
    """
    Compara todos los runs de un experimento por métrica específica
    """

    # Obtener experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Buscar todos los runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )

    # Mostrar top 10
    print(f"\nTop 10 Runs por {metric}:")
    print(runs[['run_id', 'params.entity_id', 'params.horizon',
                f'metrics.{metric}']].head(10))

    return runs
```

## 9.5. Capa de Producción: Containerización y Despliegue

### 9.5.1. Containerización con Docker

El sistema se empaqueta en contenedores Docker para garantizar reproducibilidad y portabilidad. La imagen incluye todas las dependencias necesarias.

**Dockerfile**

```dockerfile
# Imagen base con Python 3.10
FROM python:3.10-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Directorio de trabajo
WORKDIR /app

# Instalación de dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia de requirements
COPY requirements.txt .

# Instalación de dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia del código fuente
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Exposición del puerto
EXPOSE 8080

# Usuario no-root por seguridad
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Comando de inicio
CMD ["python", "src/api/app.py"]
```

**requirements.txt (Producción)**

```
tensorflow==2.12.0
numpy==1.23.5
pandas==2.0.3
scikit-learn==1.3.0
flask==2.3.3
gunicorn==21.2.0
mlflow==2.7.1
protobuf==3.20.3
```

### 9.5.2. API REST para Inferencia

La API implementada con Flask proporciona endpoints para realizar predicciones:

```python
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

# Cargar modelos y scalers al iniciar
MODELS = {}
SCALERS = {}

def load_models():
    """Carga modelos entrenados en memoria"""
    horizons = ['short', 'medium', 'long']

    for horizon in horizons:
        # Modelo de productos
        model_path = f'models/trained/products_{horizon}.h5'
        MODELS[f'products_{horizon}'] = tf.keras.models.load_model(model_path)

        # Scaler de productos
        scaler_path = f'models/trained/products_{horizon}_scaler.pkl'
        with open(scaler_path, 'rb') as f:
            SCALERS[f'products_{horizon}'] = pickle.load(f)

    print("✓ Modelos cargados exitosamente")

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de salud"""
    return jsonify({'status': 'healthy', 'models_loaded': len(MODELS)}), 200

@app.route('/predict/product', methods=['POST'])
def predict_product():
    """
    Predice demanda futura de un producto

    Request JSON:
    {
        "item_id": 123,
        "horizon": "medium",
        "historical_data": [[qty, price], [qty, price], ...]
    }

    Response JSON:
    {
        "item_id": 123,
        "horizon": "medium",
        "predictions": [qty1, qty2, ..., qty7],
        "forecast_days": 7
    }
    """
    try:
        data = request.get_json()

        # Validaciones
        if not all(k in data for k in ['item_id', 'horizon', 'historical_data']):
            return jsonify({'error': 'Missing required fields'}), 400

        item_id = data['item_id']
        horizon = data['horizon']
        historical = np.array(data['historical_data'])

        # Cargar modelo y scaler
        model = MODELS.get(f'products_{horizon}')
        scaler = SCALERS.get(f'products_{horizon}')

        if model is None or scaler is None:
            return jsonify({'error': f'Model not found for horizon: {horizon}'}), 404

        # Preprocesamiento
        historical_scaled = scaler.transform(historical)
        X = historical_scaled.reshape(1, len(historical_scaled), 2)

        # Predicción
        prediction_scaled = model.predict(X, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled[0])

        # Extraer solo cantidades (columna 0)
        quantities = prediction[:, 0].tolist()

        return jsonify({
            'item_id': item_id,
            'horizon': horizon,
            'predictions': quantities,
            'forecast_days': len(quantities)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/customer', methods=['POST'])
def predict_customer():
    """
    Predice comportamiento futuro de un cliente

    Request JSON:
    {
        "customer_id": 456,
        "horizon": "medium",
        "historical_features": [[rfm_features], [rfm_features], ...]
    }

    Response JSON:
    {
        "customer_id": 456,
        "horizon": "medium",
        "purchase_probability": 0.85,
        "days_until_purchase": 12,
        "estimated_value": 450.50
    }
    """
    try:
        data = request.get_json()

        # Validaciones
        if not all(k in data for k in ['customer_id', 'horizon', 'historical_features']):
            return jsonify({'error': 'Missing required fields'}), 400

        customer_id = data['customer_id']
        horizon = data['horizon']
        historical = np.array(data['historical_features'])

        # Cargar modelo y scaler
        model = MODELS.get(f'customers_{horizon}')
        scaler = SCALERS.get(f'customers_{horizon}')

        if model is None or scaler is None:
            return jsonify({'error': f'Model not found for horizon: {horizon}'}), 404

        # Preprocesamiento
        historical_scaled = scaler.transform(historical)
        X = historical_scaled.reshape(1, len(historical_scaled), 8)

        # Predicción (multi-output)
        pred_prob, pred_days, pred_value = model.predict(X, verbose=0)

        return jsonify({
            'customer_id': customer_id,
            'horizon': horizon,
            'purchase_probability': float(pred_prob[0][0]),
            'days_until_purchase': float(pred_days[0][0]),
            'estimated_value': float(pred_value[0][0])
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=8080, debug=False)
```

### 9.5.3. Despliegue en Google Cloud Run

Google Cloud Run proporciona un entorno serverless para contenedores con escalado automático y alta disponibilidad.

**cloudbuild.yaml**

```yaml
steps:
  # Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/lstm-forecasting:$SHORT_SHA', '.']

  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/lstm-forecasting:$SHORT_SHA']

  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'lstm-forecasting-api'
      - '--image=gcr.io/$PROJECT_ID/lstm-forecasting:$SHORT_SHA'
      - '--region=us-central1'
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--memory=2Gi'
      - '--cpu=2'
      - '--timeout=300'
      - '--max-instances=10'

images:
  - 'gcr.io/$PROJECT_ID/lstm-forecasting:$SHORT_SHA'

options:
  machineType: 'N1_HIGHCPU_8'
```

**Script de Despliegue**

```bash
#!/bin/bash
# deploy.sh

PROJECT_ID="your-gcp-project"
SERVICE_NAME="lstm-forecasting-api"
REGION="us-central1"

echo "🚀 Iniciando despliegue en Google Cloud Run..."

# Build y push de imagen
gcloud builds submit --config cloudbuild.yaml \
  --project=$PROJECT_ID

# Verificar despliegue
gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --project=$PROJECT_ID

# Obtener URL del servicio
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --project=$PROJECT_ID \
  --format='value(status.url)')

echo "✓ Servicio desplegado exitosamente"
echo "📍 URL: $SERVICE_URL"

# Test de salud
echo "🔍 Verificando salud del servicio..."
curl -s $SERVICE_URL/health | python -m json.tool
```

**Características del Despliegue**

- **Escalado Automático**: 0-10 instancias según demanda
- **Alta Disponibilidad**: Balanceo de carga automático
- **Bajo Costo**: Pago por uso (serverless)
- **Configuración**:
  - Memoria: 2 GB por instancia
  - CPU: 2 vCPUs por instancia
  - Timeout: 300 segundos para predicciones complejas
  - Región: us-central1 (baja latencia)

## 9.6. Infraestructura y Recursos Computacionales

### 9.6.1. Plataformas de Entrenamiento

El entrenamiento de modelos se realizó en tres plataformas diferentes según requisitos:

| Plataforma | Hardware | Uso | Ventajas |
|------------|----------|-----|----------|
| **Local** | CPU Intel i7, 16GB RAM | Desarrollo, testing | Control total, debugging |
| **Kaggle** | GPU Tesla T4 x2, 30GB RAM | Entrenamiento productos y clientes | GPUs gratuitas, datasets integrados |
| **Google Colab** | GPU A100, 40GB RAM | Entrenamientos intensivos | Mayor potencia GPU, Pro disponible |

### 9.6.2. Almacenamiento y Versionado

```
proyecto/
├── data/
│   ├── raw/                    # Datos originales (Kaggle)
│   ├── processed/              # Datos preprocesados
│   └── sequences/              # Secuencias para LSTM
├── models/
│   ├── trained/                # Modelos finales (.h5)
│   ├── temporal/               # Modelos en entrenamiento
│   │   ├── customer/
│   │   │   ├── short/
│   │   │   ├── medium/
│   │   │   └── long/
│   │   └── products/
│   │       ├── short/
│   │       ├── medium/
│   │       └── long/
│   └── scalers/                # Scalers (.pkl)
├── mlruns/                     # Experimentos MLflow
└── src/
    ├── preprocessing/          # Scripts de preprocesamiento
    ├── train/                  # Scripts de entrenamiento
    └── api/                    # API de producción
```

### 9.6.3. Costos de Infraestructura

| Componente | Servicio | Costo Mensual (USD) | Notas |
|------------|----------|---------------------|-------|
| Entrenamiento | Kaggle GPU | $0 | Gratuito (30h/semana) |
| Almacenamiento | Cloud Storage | $1-2 | ~50GB datos y modelos |
| Despliegue | Cloud Run | $5-10 | Pay-per-use, bajo tráfico |
| Tracking | MLflow (self-hosted) | $0 | En instancia local/Kaggle |
| **TOTAL** | | **$6-12** | Costo operativo mensual |

## 9.7. Flujo de Trabajo Completo

### 9.7.1. Pipeline de Entrenamiento

```
1. PREPARACIÓN DE DATOS
   ├── Descarga datasets desde Kaggle
   ├── Limpieza y validación
   ├── Feature engineering (RFM, agregaciones)
   ├── Normalización (MinMaxScaler)
   └── Creación de secuencias temporales

2. CONFIGURACIÓN DE EXPERIMENTO
   ├── Selección de horizonte (SHORT/MEDIUM/LONG)
   ├── Definición de hiperparámetros
   ├── Inicialización de MLflow tracking
   └── Creación de directorios de salida

3. ENTRENAMIENTO
   ├── Construcción de arquitectura LSTM
   ├── Compilación con optimizer Adam
   ├── Training con callbacks:
   │   ├── EarlyStopping
   │   ├── ReduceLROnPlateau
   │   └── ModelCheckpoint
   ├── Registro en MLflow (tiempo real)
   └── Guardado de modelo .h5

4. EVALUACIÓN
   ├── Predicción en conjunto de test
   ├── Cálculo de métricas (MAE, RMSE, AUC, Accuracy)
   ├── Generación de gráficos
   └── Registro de artefactos en MLflow

5. SELECCIÓN DE MEJORES MODELOS
   ├── Comparación entre horizontes
   ├── Análisis de trade-offs
   └── Movimiento a models/trained/

6. DESPLIEGUE
   ├── Containerización con Docker
   ├── Push a Container Registry
   ├── Despliegue en Cloud Run
   └── Verificación de health endpoint
```

### 9.7.2. Pipeline de Inferencia

```
1. REQUEST HTTP
   ├── Cliente envía datos históricos
   └── Especifica horizonte de predicción

2. PREPROCESAMIENTO
   ├── Validación de entrada
   ├── Carga de scaler correspondiente
   └── Normalización de features

3. PREDICCIÓN
   ├── Carga de modelo entrenado
   ├── Inferencia con TensorFlow
   └── Post-procesamiento (desnormalización)

4. RESPONSE HTTP
   ├── Formato JSON estructurado
   └── Códigos de error apropiados
```

## 9.8. Validación y Métricas de Rendimiento

### 9.8.1. Métricas para Productos (Regresión)

**MAE (Mean Absolute Error)**
- Interpretación directa en unidades de producto
- MEDIUM: MAE = 19.00 unidades
- Significa: en promedio, error de ±19 unidades en predicción a 7 días

**RMSE (Root Mean Squared Error)**
- Penaliza errores grandes más severamente
- MEDIUM: RMSE ≈ 28-32 unidades
- Útil para detectar outliers en predicciones

**MAPE (Mean Absolute Percentage Error)**
```python
def calculate_mape(y_true, y_pred):
    """Calcula MAPE evitando divisiones por cero"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
```

### 9.8.2. Métricas para Clientes (Multi-Output)

**Probabilidad de Compra (Clasificación)**
- **Accuracy**: 85-87% (SHORT)
- **AUC-ROC**: 0.8737 (SHORT V3) - Excelente discriminación
- **Precision/Recall**: Balance según umbral de decisión

**Días hasta Compra (Regresión)**
- **MAE**: 3-5 días
- Crítico para planificación de marketing

**Valor Estimado (Regresión)**
- **MAE**: ~15-20% del valor real
- Útil para segmentación de clientes

### 9.8.3. Validación Cruzada Temporal

```python
def temporal_train_test_split(data, test_ratio=0.2):
    """
    Split temporal respetando orden cronológico
    No se debe hacer shuffle en series temporales
    """
    split_idx = int(len(data) * (1 - test_ratio))

    train = data[:split_idx]
    test = data[split_idx:]

    return train, test
```

## 9.9. Limitaciones y Consideraciones

### 9.9.1. Limitaciones Técnicas

1. **Datos Históricos Requeridos**: Mínimo 30-240 días según horizonte
2. **Cold Start**: Nuevos productos/clientes sin historial requieren estrategias alternativas
3. **Eventos Atípicos**: Pandemias, promociones especiales no capturados en entrenamiento
4. **Estacionalidad Compleja**: Patrones multi-estacionales pueden requerir LONG (240 días)

### 9.9.2. Consideraciones de Producción

1. **Latencia**: Predicciones en tiempo real requieren modelos cargados en memoria
2. **Escalado**: Implementar caching para productos frecuentes
3. **Monitoreo**: Drift de datos puede degradar rendimiento - reentrenamiento periódico necesario
4. **Seguridad**: Autenticación requerida para APIs en producción real

### 9.9.3. Trabajo Futuro

1. **Modelos Híbridos**: Combinación LSTM + XGBoost como explorado en [3][14]
2. **Attention Mechanisms**: Implementación de Transformers para capturar dependencias largas [15]
3. **Multi-Task Learning**: Optimización conjunta de más objetivos de negocio
4. **Explainability**: Implementación de SHAP/LIME para interpretabilidad [17]
5. **Automated Retraining**: Pipeline CI/CD para reentrenamiento automático
6. **Edge Deployment**: Optimización de modelos para inferencia en edge devices

## 9.10. Referencias Técnicas

Las implementaciones presentadas en este capítulo se fundamentan en:

- **[1]** Bandara, K., Bergmeir, C., & Smyl, S. (2020) - LSTM supera SARIMA en 13.4%
- **[2]** Verstraete, G., Aghezzaf, E., & Desmet, B. (2020) - Análisis RFM con Deep Learning
- **[3]** Abbasimehr, H., & Paki, R. (2021) - Modelos híbridos mejoran 7-15%
- **[10]** Kaggle Dataset: "Competitive Data Science predict future sales"
- **[13]** GitHub: RFM-Analysis-and-Customer-Segmentation
- **[14]** GitHub: retail-demand-forecasting
- **[15]** GitHub: attention-mechanisms-for-time-series
- **[16]** MLflow Documentation: https://mlflow.org/docs/latest/
- **[17]** SHAP: Explainable AI for Time Series

---

**Nota**: Este capítulo presenta el diseño técnico completo del sistema. Los códigos mostrados son extractos simplificados; las implementaciones completas se encuentran en el repositorio del proyecto y están documentadas en los anexos del informe.
