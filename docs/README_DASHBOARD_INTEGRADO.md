# 📊 Dashboard Integrado Cliente-Producto

Sistema de análisis integrado que combina predicciones de modelos LSTM de clientes y productos para generar insights de negocio completos.

## 🎯 Características

### 1. **Análisis de Clientes** (LSTM Temporal)
- Probabilidad de que un cliente compre
- Días estimados hasta próxima compra
- Valor esperado de la compra
- Horizontes: SHORT (30→7d), MEDIUM (120→7d), LONG (240→7d)

### 2. **Análisis de Productos** (LSTM por Producto)
- Predicción de demanda futura
- Unidades esperadas a vender
- Tendencias temporales

### 3. **Análisis Cruzado Cliente-Producto**
- Matriz de afinidad basada en patrones históricos
- Top productos por cliente
- Clientes esperados por producto
- Forecast agregado combinando ambos modelos

## 📁 Estructura de Archivos

```
E:\Codigos\Proyecto Final\
├── app_analisis_integrado.py       # Backend Flask (puerto 5002)
├── dashboard_integrado.html         # Frontend interactivo
├── data/
│   └── processed/
│       └── online_retail_2.xlsx     # Dataset
├── models/
│   ├── temporal/
│   │   └── customer_v3/             # Modelos de clientes V3
│   │       ├── short/
│   │       ├── medium/
│   │       └── long/
│   └── trained/                      # Modelos de productos
│       ├── lstm_XXXXX.h5
│       └── scaler_XXXXX.pkl
```

## 🚀 Cómo Usar

### Paso 1: Verificar Requisitos

Asegúrate de tener:
- ✅ Dataset: `data/processed/online_retail_2.xlsx`
- ✅ Modelos de clientes: `models/temporal/customer_v3/{short,medium,long}/`
- ✅ Modelos de productos: `models/trained/lstm_*.h5`

### Paso 2: Instalar Dependencias

```bash
pip install flask pandas numpy tensorflow scikit-learn openpyxl
```

### Paso 3: Ejecutar el Servidor

```bash
cd "E:\Codigos\Proyecto Final"
python app_analisis_integrado.py
```

El servidor se ejecutará en: `http://localhost:5002`

### Paso 4: Abrir el Dashboard

1. Abre tu navegador
2. Ve a: `http://localhost:5002`
3. Explora las 4 pestañas disponibles

## 📊 Funcionalidades del Dashboard

### 🏠 Tab 1: Vista General

Muestra estadísticas globales:
- Total de clientes, productos y transacciones
- Modelos activos (clientes + productos)
- Información sobre horizontes temporales

### 🔥 Tab 2: Matriz Cliente-Producto

Visualiza la relación entre clientes y productos:
- Selecciona top N clientes y productos
- Ver frecuencia de compra
- Valor promedio por transacción
- Días desde última compra
- Tabla resumen con métricas agregadas

**Ejemplo de uso:**
```
Top Clientes: 20
Top Productos: 20
→ Click en "Cargar Matriz"
```

### 🎯 Tab 3: Forecast Agregado

Genera predicción combinada cliente-producto:
- Selecciona horizonte temporal (SHORT/MEDIUM/LONG)
- Define días a predecir (1-30)
- Establece probabilidad mínima de compra

**Métricas generadas:**
- Clientes esperados en el período
- Tasa de actividad (% clientes activos)
- Ingresos esperados totales
- Forecast por producto (clientes + unidades)

**Ejemplo de uso:**
```
Horizonte: MEDIUM (120→7 días)
Días a Predecir: 7
Probabilidad Mínima: 0.5
→ Click en "Generar Forecast"
```

### 👤 Tab 4: Análisis por Cliente

Análisis detallado de un cliente específico:
- Ingresa Customer ID
- Selecciona horizonte temporal
- Obtén predicción individual

**Información generada:**
- Probabilidad de compra (%)
- Días estimados hasta compra
- Valor esperado ($)
- Top 10 productos con mayor afinidad
  - Frecuencia de compra
  - Valor promedio
  - Días desde última compra

**Ejemplo de uso:**
```
Customer ID: 12345
Horizonte: MEDIUM
→ Click en "Analizar Cliente"
```

## 🔧 API Endpoints

El backend expone los siguientes endpoints REST:

### `GET /api/info`
Información general del sistema

**Response:**
```json
{
  "status": "ok",
  "dataset": {
    "total_transactions": 525461,
    "total_customers": 5942,
    "total_products": 4634,
    "date_range": {
      "start": "2009-12-01T08:26:00",
      "end": "2011-12-09T12:50:00"
    }
  },
  "customer_models": ["short", "medium", "long"],
  "product_models": {
    "total": 145,
    "codes": ["20725", "20727", ...]
  }
}
```

### `GET /api/customer-product-matrix`
Matriz de afinidad cliente-producto

**Query Params:**
- `top_customers`: int (default 50)
- `top_products`: int (default 50)

**Response:**
```json
{
  "status": "ok",
  "matrix": {
    "12345": {
      "20725": {
        "frequency": 5,
        "total_quantity": 12,
        "avg_value": 45.50,
        "last_purchase_days": 30
      }
    }
  },
  "customers": [12345, 12346, ...],
  "products": ["20725", "20727", ...]
}
```

### `POST /api/predict-customer`
Predicción para cliente individual

**Body:**
```json
{
  "customer_id": 12345,
  "horizon": "medium"
}
```

**Response:**
```json
{
  "status": "ok",
  "customer_id": 12345,
  "horizon": "medium",
  "prediction": {
    "purchase_probability": 0.75,
    "days_until_purchase": 12,
    "expected_value": 450.00
  },
  "top_products": [
    {
      "product_code": "20725",
      "frequency": 5,
      "avg_value": 45.50,
      "last_purchase_days": 30
    }
  ]
}
```

### `POST /api/aggregate-forecast`
Forecast agregado combinando clientes y productos

**Body:**
```json
{
  "horizon": "medium",
  "forecast_days": 7,
  "min_purchase_probability": 0.5
}
```

**Response:**
```json
{
  "status": "ok",
  "summary": {
    "total_expected_customers": 150,
    "active_customers_rate": 45.2,
    "total_expected_revenue": 45000.00
  },
  "product_forecast": [
    {
      "product_code": "20725",
      "expected_customers": 25,
      "expected_units": 50
    }
  ]
}
```

## 🧮 Metodología

### Matriz Cliente-Producto

La matriz se construye analizando el historial de compras:

```python
Para cada (cliente, producto):
  - frequency: número de transacciones
  - total_quantity: suma de unidades compradas
  - avg_value: valor promedio por transacción
  - last_purchase_days: días desde última compra
```

### Forecast Agregado

Combina dos fuentes de información:

1. **Modelos de Clientes** → Quién comprará
   - Filtro por probabilidad mínima
   - Estimación de días hasta compra
   - Valor esperado

2. **Patrones Históricos** → Qué comprará
   - Productos con mayor afinidad
   - Frecuencia histórica de recompra
   - Unidades promedio por transacción

3. **Modelos de Productos** → Cuánto se venderá
   - Demanda futura por producto
   - Tendencias temporales

**Fórmula simplificada:**
```
Unidades_Esperadas(producto) =
  Clientes_Activos ×
  Tasa_Recompra(producto) ×
  Unidades_Promedio(producto)
```

## 📈 Casos de Uso

### 1. Planning de Inventario
```
1. Ve a "Forecast Agregado"
2. Selecciona horizonte MEDIUM (7 días)
3. Genera forecast
4. Revisa "Unidades Esperadas" por producto
5. Ajusta stock según demanda predicha
```

### 2. Campañas de Marketing Dirigidas
```
1. Ve a "Análisis por Cliente"
2. Ingresa Customer ID de cliente VIP
3. Revisa "Top Productos con Mayor Afinidad"
4. Crea oferta personalizada de esos productos
5. Tiempo óptimo: según "Días hasta Compra"
```

### 3. Análisis de Cross-Selling
```
1. Ve a "Matriz Cliente-Producto"
2. Carga top 50 clientes y productos
3. Identifica productos frecuentes juntos
4. Diseña bundles o promociones 2x1
```

### 4. Segmentación Avanzada
```
1. Usa API para obtener matriz completa
2. Analiza clientes con alta afinidad a producto X
3. Crea segmento "Compradores de X"
4. Diseña campaña específica
```

## ⚠️ Notas Importantes

### Limitaciones Actuales

1. **Predicción de Productos por Cliente**:
   - Actualmente basada en **patrones históricos**
   - No usa modelos LSTM de productos directamente
   - Recomendación: Cliente que compró X históricamente → probablemente recomprará X

2. **Forecast Agregado**:
   - Usa heurística simple (30% tasa de recompra)
   - No integra completamente modelos LSTM
   - **TODO**: Implementar predicción real usando ambos modelos

3. **Escalabilidad**:
   - Matriz cliente-producto se carga en memoria
   - Para datasets muy grandes (>100K clientes), considerar:
     - Base de datos (SQLite/PostgreSQL)
     - Cache Redis
     - Procesamiento por lotes

### Mejoras Futuras

- [ ] Integrar predicciones LSTM reales en forecast agregado
- [ ] Agregar modelos de recomendación (Collaborative Filtering)
- [ ] Implementar análisis de secuencia (¿qué compra después de X?)
- [ ] Visualizaciones avanzadas (gráficos D3.js)
- [ ] Exportar resultados a Excel/CSV
- [ ] Sistema de alertas (clientes en riesgo de churn)

## 🐛 Troubleshooting

### Error: "No se puede conectar al servidor"
```bash
# Verifica que el servidor esté corriendo
python app_analisis_integrado.py

# Debería ver:
# [*] Cargando configuración...
# [OK] Sistema listo
# Servidor corriendo en http://localhost:5002
```

### Error: "Modelo no encontrado"
```bash
# Verifica que existan los modelos
dir models\temporal\customer_v3\medium\
# Debe contener: model_best.keras, scaler_X.pkl, etc.

dir models\trained\
# Debe contener: lstm_XXXXX.h5, scaler_XXXXX.pkl
```

### Error: "Cliente no encontrado"
```python
# Verifica IDs de clientes válidos
df = pd.read_excel('data/processed/online_retail_2.xlsx')
df[df['CustomerID'].notna()]['CustomerID'].unique()[:10]
# Usa uno de estos IDs en el dashboard
```

## 📞 Soporte

Para problemas o preguntas:
1. Revisa los logs del servidor en la consola
2. Verifica que todos los archivos estén en su lugar
3. Confirma que las dependencias están instaladas
4. Revisa la consola del navegador (F12) para errores de frontend

---

**Versión:** 1.0
**Fecha:** 2025-01
**Puerto:** 5002
**Framework:** Flask + Vanilla JS
