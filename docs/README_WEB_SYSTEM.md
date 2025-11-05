# Sistema Web de Análisis Cruzado LSTM

Sistema web completo para análisis de clientes y productos usando modelos LSTM, con todas las funcionalidades clasificadas por nivel de confiabilidad.

## 🚀 Inicio Rápido

### Iniciar el servidor

```bash
python app_cross_analysis_web.py
```

El servidor estará disponible en: **http://localhost:5001**

### Requisitos

```bash
pip install flask flask-cors pandas numpy tensorflow scikit-learn openpyxl
```

---

## 📊 Dashboard Principal

Accede a: **http://localhost:5001**

El dashboard muestra:
- 👥 **Segmentación de Clientes** (Alta/Media/Baja Prioridad)
- 🏆 **Top 10 Clientes** por probabilidad de compra
- 📦 **Top 10 Productos** más demandados
- 💰 **Forecast de Ingresos** (7 días) ⚠️
- 📊 **Forecast de Inventario** ⚠️

---

## 🔌 API Endpoints

### ✅ **ALTA CONFIABILIDAD** (Usar con confianza)

#### 1. Ranking de Clientes
```bash
GET /api/customers/ranking?top_n=10
```
**Confiabilidad:** ALTA (87.59% accuracy)
**Uso:** Priorizar contactos, segmentar clientes

**Respuesta:**
```json
{
  "ranking": [
    {
      "customer_id": 13352,
      "purchase_probability": 100.0,
      "segment": "high_value_high_prob",
      "total_spent_historical": 89.90
    }
  ],
  "metadata": {
    "confidence": "ALTA",
    "model_accuracy": 87.59
  }
}
```

---

#### 2. Segmentación de Clientes
```bash
GET /api/customers/segments
```
**Confiabilidad:** ALTA
**Uso:** Ver distribución de clientes por segmento

**Respuesta:**
```json
{
  "segments": {
    "high_value_high_prob": 5,
    "medium_prob": 5,
    "low_prob": 90
  },
  "strategies": {
    "high_value_high_prob": "Contacto VIP inmediato",
    "medium_prob": "Campaña general",
    "low_prob": "No contactar"
  }
}
```

---

#### 3. Detalle de Cliente
```bash
GET /api/customers/<customer_id>/details
```
**Confiabilidad:** ALTA para probabilidad y recomendaciones
**Uso:** Ver detalles de un cliente específico + productos recomendados

---

#### 4. Ranking de Productos
```bash
GET /api/products/ranking?top_n=20
```
**Confiabilidad:** ALTA para comparación relativa
**Uso:** Ranking de productos más demandados

⚠️ **Nota:** Las cantidades exactas NO son confiables (MAE alto). Usar solo para comparación relativa.

---

#### 5. Cross-Sell de Productos
```bash
GET /api/products/<stock_code>/cross-sell
```
**Confiabilidad:** ALTA
**Uso:** Encontrar productos relacionados

**Respuesta:**
```json
{
  "product": "85123A",
  "total_customers": 150,
  "related_products": [
    {
      "stock_code": "85099B",
      "description": "JUMBO BAG RED",
      "cross_sell_rate": 45.5
    }
  ]
}
```

---

### ⚠️ **BAJA CONFIABILIDAD** (Mejorable con más datos)

#### 6. Forecast de Ingresos
```bash
GET /api/forecast/revenue
```
**Confiabilidad:** BAJA (MAE $123 = ~40% error)
**Uso:** Solo como estimación aproximada
**TODO:** Mejorar con más datos de entrenamiento

⚠️ **Advertencia:** Los valores monetarios tienen alta variabilidad. Usar solo para comparación relativa.

---

#### 7. Forecast de Inventario
```bash
GET /api/forecast/inventory?top_n=5
```
**Confiabilidad:** BAJA (MAE 19 unidades > mean)
**Uso:** Solo para ranking relativo
**TODO:** Mejorar modelo de productos

⚠️ **Advertencia:** NO usar para planificación de inventario precisa. Solo para identificar productos de mayor demanda relativa.

---

#### 8. ROI de Campaña
```bash
POST /api/campaign/roi
{
  "cost_per_contact": 5
}
```
**Confiabilidad:** BAJA (basado en valores predichos imprecisos)
**Uso:** Solo como guía aproximada
**TODO:** Validar con conversiones reales

---

### 🔧 **UTILIDADES**

#### Estado del Sistema
```bash
GET /api/status
```

**Respuesta:**
```json
{
  "status": "operational",
  "last_analysis": "2025-11-05T17:10:45",
  "total_customers_analyzed": 100,
  "total_products_analyzed": 50,
  "models": {
    "customer_model": "V3 (7 días)",
    "customer_accuracy": 87.59,
    "product_mae": 19.17
  }
}
```

---

#### Actualizar Predicciones
```bash
POST /api/refresh
{
  "sample_size": 100
}
```

---

#### Exportar Resultados
```bash
GET /api/export/csv   # Exportar a CSV
GET /api/export/json  # Exportar a JSON
```

---

## ✅ Casos de Uso Reales

### 1. **Segmentación y Priorización de Clientes**
**Confiable:** ✅ SÍ
**Uso:**
```python
# Obtener top 10 clientes de alta prioridad
GET /api/customers/ranking?top_n=10&segment=high_value_high_prob

# Resultado:
# - Cliente #13352: 100% probabilidad
# - Cliente #15039: 95.2% probabilidad
# → Contactar INMEDIATAMENTE con oferta premium
```

---

### 2. **Timing de Campañas**
**Confiable:** ✅ Moderado (±5 días error)
**Uso:**
```python
# Ver detalles de cliente
GET /api/customers/13352/details

# Resultado:
# - Probabilidad: 100%
# - Ventana de compra: 2-7 días
# → Enviar campaña ESTA SEMANA
```

---

### 3. **Productos Populares**
**Confiable:** ✅ SÍ (para comparación relativa)
**Uso:**
```python
# Top 20 productos
GET /api/products/ranking?top_n=20

# Resultado:
# 1. JUMBO BAG RED: 14,004 unidades
# 2. WORLD WAR 2 GLIDERS: 15,963 unidades
# → JUMBO BAG tiene más demanda relativa
```

---

### 4. **Recomendaciones Cross-Sell**
**Confiable:** ✅ SÍ
**Uso:**
```python
# Ver productos relacionados a JUMBO BAG
GET /api/products/85123A/cross-sell

# Resultado:
# - 45% de clientes también compraron JUMBO BAG BLUE
# → Ofrecer bundle: "3 colores por $X"
```

---

## ❌ NO Usar Para

1. **Valores monetarios exactos**
   - ❌ "Cliente gastará exactamente $4,099.96"
   - ✅ "Cliente está en top 5 de alto valor"

2. **Cantidades exactas de inventario**
   - ❌ "Reabastece exactamente 7,026 unidades"
   - ✅ "JUMBO BAG es producto #1 en demanda"

3. **ROI exacto de campañas**
   - ❌ "ROI será 33,000%"
   - ✅ "Segmento high_value tiene mejor ROI relativo"

4. **Forecast de más de 7 días**
   - ❌ Predicciones de 30 días
   - ✅ Solo usar ventana de 7 días

---

## 📈 Métricas de los Modelos

### Modelo de Clientes (V3)
- **Accuracy:** 87.59% ✅
- **AUC:** 0.66
- **Days MAE:** 5.00 días
- **Value MAE:** $123.72 ⚠️

### Modelo de Productos
- **MAE:** 19.17 unidades ⚠️
- **RMSE:** 45.89
- **Problema:** MAE > mean = no confiable para valores absolutos

---

## 🔄 Mejoras Futuras

### Para mejorar confiabilidad BAJA:

1. **Valores monetarios:**
   - Más datos de entrenamiento
   - Features adicionales (estacionalidad, promociones)
   - Modelo específico para regresión de valor

2. **Cantidades de productos:**
   - Mejor modelo de productos (más epochs, datos)
   - Incorporar inventario histórico
   - Modelos específicos por categoría de producto

3. **ROI de campañas:**
   - Tracking de conversiones reales
   - Feedback loop con resultados de campañas
   - A/B testing

---

## 🎯 Resumen Ejecutivo

### ✅ **Funcionalidades Confiables AHORA:**
1. Ranking y priorización de clientes (87% accuracy)
2. Segmentación de clientes (Alta/Media/Baja)
3. Timing de contacto (±5 días)
4. Recomendaciones de productos cross-sell
5. Comparaciones relativas de demanda

### ⚠️ **Funcionalidades Mejorables:**
1. Forecast de ingresos exactos (40% error)
2. Cantidades exactas de inventario (190% error)
3. ROI de campañas (basado en valores imprecisos)

**Recomendación:** Usar el sistema para **decisiones relativas** (quién contactar, qué priorizar, cuándo actuar), NO para **valores absolutos** (cuánto gastará, cuántas unidades).

---

## 📞 Soporte

Para más información, revisa:
- `RESUMEN_CAPACIDADES_MODELOS.md` - Análisis detallado de capacidades
- `analisis_realista_modelos.py` - Script de análisis de confiabilidad

---

**Versión:** 1.0
**Última actualización:** 2025-11-05
**Modelos:** Customer V3 + Product SHORT (7 días)
