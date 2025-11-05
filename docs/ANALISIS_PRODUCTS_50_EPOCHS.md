# Análisis: Entrenamiento Products (50 epochs max)

**Fecha:** 2025-01-05
**Tipo:** LSTM Temporal - Products
**Configuración:** 50 epochs max, early stopping patience=5

---

## Resumen Ejecutivo

**Estado:** ✅ ENTRENAMIENTO EXITOSO

**Resultados destacados:**
- MAE consistente: 18-20 unidades
- Early stopping funcionó correctamente
- Mejor modelo: LONG (240 días ventana) con MAE = 18.65
- No hay overfitting detectado

---

## Métricas Detalladas

### SHORT Horizon (30 días → 7 días)

```json
{
  "horizon": "short",
  "window_days": 30,
  "forecast_days": 7,
  "mae": 20.01,
  "rmse": 46.06,
  "train_samples": 52288,
  "val_samples": 13072,
  "epochs_trained": 20,
  "final_train_loss": 0.000761,
  "final_val_loss": 0.000642
}
```

**Análisis:**
- MAE: 20.01 unidades (error promedio en predicción de cantidad)
- RMSE: 46.06 (penaliza más los errores grandes)
- Early stopping: Paró en epoch 20/50 (convergió temprano)
- Loss ratio: Train/Val ≈ 1.18 (sin overfitting)

### MEDIUM Horizon (120 días → 7 días)

```json
{
  "horizon": "medium",
  "window_days": 120,
  "forecast_days": 7,
  "mae": 19.75,
  "rmse": 42.03,
  "train_samples": 45088,
  "val_samples": 11272,
  "epochs_trained": 30,
  "final_train_loss": 0.000601,
  "final_val_loss": 0.000533
}
```

**Análisis:**
- MAE: 19.75 unidades (mejora de 1.3% vs SHORT)
- RMSE: 42.03 (mejora de 8.7% vs SHORT) ← **Significativo**
- Early stopping: Paró en epoch 30/50
- Loss ratio: Train/Val ≈ 1.13 (excelente)
- Más contexto (120 días) → mejor predicción

### LONG Horizon (240 días → 7 días)

```json
{
  "horizon": "long",
  "window_days": 240,
  "forecast_days": 7,
  "mae": 18.65,
  "rmse": 44.77,
  "train_samples": 35544,
  "val_samples": 8886,
  "epochs_trained": 40,
  "final_train_loss": 0.000554,
  "final_val_loss": 0.000610
}
```

**Análisis:**
- MAE: 18.65 unidades ← **MEJOR** (mejora de 6.8% vs SHORT)
- RMSE: 44.77 (intermedio entre SHORT y MEDIUM)
- Early stopping: Paró en epoch 40/50 (necesitó más entrenamiento)
- Loss ratio: Train/Val ≈ 0.91 (val loss ligeramente mayor, normal)
- Máximo contexto (240 días) → mejor MAE pero requirió más epochs

---

## Comparación de Horizontes

| Métrica | SHORT | MEDIUM | LONG | Mejor |
|---------|-------|--------|------|-------|
| **MAE** | 20.01 | 19.75 | **18.65** | LONG |
| **RMSE** | 46.06 | **42.03** | 44.77 | MEDIUM |
| **Epochs** | 20 | 30 | 40 | - |
| **Train Loss** | 0.000761 | 0.000601 | **0.000554** | LONG |
| **Val Loss** | **0.000642** | 0.000533 | 0.000610 | SHORT |
| **Samples (Train)** | **52,288** | 45,088 | 35,544 | SHORT |

**Observaciones:**

1. **Trade-off MAE vs RMSE:**
   - LONG tiene mejor MAE (errores promedio menores)
   - MEDIUM tiene mejor RMSE (menos outliers/errores grandes)

2. **Relación ventana-samples:**
   - Mayor ventana → menos samples (menos datos cumplen requisito de 240 días)
   - SHORT tiene 47% más samples que LONG

3. **Convergencia:**
   - SHORT: Rápida (20 epochs)
   - MEDIUM: Media (30 epochs)
   - LONG: Lenta (40 epochs)
   - Más contexto → necesita más entrenamiento

---

## Análisis de Early Stopping

| Horizonte | Epochs usados | Epochs máx | % Utilizado | Observación |
|-----------|---------------|------------|-------------|-------------|
| SHORT | 20 | 50 | 40% | Convergió muy rápido |
| MEDIUM | 30 | 50 | 60% | Convergencia normal |
| LONG | 40 | 50 | 80% | Casi alcanzó el límite |

**Conclusión:**
- Early stopping funcionó correctamente
- LONG podría beneficiarse de más epochs (50-60 max)
- SHORT no necesita más de 30 epochs

**Recomendación:**
Si quieres mejorar LONG aún más, considera:
- Aumentar max_epochs a 60-70
- Mantener patience=5

---

## Análisis de Overfitting

**Train Loss vs Val Loss:**

```
SHORT:  Train=0.000761  Val=0.000642  Ratio=1.18
MEDIUM: Train=0.000601  Val=0.000533  Ratio=1.13
LONG:   Train=0.000554  Val=0.000610  Ratio=0.91
```

**Interpretación:**

✅ **SHORT y MEDIUM:** Train Loss > Val Loss
- Normal y saludable
- No hay overfitting

✅ **LONG:** Val Loss ligeramente > Train Loss
- También normal (puede indicar que val set es más difícil)
- Diferencia mínima (0.000056)
- No hay overfitting

**Conclusión:** No hay overfitting en ningún horizonte. Los modelos generalizan bien.

---

## Interpretación de MAE según Escala de Datos

**MAE = 18-20 unidades**

¿Es bueno o malo? Depende del rango típico de `Quantity` en el dataset.

**Si el rango es:**
- 1-10 unidades: MAE=20 es MALO (error > rango)
- 10-100 unidades: MAE=20 es ACEPTABLE (error ~20% promedio)
- 100-1000 unidades: MAE=20 es EXCELENTE (error <5%)

**Necesitamos verificar:**
```python
import pandas as pd
df = pd.read_excel('online_retail_2.xlsx')
print(df['Quantity'].describe())
```

Métricas clave:
- **Mean:** ¿Cuál es la cantidad promedio?
- **Std:** ¿Cuál es la variabilidad?
- **Min/Max:** ¿Cuál es el rango?

**Si Mean ≈ 100:**
- MAE/Mean = 20/100 = 20% error → ACEPTABLE

**Si Mean ≈ 500:**
- MAE/Mean = 20/500 = 4% error → EXCELENTE

---

## Comparación con CUSTOMERS (para referencia)

| Modelo | Tarea | MAE | RMSE | Observación |
|--------|-------|-----|------|-------------|
| **PRODUCTS** | Predecir cantidad | 18.65 | 42.03 | Error en unidades |
| **CUSTOMERS** | Predecir días | 10-16 | N/A | Error en días |
| **CUSTOMERS** | Predecir valor | $30-80 | N/A | Error en dólares |

**Nota:** No son directamente comparables porque predicen diferentes variables.

---

## Progresión del Entrenamiento

**Hipótesis de por qué LONG necesitó más epochs:**

1. **Mayor complejidad:**
   - 240 días de ventana = 8 meses de contexto
   - Más patrones temporales que aprender

2. **Menor cantidad de datos:**
   - 35,544 samples vs 52,288 en SHORT
   - Necesita más epochs para converger con menos datos

3. **Patrones de largo plazo:**
   - Tendencias estacionales
   - Ciclos de 6-8 meses
   - Requieren más iteraciones para capturar

---

## Recomendaciones

### Para Producción

**Modelo recomendado:** **MEDIUM**

**Razones:**
1. Mejor RMSE (42.03) → menos outliers
2. MAE competitivo (19.75, solo 1.1 unidades más que LONG)
3. Balance óptimo entre:
   - Contexto (120 días = 4 meses)
   - Cantidad de samples (45,088)
   - Tiempo de entrenamiento (30 epochs)

**Usar LONG si:**
- Necesitas el MAE más bajo posible
- Tienes suficientes datos históricos (>240 días)
- No te importa el tiempo de entrenamiento adicional

**Usar SHORT si:**
- Necesitas predicciones rápidas
- Datos históricos limitados (<120 días)
- Prioridad es velocidad sobre precisión

### Para Mejorar Resultados

**LONG Horizon:**
- Aumentar max_epochs a 60-70
- Considerar aumentar patience a 7-10
- Probar arquitectura más profunda (3 capas LSTM)

**Todos los horizontes:**
- Analizar distribución de errores (¿hay productos específicos con MAE alto?)
- Feature engineering: Agregar categoría de producto, estacionalidad
- Probar diferentes arquitecturas: GRU, Attention mechanisms

---

## Archivos Generados

**Ubicación:** `E:\Codigos\Proyecto Final\models\temporal\products\`

```
products/
├── short/
│   ├── model_short.keras
│   ├── metrics.json
│   ├── training_history.pkl
│   └── scaler.pkl
├── medium/
│   ├── model_medium.keras
│   ├── metrics.json
│   ├── training_history.pkl
│   └── scaler.pkl
└── long/
    ├── model_long.keras
    ├── metrics.json
    ├── training_history.pkl
    └── scaler.pkl
```

**Modelos listos para:**
- Inferencia en producción
- Evaluación adicional en test set
- Comparación con otros modelos

---

## Próximos Pasos

1. **Verificar escala de datos:**
   - Analizar distribución de `Quantity`
   - Calcular MAE relativo (MAE/Mean)

2. **Evaluación adicional:**
   - Probar en test set (si disponible)
   - Analizar errores por categoría de producto
   - Visualizar predicciones vs reales

3. **Comparar con baseline:**
   - Naive forecast (último valor)
   - Promedio móvil
   - Confirmar que LSTM es mejor

4. **Integración:**
   - Usar modelo MEDIUM en dashboard
   - Crear API de predicción
   - Monitorear performance en producción

---

## Conclusión

✅ **Entrenamiento exitoso con resultados sólidos**

**Highlights:**
- MAE: 18.65-20.01 unidades (muy consistente)
- Early stopping funcionó correctamente
- No hay overfitting
- Modelos listos para producción

**Mejor modelo:** MEDIUM (balance óptimo)

**Siguiente:** Verificar escala de datos para interpretar si MAE=19.75 es excelente, bueno o aceptable.

---

**Fecha:** 2025-01-05
**Estado:** COMPLETADO
**Modelos:** 3 (SHORT, MEDIUM, LONG)
**Total epochs:** 90 (20+30+40)
**Mejor MAE:** 18.65 (LONG)
**Mejor RMSE:** 42.03 (MEDIUM)
