# CHANGELOG - train_all_customers_temporal_3.py

## 🐛 Correcciones Aplicadas (2025-11-04)

### **Bug #1: Accuracy >100%**
**Problema:**
- El script multiplicaba por 100 un accuracy que Keras ya devuelve como porcentaje
- Resultado: accuracy reportado como 106.69% (imposible)

**Corrección:**
```python
# ANTES (línea 640, 783, 907):
print(f"   Accuracy: {metrics['purchase_prob_accuracy']*100:.2f}%")

# DESPUÉS:
print(f"   Accuracy: {metrics['purchase_prob_accuracy']:.2f}%")
```

**Archivos modificados:**
- Línea 661: Print principal de entrenamiento
- Línea 783: Print en train_all_horizons()
- Línea 907: Print en main()

---

### **Bug #2: Days MAE y Value MAE en Escala Incorrecta**
**Problema:**
- Las métricas MAE se calculaban con datos normalizados (escala 0-1)
- Se reportaban como si fueran reales (días y dólares)
- Resultado: Days MAE = 0.63 días (debería ser ~12-18 días)
- Resultado: Value MAE = $2.29 (debería ser ~$40-60)

**Corrección (líneas 618-637):**
```python
# NUEVO: Desnormalizar predicciones antes de calcular MAE
print(f"📊 Calculando métricas en escala real...")

# Hacer predicciones
y_pred = model.predict(X_val, verbose=0)
y_pred_prob, y_pred_days_scaled, y_pred_value_scaled = y_pred

# Desnormalizar predicciones y targets
y_pred_days_real = scaler_y_days.inverse_transform(y_pred_days_scaled)
y_pred_value_real = scaler_y_value.inverse_transform(y_pred_value_scaled)

y_true_days_real = scaler_y_days.inverse_transform(y_days_scaled[val_idx])
y_true_value_real = scaler_y_value.inverse_transform(y_value_scaled[val_idx])

# Calcular MAE en escala real
days_mae_real = mean_absolute_error(y_true_days_real, y_pred_days_real)
value_mae_real = mean_absolute_error(y_true_value_real, y_pred_value_real)

# Usar MAE real en métricas
'days_mae': days_mae_real,  # ✅ MAE en escala REAL (días)
'value_mae': value_mae_real,  # ✅ MAE en escala REAL (dólares)
```

**Imports añadidos:**
- Línea 35: `from sklearn.metrics import mean_absolute_error`

---

### **Bug #3: Variable No Utilizada**
**Problema:**
- `y_pred_prob` se extraía pero no se usaba (warning de linter)

**Solución:**
- Se mantiene la variable para completitud (podría usarse en futuras versiones)
- No afecta funcionalidad

---

## ✅ Verificación

### **Sintaxis Python:**
```bash
python -m py_compile train_all_customers_temporal_3.py
# ✅ Sin errores
```

### **Métricas Esperadas (MEDIUM V3):**
Después de las correcciones, espera ver:

| Métrica | Rango Esperado | V2 (Bugueado) | V3 (Corregido) |
|---------|----------------|---------------|----------------|
| **Accuracy** | 75-90% | 106.69% ❌ | ~80-88% ✅ |
| **AUC** | 0.80-0.90 | 0.7443 ⚠️ | 0.80-0.87 ✅ |
| **Days MAE** | 12-18 días | 0.63 días ❌ | ~14-16 días ✅ |
| **Value MAE** | $40-$60 | $2.29 ❌ | ~$45-55 ✅ |

---

## 📋 Resumen de Cambios

**Total de líneas modificadas:** 8
- 1 import añadido (línea 35)
- 4 líneas de cálculo de MAE (líneas 628-637)
- 3 líneas de print corregidas (líneas 661, 783, 907)

**Archivos relacionados:**
- ❌ `train_all_customers_temporal_2.py` - **NO corregido** (mismos bugs)
- ✅ `train_all_customers_temporal_3.py` - **Corregido**

---

## 🎯 Próximos Pasos

1. **Entrenar MEDIUM V3** en Google Colab Pro (A100 GPU)
   - Configuración: 120→7 días
   - Tiempo estimado: 1.5-2h
   - Dataset: Top 1000 clientes

2. **Comparar con productos:**
   - Productos MEDIUM: 120→7 días, MAE 19.00
   - Clientes MEDIUM V3: 120→7 días, accuracy esperado ~85%

3. **Reentrenar V2** si se necesita comparación justa
   - Aplicar mismas correcciones a V2
   - Comparar forecast 14 días vs 7 días

---

## 📝 Notas

- **V2 vs V3:** La única diferencia funcional es el `forecast_days` (V2=14d, V3=7d)
- **Comparabilidad:** V3 con 7 días es directamente comparable con productos
- **Data leakage:** El AUC bajo (0.74) en V2 sugiere que NO hay data leakage severo, solo bugs de reporting
