# 📊 Configuraciones de Modelos de Clientes LSTM

## Resumen de Versiones

Este proyecto tiene 3 versiones del sistema de entrenamiento para evaluar diferentes configuraciones de ventanas temporales.

---

## 📋 Tabla Comparativa

| Script | SHORT | MEDIUM | LONG | Objetivo |
|--------|-------|--------|------|----------|
| **V1 (original)** | 30→7d | 120→30d | 240→60d | Baseline - Ventanas proporcionales |
| **V2 (cortas)** | 30→7d | 120→14d | 240→14d | Reducir forecast, mantener contexto |
| **V3 (uniforme)** | 30→7d | 120→7d | 240→7d | Mismo forecast, comparar contexto |

---

## 🔍 Detalle de Configuraciones

### **V1 - Original** ([train_all_customers_temporal.py](src/train/train_all_customers_temporal.py))

**Hipótesis:** Ventanas proporcionales (ratio 4:1) capturan patrones a diferentes escalas temporales.

| Horizonte | Window | Forecast | Ratio | LSTM Units | Epochs | Batch Size |
|-----------|--------|----------|-------|------------|--------|------------|
| SHORT | 30 días | 7 días | 4:1 | [64, 32] | 20 | 32 |
| MEDIUM | 120 días | 30 días | 4:1 | [128, 64] | 30 | 64 |
| LONG | 240 días | 60 días | 4:1 | [128, 64, 32] | 40 | 64 |

**Resultados conocidos:**
- ✅ **SHORT**: AUC 0.8737, Acc 52.37% → Excelente
- ❌ **MEDIUM**: AUC 0.6393, Acc 47.09% → Malo (peor que azar)
- ⏳ **LONG**: Entrenando en Kaggle (~5 horas)

---

### **V2 - Ventanas Cortas** ([train_all_customers_temporal_2.py](src/train/train_all_customers_temporal_2.py))

**Hipótesis:** El problema de V1 es que predecir 30-60 días es muy difícil. Reduciendo forecast a 14 días mantendremos ventanas históricas largas (contexto) pero con predicción más cercana (más fácil).

| Horizonte | Window | Forecast | Ratio | LSTM Units | Epochs | Batch Size |
|-----------|--------|----------|-------|------------|--------|------------|
| SHORT | 30 días | 7 días | 4:1 | [64, 32] | 20 | 32 |
| MEDIUM | 120 días | **14 días** ⭐ | 8.5:1 | [128, 64] | 30 | 64 |
| LONG | 240 días | **14 días** ⭐ | 17:1 | [128, 64, 32] | 40 | 64 |

**Cambios:**
- MEDIUM: 30d → **14d** forecast
- LONG: 60d → **14d** forecast

**Expectativa:**
- MEDIUM_v2 debería tener **mejor AUC que MEDIUM_v1** (0.64 → 0.75+?)
- LONG_v2 debería ser **mejor que LONG_v1** (aún por ver)

---

### **V3 - Forecast Uniforme** ([train_all_customers_temporal_3.py](src/train/train_all_customers_temporal_3.py))

**Hipótesis:** Para comparar justamente el impacto del contexto histórico, todos los modelos deben predecir el **mismo horizonte** (7 días).

| Horizonte | Window | Forecast | Ratio | LSTM Units | Epochs | Batch Size |
|-----------|--------|----------|-------|------------|--------|------------|
| SHORT | 30 días | 7 días | 4:1 | [64, 32] | 20 | 32 |
| MEDIUM | 120 días | **7 días** ⭐ | 17:1 | [128, 64] | 30 | 64 |
| LONG | 240 días | **7 días** ⭐ | 34:1 | [128, 64, 32] | 40 | 64 |

**Cambios:**
- MEDIUM: 30d → **7d** forecast (mismo que SHORT)
- LONG: 60d → **7d** forecast (mismo que SHORT)

**Expectativa:**
- MEDIUM_v3 debería ser **mejor o igual que SHORT** (más contexto histórico)
- LONG_v3 debería ser **el mejor de todos** (máximo contexto)
- Permite responder: ¿Ayuda tener más historial para predecir 7 días?

---

## 🧪 Plan de Experimentos

### Fase 1: Entrenar V2 (localmente en PC)
```bash
# MEDIUM_v2
python src/train/train_all_customers_temporal_2.py
# Seleccionar opción 3 (solo MEDIUM)

# LONG_v2 (si cabe en RAM, sino usar Kaggle)
python src/train/train_all_customers_temporal_2.py
# Seleccionar opción 4 (solo LONG)
```

### Fase 2: Entrenar V3 (localmente en PC)
```bash
# MEDIUM_v3
python src/train/train_all_customers_temporal_3.py
# Seleccionar opción 3 (solo MEDIUM)

# LONG_v3
python src/train/train_all_customers_temporal_3.py
# Seleccionar opción 4 (solo LONG)
```

### Fase 3: Análisis Comparativo

Comparar las 9 configuraciones:

| Modelo | Window→Forecast | AUC | Accuracy | Conclusión |
|--------|-----------------|-----|----------|------------|
| SHORT_v1 | 30→7 | 0.8737 | 52.37% | ✅ Baseline excelente |
| MEDIUM_v1 | 120→30 | 0.6393 | 47.09% | ❌ Malo |
| LONG_v1 | 240→60 | ??? | ??? | ⏳ Entrenando |
| MEDIUM_v2 | 120→14 | ??? | ??? | 🔬 Por entrenar |
| LONG_v2 | 240→14 | ??? | ??? | 🔬 Por entrenar |
| MEDIUM_v3 | 120→7 | ??? | ??? | 🔬 Por entrenar |
| LONG_v3 | 240→7 | ??? | ??? | 🔬 Por entrenar |

---

## 🎯 Preguntas que responderemos

1. **¿Importa más el contexto histórico o la cercanía de la predicción?**
   - Comparar V1 vs V2 vs V3

2. **¿Hay un punto óptimo de window/forecast?**
   - Analizar ratios 4:1 vs 8:1 vs 17:1 vs 34:1

3. **¿Más contexto siempre es mejor para predicción cercana?**
   - Comparar SHORT_v3 vs MEDIUM_v3 vs LONG_v3 (todos predicen 7 días)

4. **¿Cuál configuración usar en producción?**
   - Balance entre accuracy y utilidad práctica

---

## 📁 Estructura de Salida

Los modelos se guardan en directorios independientes:

```
models/temporal/customer/
├── short/              # V1 SHORT (30→7)
├── medium/             # V1 MEDIUM (120→30)
├── long/               # V1 LONG (240→60)
```

**Nota:** V2 y V3 se guardan en las mismas carpetas pero puedes renombrarlas manualmente después:
- `medium/` → `medium_v2/` (120→14)
- `long/` → `long_v2/` (240→14)
- Etc.

O modificar `output_dir` en cada script antes de entrenar.

---

## ⏱️ Tiempo Estimado de Entrenamiento

### En PC Local (CPU):
- SHORT: ~1.5 horas
- MEDIUM: ~12-15 horas
- LONG: ~20-25 horas

### En Kaggle (GPU T4 x2):
- SHORT: ~15-20 min
- MEDIUM: ~45-60 min
- LONG: ~5 horas (con 1000 clientes, batch 16)

**Recomendación:** Entrenar MEDIUM en PC, LONG en Kaggle.

---

## 📝 Notas Importantes

1. **MEDIUM y LONG generan menos samples** debido a ventanas más grandes
2. **LONG en Kaggle usa 1000 clientes** (vs 2127) por límites de RAM
3. **Batch size de LONG reducido** a 16 (vs 64) para evitar OOM
4. **Early stopping activado** (patience=20) puede terminar antes de max epochs

---

## 🚀 Próximos Pasos

1. ✅ Crear scripts V2 y V3
2. ⏳ Esperar que termine LONG_v1 en Kaggle
3. 🔄 Entrenar MEDIUM_v2 y MEDIUM_v3 localmente
4. 🔄 Entrenar LONG_v2 y LONG_v3 en Kaggle
5. 📊 Comparar todos los resultados
6. 🎯 Elegir configuración óptima para producción
