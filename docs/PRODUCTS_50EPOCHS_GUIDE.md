# 🚀 Products 50 Epochs - Guía de Entrenamiento

## 📋 Propósito

Entrenar modelos de Products con **50 épocas** en todos los horizontes para comparar con el baseline:

| Horizonte | Baseline Epochs | 50 Epochs | Diferencia |
|-----------|----------------|-----------|------------|
| **SHORT** | 20 | **50** | +30 epochs (+150%) |
| **MEDIUM** | 30 | **50** | +20 epochs (+67%) |
| **LONG** | 40 | **50** | +10 epochs (+25%) |

**Objetivo:** Ver si más épocas mejoran el MAE sin causar overfitting.

---

## ⚡ Ejecutar Rápido

### **Opción 1: Script Batch (Windows)**

```bash
# Ejecutar desde: E:\Codigos\Proyecto Final

train_products_50epochs.bat
```

Selecciona:
- `1` = Entrenar TODOS (SHORT + MEDIUM + LONG) ← Recomendado
- `2` = Solo SHORT
- `3` = Solo MEDIUM
- `4` = Solo LONG

### **Opción 2: Línea de Comandos**

```bash
cd "E:\Codigos\Proyecto Final"

# Entrenar TODOS (recomendado)
python src\train\train_products_temporal_50epochs.py --horizon all --platform local --script-version v1_50epochs

# Solo MEDIUM (para comparar rápido)
python src\train\train_products_temporal_50epochs.py --horizon medium --platform local --script-version v1_50epochs
```

---

## 📂 Archivos Generados

### **NO sobreescribe modelos originales**

Los modelos se guardan en un directorio **separado**:

```
E:\Codigos\Proyecto Final\
├── models\temporal\
│   ├── products\                    ← Modelos baseline (originales)
│   │   ├── short\   (20 epochs)
│   │   ├── medium\  (30 epochs)
│   │   └── long\    (40 epochs)
│   │
│   └── products_50epochs\           ← Modelos NUEVOS (50 epochs)
│       ├── short\   (50 epochs)    ← SE CREAN AQUÍ
│       ├── medium\  (50 epochs)    ← SE CREAN AQUÍ
│       └── long\    (50 epochs)    ← SE CREAN AQUÍ
```

---

## 📊 MLflow Tracking

### **Experimento Separado**

Los runs se guardan en un experimento **diferente**:

```
MLflow Experimentos:
├── products_temporal              ← Baseline (20/30/40 epochs)
└── products_temporal_50epochs     ← Nuevos (50 epochs) ✨
```

### **Parámetros Logged**

Cada run tiene:
- `version`: "50_epochs" (identificador)
- `epochs_config`: 50 (para todos)
- `horizon`: short/medium/long
- Resto igual que baseline

---

## 🔬 Comparación en MLflow

### **Paso 1: Iniciar MLflow UI**

```bash
cd "E:\Codigos\Proyecto Final"
mlflow ui
```

Abre: http://localhost:5000

### **Paso 2: Ver ambos experimentos**

1. **Baseline (original):**
   - Experimento: `products_temporal`
   - Runs:
     - `products_short_local` (20 epochs)
     - `products_medium_local` (30 epochs)
     - `products_long_local` (40 epochs)

2. **50 Epochs (nuevo):**
   - Experimento: `products_temporal_50epochs`
   - Runs:
     - `products_short_local_50ep` (50 epochs)
     - `products_medium_local_50ep` (50 epochs)
     - `products_long_local_50ep` (50 epochs)

### **Paso 3: Comparar Métricas**

**Opción A: Comparar dentro de cada experimento**

1. Ve a `products_temporal_50epochs`
2. Ordena por `mae` (ascendente)
3. Compara SHORT vs MEDIUM vs LONG (todos con 50 epochs)

**Opción B: Comparar baseline vs 50 epochs**

Para MEDIUM por ejemplo:

1. Anota MAE de `products_temporal` → MEDIUM: **19.00**
2. Anota MAE de `products_temporal_50epochs` → MEDIUM: **[NUEVO VALOR]**
3. Calcula diferencia:
   ```
   Mejora = (19.00 - NUEVO_MAE) / 19.00 * 100
   ```

---

## ⏱️ Tiempos de Entrenamiento Estimados

Con early stopping (puede terminar antes):

| Horizonte | Epochs Max | Tiempo Estimado | Muestras |
|-----------|-----------|-----------------|----------|
| **SHORT** | 50 | ~15-20 min | 52,000 |
| **MEDIUM** | 50 | ~20-25 min | 52,000 |
| **LONG** | 50 | ~25-30 min | 52,000 |
| **TODOS** | - | **~60-75 min** | - |

**Nota:** Con early stopping (patience=15), puede converger antes de 50 epochs.

---

## 📈 Resultados Esperados

### **Baseline Actual (ya entrenado):**

| Horizonte | Epochs | MAE | RMSE | Converged |
|-----------|--------|-----|------|-----------|
| SHORT | 20 | 19.93 | 46.09 | Epoch ~18 |
| MEDIUM | 30 | **19.00** | **42.10** | Epoch ~28 ✅ |
| LONG | 40 | **18.51** | 44.85 | Epoch ~35 |

**MEDIUM es el mejor balance** (menor MAE sin overfit)

### **50 Epochs - Escenarios Posibles:**

#### **Escenario 1: Mejora moderada (esperado)**

| Horizonte | MAE 50ep | vs Baseline | Conclusión |
|-----------|----------|-------------|------------|
| SHORT | ~19.50 | -2% | Mejora leve |
| MEDIUM | ~18.70 | **-1.6%** ✅ | Mejora útil |
| LONG | ~18.30 | -1.1% | Mejora marginal |

**Conclusión:** MEDIUM con 50 epochs podría ser el mejor modelo.

#### **Escenario 2: Sin mejora (posible)**

| Horizonte | MAE 50ep | vs Baseline | Conclusión |
|-----------|----------|-------------|------------|
| SHORT | ~19.95 | +0.1% | Sin cambio |
| MEDIUM | ~19.10 | +0.5% | Sin cambio |
| LONG | ~18.60 | +0.5% | Sin cambio |

**Conclusión:** Early stopping ya estaba optimal. Usar baseline.

#### **Escenario 3: Overfit (no deseado)**

| Horizonte | MAE 50ep | RMSE 50ep | Conclusión |
|-----------|----------|-----------|------------|
| LONG | ~18.20 | **46.00** | Overfit (RMSE subió) |

**Conclusión:** 50 epochs es demasiado para LONG. Usar baseline (40 epochs).

---

## ✅ Checklist de Análisis

Después de entrenar, revisar:

### **Para cada horizonte:**

- [ ] **MAE mejoró?**
  - Si mejora >1% → Considerar usar modelo 50 epochs
  - Si mejora <1% → No vale la pena (usar baseline)

- [ ] **RMSE se mantuvo o bajó?**
  - Si RMSE bajó → ✅ Mejora real
  - Si RMSE subió → ❌ Overfitting (no usar)

- [ ] **Converged en qué epoch?**
  - Si convergió <40 epochs → No necesitaba 50
  - Si convergió ~45-50 → Necesitaba más épocas

### **Decisión final:**

```
Si MEDIUM 50epochs tiene:
  - MAE < 19.00 (mejora)
  - RMSE ≤ 42.10 (sin overfit)
  - Convergió en epoch 40-50

  → ✅ Usar MEDIUM 50epochs para producción

Si no cumple lo anterior:
  → ✅ Usar MEDIUM baseline (30 epochs) actual
```

---

## 🎯 Próximos Pasos

### **Después de entrenar:**

1. **Abrir MLflow:**
   ```bash
   mlflow ui
   ```

2. **Comparar métricas:**
   - Baseline vs 50 epochs
   - Anotar en `ANALISIS_COMPARATIVO.md`

3. **Decidir modelo final:**
   - Si 50 epochs mejor → Copiar a `production/`
   - Si baseline mejor → Mantener baseline

4. **Actualizar documentación:**
   - `CONFIGURACIONES_MODELOS.md`
   - `docs/PROCESO_TRABAJO_FINAL.txt`

---

## 📝 Comandos Útiles

### **Ver métricas guardadas:**

```bash
# Baseline MEDIUM
type models\temporal\products\medium\metrics.json

# 50 epochs MEDIUM
type models\temporal\products_50epochs\medium\metrics.json
```

### **Comparar manualmente:**

```python
import json

# Cargar baseline
with open('models/temporal/products/medium/metrics.json') as f:
    baseline = json.load(f)

# Cargar 50 epochs
with open('models/temporal/products_50epochs/medium/metrics.json') as f:
    epochs50 = json.load(f)

# Comparar
print(f"Baseline MAE: {baseline['mae']:.2f}")
print(f"50 Epochs MAE: {epochs50['mae']:.2f}")
print(f"Mejora: {(baseline['mae'] - epochs50['mae']):.2f} ({((baseline['mae'] - epochs50['mae'])/baseline['mae']*100):.1f}%)")
```

---

## 🆘 Troubleshooting

### **Error: "Directory already exists"**

Es normal. El script crea el directorio si no existe. Si ya existe, lo usa.

### **Early stopping se activa muy pronto**

Si early stopping se activa en epoch 20-25:
- No necesitabas 50 epochs
- El modelo ya había convergido
- Baseline era óptimo

### **Out of memory**

Reduce batch size en el script:
```python
MEDIUM = {
    'batch_size': 32,  # En vez de 64
}
```

---

## 🎓 Interpretación de Resultados

### **Si MAE mejora pero RMSE empeora:**

Ejemplo:
```
Baseline:  MAE 19.00, RMSE 42.10
50 Epochs: MAE 18.70, RMSE 43.50
```

**Análisis:** Está ajustando mejor en promedio (MAE) pero peor en outliers (RMSE).

**Decisión:** Usar baseline (mejor balance).

### **Si ambos mejoran:**

Ejemplo:
```
Baseline:  MAE 19.00, RMSE 42.10
50 Epochs: MAE 18.70, RMSE 41.80
```

**Análisis:** ✅ Mejora real en ambas métricas.

**Decisión:** Usar 50 epochs.

### **Si ambos empeoran:**

Ejemplo:
```
Baseline:  MAE 19.00, RMSE 42.10
50 Epochs: MAE 19.30, RMSE 43.00
```

**Análisis:** Overfitting o run malo.

**Decisión:** Usar baseline. Opcionalmente re-entrenar 50 epochs para verificar.

---

## ✅ Resumen

| Aspecto | Detalle |
|---------|---------|
| **Script** | `train_products_temporal_50epochs.py` |
| **Batch** | `train_products_50epochs.bat` |
| **Output** | `models/temporal/products_50epochs/` |
| **MLflow** | Experimento: `products_temporal_50epochs` |
| **Epochs** | 50 para SHORT, MEDIUM, LONG |
| **Tiempo** | ~60-75 min para TODOS |
| **Objetivo** | Comparar con baseline (20/30/40) |

**¿Listo para entrenar?** 🚀

```bash
train_products_50epochs.bat
```
