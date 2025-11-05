# LSTM Customer & Product Prediction System

Sistema de predicción de clientes y productos usando modelos LSTM (Long Short-Term Memory), con análisis cruzado y dashboard web interactivo.

## Descripción

Este proyecto predice:
1. **Clientes:** Quién comprará, cuándo, y cuánto gastará
2. **Productos:** Qué productos tendrán demanda
3. **Análisis Cruzado:** Recomendaciones de productos por cliente

### Modelos Entrenados

- **Customer Model V3:** 87.59% accuracy, horizonte 7 días
- **Product Model:** MAE 19.17 unidades, horizonte 7 días

---

## Inicio Rápido

### 1. Requisitos

```bash
Python 3.8+
pip install -r requirements.txt
```

### 2. Iniciar Dashboard Web

```bash
python app_cross_analysis_web.py
```

Accede a: **http://localhost:5001**

### 3. Funcionalidades

#### ✅ Alta Confiabilidad (Usar con confianza)
- Ranking de clientes por probabilidad de compra (87% accuracy)
- Segmentación: Alta/Media/Baja prioridad
- Recomendaciones de productos cross-sell
- Comparación relativa de demanda de productos

#### ⚠️ Baja Confiabilidad (Mejorable)
- Forecast de ingresos ($123 MAE = 40% error)
- Cantidades exactas de inventario
- ROI de campañas

Ver documentación completa en [docs/README_WEB_SYSTEM.md](docs/README_WEB_SYSTEM.md)

---

## Estructura del Proyecto

```
.
├── app_cross_analysis_web.py    # Sistema web principal
├── src/                         # Código fuente
│   ├── analysis/               # Análisis cruzado
│   ├── train/                  # Scripts de entrenamiento
│   └── services/               # Servicios (preprocessing, etc.)
├── models/                      # Modelos entrenados (no en Git)
│   ├── temporal/customer_v3/   # Modelo de clientes V3
│   └── temporal/products_50epochs/ # Modelo de productos
├── data/                        # Datos (no en Git)
│   ├── raw/                    # Datos crudos
│   └── processed/              # Datos procesados
├── templates/                   # HTML para dashboard
├── notebooks/                   # Jupyter notebooks
├── scripts/                     # Scripts de análisis
│   ├── analysis/               # Análisis de resultados
│   └── training/               # Scripts de entrenamiento auxiliares
├── docs/                        # Documentación
└── output/                      # Resultados de análisis
```

---

## API Endpoints

### Alta Confiabilidad

```bash
# Top 10 clientes
GET /api/customers/ranking?top_n=10

# Segmentación
GET /api/customers/segments

# Detalle de cliente
GET /api/customers/<id>/details

# Ranking de productos
GET /api/products/ranking?top_n=20

# Cross-sell
GET /api/products/<code>/cross-sell
```

### Mejorable (usar con precaución)

```bash
# Forecast de ingresos
GET /api/forecast/revenue

# Forecast de inventario
GET /api/forecast/inventory

# ROI de campaña
POST /api/campaign/roi
```

---

## Casos de Uso

### 1. Priorización de Clientes

```python
# Obtener top clientes de alta probabilidad
GET /api/customers/ranking?top_n=10&segment=high_value_high_prob

# Resultado: Cliente #13352 tiene 100% probabilidad
# Acción: Contactar INMEDIATAMENTE con oferta premium
```

### 2. Timing de Campañas

```python
# Ver cuándo contactar
GET /api/customers/13352/details

# Resultado: Ventana de compra 2-7 días
# Acción: Enviar campaña ESTA SEMANA
```

### 3. Recomendaciones Cross-Sell

```python
# Ver productos relacionados
GET /api/products/85123A/cross-sell

# Resultado: 45% de clientes también compraron JUMBO BAG BLUE
# Acción: Ofrecer bundle "3 colores por $X"
```

---

## Documentación

### Esencial
- [README_WEB_SYSTEM.md](docs/README_WEB_SYSTEM.md) - Sistema web completo
- [RESUMEN_CAPACIDADES_MODELOS.md](docs/RESUMEN_CAPACIDADES_MODELOS.md) - Análisis de confiabilidad
- [CONFIGURACIONES_MODELOS.md](docs/CONFIGURACIONES_MODELOS.md) - Configuraciones de modelos

### Análisis
- [ANALISIS_COMPARATIVO.md](docs/ANALISIS_COMPARATIVO.md) - Comparación de modelos
- `scripts/analysis/analisis_realista_modelos.py` - Script de análisis de confiabilidad

---

## Modelos y Datos

### Modelos

Los modelos entrenados (**NO están en Git** por tamaño):
- `models/temporal/customer_v3/medium/` - Modelo de clientes V3
- `models/temporal/products_50epochs/short/` - Modelo de productos

Para obtener los modelos, contactar al equipo o entrenar desde cero.

### Datos

Dataset: **Online Retail II** (UCI Machine Learning Repository)
- 811,893 transacciones
- 5,924 clientes únicos
- 4,645 productos únicos
- Período: 2009-2011

**Los datos NO están en Git** por privacidad/tamaño.

---

## Entrenamiento de Modelos

### Customer Model

```bash
# Ver notebooks en notebooks/
# - LSTM_Customer_V3_Kaggle.ipynb (recomendado)
# - LSTM_Customer_V2_Kaggle.ipynb

# O usar scripts en src/train/
python src/train/train_customer_single_horizon.py
```

### Product Model

```bash
# Ver notebooks/
# - LSTM_Products_SHORT_Kaggle.ipynb
# - LSTM_Products_MEDIUM_Kaggle.ipynb

python src/train/train_products_temporal.py
```

---

## Deployment

### Docker

```bash
docker build -t lstm-prediction .
docker run -p 5001:5001 lstm-prediction
```

### Google Cloud Run

Ver [CLOUD_RUN_SETUP.md](docs/CLOUD_RUN_SETUP.md) (si disponible)

---

## Métricas de los Modelos

### Customer Model V3
| Métrica | Valor | Confiabilidad |
|---------|-------|---------------|
| Accuracy (probabilidad) | 87.59% | ✅ ALTA |
| AUC | 0.66 | ✅ ALTA |
| Days MAE | 5.00 días | ⚡ MEDIA (71% error en 7 días) |
| Value MAE | $123.72 | ⚠️ BAJA (40% error) |

### Product Model
| Métrica | Valor | Confiabilidad |
|---------|-------|---------------|
| MAE | 19.17 unidades | ⚠️ BAJA (>mean) |
| RMSE | 45.89 | ⚠️ BAJA |

**Conclusión:** Excelente para **ranking y segmentación**, NO para **valores absolutos exactos**.

---

## Tecnologías

- **Backend:** Python 3, Flask
- **ML:** TensorFlow 2.x, Keras
- **Data:** Pandas, NumPy, Scikit-learn
- **Frontend:** HTML5, JavaScript (Vanilla)
- **Deployment:** Docker, Google Cloud Run

---

## Contribuir

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

---

## Licencia

Este proyecto es parte de un trabajo académico/profesional.

---

## Contacto

Para preguntas sobre el proyecto, modelos, o datos, contactar al equipo.

---

## Notas Importantes

### ⚠️ Limitaciones Conocidas

1. **Valores monetarios** tienen ~40% error (MAE $123)
2. **Cantidades de productos** tienen alta variabilidad (MAE > mean)
3. **Predicciones solo hasta 7 días** (horizonte del modelo)
4. **Clientes nuevos** (sin historial) tienen predicciones poco confiables

### ✅ Fortalezas

1. **Ranking de clientes** muy confiable (87% accuracy)
2. **Segmentación** efectiva para priorizar contactos
3. **Recomendaciones cross-sell** basadas en patrones reales
4. **Comparaciones relativas** entre productos confiables

---

## Roadmap

### Mejoras Planificadas

- [ ] Mejorar predicción de valores monetarios (más datos)
- [ ] Modelo de productos con mayor accuracy
- [ ] Tracking de conversiones reales para validar ROI
- [ ] Frontend con React/Vue para mejor UX
- [ ] API REST completa con autenticación
- [ ] Dashboard de métricas en tiempo real
- [ ] Integración con CRM

---

## Changelog

### v1.0 (2025-11-05)
- Sistema web completo con Flask
- Dashboard interactivo
- 11 API endpoints (5 alta confiabilidad, 3 mejorables, 3 utilidades)
- Documentación completa
- Análisis de confiabilidad de modelos

---

**Última actualización:** 2025-11-05
