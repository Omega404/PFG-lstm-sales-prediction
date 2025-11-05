# 📊 Sistema de Predicción de Ventas - LSTM

Sistema web simple para consultar predicciones de ventas de productos usando modelos LSTM entrenados.

## 🚀 Cómo usar

### 1. Instalar dependencias

```bash
pip install flask flask-cors tensorflow pandas openpyxl
```

### 2. Iniciar el servidor backend

```bash
python app_prediccion.py
```

El servidor se iniciará en `http://localhost:5000`

### 3. Abrir la interfaz web

Abre en tu navegador:
```
http://localhost:5000
```

O simplemente abre el archivo `prediccion.html` directamente.

## 📝 Funcionalidades

### ✅ Consultar Predicción
1. Ingresa un código de producto (ej: `20723`, `22112`, `85123A`)
2. Selecciona el horizonte temporal:
   - **Corto Plazo**: Predicción para 2 semanas
   - **Medio Plazo**: Predicción para 1 mes
   - **Largo Plazo**: Predicción para 2 meses
3. Haz clic en "Predecir Ventas"

### ✅ Ver Productos Disponibles
- Haz clic en "Ver Productos Disponibles"
- Muestra todos los productos con modelos entrenados
- Haz clic en cualquier producto para seleccionarlo

### ✅ Visualización
- Gráfico de ventas históricas (últimos 30 días)
- Métricas clave: promedio, máximo, total histórico
- Precio promedio del producto

## 🏗️ Estructura de Archivos

```
Proyecto Final/
├── app_prediccion.py          # Backend Flask (API REST)
├── prediccion.html            # Frontend (interfaz web)
├── data/
│   └── online_retail.xlsx     # Dataset
├── models/
│   ├── temporal/              # Modelos temporales (short/medium/long)
│   │   ├── lstm_20723_short.h5
│   │   ├── scaler_20723_short.pkl
│   │   └── ...
│   └── trained/               # Modelos simples
│       ├── lstm_20723.h5
│       ├── scaler_20723.pkl
│       └── ...
```

## 🔌 API Endpoints

### `GET /api/products`
Lista todos los productos con modelos disponibles.

**Response:**
```json
{
  "success": true,
  "count": 35,
  "products": [
    {
      "ProductCode": "20723",
      "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
      "TotalSales": 2458,
      "TotalRevenue": 5124.32,
      "AvgPrice": 2.08
    }
  ]
}
```

### `POST /api/predict`
Realiza predicción para un producto específico.

**Request:**
```json
{
  "product_code": "20723",
  "horizon": "short"
}
```

**Response:**
```json
{
  "success": true,
  "product_code": "20723",
  "horizon": "short",
  "prediction": {
    "value": 145.23,
    "units": 145,
    "confidence": "medium"
  },
  "historical": {
    "last_30_days": [12, 15, 8, ...],
    "avg_daily": 10.5,
    "max_daily": 45
  },
  "product_info": { ... }
}
```

### `GET /api/health`
Health check del servidor.

## 🎨 Características de la UI

- ✨ Diseño moderno con gradientes
- 📊 Gráficos interactivos con Chart.js
- ⚡ Animaciones suaves
- 📱 Responsive (funciona en móviles)
- 🎯 Autocompletado de productos
- ⌨️ Atajos de teclado (Enter para buscar)

## 🔧 Notas Técnicas

### Caché de Modelos
Los modelos se cargan en memoria al hacer la primera predicción y se mantienen en caché para mejorar el rendimiento.

### Formato de Secuencias
El sistema genera secuencias de los últimos 30 días para cada producto (simplificado). En producción deberías usar el mismo pipeline de preprocessing que en el entrenamiento.

### Escalado
Se aplica el mismo `RobustScaler` usado durante el entrenamiento.

## 🐛 Troubleshooting

### Error: "Modelo no encontrado"
- Verifica que el producto tenga un modelo entrenado en `models/temporal/` o `models/trained/`
- Revisa que los archivos `.h5` y `.pkl` existan

### Error: "No se encontraron datos"
- El producto debe existir en `data/online_retail.xlsx`
- Verifica que el código de producto sea correcto

### Error de conexión
- Asegúrate de que el backend Flask esté corriendo
- Verifica que el puerto 5000 esté disponible
- Comprueba la URL en `prediccion.html` (línea con `API_URL`)

## 📈 Mejoras Futuras

- [ ] Autenticación de usuarios
- [ ] Exportar predicciones a CSV/Excel
- [ ] Comparar múltiples productos
- [ ] Predicciones batch (múltiples productos a la vez)
- [ ] Intervalos de confianza
- [ ] Análisis de sensibilidad
- [ ] Deploy en producción (Docker, AWS, etc.)

## 📄 Licencia

Proyecto académico - PFG LSTM 2025
