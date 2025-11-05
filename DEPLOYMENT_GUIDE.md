# Guía de Deployment - LSTM Prediction System

## Estado Actual

Ya tienes un servicio desplegado en Google Cloud Run:
- **URL**: https://lstm-prediction-service-m42t7bzy3q-uc.a.run.app
- **Aplicación**: app_prediccion_lstm.py (versión antigua)
- **Estado**: Funcional pero con app antigua

## Problema: Modelos y Datos No Están en Git

Tu `.gitignore` excluye:
```
models/**/*.h5
models/**/*.keras
models/**/*.pkl
data/*.xlsx
data/*.csv
```

**Esto significa que NO se pueden desplegar directamente desde Git.**

---

## OPCIÓN 1: Railway (⭐ RECOMENDADA - MÁS FÁCIL)

### Ventajas:
- ✅ Deployment automático desde GitHub
- ✅ Configuración zero
- ✅ Tier gratuito generoso ($5/mes sin tarjeta)
- ✅ Logs en tiempo real
- ✅ HTTPS automático

### Pasos:

#### 1. Preparar Repositorio

Primero, necesitas subir los modelos a Google Cloud Storage o similar, y modificar `app_cross_analysis_web.py` para descargarlos al inicio.

**Crear script de descarga de modelos:**

```python
# Agregar al inicio de app_cross_analysis_web.py
import os
import urllib.request

def download_models_if_needed():
    """Descargar modelos desde Cloud Storage si no existen localmente"""
    models_to_download = [
        {
            'url': 'https://storage.googleapis.com/YOUR_BUCKET/customer_v3/model_best.keras',
            'path': 'models/temporal/customer_v3/medium/model_best.keras'
        },
        {
            'url': 'https://storage.googleapis.com/YOUR_BUCKET/customer_v3/scaler_X.pkl',
            'path': 'models/temporal/customer_v3/medium/scaler_X.pkl'
        },
        # ... más modelos
    ]

    for model in models_to_download:
        if not os.path.exists(model['path']):
            print(f"Descargando {model['path']}...")
            os.makedirs(os.path.dirname(model['path']), exist_ok=True)
            urllib.request.urlretrieve(model['url'], model['path'])
            print(f"✓ {model['path']} descargado")

# Llamar al inicio
download_models_if_needed()
```

#### 2. Subir Modelos a Google Cloud Storage

```bash
# Crear bucket
gsutil mb gs://lstm-prediction-models-pfg

# Subir modelos
gsutil -m cp -r models/temporal/customer_v3/ gs://lstm-prediction-models-pfg/customer_v3/
gsutil -m cp -r models/temporal/products_50epochs/ gs://lstm-prediction-models-pfg/products_50epochs/

# Subir datos
gsutil cp data/processed/online_retail_2.xlsx gs://lstm-prediction-models-pfg/data/online_retail_2.xlsx

# Hacer público (para que Railway pueda descargar)
gsutil iam ch allUsers:objectViewer gs://lstm-prediction-models-pfg
```

#### 3. Desplegar en Railway

1. Ve a https://railway.app
2. Crea cuenta con GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Selecciona tu repositorio `lstm-sales-prediction-pfg`
5. Railway detectará automáticamente Python y desplegará

**Variables de entorno en Railway:**
- `PORT`: 8080 (Railway lo configura automático)
- `PYTHON_VERSION`: 3.11

#### 4. Verificar Deployment

Railway te dará una URL como:
```
https://lstm-prediction-pfg.up.railway.app
```

Accede a `/api/status` para verificar.

---

## OPCIÓN 2: Google Cloud Run con Cloud Storage

### Ventajas:
- ✅ Escalabilidad profesional
- ✅ Integración nativa con GCP
- ✅ Control total

### Desventajas:
- ⚠️ Más complejo de configurar
- ⚠️ Requiere billing habilitado

### Pasos:

#### 1. Subir Modelos a Cloud Storage (mismo paso de arriba)

```bash
gsutil mb gs://lstm-prediction-models-pfg
gsutil -m cp -r models/temporal/customer_v3/ gs://lstm-prediction-models-pfg/customer_v3/
gsutil -m cp -r models/temporal/products_50epochs/ gs://lstm-prediction-models-pfg/products_50epochs/
gsutil cp data/processed/online_retail_2.xlsx gs://lstm-prediction-models-pfg/data/online_retail_2.xlsx
gsutil iam ch allUsers:objectViewer gs://lstm-prediction-models-pfg
```

#### 2. Modificar app_cross_analysis_web.py

Agregar al inicio:

```python
from google.cloud import storage
import os

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Descargar archivo desde Cloud Storage"""
    if os.path.exists(destination_file_name):
        print(f"✓ {destination_file_name} ya existe")
        return

    print(f"Descargando gs://{bucket_name}/{source_blob_name}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob.download_to_filename(destination_file_name)
    print(f"✓ Descargado a {destination_file_name}")

def setup_models():
    """Descargar todos los modelos necesarios"""
    bucket_name = 'lstm-prediction-models-pfg'

    models = [
        ('customer_v3/model_best.keras', 'models/temporal/customer_v3/medium/model_best.keras'),
        ('customer_v3/scaler_X.pkl', 'models/temporal/customer_v3/medium/scaler_X.pkl'),
        ('customer_v3/scaler_y_days.pkl', 'models/temporal/customer_v3/medium/scaler_y_days.pkl'),
        ('customer_v3/scaler_y_value.pkl', 'models/temporal/customer_v3/medium/scaler_y_value.pkl'),
        ('customer_v3/metrics.json', 'models/temporal/customer_v3/medium/metrics.json'),
        ('products_50epochs/model_best.keras', 'models/temporal/products_50epochs/short/model_best.keras'),
        ('products_50epochs/scaler_X.pkl', 'models/temporal/products_50epochs/short/scaler_X.pkl'),
        ('products_50epochs/scaler_y.pkl', 'models/temporal/products_50epochs/short/scaler_y.pkl'),
        ('products_50epochs/metrics.json', 'models/temporal/products_50epochs/short/metrics.json'),
        ('data/online_retail_2.xlsx', 'data/processed/online_retail_2.xlsx'),
    ]

    for source, dest in models:
        download_from_gcs(bucket_name, source, dest)

# Llamar al inicio (antes de crear la app)
setup_models()
```

#### 3. Agregar google-cloud-storage a requirements.txt

```bash
echo "google-cloud-storage>=2.10.0" >> requirements.txt
```

#### 4. Desplegar

```bash
cd "E:\Codigos\Proyecto Final"

gcloud run deploy lstm-prediction-service \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --timeout 300 \
  --memory 4Gi \
  --cpu 2
```

---

## OPCIÓN 3: Usar Git LFS para Modelos (Más simple pero con límites)

### Ventajas:
- ✅ Todo en un solo repositorio
- ✅ Deployment directo desde Git

### Desventajas:
- ⚠️ Límite de 1GB gratis en GitHub
- ⚠️ $5/mes por 50GB adicionales

### Pasos:

```bash
# 1. Instalar Git LFS
git lfs install

# 2. Trackear archivos grandes
git lfs track "models/**/*.keras"
git lfs track "models/**/*.pkl"
git lfs track "models/**/*.h5"
git lfs track "data/*.xlsx"

# 3. Agregar y commitear
git add .gitattributes
git add models/ data/
git commit -m "Agregar modelos con Git LFS"
git push origin master

# 4. Desplegar normalmente desde Git
```

---

## RECOMENDACIÓN FINAL

**Para desarrollo rápido y fácil:**
→ **Railway** (Opción 1)
- Menos pasos
- Deployment en 5 minutos
- Tier gratuito sin tarjeta

**Para producción profesional:**
→ **Google Cloud Run + Cloud Storage** (Opción 2)
- Escalabilidad ilimitada
- Mejor rendimiento
- Control total

---

## Servicio Actual

Tu servicio actual en Cloud Run:
- **URL**: https://lstm-prediction-service-m42t7bzy3q-uc.a.run.app
- **Aplicación**: app_prediccion_lstm.py (antigua)

Para actualizar a la nueva app (app_cross_analysis_web.py), sigue Opción 2.

---

## Checklist

### Antes de desplegar:

- [ ] Modelos subidos a Cloud Storage o públicos
- [ ] app_cross_analysis_web.py tiene función para descargar modelos
- [ ] requirements.txt actualizado con todas las dependencias
- [ ] Dockerfile correcto para la nueva app
- [ ] Puerto configurable (os.environ.get('PORT', 5001))
- [ ] Git commit y push al repositorio

### Después de desplegar:

- [ ] Verificar `/api/status` responde
- [ ] Verificar `/api/customers/ranking` funciona
- [ ] Verificar dashboard carga en `/`
- [ ] Verificar logs no tienen errores
- [ ] Probar refresh de análisis con `/api/refresh`

---

## Troubleshooting

### Error: "Build failed"
→ Verificar que Dockerfile tenga todos los archivos necesarios
→ Verificar que .dockerignore no excluya archivos críticos

### Error: "FileNotFoundError: models/..."
→ Modelos no están en la imagen
→ Implementar descarga de Cloud Storage (Opción 2)

### Error: "Memory limit exceeded"
→ Aumentar memoria: `--memory 4Gi`
→ TensorFlow consume mucha RAM al cargar modelos

### Error: "Connection timeout"
→ Aumentar timeout: `--timeout 300`
→ Carga de modelos tarda varios segundos

---

## URLs Útiles

- **Consola Google Cloud**: https://console.cloud.google.com
- **Cloud Run Dashboard**: https://console.cloud.google.com/run
- **Cloud Storage**: https://console.cloud.google.com/storage
- **Railway**: https://railway.app
- **Repositorio GitHub**: (tu repositorio)

---

**Última actualización:** 2025-11-05
