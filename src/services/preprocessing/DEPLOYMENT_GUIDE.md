# 🚀 Guía de Deployment a Google Cloud Run

Guía completa paso a paso para desplegar el sistema LSTM Prediction Service en Google Cloud Run.

---

## 📋 Requisitos Previos

### 1. Cuenta de Google Cloud Platform (GCP)
- ✅ Cuenta activa de GCP
- ✅ Proyecto creado
- ✅ Billing habilitado
- ✅ APIs necesarias habilitadas:
  - Cloud Run API
  - Cloud Build API
  - Container Registry API

### 2. Herramientas Instaladas
```bash
# Google Cloud SDK
# Descargar de: https://cloud.google.com/sdk/docs/install

# Verificar instalación
gcloud --version

# Docker (opcional, para test local)
docker --version
```

### 3. Autenticación
```bash
# Login en GCP
gcloud auth login

# Configurar proyecto por defecto
gcloud config set project TU-PROJECT-ID

# Verificar configuración
gcloud config list
```

---

## 🔧 Configuración Inicial

### Paso 1: Habilitar APIs Necesarias

```bash
# Habilitar Cloud Run API
gcloud services enable run.googleapis.com

# Habilitar Cloud Build API
gcloud services enable cloudbuild.googleapis.com

# Habilitar Container Registry API
gcloud services enable containerregistry.googleapis.com
```

### Paso 2: Configurar Variables

Edita los scripts de deployment (`deploy.sh` o `deploy.bat`):

```bash
PROJECT_ID="tu-proyecto-gcp"           # ⚠️ CAMBIAR
SERVICE_NAME="lstm-prediction-service"
REGION="us-central1"                   # us-central1, europe-west1, asia-southeast1
```

**Regiones recomendadas:**
- `us-central1` (Iowa) - Más barato, buena latencia USA
- `us-east1` (Carolina del Sur) - Económico
- `europe-west1` (Bélgica) - Europa
- `asia-southeast1` (Singapur) - Asia

---

## 🚀 Deployment Automático

### Opción 1: Script Automático (Linux/Mac)

```bash
# Ir al directorio de deployment
cd src/services/preprocessing

# Dar permisos de ejecución
chmod +x deploy.sh

# Ejecutar deployment
./deploy.sh
```

### Opción 2: Script Automático (Windows)

```cmd
REM Ir al directorio
cd src\services\preprocessing

REM Ejecutar deployment
deploy.bat
```

El script hará automáticamente:
1. ✅ Validar configuración
2. ✅ Build de imagen Docker con Cloud Build
3. ✅ Deploy a Cloud Run
4. ✅ Configurar auto-scaling
5. ✅ Health check
6. ✅ Test de funcionamiento

---

## 🛠️ Deployment Manual (Paso a Paso)

### Paso 1: Build de Imagen Docker

```bash
# Ir al directorio raíz del proyecto
cd /path/to/Proyecto Final

# Build con Cloud Build (recomendado)
gcloud builds submit \
    --tag gcr.io/TU-PROJECT-ID/lstm-prediction-service \
    --timeout=20m

# O build local (si tienes Docker instalado)
docker build -t gcr.io/TU-PROJECT-ID/lstm-prediction-service .
docker push gcr.io/TU-PROJECT-ID/lstm-prediction-service
```

### Paso 2: Deploy a Cloud Run

```bash
gcloud run deploy lstm-prediction-service \
    --image gcr.io/TU-PROJECT-ID/lstm-prediction-service \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300s \
    --max-instances 10 \
    --min-instances 0 \
    --allow-unauthenticated \
    --port 8080 \
    --set-env-vars="ENVIRONMENT=production,TF_CPP_MIN_LOG_LEVEL=2"
```

### Paso 3: Obtener URL del Servicio

```bash
gcloud run services describe lstm-prediction-service \
    --platform managed \
    --region us-central1 \
    --format 'value(status.url)'
```

---

## ✅ Verificación y Testing

### 1. Health Check

```bash
curl https://TU-SERVICIO.run.app/api/health
```

**Respuesta esperada:**
```json
{
  "success": true,
  "status": "running",
  "mode": "LSTM Deep Learning",
  "models_loaded": 0,
  "models_available": 20,
  "tensorflow_version": "2.15.0"
}
```

### 2. Listar Productos

```bash
curl https://TU-SERVICIO.run.app/api/products
```

### 3. Test de Predicción

```bash
curl -X POST https://TU-SERVICIO.run.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"product_code":"20723"}'
```

**Respuesta esperada:**
```json
{
  "success": true,
  "product_code": "20723",
  "prediction": {
    "day_1": 38.7,
    "day_2": 34.9,
    "day_3": 32.0,
    "total_3_days": 105.6
  }
}
```

### 4. Interfaz Web

Abre en navegador:
```
https://TU-SERVICIO.run.app/
```

---

## 📊 Monitoreo y Logs

### Ver Logs en Tiempo Real

```bash
gcloud run services logs read lstm-prediction-service \
    --region us-central1 \
    --limit 50 \
    --follow
```

### Ver Métricas

```bash
# Descripción del servicio
gcloud run services describe lstm-prediction-service \
    --region us-central1

# Ir a la consola web
# https://console.cloud.google.com/run
```

### Métricas en GCP Console

1. Ir a: https://console.cloud.google.com/run
2. Seleccionar tu servicio
3. Ver tabs:
   - **Métricas**: Requests, latencia, CPU, memoria
   - **Logs**: Logs detallados
   - **Revisiones**: Historial de deployments

---

## 🔄 Actualizaciones

### Re-deployar Nueva Versión

```bash
# Opción 1: Ejecutar script de nuevo
./deploy.sh

# Opción 2: Deploy manual
gcloud builds submit --tag gcr.io/TU-PROJECT-ID/lstm-prediction-service
gcloud run deploy lstm-prediction-service \
    --image gcr.io/TU-PROJECT-ID/lstm-prediction-service \
    --region us-central1
```

### Rollback a Versión Anterior

```bash
# Listar revisiones
gcloud run revisions list --service lstm-prediction-service --region us-central1

# Hacer rollback
gcloud run services update-traffic lstm-prediction-service \
    --to-revisions REVISION-NAME=100 \
    --region us-central1
```

---

## ⚙️ Configuración Avanzada

### Auto-scaling

```bash
# Configurar auto-scaling
gcloud run services update lstm-prediction-service \
    --min-instances 1 \
    --max-instances 20 \
    --region us-central1
```

### Variables de Entorno

```bash
gcloud run services update lstm-prediction-service \
    --set-env-vars="VAR1=value1,VAR2=value2" \
    --region us-central1
```

### Dominio Personalizado

```bash
# Mapear dominio
gcloud run domain-mappings create \
    --service lstm-prediction-service \
    --domain tu-dominio.com \
    --region us-central1
```

### Autenticación (Opcional)

```bash
# Requiere autenticación
gcloud run services update lstm-prediction-service \
    --no-allow-unauthenticated \
    --region us-central1
```

---

## 💰 Costos Estimados

### Pricing de Cloud Run

**Tier Gratuito (mensual):**
- 2 millones requests
- 360,000 GB-segundos
- 180,000 vCPU-segundos

**Después del tier gratuito:**
- Requests: $0.40 por millón
- CPU: $0.00002400 por vCPU-segundo
- Memoria: $0.00000250 por GB-segundo

**Ejemplo (configuración 2CPU/2GB):**
- 100,000 requests/mes
- 2 segundos promedio por request
- **Costo estimado:** ~$5-10/mes

### Optimización de Costos

1. **Min instances = 0**: Solo paga cuando hay requests
2. **Timeout bajo**: 60-120s para APIs simples
3. **Región económica**: us-central1
4. **Cold starts**: Acepta ~2s de latencia inicial

---

## 🧹 Limpieza y Eliminación

### Eliminar Servicio

```bash
gcloud run services delete lstm-prediction-service \
    --region us-central1 \
    --quiet
```

### Eliminar Imágenes

```bash
# Listar imágenes
gcloud container images list --repository=gcr.io/TU-PROJECT-ID

# Eliminar imagen específica
gcloud container images delete gcr.io/TU-PROJECT-ID/lstm-prediction-service
```

---

## 🐛 Troubleshooting

### Error: "Permission denied"

```bash
# Dar permisos necesarios
gcloud projects add-iam-policy-binding TU-PROJECT-ID \
    --member="user:tu-email@gmail.com" \
    --role="roles/run.admin"
```

### Error: "Out of memory"

```bash
# Aumentar memoria
gcloud run services update lstm-prediction-service \
    --memory 4Gi \
    --region us-central1
```

### Error: "Timeout"

```bash
# Aumentar timeout
gcloud run services update lstm-prediction-service \
    --timeout 600s \
    --region us-central1
```

### Ver Logs de Error

```bash
# Logs detallados
gcloud run services logs read lstm-prediction-service \
    --region us-central1 \
    --limit 100
```

---

## 📚 Recursos Adicionales

- **Documentación oficial**: https://cloud.google.com/run/docs
- **Pricing calculator**: https://cloud.google.com/products/calculator
- **Best practices**: https://cloud.google.com/run/docs/best-practices
- **Samples**: https://github.com/GoogleCloudPlatform/cloud-run-samples

---

## 🎯 Checklist de Deployment

- [ ] Cuenta GCP activa con billing
- [ ] Google Cloud SDK instalado
- [ ] Autenticado con `gcloud auth login`
- [ ] Proyecto configurado
- [ ] APIs habilitadas (Run, Build, Container Registry)
- [ ] Variables configuradas en script
- [ ] Modelos entrenados en `models/trained/`
- [ ] Dataset en `data/processed/`
- [ ] Script ejecutado sin errores
- [ ] Health check exitoso
- [ ] Test de predicción OK
- [ ] Interfaz web accesible

---

## 📞 Soporte

Si encuentras problemas:

1. Revisa logs: `gcloud run services logs read ...`
2. Verifica configuración: `gcloud config list`
3. Consulta documentación oficial
4. Revisa issues en GitHub del proyecto
