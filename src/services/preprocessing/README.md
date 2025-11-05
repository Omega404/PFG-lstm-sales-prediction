# 🚀 Google Cloud Run Deployment

Archivos necesarios para desplegar el sistema LSTM Prediction Service en Google Cloud Run.

## 📁 Archivos Incluidos

| Archivo | Descripción |
|---------|-------------|
| `Dockerfile` | Imagen Docker optimizada para Cloud Run |
| `requirements.txt` | Dependencias de Python |
| `.dockerignore` | Archivos a excluir del build |
| `deploy.sh` | Script de deployment automático (Linux/Mac) |
| `deploy.bat` | Script de deployment automático (Windows) |
| `cloudbuild.yaml` | Configuración de Cloud Build |
| `DEPLOYMENT_GUIDE.md` | Guía completa paso a paso |

## ⚡ Quick Start

### 1. Configurar Proyecto

Edita `deploy.sh` o `deploy.bat`:
```bash
PROJECT_ID="tu-proyecto-gcp"  # ⚠️ CAMBIAR ESTO
```

### 2. Habilitar APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 3. Ejecutar Deployment

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

**Windows:**
```cmd
deploy.bat
```

### 4. Obtener URL

El script mostrará la URL al finalizar:
```
https://lstm-prediction-service-xxxxx.run.app
```

## 📊 Características del Deployment

- ✅ **Auto-scaling**: 0-10 instancias
- ✅ **Memoria**: 2GB RAM
- ✅ **CPU**: 2 vCPUs
- ✅ **Timeout**: 300 segundos
- ✅ **Acceso público**: Sin autenticación
- ✅ **Health checks**: Automáticos
- ✅ **SSL/HTTPS**: Incluido gratis

## 💰 Costos Estimados

**Tier Gratuito:**
- 2M requests/mes
- 360,000 GB-segundos

**Después del tier gratuito:**
- ~$5-10/mes con tráfico moderado

## 🔧 Comandos Útiles

```bash
# Ver logs
gcloud run services logs read lstm-prediction-service --region us-central1

# Ver estado
gcloud run services describe lstm-prediction-service --region us-central1

# Actualizar
./deploy.sh

# Eliminar
gcloud run services delete lstm-prediction-service --region us-central1
```

## 📚 Documentación Completa

Lee [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) para:
- Configuración detallada
- Troubleshooting
- Configuración avanzada
- Monitoreo y logs
- Best practices

## 🎯 Testing Rápido

```bash
# Health check
curl https://TU-SERVICIO.run.app/api/health

# Listar productos
curl https://TU-SERVICIO.run.app/api/products

# Predicción
curl -X POST https://TU-SERVICIO.run.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"product_code":"20723"}'
```

## 🐛 Problemas Comunes

**"Permission denied"**
```bash
gcloud auth login
gcloud config set project TU-PROJECT-ID
```

**"Out of memory"**
```bash
# Editar deploy.sh y cambiar:
MEMORY="4Gi"
```

**"Build timeout"**
```bash
# Usar Cloud Build en lugar de build local
```

## 📞 Soporte

- 📖 [Guía Completa](./DEPLOYMENT_GUIDE.md)
- 🌐 [Cloud Run Docs](https://cloud.google.com/run/docs)
- 💬 Issues en GitHub del proyecto

---

**¿Primera vez con Cloud Run?** → Lee [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
