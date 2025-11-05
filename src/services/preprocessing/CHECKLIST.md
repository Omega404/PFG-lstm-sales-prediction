# ✅ Checklist de Deployment a Google Cloud Run

Usa este checklist para asegurar un deployment exitoso.

## 📋 Pre-Deployment

### Cuenta y Proyecto
- [ ] Cuenta de Google Cloud activa
- [ ] Proyecto GCP creado
- [ ] Billing habilitado en el proyecto
- [ ] Nombre del proyecto anotado: `________________`

### Herramientas
- [ ] Google Cloud SDK instalado
- [ ] Autenticado con `gcloud auth login`
- [ ] Proyecto configurado con `gcloud config set project`
- [ ] Verificado con `gcloud config list`

### APIs Habilitadas
```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
```
- [ ] Cloud Run API
- [ ] Cloud Build API
- [ ] Container Registry API

### Configuración Local
- [ ] Modelos LSTM entrenados en `models/trained/`
- [ ] Dataset en `data/processed/product_demand.xlsx`
- [ ] Variable `PROJECT_ID` actualizada en `deploy.sh` o `deploy.bat`
- [ ] Variable `REGION` configurada (default: `us-central1`)
- [ ] Variable `SERVICE_NAME` definida (default: `lstm-prediction-service`)

## 🚀 Durante el Deployment

### Build
- [ ] Script de deployment ejecutado
- [ ] Build iniciado sin errores
- [ ] Imagen Docker creada exitosamente
- [ ] Imagen subida a Container Registry

### Deploy
- [ ] Deploy a Cloud Run iniciado
- [ ] Servicio creado/actualizado
- [ ] URL del servicio obtenida
- [ ] URL anotada: `https://_____________________.run.app`

### Configuración
- [ ] Memoria: 2Gi
- [ ] CPU: 2 vCPUs
- [ ] Timeout: 300s
- [ ] Max instances: 10
- [ ] Min instances: 0
- [ ] Puerto: 8080
- [ ] Acceso público: Habilitado

## ✅ Post-Deployment

### Verificación Básica
- [ ] Health check responde correctamente
  ```bash
  curl https://TU-SERVICIO.run.app/api/health
  ```
- [ ] Endpoint de productos funciona
  ```bash
  curl https://TU-SERVICIO.run.app/api/products
  ```
- [ ] Predicción funciona
  ```bash
  curl -X POST https://TU-SERVICIO.run.app/api/predict \
    -H "Content-Type: application/json" \
    -d '{"product_code":"20723"}'
  ```

### Interfaz Web
- [ ] URL principal abre en navegador
- [ ] Página carga correctamente
- [ ] Botón "Ver Productos" funciona
- [ ] Lista de productos se muestra
- [ ] Predicción desde interfaz funciona
- [ ] Gráfico se visualiza correctamente

### Monitoreo
- [ ] Logs visibles en consola GCP
- [ ] Métricas disponibles en dashboard
- [ ] No hay errores en logs recientes

## 📊 Testing Completo

### Endpoints
- [ ] `GET /` - Interfaz web
- [ ] `GET /api/health` - Health check
- [ ] `GET /api/products` - Lista de productos
- [ ] `POST /api/predict` - Predicción LSTM

### Productos de Test
Probar predicciones con:
- [ ] Producto 20723
- [ ] Producto 20727
- [ ] Producto 22112
- [ ] Producto personalizado: `______`

### Performance
- [ ] Tiempo de respuesta < 5 segundos
- [ ] Predicciones consistentes
- [ ] Sin errores 500 en logs
- [ ] Cold start aceptable (< 10s primera request)

## 📈 Optimización

### Opcional - Mejoras
- [ ] Dominio personalizado configurado
- [ ] Min instances > 0 (si necesitas baja latencia)
- [ ] Logging estructurado configurado
- [ ] Alertas configuradas en GCP
- [ ] Budget alert configurado

### Monitoreo Continuo
- [ ] Dashboard de métricas revisado
- [ ] Logs sin errores frecuentes
- [ ] Costos dentro de presupuesto
- [ ] Auto-scaling funcionando correctamente

## 🔒 Seguridad (Opcional)

- [ ] Autenticación configurada (si es privado)
- [ ] Variables sensibles en Secret Manager
- [ ] IAM roles configurados correctamente
- [ ] VPC connector (si aplica)

## 📝 Documentación

- [ ] URL de producción documentada
- [ ] Credenciales de acceso guardadas (si aplica)
- [ ] Proceso de actualización documentado
- [ ] Contactos de soporte definidos

## 🎉 Deployment Completo

Si todo está marcado, ¡deployment exitoso! 🚀

**URL de Producción:** `_______________________________`

**Fecha de Deployment:** `_______________`

**Versión:** `_______________`

---

## 🐛 Si algo falló

1. **Revisa logs:**
   ```bash
   gcloud run services logs read lstm-prediction-service --region us-central1 --limit 100
   ```

2. **Verifica configuración:**
   ```bash
   gcloud run services describe lstm-prediction-service --region us-central1
   ```

3. **Consulta la guía:** [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)

4. **Rollback si es necesario:**
   ```bash
   gcloud run revisions list --service lstm-prediction-service --region us-central1
   ```

---

## 📞 Contacto y Soporte

- **Documentación:** [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- **Issues:** GitHub del proyecto
- **GCP Support:** https://cloud.google.com/support
