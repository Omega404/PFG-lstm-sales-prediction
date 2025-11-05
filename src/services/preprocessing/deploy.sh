#!/bin/bash

# ============================================================================
# SCRIPT DE DEPLOYMENT A GOOGLE CLOUD RUN
# ============================================================================
# Automatiza el proceso completo de build y deploy

set -e  # Exit on error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "========================================================================"
echo "  DEPLOYMENT A GOOGLE CLOUD RUN - LSTM PREDICTION SERVICE"
echo "========================================================================"
echo -e "${NC}"

# ============================================================================
# CONFIGURACION
# ============================================================================

# Variables (PERSONALIZA ESTOS VALORES)
PROJECT_ID="tu-proyecto-gcp"
SERVICE_NAME="lstm-prediction-service"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
MEMORY="2Gi"
CPU="2"
MAX_INSTANCES="10"
MIN_INSTANCES="0"
TIMEOUT="300s"

echo -e "${YELLOW}Configuracion:${NC}"
echo "  Project ID: ${PROJECT_ID}"
echo "  Service: ${SERVICE_NAME}"
echo "  Region: ${REGION}"
echo "  Image: ${IMAGE_NAME}"
echo ""

# ============================================================================
# VALIDACIONES
# ============================================================================

echo -e "${BLUE}[1/6] Validando configuracion...${NC}"

# Verificar que gcloud este instalado
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}ERROR: gcloud CLI no esta instalado${NC}"
    echo "Instala desde: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Verificar autenticacion
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo -e "${RED}ERROR: No estas autenticado en gcloud${NC}"
    echo "Ejecuta: gcloud auth login"
    exit 1
fi

# Verificar proyecto
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
    echo -e "${YELLOW}Cambiando a proyecto: ${PROJECT_ID}${NC}"
    gcloud config set project ${PROJECT_ID}
fi

echo -e "${GREEN}Validacion OK${NC}\n"

# ============================================================================
# BUILD DE IMAGEN DOCKER
# ============================================================================

echo -e "${BLUE}[2/6] Construyendo imagen Docker...${NC}"

# Ir al directorio raiz del proyecto
cd ../../..

# Build con Cloud Build (mas rapido y optimizado)
gcloud builds submit \
    --tag ${IMAGE_NAME} \
    --timeout=20m \
    --machine-type=e2-highcpu-8 \
    .

echo -e "${GREEN}Imagen construida exitosamente${NC}\n"

# ============================================================================
# DEPLOY A CLOUD RUN
# ============================================================================

echo -e "${BLUE}[3/6] Desplegando a Cloud Run...${NC}"

gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --memory ${MEMORY} \
    --cpu ${CPU} \
    --timeout ${TIMEOUT} \
    --max-instances ${MAX_INSTANCES} \
    --min-instances ${MIN_INSTANCES} \
    --allow-unauthenticated \
    --port 8080 \
    --set-env-vars="ENVIRONMENT=production,TF_CPP_MIN_LOG_LEVEL=2"

echo -e "${GREEN}Deploy completado${NC}\n"

# ============================================================================
# OBTENER URL DEL SERVICIO
# ============================================================================

echo -e "${BLUE}[4/6] Obteniendo URL del servicio...${NC}"

SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)')

echo -e "${GREEN}URL del servicio: ${SERVICE_URL}${NC}\n"

# ============================================================================
# HEALTH CHECK
# ============================================================================

echo -e "${BLUE}[5/6] Verificando health del servicio...${NC}"

sleep 5  # Esperar a que el servicio este listo

HEALTH_RESPONSE=$(curl -s "${SERVICE_URL}/api/health")

if echo "$HEALTH_RESPONSE" | grep -q '"success": true'; then
    echo -e "${GREEN}Health check OK${NC}"
    echo "Response: ${HEALTH_RESPONSE}"
else
    echo -e "${RED}Health check FAILED${NC}"
    echo "Response: ${HEALTH_RESPONSE}"
fi

echo ""

# ============================================================================
# TEST BASICO
# ============================================================================

echo -e "${BLUE}[6/6] Ejecutando test basico...${NC}"

# Test de lista de productos
PRODUCTS_RESPONSE=$(curl -s "${SERVICE_URL}/api/products")
PRODUCTS_COUNT=$(echo "$PRODUCTS_RESPONSE" | grep -o '"count": [0-9]*' | grep -o '[0-9]*')

echo "Productos disponibles: ${PRODUCTS_COUNT}"

# Test de prediccion
echo "Probando prediccion para producto 20723..."
PREDICTION_RESPONSE=$(curl -s -X POST "${SERVICE_URL}/api/predict" \
    -H "Content-Type: application/json" \
    -d '{"product_code":"20723"}')

if echo "$PREDICTION_RESPONSE" | grep -q '"success": true'; then
    echo -e "${GREEN}Prediccion OK${NC}"
else
    echo -e "${RED}Prediccion FAILED${NC}"
fi

echo ""

# ============================================================================
# RESUMEN FINAL
# ============================================================================

echo -e "${GREEN}"
echo "========================================================================"
echo "  DEPLOYMENT COMPLETADO EXITOSAMENTE"
echo "========================================================================"
echo -e "${NC}"
echo "URL del servicio:"
echo -e "${BLUE}${SERVICE_URL}${NC}"
echo ""
echo "Endpoints disponibles:"
echo "  GET  ${SERVICE_URL}/api/health"
echo "  GET  ${SERVICE_URL}/api/products"
echo "  POST ${SERVICE_URL}/api/predict"
echo ""
echo "Interfaz web:"
echo "  ${SERVICE_URL}/"
echo ""
echo "Comandos utiles:"
echo "  Ver logs:     gcloud run services logs read ${SERVICE_NAME} --region ${REGION}"
echo "  Ver metricas: gcloud run services describe ${SERVICE_NAME} --region ${REGION}"
echo "  Eliminar:     gcloud run services delete ${SERVICE_NAME} --region ${REGION}"
echo ""
