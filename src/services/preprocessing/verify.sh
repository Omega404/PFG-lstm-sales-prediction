#!/bin/bash

# ============================================================================
# SCRIPT DE VERIFICACION PRE-DEPLOYMENT
# ============================================================================
# Verifica que todo este listo antes de hacer deployment

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

echo -e "${BLUE}"
echo "========================================================================"
echo "  VERIFICACION PRE-DEPLOYMENT"
echo "========================================================================"
echo -e "${NC}\n"

# ============================================================================
# VERIFICACIONES
# ============================================================================

echo -e "${BLUE}[1] Verificando herramientas instaladas...${NC}"

# gcloud
if command -v gcloud &> /dev/null; then
    GCLOUD_VERSION=$(gcloud --version | head -n 1)
    echo -e "${GREEN}✓${NC} gcloud instalado: $GCLOUD_VERSION"
else
    echo -e "${RED}✗${NC} gcloud NO instalado"
    ((ERRORS++))
fi

# python
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    echo -e "${GREEN}✓${NC} Python instalado: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python NO instalado"
    ((ERRORS++))
fi

echo ""

# ============================================================================

echo -e "${BLUE}[2] Verificando autenticación GCP...${NC}"

if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
    echo -e "${GREEN}✓${NC} Autenticado como: $ACCOUNT"
else
    echo -e "${RED}✗${NC} No autenticado en gcloud"
    echo "   Ejecuta: gcloud auth login"
    ((ERRORS++))
fi

PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -n "$PROJECT" ]; then
    echo -e "${GREEN}✓${NC} Proyecto configurado: $PROJECT"
else
    echo -e "${YELLOW}!${NC} No hay proyecto configurado"
    echo "   Ejecuta: gcloud config set project TU-PROJECT-ID"
    ((WARNINGS++))
fi

echo ""

# ============================================================================

echo -e "${BLUE}[3] Verificando archivos necesarios...${NC}"

# Ir al directorio raiz
cd ../../..

FILES_TO_CHECK=(
    "app_prediccion_lstm.py"
    "prediccion_lstm.html"
    "data/processed/product_demand.xlsx"
)

for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file NO encontrado"
        ((ERRORS++))
    fi
done

echo ""

# ============================================================================

echo -e "${BLUE}[4] Verificando modelos entrenados...${NC}"

MODEL_COUNT=$(find models/trained -name "lstm_*.h5" 2>/dev/null | wc -l)
SCALER_COUNT=$(find models/trained -name "scaler_*.pkl" 2>/dev/null | wc -l)

if [ "$MODEL_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Modelos LSTM encontrados: $MODEL_COUNT"
else
    echo -e "${RED}✗${NC} No hay modelos LSTM entrenados"
    ((ERRORS++))
fi

if [ "$SCALER_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Scalers encontrados: $SCALER_COUNT"
else
    echo -e "${RED}✗${NC} No hay scalers"
    ((ERRORS++))
fi

if [ "$MODEL_COUNT" -ne "$SCALER_COUNT" ]; then
    echo -e "${YELLOW}!${NC} Advertencia: Numero de modelos != numero de scalers"
    ((WARNINGS++))
fi

echo ""

# ============================================================================

echo -e "${BLUE}[5] Verificando APIs de GCP...${NC}"

if [ -n "$PROJECT" ]; then
    APIS=(
        "run.googleapis.com"
        "cloudbuild.googleapis.com"
        "containerregistry.googleapis.com"
    )

    for api in "${APIS[@]}"; do
        if gcloud services list --enabled --filter="name:$api" --format="value(name)" 2>/dev/null | grep -q "$api"; then
            echo -e "${GREEN}✓${NC} $api habilitada"
        else
            echo -e "${YELLOW}!${NC} $api NO habilitada"
            echo "   Habilita con: gcloud services enable $api"
            ((WARNINGS++))
        fi
    done
else
    echo -e "${YELLOW}!${NC} No se puede verificar APIs (no hay proyecto configurado)"
fi

echo ""

# ============================================================================

echo -e "${BLUE}[6] Verificando configuracion de deployment...${NC}"

cd src/services/preprocessing

DEPLOY_SCRIPT="deploy.sh"
if [ -f "$DEPLOY_SCRIPT" ]; then
    if grep -q 'PROJECT_ID="tu-proyecto-gcp"' "$DEPLOY_SCRIPT"; then
        echo -e "${RED}✗${NC} PROJECT_ID no esta configurado en $DEPLOY_SCRIPT"
        echo "   Edita el archivo y cambia PROJECT_ID"
        ((ERRORS++))
    else
        CONFIGURED_PROJECT=$(grep 'PROJECT_ID=' "$DEPLOY_SCRIPT" | head -n 1 | cut -d'"' -f2)
        echo -e "${GREEN}✓${NC} PROJECT_ID configurado: $CONFIGURED_PROJECT"
    fi
else
    echo -e "${YELLOW}!${NC} $DEPLOY_SCRIPT no encontrado"
fi

echo ""

# ============================================================================

echo -e "${BLUE}[7] Verificando tamaño de archivos...${NC}"

cd ../../..

DATASET_SIZE=$(du -sh data/processed/product_demand.xlsx 2>/dev/null | cut -f1)
MODELS_SIZE=$(du -sh models/trained 2>/dev/null | cut -f1)

if [ -n "$DATASET_SIZE" ]; then
    echo -e "${GREEN}✓${NC} Dataset: $DATASET_SIZE"
fi

if [ -n "$MODELS_SIZE" ]; then
    echo -e "${GREEN}✓${NC} Modelos: $MODELS_SIZE"

    # Advertir si es muy grande
    MODELS_SIZE_MB=$(du -sm models/trained 2>/dev/null | cut -f1)
    if [ "$MODELS_SIZE_MB" -gt 500 ]; then
        echo -e "${YELLOW}!${NC} Advertencia: Modelos muy grandes (>500MB)"
        echo "   Considera optimizar o usar Cloud Storage"
        ((WARNINGS++))
    fi
fi

echo ""

# ============================================================================
# RESUMEN
# ============================================================================

echo -e "${BLUE}"
echo "========================================================================"
echo "  RESUMEN DE VERIFICACION"
echo "========================================================================"
echo -e "${NC}"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ TODO LISTO PARA DEPLOYMENT${NC}\n"
    echo "Ejecuta el deployment con:"
    echo "  cd src/services/preprocessing"
    echo "  ./deploy.sh"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}✓ LISTO CON ADVERTENCIAS${NC}"
    echo -e "  Advertencias: $WARNINGS\n"
    echo "Puedes proceder con el deployment, pero revisa las advertencias."
    exit 0
else
    echo -e "${RED}✗ HAY ERRORES QUE CORREGIR${NC}"
    echo -e "  Errores: $ERRORS"
    echo -e "  Advertencias: $WARNINGS\n"
    echo "Corrige los errores antes de hacer deployment."
    exit 1
fi
