@echo off
REM ============================================================================
REM SCRIPT DE DEPLOYMENT A GOOGLE CLOUD RUN (Windows)
REM ============================================================================
REM Version para Windows del script de deployment

setlocal enabledelayedexpansion

echo ========================================================================
echo   DEPLOYMENT A GOOGLE CLOUD RUN - LSTM PREDICTION SERVICE
echo ========================================================================
echo.

REM ============================================================================
REM CONFIGURACION
REM ============================================================================

set PROJECT_ID=lstm-sales-prediction-pfg
set SERVICE_NAME=lstm-prediction-service
set REGION=us-central1
set IMAGE_NAME=gcr.io/%PROJECT_ID%/%SERVICE_NAME%
set MEMORY=2Gi
set CPU=2
set MAX_INSTANCES=10
set MIN_INSTANCES=0
set TIMEOUT=300s

echo Configuracion:
echo   Project ID: %PROJECT_ID%
echo   Service: %SERVICE_NAME%
echo   Region: %REGION%
echo   Image: %IMAGE_NAME%
echo.

REM ============================================================================
REM VALIDACIONES
REM ============================================================================

echo [1/6] Validando configuracion...

REM Verificar gcloud
where gcloud >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: gcloud CLI no esta instalado
    echo Instala desde: https://cloud.google.com/sdk/docs/install
    exit /b 1
)

REM Configurar proyecto
gcloud config set project %PROJECT_ID%

echo Validacion OK
echo.

REM ============================================================================
REM BUILD DE IMAGEN DOCKER
REM ============================================================================

echo [2/6] Construyendo imagen Docker...

REM Ir al directorio raiz
cd ..\..\..

REM Build con Cloud Build
gcloud builds submit --tag %IMAGE_NAME% --timeout=20m --machine-type=e2-highcpu-8

if %ERRORLEVEL% neq 0 (
    echo ERROR: Build fallo
    exit /b 1
)

echo Imagen construida exitosamente
echo.

REM ============================================================================
REM DEPLOY A CLOUD RUN
REM ============================================================================

echo [3/6] Desplegando a Cloud Run...

gcloud run deploy %SERVICE_NAME% ^
    --image %IMAGE_NAME% ^
    --platform managed ^
    --region %REGION% ^
    --memory %MEMORY% ^
    --cpu %CPU% ^
    --timeout %TIMEOUT% ^
    --max-instances %MAX_INSTANCES% ^
    --min-instances %MIN_INSTANCES% ^
    --allow-unauthenticated ^
    --port 8080 ^
    --set-env-vars="ENVIRONMENT=production,TF_CPP_MIN_LOG_LEVEL=2"

if %ERRORLEVEL% neq 0 (
    echo ERROR: Deploy fallo
    exit /b 1
)

echo Deploy completado
echo.

REM ============================================================================
REM OBTENER URL
REM ============================================================================

echo [4/6] Obteniendo URL del servicio...

for /f "delims=" %%i in ('gcloud run services describe %SERVICE_NAME% --platform managed --region %REGION% --format "value(status.url)"') do set SERVICE_URL=%%i

echo URL del servicio: %SERVICE_URL%
echo.

REM ============================================================================
REM HEALTH CHECK
REM ============================================================================

echo [5/6] Verificando health del servicio...
timeout /t 5 /nobreak >nul

curl -s "%SERVICE_URL%/api/health"
echo.
echo.

REM ============================================================================
REM TEST BASICO
REM ============================================================================

echo [6/6] Ejecutando test basico...

curl -s "%SERVICE_URL%/api/products"
echo.

curl -s -X POST "%SERVICE_URL%/api/predict" -H "Content-Type: application/json" -d "{\"product_code\":\"20723\"}"
echo.
echo.

REM ============================================================================
REM RESUMEN
REM ============================================================================

echo ========================================================================
echo   DEPLOYMENT COMPLETADO
echo ========================================================================
echo.
echo URL del servicio:
echo   %SERVICE_URL%
echo.
echo Endpoints:
echo   GET  %SERVICE_URL%/api/health
echo   GET  %SERVICE_URL%/api/products
echo   POST %SERVICE_URL%/api/predict
echo.
echo Interfaz web:
echo   %SERVICE_URL%/
echo.

endlocal
