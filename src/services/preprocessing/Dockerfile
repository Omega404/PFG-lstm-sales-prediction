# ============================================================================
# DOCKERFILE PARA GOOGLE CLOUD RUN - LSTM PREDICTION SERVICE
# ============================================================================
# Imagen optimizada para deployment de modelos LSTM en produccion
# Compatible con Cloud Run y arquitectura serverless

# Usar imagen base oficial de Python
FROM python:3.11-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

# Crear directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (mejor cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicacion
COPY app_prediccion_lstm.py .
COPY prediccion_lstm.html .

# Copiar datos y modelos
COPY data/ ./data/
COPY models/trained/ ./models/trained/

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Exponer puerto (Cloud Run usa variable $PORT)
EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')"

# Comando de inicio
CMD exec python app_prediccion_lstm.py
