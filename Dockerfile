# ============================================================================
# DOCKERFILE PARA GOOGLE CLOUD RUN - LSTM PREDICTION SERVICE
# ============================================================================

FROM python:3.11-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicacion
COPY app_prediccion_lstm.py .
COPY prediccion_lstm.html .

# Crear directorios necesarios
RUN mkdir -p data/processed models/trained

# Copiar datos y modelos explícitamente
COPY data/processed/product_demand.xlsx ./data/processed/product_demand.xlsx
COPY models/trained/*.h5 ./models/trained/
COPY models/trained/*.pkl ./models/trained/

# Crear usuario no-root
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

# Comando de inicio
CMD exec python app_prediccion_lstm.py
