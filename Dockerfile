# ============================================================================
# DOCKERFILE PARA GOOGLE CLOUD RUN - LSTM CROSS ANALYSIS WEB SERVICE
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

# Copiar código fuente
COPY app_cross_analysis_web.py .
COPY src/ ./src/
COPY templates/ ./templates/

# Crear directorios necesarios
RUN mkdir -p data/processed models/temporal/customer_v3/medium models/temporal/products_50epochs/short

# NOTA: Los modelos y datos deben descargarse al inicio o montarse desde Cloud Storage
# No se incluyen en la imagen por su tamaño (excluidos en .gitignore)

# Crear usuario no-root
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

# Comando de inicio - Cloud Run usa PORT=8080
CMD exec python app_cross_analysis_web.py
