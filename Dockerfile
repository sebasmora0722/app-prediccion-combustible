FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Herramientas de compilación (xgboost/scikit-learn suelen requerirlas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala deps primero para cachear
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copia todo (incluye data/, modelos/, static/, .streamlit/)
COPY . /app

# Streamlit debe escuchar en 0.0.0.0 y puerto 8080
ENV PORT=8080
CMD ["streamlit", "run", "codigo_prueba.py", "--server.port=8080", "--server.address=0.0.0.0"]
