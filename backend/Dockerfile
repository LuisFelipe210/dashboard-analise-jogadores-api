# backend/Dockerfile
FROM python:3.12-slim

# Evita .pyc e buffers no stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/artifacts

# Dependências de runtime (OpenMP/Fortran usadas por numpy/scipy/sklearn)
# build-essential só se faltar wheel — pode remover se tudo vier como wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Primeiro as deps (melhor cache)
COPY requirements.txt .
RUN python -m pip install -U pip setuptools wheel \
 && pip install -r requirements.txt

# Código e artefatos
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
