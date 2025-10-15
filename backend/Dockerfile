# backend/Dockerfile
FROM python:3.9-slim

# Variável de ambiente para o Flask
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8000

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY . .

EXPOSE 8000

# Comando para iniciar o servidor de produção (waitress)
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8000", "main:app"]