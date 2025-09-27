FROM python:3.13-slim

RUN apt-get update && apt-get install -y libgomp1

WORKDIR /app

# Copia primeiro o arquivo de dependências
COPY requirements.txt .

# Instala as dependências (esta camada será cacheada)
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do código da aplicação
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]