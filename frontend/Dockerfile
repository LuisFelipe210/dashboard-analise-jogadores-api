FROM python:3.13-slim

WORKDIR /app

# Copia primeiro o arquivo de dependências
COPY requirements.txt .

# Instala as dependências (esta camada será cacheada)
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do código da aplicação
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py"]