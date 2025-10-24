# Dashboard de Análise e Previsão de Jogadores

Este projeto consiste em uma solução completa de Machine Learning, que inclui uma API de Backend utilizando FastAPI para previsões e um Dashboard Interativo para análise de dados e interação com os modelos. A aplicação permite que os usuários carreguem planilhas de dados, realizem previsões com modelos de aprendizado de máquina e visualizem gráficos interativos com os resultados.

## Funcionalidades

### Backend (FastAPI):

- **API de Previsões**: Fornece endpoints para enviar dados e realizar previsões sobre o desempenho de jogadores.
- **Processamento de Dados**: Carrega e processa dados de entrada, realiza previsões utilizando modelos treinados e retorna os resultados.
- **Suporte a Upload de Arquivos**: Permite fazer upload de arquivos XLSX com dados de jogadores para previsões em massa.
- **Resumo por Faixas**: Gera resumos por faixas de valores para os alvos preditivos (como Target1, Target2, Target3).

### Frontend (React + Plotly):

- **Carregamento e Exibição de Dados**: Permite que os usuários façam upload de planilhas e visualizem os dados em formato tabular.
- **Análise e Previsão de Jogadores**: Oferece visualizações interativas, como gráficos de dispersão, gráficos de resíduos e radar com o desempenho dos jogadores.
- **Filtros e Detalhes de Jogadores**: Os usuários podem filtrar os jogadores por identificador e visualizar métricas detalhadas como RMSE, R², MAE, entre outros.
- **Gráficos Interativos**: Gráficos de comparação entre os valores reais e previstos (Real vs Predito) e visualizações de agrupamento de faixas.

## Tecnologias Utilizadas

### Backend
- **FastAPI**: Framework rápido e moderno para criar APIs com Python.
- **scikit-learn**: Biblioteca de aprendizado de máquina utilizada para treinar e gerar previsões com os modelos.
- **Joblib**: Biblioteca para serializar objetos Python, usada para carregar os modelos de Machine Learning.
- **Pandas**: Biblioteca para manipulação de dados.
- **Uvicorn**: Servidor ASGI usado para rodar a API FastAPI.

### Frontend
- **React**: Biblioteca para construção da interface interativa.
- **Material-UI (MUI)**: Biblioteca de componentes React para criar a interface de usuário com design responsivo.
- **Plotly.js**: Biblioteca para criar gráficos interativos de dados.
- **Axios**: Biblioteca para realizar requisições HTTP à API backend.

### Docker
- **Docker**: Contêineres usados para isolar e empacotar a aplicação frontend e backend.

### Outros
- **Nginx**: Servidor web utilizado para hospedar o frontend e redirecionar requisições para o backend.
- **XLSX**: Formato de arquivo usado para importar dados e previsões.

## Estrutura do Projeto
```
/backend                # Código do backend (API FastAPI)
  /artifacts            # Diretório contendo os artefatos dos modelos
  /models               # Modelos treinados salvos com joblib
  /main.py              # Arquivo principal do backend com as definições da API
  /Dockerfile           # Dockerfile para rodar o backend

/frontend               # Código do frontend (React)
  /public               # Arquivos públicos do frontend (HTML, CSS)
  /src                  # Código-fonte do React (Componentes, hooks, etc.)
  /Dockerfile           # Dockerfile para rodar o frontend

/nginx                  # Arquivos de configuração do Nginx

/.env                   # Arquivo de variáveis de ambiente (para configurar API_URL, etc.)
/docker-compose.yml      # Arquivo de configuração para rodar os containers com Docker Compose
/requirements.txt        # Dependências Python do backend
/package.json            # Dependências do frontend (React)
```

## Instalação

### Backend

1. Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/projeto.git
    cd projeto
    ```

2. Instale as dependências do backend:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

3. Certifique-se de ter os artefatos necessários (modelos e arquivos). Se não tiver, será necessário executar o script de treinamento para gerar os artefatos.

4. Inicie o servidor backend com o Uvicorn:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

O servidor estará disponível em [http://localhost:8000](http://localhost:8000).

### Frontend

1. Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/projeto.git
    cd projeto
    ```

2. Instale as dependências do frontend:
    ```bash
    cd frontend
    npm install
    ```

3. Inicie o servidor de desenvolvimento:
    ```bash
    npm run dev
    ```

O frontend estará disponível em [http://localhost:5173](http://localhost:5173).

### Docker (para ambiente isolado)

Se preferir rodar a aplicação com Docker, pode usar o docker-compose para facilitar a configuração.

1. Certifique-se de que o Docker e o Docker Compose estão instalados em sua máquina.

2. Na raiz do projeto, execute:
    ```bash
    docker-compose up --build
    ```

O Docker Compose irá construir e iniciar os containers para o backend e frontend. O dashboard estará disponível em [http://localhost](http://localhost) e o backend em [http://localhost:8000](http://localhost:8000).

## Endpoints da API

- **GET /health**: Verifica se o backend está funcionando.
- **GET /predict/schema**: Retorna o esquema das colunas esperadas para o upload do XLSX.
- **POST /predict**: Envia dados JSON para previsão. Espera um JSON com as linhas de dados no corpo.
- **POST /predict/file**: Envia um arquivo XLSX para o backend para realizar previsões.
- **POST /radar**: Gera o perfil de radar para um jogador específico.
- **POST /targets/buckets**: Retorna a contagem de previsões por faixas de valores (low, high, etc.).

## Como Usar

### Carregar um arquivo XLSX

1. No dashboard, clique no botão "Carregar Excel" e selecione um arquivo XLSX contendo os dados de jogadores.
2. Após o carregamento, os resultados das previsões serão exibidos no painel, incluindo os valores previstos para Target1, Target2 e Target3.

### Realizar Previsões

1. Após carregar o arquivo, clique em "Realizar Análise" para gerar as previsões.
2. Os resultados serão exibidos no formato de tabela, e gráficos interativos mostrarão a comparação entre os valores reais e previstos, além de outros gráficos estatísticos.

### Interagir com os Resultados

O painel permite que você selecione jogadores individuais para visualizar detalhes adicionais e comparações de radar.

## Contribuições

Contribuições são bem-vindas! Se você deseja melhorar este projeto, faça um fork, crie uma branch e submeta um pull request. Agradecemos por qualquer melhoria, sugestão ou correção.
