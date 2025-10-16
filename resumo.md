# 📊 Análise Detalhada do Projeto de Dashboard e API de Machine Learning

## 🎯 Visão Geral

Este projeto consiste em uma solução de Machine Learning completa e bem arquitetada, com um pipeline de treinamento de modelo, uma API de backend para inferência (FastAPI) e um dashboard interativo para visualização e previsão (Streamlit). A aplicação é containerizada com Docker, garantindo portabilidade e facilidade de implantação.

---

## 📁 Análise por Arquivo

### 📖 README.md
**Tipo:** Documentação

O README é claro, bem estruturado e fornece uma excelente visão geral do projeto. A inclusão de um diagrama de arquitetura (Mermaid) é um grande diferencial. As instruções de instalação e execução são detalhadas e fáceis de seguir. É um ótimo ponto de partida para qualquer pessoa que queira entender e executar o projeto.

---

### 🐳 docker-compose.yml
**Tipo:** Orquestração de Contêineres

#### Configuração Geral
- **Versão:** `3.8` - Define a versão da sintaxe do Docker Compose

#### Serviço: Backend
| Configuração | Descrição |
|--------------|-----------|
| **build** | Constrói imagem a partir do `Dockerfile` em `./backend` |
| **container_name** | Nomeia o contêiner como `backend_api` |
| **environment** | Injeta `API_KEY` do arquivo `.env` |
| **ports** | Mapeia porta 8000:8000 para acesso externo |
| **volumes** | Monta `./backend` em `/app` para hot-reload |
| **networks** | Conecta à `app_network` |

#### Serviço: Frontend
| Configuração | Descrição |
|--------------|-----------|
| **build** | Constrói imagem do `Dockerfile` em `./frontend` |
| **depends_on** | Garante inicialização do backend primeiro |
| **command** | Cria `.streamlit/secrets.toml` e injeta `API_KEY` de forma segura |
| **ports** | Mapeia porta 8501:8501 |

#### Rede
- **app_network:** Rede bridge customizada que isola a comunicação entre contêineres e permite uso de nomes de serviços (ex: `http://backend:8000`)

---

### 🤖 train_model.py
**Tipo:** Script de Treinamento de ML

#### Pipeline de Treinamento

**Passo 1: Carregamento e Pré-processamento**
- Lê o CSV `JogadoresV1.csv`
- Limpa nomes das colunas
- Usa `force_clean_and_convert_string()` para conversão robusta de dados textuais
- Realiza One-Hot Encoding em features categóricas de cores

**Passo 2: Engenharia de Features**
- Cria features derivadas: `Indice_Sono_T1`, `Eficiencia_Total`, `Gap_F11_F07`
- Extrai mais informações dos dados brutos
- Melhora performance preditiva do modelo

**Passo 3: Imputação e Clusterização**
- Trata valores ausentes (NaN):
  - Mediana para dados numéricos
  - Moda para categóricos
- Pipeline scikit-learn:
  1. `StandardScaler` - Padronização
  2. `PCA` - Redução de dimensionalidade
  3. `KMeans` (2 clusters) - Segmentação de perfis

**Passo 4: Divisão Treino-Validação-Teste**
- Estratégia de divisão:
  - 64% Treino
  - 16% Validação
  - 20% Teste
- Garante avaliação robusta e imparcial

**Passo 5: Treinamento para Avaliação**
- Modelo: `StackingRegressor`
  - Base: `RandomForest` + `RidgeCV`
  - Meta-estimador: `Lasso`
- Gera `frontend/jogadores_com_clusters.csv` para dashboard

**Passo 6: Treinamento Final e Salvamento**
- ✅ Boa prática MLOps: retreina com 100% dos dados
- Salva artefatos em `backend/model_artifacts/`:
  - Modelos `.joblib`
  - Pipeline de cluster
  - Perfil dos clusters (CSV)
  - Configuração (JSON)

---

### ⚡ backend/main.py
**Tipo:** API Backend (FastAPI)

#### Estrutura da API

**Inicialização**
```python
app = FastAPI(...)
```

**Carregamento de Artefatos**
- Carrega todos os modelos na inicialização
- Bloco `try-except` previne quebra se artefatos não encontrados

**Modelo de Dados**
- `PlayerDataRequest(BaseModel)`: Validação automática via Pydantic

**Pré-processamento**
- `preprocess_input_data()`: Replica EXATAMENTE os passos do treinamento
- Previne "training-serving skew"

#### Endpoint `/predict`

**Segurança**
- Valida `x-api-key` no header

**Fluxo de Processamento**
1. Converte JSON → DataFrame
2. Aplica pré-processamento
3. Alinha colunas com `df.reindex()` (fill_value=0)
4. Prevê cluster do jogador
5. Adiciona features de cluster
6. Prevê 3 targets (Target1, Target2, Target3)
7. Retorna JSON com:
   - Previsões
   - Cluster atribuído
   - Perfil do jogador
   - Perfil do cluster

---

### 🎨 frontend/dashboard.py
**Tipo:** Dashboard Frontend (Streamlit)

#### Configuração

**Página**
```python
st.set_page_config(..., layout='wide')
```

**Variáveis de Ambiente**
- `API_URL`: Do ambiente Docker
- `API_KEY`: De `st.secrets`

#### Funções Principais

**`load_data()` + Cache**
- Carrega `jogadores_com_clusters.csv`
- `@st.cache_data` otimiza performance

**`run_prediction()`**
Fluxo completo de previsão:
1. Obtém arquivo do `st.session_state`
2. Lê Excel → DataFrame
3. Exibe `st.spinner` (feedback visual)
4. Limpa NaNs/Infs
5. Monta payload + headers (X-API-KEY)
6. POST para API backend
7. Trata resposta:
   - ✅ 200: Armazena previsões
   - ❌ 403: Erro de autenticação
   - ❌ Outro: Erro genérico
8. Usa `st.session_state` para persistência

**`plot_real_vs_previsto()`**
- Gráfico de dispersão interativo (Plotly Express)
- Linha de tendência incluída

**`plot_radar_chart()`**
- Gráfico de radar (Plotly Graph Objects)
- Compara jogador vs. média do cluster

#### Layout

**Aba: Análise de Desempenho**
- Métricas: RMSE e R² (`st.metric`)
- Gráficos de dispersão
- Distribuição de clusters (barras)

**Aba: Previsão para Novos Jogadores**
- `st.file_uploader`: Upload de arquivos
- `st.button`: Trigger de análise
- Feedback: `st.success` / `st.error`
- Tabelas: `st.dataframe`
- `st.selectbox`: Análise individual detalhada

---

## 🔄 Guia de Migração para React

### Por que migrar?
- ✅ Mais flexibilidade de UI/UX
- ✅ Vasto ecossistema de bibliotecas
- ✅ Melhor gerenciamento de estado
- ✅ Ideal para aplicações complexas

### Passo 1: Configuração do Ambiente

```bash
# Instalar Node.js e npm/Yarn
npx create-react-app nome-do-app

# Instalar dependências
npm install axios react-bootstrap bootstrap plotly.js react-plotly.js xlsx
```

### Passo 2: Estruturação de Componentes

Crie a pasta `components/` com:

| Componente | Responsabilidade |
|------------|------------------|
| `FileUpload.js` | Upload de arquivos e botão de análise |
| `PredictionsTable.js` | Tabela de resultados |
| `PlayerAnalysis.js` | Análise detalhada individual |
| `RadarChart.js` | Gráfico de radar (react-plotly.js) |

### Passo 3: Conexão com API

**`services/api.js`**
```javascript
import axios from 'axios';

const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
    'X-API-KEY': process.env.REACT_APP_API_KEY
  }
});

export const getPredictions = (playerData) => {
  return apiClient.post('/predict', { data: playerData });
};
```

**Uso no componente:**
- `useState` para gerenciar estado (arquivo, loading, erro)
- `async/await` para chamar `getPredictions`

### Passo 4: Gerenciamento de Estado

**`App.js`**
- `useState` para array de previsões
- `useState` para jogador selecionado
- Props para componentes filhos

### Passo 5: Atualização do Docker

**Novo `Dockerfile` multi-stage:**
1. **Stage 1:** Build da aplicação React (`npm run build`)
2. **Stage 2:** Servir arquivos estáticos (nginx)

**Atualizar `docker-compose.yml`:**
- Apontar para novo Dockerfile
- Ajustar variáveis de ambiente

---

## 🎯 Conclusão

Este projeto demonstra excelentes práticas de:
- ✅ Arquitetura de microsserviços
- ✅ MLOps (treinamento, versionamento, deployment)
- ✅ Containerização (Docker)
- ✅ API design (FastAPI + Pydantic)
- ✅ UX interativa (Streamlit/React)
- ✅ Segurança (API keys, validação)

A migração para React oferece ainda mais escalabilidade e flexibilidade para o futuro do projeto