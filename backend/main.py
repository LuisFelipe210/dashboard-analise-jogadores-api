# Desafio-final/backend/main.py
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from typing import List, Dict, Any, Optional

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Inicialização da Aplicação ---
app = FastAPI(
    title="API de Previsão de Desempenho de Jogadores",
    description="Uma API que prevê os Targets 1, 2 e 3 e segmenta novos jogadores em clusters.",
    version="1.1.0" # Versão atualizada
)

# --- Segurança: Configuração da Chave de API ---
API_KEY = os.getenv("API_KEY", "default-secret-key") # Use uma chave segura em produção
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(status_code=403, detail="Chave de API inválida ou ausente.")

# --- Carregamento dos Artefatos do Modelo ---
try:
    logger.info("Carregando artefatos do modelo...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    artifacts_path = os.path.join(BASE_DIR, "model_artifacts")

    scaler = joblib.load(os.path.join(artifacts_path, "scaler.joblib"))
    kmeans = joblib.load(os.path.join(artifacts_path, "kmeans.joblib"))
    model_t1 = joblib.load(os.path.join(artifacts_path, "lgbm_target1.joblib"))
    model_t2 = joblib.load(os.path.join(artifacts_path, "lgbm_target2.joblib"))
    model_t3 = joblib.load(os.path.join(artifacts_path, "lgbm_target3.joblib"))

    with open(os.path.join(artifacts_path, "model_config.json"), 'r') as f:
        model_config = json.load(f)

    with open(os.path.join(artifacts_path, "imputation_values.json"), 'r') as f:
        imputation_values = json.load(f)

    perfil_clusters = pd.read_csv(os.path.join(artifacts_path, "perfil_clusters.csv"), index_col='cluster')
    logger.info("Artefatos carregados com sucesso.")

except FileNotFoundError as e:
    logger.error(f"Erro fatal ao carregar artefatos: {e}. Execute 'train_model.py' primeiro.")
    raise RuntimeError(f"Erro ao carregar artefatos do modelo: {e}. Execute o script 'train_model.py' primeiro.")

# --- Definição dos Modelos de Dados (Pydantic) ---
# Modelo para validação de cada jogador individualmente
class PlayerInput(BaseModel):
    # Adicione aqui as colunas mais importantes com seus tipos esperados
    # Exemplo:
    T0101: Optional[float] = Field(None, description="Exemplo de feature numérica")
    T0102: Optional[float] = Field(None, description="Exemplo de outra feature")
    # Adicione todas as outras colunas esperadas do seu dataset
    # Deixando como um dicionário genérico por enquanto para flexibilidade
    class Config:
        extra = 'allow' # Permite colunas extras que não foram definidas

class PredictionPayload(BaseModel):
    data: List[PlayerInput]

# --- Endpoint de "Saúde" da API ---
@app.get("/", tags=["Status"])
def read_root():
    """Verifica se a API está online."""
    return {"status": "ok", "message": "API de Previsão de Jogadores está ativa!"}

# --- Endpoint de Previsão (Protegido) ---
@app.post("/predict", tags=["Prediction"])
def predict(payload: PredictionPayload, api_key: str = Security(get_api_key)):
    """
    Recebe dados de novos jogadores, realiza o pré-processamento e retorna as previsões.
    Este endpoint é protegido e requer uma chave de API no cabeçalho X-API-KEY.
    """
    logger.info(f"Recebida requisição de previsão para {len(payload.data)} jogadores.")
    try:
        # Pydantic já validou a estrutura. Agora convertemos para DataFrame.
        # O .dict() foi depreciado, use .model_dump()
        df_new = pd.DataFrame([p.model_dump() for p in payload.data])


        identifiers = df_new['Código de Acesso'] if 'Código de Acesso' in df_new.columns else pd.Series(
            range(len(df_new)), name="Player_ID")

        # 1. Pré-processamento e Imputação
        logger.info("Iniciando pré-processamento...")
        for col, value in imputation_values.items():
            if col in df_new.columns:
                df_new[col].fillna(value, inplace=True)

        # Garantir que não sobrem nulos nas features importantes
        for col in model_config['features_para_cluster']:
             if col not in df_new.columns:
                 df_new[col] = 0 # Adiciona a coluna se estiver faltando
             df_new[col].fillna(0, inplace=True)
        for col in model_config['categorical_features']:
             if col not in df_new.columns:
                 df_new[col] = 'missing' # Adiciona a coluna se estiver faltando
             df_new[col].fillna('missing', inplace=True)


        # 2. Predição do Cluster
        logger.info("Prevendo clusters...")
        features_cluster = df_new[model_config['features_para_cluster']]
        features_scaled = scaler.transform(features_cluster)
        df_new['cluster'] = kmeans.predict(features_scaled)

        # 3. Preparação para a Regressão
        logger.info("Preparando dados para regressão...")
        df_encoded = pd.get_dummies(df_new, drop_first=True)
        df_aligned = df_encoded.reindex(columns=model_config['model_columns'], fill_value=0)

        X_new = df_aligned[model_config['top_features_regressao']]

        # 4. Predição dos Targets
        logger.info("Prevendo targets...")
        pred_t1 = model_t1.predict(X_new)
        pred_t2 = model_t2.predict(X_new)
        pred_t3 = model_t3.predict(X_new)

        # 5. Montar a resposta
        logger.info("Montando resposta...")
        results = []
        for i in range(len(df_new)):
            player_profile = df_new.iloc[i][model_config['features_para_cluster']].to_dict()
            cluster_profile = perfil_clusters.loc[df_new['cluster'].iloc[i]].to_dict()

            results.append({
                "identifier": identifiers.iloc[i],
                "predicted_cluster": int(df_new['cluster'].iloc[i]),
                "predicted_target1": round(pred_t1[i], 2),
                "predicted_target2": round(pred_t2[i], 2),
                "predicted_target3": round(pred_t3[i], 2),
                "player_profile": player_profile,
                "cluster_average_profile": cluster_profile
            })
        logger.info("Previsão concluída com sucesso.")
        return {"predictions": results}

    except KeyError as e:
        logger.error(f"KeyError: {e}")
        raise HTTPException(status_code=400, detail=f"Coluna esperada não encontrada nos dados de entrada: {e}")
    except Exception as e:
        logger.error(f"Erro inesperado: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {e}")