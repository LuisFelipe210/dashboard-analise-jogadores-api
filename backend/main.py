from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import List , Dict , Any

# --- Inicialização da Aplicação ---
app = FastAPI(
    title="API de Previsão de Desempenho de Jogadores" ,
    description="Uma API que prevê os Targets 1, 2 e 3 e segmenta novos jogadores em clusters." ,
    version="1.0.0"
)

# --- Carregamento dos Artefatos do Modelo ---
# Carregar modelos e configurações na inicialização para evitar recarregá-los a cada requisição
try:
    artifacts_path = "backend/model_artifacts"
    scaler = joblib.load(os.path.join(artifacts_path , "scaler.joblib"))
    kmeans = joblib.load(os.path.join(artifacts_path , "kmeans.joblib"))
    model_t1 = joblib.load(os.path.join(artifacts_path , "lgbm_target1.joblib"))
    model_t2 = joblib.load(os.path.join(artifacts_path , "lgbm_target2.joblib"))
    model_t3 = joblib.load(os.path.join(artifacts_path , "lgbm_target3.joblib"))

    with open(os.path.join(artifacts_path , "model_config.json") , 'r') as f:
        model_config = json.load(f)

    perfil_clusters = pd.read_csv(os.path.join(artifacts_path , "perfil_clusters.csv") , index_col='cluster')

except FileNotFoundError as e:
    raise RuntimeError(f"Erro ao carregar artefatos do modelo: {e}. Execute o script 'train_model.py' primeiro.")


# --- Definição dos Modelos de Dados (Pydantic) ---
# Define a estrutura esperada para um jogador nos dados de entrada
class PlayerData(BaseModel):
    data: List [Dict [str , Any]]


# --- Endpoint de "Saúde" da API ---
@app.get("/" , tags=["Status"])
def read_root():
    """Verifica se a API está online."""
    return {"status": "ok" , "message": "API de Previsão de Jogadores está ativa!"}


# --- Endpoint de Previsão ---
@app.post("/predict" , tags=["Prediction"])
def predict(player_data: PlayerData):
    """
    Recebe dados de novos jogadores, realiza o pré-processamento e retorna as previsões.
    """
    try:
        # Converter dados de entrada para DataFrame
        df_new = pd.DataFrame(player_data.data)

        # Guardar o identificador se existir
        identifiers = df_new ['Código de Acesso'] if 'Código de Acesso' in df_new.columns else pd.Series(
            range(len(df_new)) , name="Player_ID")

        # 1. Pré-processamento e Imputação (similar ao treino)
        colunas_numericas = model_config ['features_para_cluster']  # Usar as mesmas do treino
        colunas_categoricas = [c for c in df_new.columns if c not in colunas_numericas and c != 'Código de Acesso']

        for col in colunas_numericas:
            df_new [col].fillna(0 , inplace=True)  # Preenchimento simples para novos dados
        for col in colunas_categoricas:
            df_new [col].fillna('missing' , inplace=True)

        # 2. Predição do Cluster
        features_cluster = df_new [model_config ['features_para_cluster']]
        features_scaled = scaler.transform(features_cluster)
        df_new ['cluster'] = kmeans.predict(features_scaled)

        # 3. Preparação para a Regressão
        df_encoded = pd.get_dummies(df_new , drop_first=True)
        # Alinhar colunas com as do modelo (adiciona colunas faltantes com 0, remove extras)
        df_aligned = df_encoded.reindex(columns=model_config ['model_columns'] , fill_value=0)

        X_new = df_aligned [model_config ['top_features_regressao']]

        # 4. Predição dos Targets
        pred_t1 = model_t1.predict(X_new)
        pred_t2 = model_t2.predict(X_new)
        pred_t3 = model_t3.predict(X_new)

        # 5. Montar a resposta
        results = []
        for i in range(len(df_new)):
            player_profile = df_new.iloc [i] [model_config ['features_para_cluster']].to_dict()
            cluster_profile = perfil_clusters.loc [df_new ['cluster'].iloc [i]].to_dict()

            results.append({
                "identifier": identifiers.iloc [i] ,
                "predicted_cluster": int(df_new ['cluster'].iloc [i]) ,
                "predicted_target1": round(pred_t1 [i] , 2) ,
                "predicted_target2": round(pred_t2 [i] , 2) ,
                "predicted_target3": round(pred_t3 [i] , 2) ,
                "player_profile": player_profile ,
                "cluster_average_profile": cluster_profile
            })

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500 , detail=str(e))
