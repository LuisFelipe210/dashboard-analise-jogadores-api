# /backend/main.py (Versão 2.0 - Compatível com Modelo Otimizado)

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import pandas as pd
import numpy as np
import json
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.core.frame")

app = FastAPI(title="API de Análise de Jogadores v4.0 (Otimizada)")
API_KEY = os.getenv("API_KEY", "default-secret-key")

# --- Carregamento dos Artefatos ---
try:
    ARTIFACTS_DIR = "model_artifacts"
    models = {
        'Target1': joblib.load(os.path.join(ARTIFACTS_DIR, "best_model_Target1.joblib")),
        'Target2': joblib.load(os.path.join(ARTIFACTS_DIR, "best_model_Target2.joblib")),
        'Target3': joblib.load(os.path.join(ARTIFACTS_DIR, "best_model_Target3.joblib"))
    }
    cluster_pipeline = joblib.load(os.path.join(ARTIFACTS_DIR, "cluster_pipeline.joblib"))
    perfil_clusters_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "perfil_clusters.csv"), index_col=0)
    with open(os.path.join(ARTIFACTS_DIR, "model_config.json"), 'r') as f:
        model_config = json.load(f)
    
    FINAL_MODEL_COLUMNS = model_config["final_model_columns"]
    NUMERICAL_FEATURES_FOR_CLUSTERING = model_config["numerical_features_for_clustering"]
    CATEGORICAL_FEATURES_OHE = model_config["categorical_features_ohe"]
    SELECTED_FEATURES_BY_TARGET = model_config["selected_features_by_target"]

except FileNotFoundError as e:
    print(f"Erro Crítico ao carregar artefatos: {e}")
    models, cluster_pipeline, perfil_clusters_df, FINAL_MODEL_COLUMNS, NUMERICAL_FEATURES_FOR_CLUSTERING, CATEGORICAL_FEATURES_OHE, SELECTED_FEATURES_BY_TARGET = {}, None, None, [], [], [], {}

class PlayerDataRequest(BaseModel):
    data: List[Dict[str, Any]]

def preprocess_input_data(data: pd.DataFrame, categorical_cols_from_training: list) -> pd.DataFrame:
    # Esta função é idêntica à do script de treinamento para evitar divergências.
    df_processed = data.copy()
    df_processed.columns = df_processed.columns.str.strip().str.replace(' ', '_', regex=False).str.replace('[^a-zA-Z0-9_]', '', regex=True)

    def force_clean_and_convert_string(series):
        series = series.astype(str).str.replace('"', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        series = series.replace(['-1.0', 'nan', 'N/A', 'NaN'], np.nan)
        return pd.to_numeric(series, errors='coerce')

    object_cols = df_processed.select_dtypes(include='object').columns
    cols_to_exclude_conv = ['Cdigo_de_Acesso'] + [col for col in object_cols if col.startswith('Cor') or col.startswith('F0207')]
    for col in object_cols:
        if col not in cols_to_exclude_conv:
            df_processed[col] = force_clean_and_convert_string(df_processed[col])

    cols_to_drop = ['T1205Expl', 'T1199Expl', 'F0299__Explicao_Tempo', 'TempoTotalExpl', 'PTempoTotalExpl', 'T1210Expl', 'T0499__Explicao_Tempo', 'DataHora_ltimo']
    df_processed.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    color_cols = [col for col in df_processed.columns if col.startswith('Cor') or col == 'F0207']
    if color_cols:
        df_color_ohe = pd.get_dummies(df_processed[color_cols].astype(str), prefix=color_cols, dummy_na=False)
        df_color_ohe = df_color_ohe.reindex(columns=categorical_cols_from_training, fill_value=0)
        df_processed = pd.concat([df_processed.drop(columns=color_cols, errors='ignore'), df_color_ohe], axis=1)

    Sono_col = next((col for col in df_processed.columns if 'QtdHorasSono' in col), None)
    if Sono_col and 'Acordar' in df_processed.columns:
        df_processed['Indice_Sono_T1'] = df_processed[Sono_col] * (df_processed['Acordar'].max() - df_processed['Acordar'])
    if 'F1103' in df_processed.columns and 'F0713' in df_processed.columns:
        df_processed['F_Oposto_T3'] = df_processed['F1103'] - df_processed['F0713']
    F07_cols = [col for col in df_processed.columns if col.startswith('F07') and len(col) == 5]
    if F07_cols: df_processed['F07_Media'] = df_processed[F07_cols].mean(axis=1)
    F11_cols = [col for col in df_processed.columns if col.startswith('F11') and len(col) == 5]
    if F11_cols: df_processed['F11_Media'] = df_processed[F11_cols].mean(axis=1)
    if 'P09' in df_processed.columns and 'T09' in df_processed.columns:
        df_processed['Eficiencia_P09_T09'] = df_processed['P09'] / df_processed['T09'].replace(0, 1e-6)
    p_cols = [col for col in df_processed.columns if col.startswith('P') and len(col) == 3]
    t_cols = [col for col in df_processed.columns if col.startswith('T') and len(col) == 3]
    if p_cols and t_cols:
        df_processed['Eficiencia_Total'] = df_processed[p_cols].sum(axis=1, skipna=True) / df_processed[t_cols].sum(axis=1, skipna=True).replace(0, 1e-6)
    if 'F11_Media' in df_processed.columns and 'F07_Media' in df_processed.columns:
        df_processed['Gap_F11_F07'] = df_processed['F11_Media'] - df_processed['F07_Media']
    if 'QtdHorasDormi' in df_processed.columns and 'QtdHorasSono' in df_processed.columns:
        df_processed['Sono_Ineficiencia'] = df_processed['QtdHorasDormi'] - df_processed['QtdHorasSono']
    if 'PTempoTotal' in df_processed.columns and 'TempoTotal' in df_processed.columns:
        df_processed['Indice_Final_PT_T'] = df_processed['PTempoTotal'] / df_processed['TempoTotal'].replace(0, 1e-6)
    all_p_cols = [col for col in df_processed.columns if col.startswith('P') and (len(col) == 3 or len(col) == 4)]
    for p_col in all_p_cols:
        t_col = 'T' + p_col[1:]
        new_col_name = f'Eficiencia_{p_col}_{t_col}'
        if t_col in df_processed.columns and 'Expl' not in t_col and new_col_name not in df_processed.columns:
            df_processed[new_col_name] = df_processed[p_col] / df_processed[t_col].replace(0, 1e-6)

    df_processed.drop(columns=[col for col in df_processed.columns if '_Limpo' in col or col.startswith('Soma_') or 'Acordar_Invertido' in col], errors='ignore', inplace=True)
    numerical_cols = df_processed.select_dtypes(include=np.number).columns
    for col in [c for c in numerical_cols if c not in ['Target1', 'Target2', 'Target3']]:
        df_processed.loc[df_processed[col] < 0, col] = np.nan
    for col in df_processed.columns:
        if df_processed[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    return df_processed

@app.post("/predict", tags=["Predição"])
def predict(request_data: PlayerDataRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Chave de API inválida.")
    if not all(models.values()) or cluster_pipeline is None:
        raise HTTPException(status_code=500, detail="Modelos não carregados corretamente no servidor.")

    input_df_raw = pd.DataFrame(request_data.data)
    identifiers = input_df_raw['Código de Acesso'].copy()

    df_processed = preprocess_input_data(input_df_raw, CATEGORICAL_FEATURES_OHE)
    
    # Clusterização
    cluster_features_df = df_processed.reindex(columns=NUMERICAL_FEATURES_FOR_CLUSTERING, fill_value=0)
    cluster_predictions = cluster_pipeline.predict(cluster_features_df)
    
    df_processed['cluster'] = cluster_predictions
    df_processed_ohe = pd.get_dummies(df_processed, columns=['cluster'], prefix='Cluster')

    # Alinhamento final das colunas para os modelos de target
    df_aligned = df_processed_ohe.reindex(columns=FINAL_MODEL_COLUMNS, fill_value=0)
    for col in df_aligned.select_dtypes(include='bool').columns:
        df_aligned[col] = df_aligned[col].astype(int)

    predictions_list = []
    for i in range(len(df_aligned)):
        player_data = df_aligned.iloc[[i]]
        
        # Previsão para cada target usando suas features específicas
        pred_target1 = models['Target1'].predict(player_data[SELECTED_FEATURES_BY_TARGET['Target1']])[0]
        pred_target2 = models['Target2'].predict(player_data[SELECTED_FEATURES_BY_TARGET['Target2']])[0]
        pred_target3 = models['Target3'].predict(player_data[SELECTED_FEATURES_BY_TARGET['Target3']])[0]

        cluster_id = cluster_predictions[i]
        player_profile = df_aligned.iloc[i][NUMERICAL_FEATURES_FOR_CLUSTERING].to_dict()
        cluster_profile = perfil_clusters_df.loc[cluster_id].to_dict()

        predictions_list.append({
            "identifier": identifiers.iloc[i],
            "predicted_cluster": int(cluster_id),
            "predicted_target1": round(pred_target1, 2),
            "predicted_target2": round(pred_target2, 2),
            "predicted_target3": round(pred_target3, 2),
            "player_profile": player_profile,
            "cluster_average_profile": cluster_profile
        })
    return {"predictions": predictions_list}