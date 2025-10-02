from fastapi import FastAPI , Header , HTTPException
from pydantic import BaseModel
from typing import List , Dict , Any
import joblib
import pandas as pd
import numpy as np
import json
import os
import warnings

warnings.filterwarnings("ignore" , category=FutureWarning , module="pandas.core.frame")

app = FastAPI(title="API de Análise de Jogadores v3.2 Final")
API_KEY = os.getenv("API_KEY" , "default-secret-key")

try:
    ARTIFACTS_DIR = "model_artifacts"
    models = {
        'target1': joblib.load(os.path.join(ARTIFACTS_DIR , "stacking_model_target1.joblib")) ,
        'target2': joblib.load(os.path.join(ARTIFACTS_DIR , "stacking_model_target2.joblib")) ,
        'target3': joblib.load(os.path.join(ARTIFACTS_DIR , "stacking_model_target3.joblib"))
    }
    cluster_pipeline = joblib.load(os.path.join(ARTIFACTS_DIR , "cluster_pipeline.joblib"))
    perfil_clusters_df = pd.read_csv(os.path.join(ARTIFACTS_DIR , "perfil_clusters.csv") , index_col=0)
    with open(os.path.join(ARTIFACTS_DIR , "model_config.json") , 'r') as f:
        model_config = json.load(f)
    FINAL_MODEL_COLUMNS = model_config ["final_model_columns"]
    NUMERICAL_FEATURES_FOR_CLUSTERING = model_config ["numerical_features_for_clustering"]
    CATEGORICAL_FEATURES = model_config ["categorical_features"]
except FileNotFoundError as e:
    print(f"Erro Crítico ao carregar artefatos: {e}")
    models , cluster_pipeline , perfil_clusters_df , FINAL_MODEL_COLUMNS , NUMERICAL_FEATURES_FOR_CLUSTERING , CATEGORICAL_FEATURES = {} , None , None , [] , [] , []


class PlayerDataRequest(BaseModel):
    data: List [Dict [str , Any]]


def preprocess_input_data(data: pd.DataFrame , categorical_cols_from_training: list) -> pd.DataFrame:
    df_processed = data.copy()
    df_processed.columns = df_processed.columns.str.strip().str.replace(' ' , '_' , regex=False).str.replace(
        '[^a-zA-Z0-9_]' , '' , regex=True)

    def force_clean_and_convert_string(series):
        series = series.astype(str).str.replace('"' , '' , regex=False).str.replace('.' , '' , regex=False).str.replace(
            ',' , '.' , regex=False)
        series = series.replace(['-1.0' , 'nan' , 'N/A' , 'NaN'] , np.nan)
        return pd.to_numeric(series , errors='coerce')

    object_cols = df_processed.select_dtypes(include='object').columns
    cols_to_exclude_conv = ['Cdigo_de_Acesso'] + [col for col in object_cols if
                                                  col.startswith('Cor') or col.startswith('F0207')]
    for col in object_cols:
        if col not in cols_to_exclude_conv:
            df_processed [col] = force_clean_and_convert_string(df_processed [col])

    cols_to_drop = ['T1205Expl' , 'T1199Expl' , 'F0299__Explicao_Tempo' , 'TempoTotalExpl' , 'PTempoTotalExpl' ,
                    'T1210Expl' , 'T0499__Explicao_Tempo' , 'DataHora_ltimo']
    df_processed.drop(columns=cols_to_drop , errors='ignore' , inplace=True)

    # LÓGICA DE ALINHAMENTO DAS CORES
    color_cols = [col for col in df_processed.columns if col.startswith('Cor') or col == 'F0207']
    if color_cols:
        df_color_ohe = pd.get_dummies(df_processed [color_cols].astype(str) , prefix=color_cols , dummy_na=False)
        # Reindexa para garantir que tenha exatamente as mesmas colunas do treino
        df_color_ohe = df_color_ohe.reindex(columns=categorical_cols_from_training , fill_value=0)
        df_processed = pd.concat([df_processed.drop(columns=color_cols , errors='ignore') , df_color_ohe] , axis=1)

    # ... (código de engenharia de features idêntico ao anterior) ...
    Sono_col = next((col for col in df_processed.columns if 'QtdHorasSono' in col) , None)
    if Sono_col and 'Acordar' in df_processed.columns:
        df_processed ['Indice_Sono_T1'] = df_processed [Sono_col] * (
                    df_processed ['Acordar'].max() - df_processed ['Acordar'])
    if 'F1103' in df_processed.columns and 'F0713' in df_processed.columns:
        df_processed ['F_Oposto_T3'] = df_processed ['F1103'] - df_processed ['F0713']
    F07_cols = [col for col in df_processed.columns if col.startswith('F07') and len(col) == 5]
    if F07_cols: df_processed ['F07_Media'] = df_processed [F07_cols].mean(axis=1)
    F11_cols = [col for col in df_processed.columns if col.startswith('F11') and len(col) == 5]
    if F11_cols: df_processed ['F11_Media'] = df_processed [F11_cols].mean(axis=1)
    if 'P09' in df_processed.columns and 'T09' in df_processed.columns:
        df_processed ['Eficiencia_P09_T09'] = df_processed ['P09'] / df_processed ['T09'].replace(0 , 1e-6)
    p_cols = [col for col in df_processed.columns if col.startswith('P') and len(col) == 3]
    t_cols = [col for col in df_processed.columns if col.startswith('T') and len(col) == 3]
    if p_cols and t_cols:
        df_processed ['Eficiencia_Total'] = df_processed [p_cols].sum(axis=1 , skipna=True) / df_processed [t_cols].sum(
            axis=1 , skipna=True).replace(0 , 1e-6)
    if 'F11_Media' in df_processed.columns and 'F07_Media' in df_processed.columns:
        df_processed ['Gap_F11_F07'] = df_processed ['F11_Media'] - df_processed ['F07_Media']
    if 'QtdHorasDormi' in df_processed.columns and 'QtdHorasSono' in df_processed.columns:
        df_processed ['Sono_Ineficiencia'] = df_processed ['QtdHorasDormi'] - df_processed ['QtdHorasSono']
    if 'PTempoTotal' in df_processed.columns and 'TempoTotal' in df_processed.columns:
        df_processed ['Indice_Final_PT_T'] = df_processed ['PTempoTotal'] / df_processed ['TempoTotal'].replace(0 ,
                                                                                                                1e-6)
    all_p_cols = [col for col in df_processed.columns if col.startswith('P') and (len(col) == 3 or len(col) == 4)]
    for p_col in all_p_cols:
        t_col = 'T' + p_col [1:]
        new_col_name = f'Eficiencia_{p_col}_{t_col}'
        if t_col in df_processed.columns and 'Expl' not in t_col and new_col_name not in df_processed.columns:
            df_processed [new_col_name] = df_processed [p_col] / df_processed [t_col].replace(0 , 1e-6)
    df_processed.drop(columns=[col for col in df_processed.columns if
                               '_Limpo' in col or col.startswith('Soma_') or 'Acordar_Invertido' in col] ,
                      errors='ignore' , inplace=True)
    numerical_cols = df_processed.select_dtypes(include=np.number).columns
    for col in [c for c in numerical_cols if c not in ['Target1' , 'Target2' , 'Target3']]:
        df_processed.loc [df_processed [col] < 0 , col] = np.nan
    for col in df_processed.columns:
        if df_processed [col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_processed [col]):
                df_processed [col] = df_processed [col].fillna(df_processed [col].median())
            else:
                df_processed [col] = df_processed [col].fillna(df_processed [col].mode() [0])
    return df_processed


@app.post("/predict" , tags=["Predição"])
def predict(request_data: PlayerDataRequest , x_api_key: str = Header(None)):
    if x_api_key != API_KEY: raise HTTPException(status_code=403 , detail="Chave de API inválida.")
    if not all(models.values()) or cluster_pipeline is None: raise HTTPException(status_code=500 ,
                                                                                 detail="Modelos não carregados.")

    input_df_raw = pd.DataFrame(request_data.data)
    identifiers = input_df_raw ['Código de Acesso'].copy()

    df_processed = preprocess_input_data(input_df_raw , CATEGORICAL_FEATURES)
    df_aligned = df_processed.reindex(columns=FINAL_MODEL_COLUMNS , fill_value=0)
    for col in df_aligned.select_dtypes(include='bool').columns:
        df_aligned [col] = df_aligned [col].astype(int)

    cluster_features = df_aligned [NUMERICAL_FEATURES_FOR_CLUSTERING]
    cluster_predictions = cluster_pipeline.predict(cluster_features)

    for i in range(2):  # Alterado para 2 clusters
        cluster_col_name = f'Cluster_{i}'
        if cluster_col_name in df_aligned.columns:
            df_aligned [cluster_col_name] = (cluster_predictions == i).astype(int)

    predictions_list = []
    for i in range(len(df_aligned)):
        player_data = df_aligned.iloc [[i]]
        pred_target1 = models ['target1'].predict(player_data) [0]
        pred_target2 = models ['target2'].predict(player_data) [0]
        pred_target3 = models ['target3'].predict(player_data) [0]
        cluster_id = cluster_predictions [i]
        player_profile = df_aligned.iloc [i] [NUMERICAL_FEATURES_FOR_CLUSTERING].to_dict()
        cluster_profile = perfil_clusters_df.loc [cluster_id].to_dict()

        predictions_list.append({
            "identifier": identifiers.iloc [i] ,
            "predicted_cluster": int(cluster_id) ,
            "predicted_target1": round(pred_target1 , 2) ,
            "predicted_target2": round(pred_target2 , 2) ,
            "predicted_target3": round(pred_target3 , 2) ,
            "player_profile": player_profile ,
            "cluster_average_profile": cluster_profile
        })
    return {"predictions": predictions_list}