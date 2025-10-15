# backend/main.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Body, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings('ignore', category=UserWarning)

# ===================================================================
# CONFIGURAÇÃO E CARREGAMENTO DE ARTEFATOS
# ===================================================================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
API_KEY = os.getenv("API_KEY", "fallback_key_if_not_set")

app = FastAPI(title="API de Previsão de Performance de Jogadores", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Carregamento seguro de artefatos
def _safe_load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Artefato essencial não encontrado: {path}")
    if path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return joblib.load(path)

try:
    print("Carregando artefatos do modelo...")
    COLUMN_INFO = _safe_load(ARTIFACTS_DIR / 'column_info.json')
    NUMERIC_IMPUTER = _safe_load(ARTIFACTS_DIR / 'numeric_imputer.joblib')
    CATEGORICAL_IMPUTER = _safe_load(ARTIFACTS_DIR / 'categorical_imputer.joblib')
    SCALER = _safe_load(ARTIFACTS_DIR / 'scaler.joblib')
    ENCODER_OHE = _safe_load(ARTIFACTS_DIR / 'encoder_ohe.joblib')
    KMEANS_MODEL = _safe_load(ARTIFACTS_DIR / 'kmeans_model.joblib')

    MODELS = {
        'Target1': _safe_load(ARTIFACTS_DIR / 'best_model_Target1.joblib'),
        'Target2': _safe_load(ARTIFACTS_DIR / 'best_model_Target2.joblib'),
        'Target3': _safe_load(ARTIFACTS_DIR / 'best_model_Target3.joblib'),
    }
    print("Artefatos carregados com sucesso.")
except FileNotFoundError as e:
    print(f"Erro Crítico: {e}. Execute o script de treinamento primeiro.")
    # Em um ambiente real, isso poderia impedir o app de iniciar.
    # Por simplicidade, deixamos as variáveis como None.
    COLUMN_INFO, NUMERIC_IMPUTER, CATEGORICAL_IMPUTER, SCALER, ENCODER_OHE, KMEANS_MODEL, MODELS = [None] * 7

# ===================================================================
# PIPELINE DE PREVISÃO
# ===================================================================
def prediction_pipeline(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aplica todo o pipeline de pré-processamento do notebook em novos dados."""
    if not COLUMN_INFO:
        raise HTTPException(status_code=500, detail="O modelo não está carregado.")

    df = pd.DataFrame(rows)
    df.replace("N/A", np.nan, inplace=True)

    # Garante que todas as colunas esperadas na fase de imputação existam
    for col in COLUMN_INFO['all_features_imputed']:
        if col not in df.columns:
            df[col] = np.nan

    # 1. Imputação
    df[COLUMN_INFO['numerical_features']] = NUMERIC_IMPUTER.transform(df[COLUMN_INFO['numerical_features']])
    df[COLUMN_INFO['categorical_features']] = CATEGORICAL_IMPUTER.transform(df[COLUMN_INFO['categorical_features']])

    # 2. Tratamento de valores fora do intervalo
    imputer_most_frequent = SimpleImputer(strategy='most_frequent')
    for col, (lower, upper) in COLUMN_INFO['valid_ranges'].items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce')
            mask = (s < lower) | (s > upper)
            if mask.any():
                df.loc[mask, col] = np.nan
                df[[col]] = imputer_most_frequent.fit_transform(df[[col]])
    
    # 3. Tratamento de valores negativos
    for col in COLUMN_INFO['numerical_features']:
        s = pd.to_numeric(df[col], errors='coerce')
        mask = s < 0
        if mask.any():
            # Usa a mediana do imputer numérico como fallback
            median_val = pd.DataFrame(NUMERIC_IMPUTER.transform(df[COLUMN_INFO['numerical_features']]), columns=COLUMN_INFO['numerical_features'])[col].median()
            df.loc[mask, col] = median_val

    # 4. Scaling
    df[COLUMN_INFO['numerical_features']] = SCALER.transform(df[COLUMN_INFO['numerical_features']])
    
    # 5. One-Hot Encoding
    ohe_cols = COLUMN_INFO['ohe_cols']
    encoded_data = ENCODER_OHE.transform(df[ohe_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=ENCODER_OHE.get_feature_names_out(ohe_cols), index=df.index)
    
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    # Garante que todas as colunas finais do modelo existam na ordem correta
    final_cols = COLUMN_INFO['final_model_columns']
    for col in final_cols:
        if col not in df.columns:
            df[col] = 0
            
    return df[final_cols]

# ===================================================================
# ENDPOINTS DA API
# ===================================================================
def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": all(v is not None for v in [COLUMN_INFO, MODELS])}

@app.get("/predict/schema")
def predict_schema():
    if not COLUMN_INFO:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")
    # Retorna o schema inicial (colunas antes do OHE)
    initial_cols = set(COLUMN_INFO['numerical_features']) | set(COLUMN_INFO['categorical_features'])
    return sorted(list(initial_cols))

@app.post("/predict")
def predict(
    payload: Dict[str, List[Dict[str, Any]]] = Body(...),
    x_api_key: str = Header(None, alias="X-API-KEY")
):
    verify_api_key(x_api_key)
    rows = payload.get("data")
    if not rows:
        raise HTTPException(status_code=422, detail="Payload inválido. Esperado: {'data': [...]}.")

    # Pipeline de pré-processamento completo
    X_processed = prediction_pipeline(rows)

    # Previsão de Cluster
    try:
        cluster_preds = KMEANS_MODEL.predict(X_processed).tolist()
    except Exception:
        cluster_preds = [None] * len(rows)

    # Previsão dos Targets
    predictions = {
        'cluster': cluster_preds,
        'target1': [None] * len(rows),
        'target2': [None] * len(rows),
        'target3': [None] * len(rows),
    }

    for target_name, model in MODELS.items():
        try:
            selected_features = COLUMN_INFO['selected_features_by_target'][target_name]
            preds = model.predict(X_processed[selected_features]).tolist()
            predictions[target_name.lower()] = preds
        except Exception as e:
            print(f"Erro ao prever {target_name}: {e}")

    return {"predictions": predictions, "count": len(rows)}

# Outros endpoints podem ser adaptados de forma similar se necessário
@app.get("/overview")
def overview():
    if not COLUMN_INFO:
        return {"n_expected_features": 0}
    return {"n_expected_features": len(COLUMN_INFO['final_model_columns'])}

@app.get("/clusters/profile")
def clusters_profile():
    # Este endpoint depende de um arquivo CSV que não é mais gerado pelo novo script.
    # Retornando uma mensagem informativa.
    return {"message": "Endpoint de perfil de cluster não implementado para este modelo."}