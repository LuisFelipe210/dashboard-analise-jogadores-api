# backend/main.py
from __future__ import annotations
import os
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Body, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sklearn.impute import SimpleImputer  # necessário

warnings.filterwarnings("ignore", category=UserWarning)

# ===================================================================
# CONFIGURAÇÃO E CARREGAMENTO DE ARTEFATOS
# ===================================================================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
API_KEY = os.getenv("API_KEY", "fallback_key_if_not_set")

app = FastAPI(title="API de Previsão de Performance de Jogadores", version="2.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def _safe_load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Artefato essencial não encontrado: {path}")
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        return joblib.load(path)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Falha ao carregar {path.name}: {e}. "
            "Verifique se o ambiente contém as libs do treinamento (ex.: scikit-learn)."
        ) from e

# Carrega artefatos no import (mantendo seu padrão atual)
try:
    print("Carregando artefatos do modelo...")
    COLUMN_INFO = _safe_load(ARTIFACTS_DIR / "column_info.json")
    NUMERIC_IMPUTER = _safe_load(ARTIFACTS_DIR / "numeric_imputer.joblib")
    CATEGORICAL_IMPUTER = _safe_load(ARTIFACTS_DIR / "categorical_imputer.joblib")
    SCALER = _safe_load(ARTIFACTS_DIR / "scaler.joblib")
    ENCODER_OHE = _safe_load(ARTIFACTS_DIR / "encoder_ohe.joblib")
    KMEANS_MODEL = _safe_load(ARTIFACTS_DIR / "kmeans_model.joblib")  # opcional

    MODELS = {
        "Target1": _safe_load(ARTIFACTS_DIR / "best_model_Target1.joblib"),
        "Target2": _safe_load(ARTIFACTS_DIR / "best_model_Target2.joblib"),
        "Target3": _safe_load(ARTIFACTS_DIR / "best_model_Target3.joblib"),
    }
    print("Artefatos carregados com sucesso.")
except FileNotFoundError as e:
    print(f"Erro Crítico: {e}. Execute o script de treinamento primeiro.")
    COLUMN_INFO = NUMERIC_IMPUTER = CATEGORICAL_IMPUTER = SCALER = ENCODER_OHE = KMEANS_MODEL = None
    MODELS = {}

# ===================================================================
# HELPERS
# ===================================================================

def _require_model_loaded():
    if not COLUMN_INFO or not MODELS or any(v is None for v in [NUMERIC_IMPUTER, CATEGORICAL_IMPUTER, SCALER, ENCODER_OHE]):
        raise HTTPException(status_code=500, detail="Modelo não carregado/artefatos ausentes.")

def _to_percent(values: List[float] | np.ndarray) -> np.ndarray:
    """Converte predições para porcentagem. Se o maior valor <= 1, assume 0-1 e multiplica por 100."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    maxv = np.nanmax(arr)
    if np.isnan(maxv):
        return arr
    return arr * 100.0 if maxv <= 1.0 else arr

def _bucket_counts(percs: np.ndarray, low: float = 30.0, high: float = 60.0) -> Dict[str, int]:
    """Conta <low, [low, high], >high. Inclusivo nas bordas do intervalo médio."""
    percs = np.asarray(percs, dtype=float)
    lt = int(np.sum(percs < low))
    mid = int(np.sum((percs >= low) & (percs <= high)))
    gt = int(np.sum(percs > high))
    return {"<30": lt, "30-60": mid, ">60": gt}

def _extract_rows(payload: Any) -> List[Dict[str, Any]]:
    """Aceita {'data': [...]} ou um array JSON puro [...]."""
    if isinstance(payload, dict) and "data" in payload:
        rows = payload["data"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise HTTPException(
            status_code=422,
            detail="Payload inválido. Use {'data': [...]} ou um array JSON."
        )
    if not isinstance(rows, list) or len(rows) == 0:
        raise HTTPException(status_code=422, detail="Nenhuma linha enviada para previsão.")
    return rows

# ========================= RADAR HELPERS (NOVO) =========================

def _ensure_all_imputed_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in COLUMN_INFO["all_features_imputed"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def _scale_0_100(val: float, rng: Tuple[float, float]) -> float:
    lo, hi = float(rng[0]), float(rng[1])
    if hi <= lo:
        return 50.0
    x = (float(val) - lo) / (hi - lo)
    return float(max(0.0, min(1.0, x)) * 100.0)

def _choose_radar_labels_for_target(target: str, k: int = 5) -> List[str]:
    # 1) preferir os top-5 salvos pelo treino
    top5_map = COLUMN_INFO.get("radar_top5_by_target", {})
    chosen = list(top5_map.get(target, []))
    if chosen:
        return chosen[:k]
    # 2) fallback: usar selected_features_by_target filtrando por valid_ranges
    selected = COLUMN_INFO["selected_features_by_target"].get(target, [])
    valid = set(COLUMN_INFO.get("valid_ranges", {}).keys())
    filtered = [f for f in selected if f in valid]
    if len(filtered) >= k:
        return filtered[:k]
    # 3) último fallback fixo (se nada acima existir)
    fallback = ["P01", "P02", "P03", "Acordar", "QtdHorasDormi"]
    return fallback[:k]

def _impute_original_and_process_one(row: Dict[str, Any]) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Retorna:
      - orig_num_imputed: pd.Series com as features numéricas no ESPAÇO ORIGINAL (antes do scaler) imputadas
      - orig_cat_like_num: pd.Series para categóricas que também têm faixas numéricas em valid_ranges (ex.: Likert)
      - X_processed: DataFrame 1xN no mesmo espaço final dos modelos/cluster (pós-scaler + OHE + ordem)
    """
    df = pd.DataFrame([row]).replace("N/A", np.nan)
    df = _ensure_all_imputed_columns(df)

    # 1) Imputação por tipo (mesmo do pipeline)
    df[COLUMN_INFO["numerical_features"]] = NUMERIC_IMPUTER.transform(
        df[COLUMN_INFO["numerical_features"]]
    )
    df[COLUMN_INFO["categorical_features"]] = CATEGORICAL_IMPUTER.transform(
        df[COLUMN_INFO["categorical_features"]]
    )

    # 2) Tratamento de faixa (clip + fillna com mediana do intervalo)
    for col, (lower, upper) in COLUMN_INFO["valid_ranges"].items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.clip(lower=lower, upper=upper)
            mid = (float(lower) + float(upper)) / 2.0
            s = s.fillna(mid)
            df[col] = s

    # 3) Negativos nas numéricas -> mediana (no espaço ORIGINAL)
    for col in COLUMN_INFO["numerical_features"]:
        s = pd.to_numeric(df[col], errors="coerce")
        mask = s < 0
        if mask.any():
            med = np.nanmedian(s[s >= 0])
            if np.isnan(med):
                med = 0.0
            df.loc[mask, col] = med

    # --- snapshot ORIGINAL imputado (antes do scaler) ---
    orig_num_imputed = df[COLUMN_INFO["numerical_features"]].iloc[0].copy()

    cats_with_range = [c for c in COLUMN_INFO["categorical_features"]
                       if c in COLUMN_INFO.get("valid_ranges", {}) and c in df.columns]
    if cats_with_range:
        # garantir numéricas
        orig_cat_like_num = df[cats_with_range].apply(pd.to_numeric, errors="coerce").iloc[0].copy()
    else:
        orig_cat_like_num = pd.Series(dtype=float)

    # 4) Scaler nas numéricas (para o espaço dos modelos/cluster)
    df[COLUMN_INFO["numerical_features"]] = SCALER.transform(
        df[COLUMN_INFO["numerical_features"]]
    )

    # 5) OHE e ordem final (igual ao pipeline)
    ohe_cols = COLUMN_INFO["ohe_cols"]
    enc = ENCODER_OHE.transform(df[ohe_cols])
    enc_df = pd.DataFrame(
        enc,
        columns=ENCODER_OHE.get_feature_names_out(ohe_cols),
        index=df.index
    )
    df2 = pd.concat([df.drop(columns=ohe_cols), enc_df], axis=1)

    final_cols = COLUMN_INFO["final_model_columns"]
    for col in final_cols:
        if col not in df2.columns:
            df2[col] = 0
    X_processed = df2[final_cols]

    return orig_num_imputed, orig_cat_like_num, X_processed


def _cluster_center_num_orig(cluster_id: int) -> Dict[str, float]:
    """
    Traz o centro do cluster para a escala ORIGINAL das features NUMÉRICAS.
    Retorna {num_feature: valor_no_espaco_original}.
    """
    try:
        if KMEANS_MODEL is None:
            return {}
        center = np.array(KMEANS_MODEL.cluster_centers_[cluster_id], dtype=float)

        final_cols = COLUMN_INFO["final_model_columns"]
        num_cols = COLUMN_INFO["numerical_features"]
        num_idx = [final_cols.index(c) for c in num_cols]

        center_num_scaled = center[num_idx].reshape(1, -1)
        center_num_orig = SCALER.inverse_transform(center_num_scaled).flatten()
        return {c: float(v) for c, v in zip(num_cols, center_num_orig)}
    except Exception:
        return {}

# ===================================================================
# PIPELINE DE PREVISÃO
# ===================================================================
def prediction_pipeline(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aplica o pipeline de pré-processamento e retorna X na ordem final esperada."""
    _require_model_loaded()

    df = pd.DataFrame(rows)
    df.replace("N/A", np.nan, inplace=True)

    # Garante todas as colunas esperadas na fase de imputação
    for col in COLUMN_INFO["all_features_imputed"]:
        if col not in df.columns:
            df[col] = np.nan

    # 1) Imputação
    df[COLUMN_INFO["numerical_features"]] = NUMERIC_IMPUTER.transform(df[COLUMN_INFO["numerical_features"]])
    df[COLUMN_INFO["categorical_features"]] = CATEGORICAL_IMPUTER.transform(df[COLUMN_INFO["categorical_features"]])

    # 2) Tratamento de ranges (outliers -> moda)
    imputer_most_frequent = SimpleImputer(strategy="most_frequent")
    for col, (lower, upper) in COLUMN_INFO["valid_ranges"].items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            mask = (s < lower) | (s > upper)
            if mask.any():
                df.loc[mask, col] = np.nan
                df[[col]] = imputer_most_frequent.fit_transform(df[[col]])

    # 3) Negativos (usa mediana após imputação numérica)
    for col in COLUMN_INFO["numerical_features"]:
        s = pd.to_numeric(df[col], errors="coerce")
        mask = s < 0
        if mask.any():
            median_val = np.nanmedian(pd.to_numeric(df[col], errors="coerce"))
            if np.isnan(median_val):
                median_val = 0.0
            df.loc[mask, col] = median_val

    # 4) Scaling
    df[COLUMN_INFO["numerical_features"]] = SCALER.transform(df[COLUMN_INFO["numerical_features"]])

    # 5) One-Hot Encoding
    ohe_cols = COLUMN_INFO["ohe_cols"]
    encoded_data = ENCODER_OHE.transform(df[ohe_cols])
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=ENCODER_OHE.get_feature_names_out(ohe_cols),
        index=df.index
    )
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    # 6) Ordem final
    final_cols = COLUMN_INFO["final_model_columns"]
    for col in final_cols:
        if col not in df.columns:
            df[col] = 0
    return df[final_cols]

def run_predictions(X_processed: pd.DataFrame) -> Dict[str, List[Optional[float]]]:
    """Roda as predições de cluster e dos alvos. Retorna dict com listas."""
    preds = {
        "cluster": [],
        "target1": [],
        "target2": [],
        "target3": [],
    }

    # Cluster (se existir)
    try:
        if KMEANS_MODEL is not None:
            preds["cluster"] = KMEANS_MODEL.predict(X_processed).tolist()
        else:
            preds["cluster"] = [None] * len(X_processed)
    except Exception:
        preds["cluster"] = [None] * len(X_processed)

    # Targets
    for target_name, model in MODELS.items():
        key = target_name.lower()  # "target1", "target2", "target3"
        try:
            selected = COLUMN_INFO["selected_features_by_target"][target_name]
            yhat = model.predict(X_processed[selected]).tolist()
            preds[key] = yhat
        except Exception as e:
            print(f"Erro ao prever {target_name}: {e}")
            preds[key] = [None] * len(X_processed)

    return preds

def summarize_targets(preds: Dict[str, List[Optional[float]]], thresholds: Tuple[float, float] = (30.0, 60.0)) -> Dict[str, Any]:
    """Gera contagens por faixas para Target1/2/3 em porcentagem."""
    low, high = thresholds
    out: Dict[str, Any] = {"thresholds": {"low": low, "high": high, "unit": "percent"}}

    for t in ("target1", "target2", "target3"):
        vals = np.array([v for v in preds.get(t, []) if v is not None], dtype=float)
        percs = _to_percent(vals)
        out[t.capitalize()] = _bucket_counts(percs, low=low, high=high)
    return out

# ===================================================================
# ENDPOINTS
# ===================================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bool(COLUMN_INFO and MODELS),
        "has_artifacts": all(v is not None for v in [COLUMN_INFO, NUMERIC_IMPUTER, CATEGORICAL_IMPUTER, SCALER, ENCODER_OHE]),
    }

@app.get("/predict/schema")
def predict_schema():
    _require_model_loaded()
    initial_cols = set(COLUMN_INFO["numerical_features"]) | set(COLUMN_INFO["categorical_features"])
    return sorted(list(initial_cols))

@app.post("/predict")
def predict(
    payload: Any = Body(...),
):
    rows = _extract_rows(payload)               # <-- garante 'rows'
    X_processed = prediction_pipeline(rows)
    predictions = run_predictions(X_processed)
    bucket_summary = summarize_targets(predictions, thresholds=(30.0, 60.0))
    return {"predictions": predictions, "bucket_summary": bucket_summary, "count": len(rows)}

@app.post("/predict/file")
async def predict_file(
    file: UploadFile = File(...),
):
    """
    Faz upload de um CSV e retorna as predições + resumo por faixas.
    Requisitos:
      - O CSV deve conter as colunas do /predict/schema (as que o treinamento espera antes do OHE).
    """
    _require_model_loaded()

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Envie um arquivo .csv")

    content = await file.read()
    try:
        # Autodetecta separador (sep=None) com engine='python'
        df = pd.read_csv(BytesIO(content), sep=None, engine="python")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler CSV: {e}")

    rows = df.to_dict(orient="records")
    X_processed = prediction_pipeline(rows)
    predictions = run_predictions(X_processed)
    bucket_summary = summarize_targets(predictions, thresholds=(30.0, 60.0))

    return {
        "filename": file.filename,
        "predictions": predictions,
        "bucket_summary": bucket_summary,
        "count": len(rows),
    }

@app.post("/targets/buckets")
def targets_buckets(
    payload: Dict[str, Any] = Body(...),
):
    """
    Gera apenas o resumo por faixas. Aceita:
      - {'data': [...]}  -> roda pipeline + modelos e sumariza
      - {'predictions': {'target1': [...], 'target2': [...], 'target3': [...]}} -> apenas sumariza
      - opcional: 'low', 'high' para customizar os limites (padrão 30/60)
    """
    _require_model_loaded()

    low = float(payload.get("low", 30.0))
    high = float(payload.get("high", 60.0))

    if "predictions" in payload:
        preds = payload["predictions"]
        summary = summarize_targets(preds, thresholds=(low, high))
        return {"bucket_summary": summary}

    rows = payload.get("data")
    if not rows:
        raise HTTPException(status_code=422, detail="Envie 'data' ou 'predictions' no corpo.")
    X_processed = prediction_pipeline(rows)
    predictions = run_predictions(X_processed)
    summary = summarize_targets(predictions, thresholds=(low, high))
    return {"bucket_summary": summary, "count": len(rows)}

@app.get("/overview")
def overview():
    if not COLUMN_INFO:
        return {"n_expected_features": 0}
    return {"n_expected_features": len(COLUMN_INFO["final_model_columns"])}

@app.get("/clusters/profile")
def clusters_profile():
    return {"message": "Este endpoint foi substituído pelo agrupamento por faixas via /predict e /targets/buckets."}

# =========================== RADAR ENDPOINT (NOVO) ===========================
@app.post("/radar")
def radar_profile(
    payload: Dict[str, Any] = Body(...),
    target: str = "Target1",
):
    """
    Monta o perfil de radar para um jogador (player) e um target específico.
    Entrada:
      {
        "player": { "<col>": <valor>, ... }
      }
    Query param:
      - target=Target1|Target2|Target3  (padrão Target1)
    Saída:
      {
        "target": "Target1",
        "cluster_id": 0|1|...|null,
        "labels": [f1..f5],
        "player_profile": [..0-100..],
        "cluster_average_profile": [..0-100..],
        "scale": "range_0_100",
        "source": "radar"
      }
    """
    _require_model_loaded()
    if target not in ("Target1", "Target2", "Target3"):
        raise HTTPException(status_code=422, detail="Parâmetro 'target' deve ser Target1, Target2 ou Target3.")

    # aceita {'player': {...}} ou {'data': [ {...} ]} (usa a primeira linha)
    if "player" in payload and isinstance(payload["player"], dict):
        row = payload["player"]
    elif "data" in payload and isinstance(payload["data"], list) and payload["data"]:
        row = payload["data"][0]
    else:
        raise HTTPException(status_code=422, detail="Envie {'player': {...}} ou {'data': [ {...} ]}")

    # 1) imputar (original) e processar (final) para cluster/pred
    orig_num, orig_cat_like_num, Xp = _impute_original_and_process_one(row)

    # 2) escolher eixos (Top-5 por target vindos do treino)
    labels = _choose_radar_labels_for_target(target, k=5)

    # 3) valores do jogador em escala ORIGINAL (numéricas + categóricas com faixa)
    vals_player = []
    vr = COLUMN_INFO.get("valid_ranges", {})
    for f in labels:
        if f in orig_num.index:
            vals_player.append(float(orig_num[f]))
        elif f in orig_cat_like_num.index:
            vals_player.append(float(orig_cat_like_num[f]))
        else:
            vals_player.append(np.nan)

    player_profile = [_scale_0_100(v, vr.get(f, (0.0, 1.0))) if not np.isnan(v) else 50.0 for f, v in zip(labels, vals_player)]

    # 4) cluster id e centro aproximado (médias) no espaço ORIGINAL das numéricas
    cluster_id = None
    center_num_map: Dict[str, float] = {}
    try:
        if KMEANS_MODEL is not None:
            cluster_id = int(KMEANS_MODEL.predict(Xp)[0])
            center_num_map = _cluster_center_num_orig(cluster_id)
    except Exception:
        center_num_map = {}

    # 5) cluster_average_profile alinhado aos labels (fallback: usa valor do jogador)
    cluster_vals = []
    for f in labels:
        if f in center_num_map:
            cluster_vals.append(center_num_map[f])
        elif f in orig_cat_like_num.index:
            # sem média categórica por cluster -> usa o próprio do jogador (até salvarmos médias por cluster)
            cluster_vals.append(float(orig_cat_like_num[f]))
        elif f in orig_num.index:
            cluster_vals.append(float(orig_num[f]))
        else:
            cluster_vals.append(np.nan)

    cluster_average_profile = [_scale_0_100(v, vr.get(f, (0.0, 1.0))) if not np.isnan(v) else 50.0 for f, v in zip(labels, cluster_vals)]

    return {
        "target": target,
        "cluster_id": cluster_id,
        "labels": labels,
        "player_profile": player_profile,
        "cluster_average_profile": cluster_average_profile,
        "scale": "range_0_100",
        "source": "radar",
    }
