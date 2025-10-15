# backend/main.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Body, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------------------------
# Configuração básica
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "model_artifacts"

API_KEY = os.getenv("API_KEY", "")  # defina no ambiente: $env:API_KEY="sua-chave" (Windows PowerShell)

# CORS liberado para desenvolvimento
app = FastAPI(title="Jogadores API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key", "X-API-KEY"],
)

# -------------------------------------------------------------------
# Carregamento de artefatos
# -------------------------------------------------------------------
def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_joblib_load(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)

# Modelos de regressão
MODEL_T1 = _safe_joblib_load(ARTIFACTS_DIR / "stacking_model_target1.joblib")
MODEL_T2 = _safe_joblib_load(ARTIFACTS_DIR / "stacking_model_target2.joblib")
MODEL_T3 = _safe_joblib_load(ARTIFACTS_DIR / "stacking_model_target3.joblib")

# Pipeline de cluster
CLUSTER_PIPE = _safe_joblib_load(ARTIFACTS_DIR / "cluster_pipeline.joblib")

# Config do treino
MODEL_CONFIG: Dict[str, Any] = {}
config_path = ARTIFACTS_DIR / "model_config.json"
if config_path.exists():
    MODEL_CONFIG = _load_json(config_path)

EXPECTED: List[str] = []
expected_path = ARTIFACTS_DIR / "expected_features.json"
if expected_path.exists():
    EXPECTED = _load_json(expected_path)
else:
    # fallback para compatibilidade: tentar pegar das colunas finais salvas no config
    EXPECTED = MODEL_CONFIG.get("final_model_columns", [])

NUM_FOR_CLUSTER: List[str] = MODEL_CONFIG.get("numerical_features_for_clustering", [])
CATEGORICAL_FEATURES: List[str] = MODEL_CONFIG.get("categorical_features", [])  # pode não ser usado

# perfil de clusters (opcional)
PERFIL_CLUSTERS_CSV = ARTIFACTS_DIR / "perfil_clusters.csv"
PERFIL_CLUSTERS_DF = None
if PERFIL_CLUSTERS_CSV.exists():
    try:
        PERFIL_CLUSTERS_DF = pd.read_csv(PERFIL_CLUSTERS_CSV)
    except Exception:
        PERFIL_CLUSTERS_DF = None

# -------------------------------------------------------------------
# Utilidades de coerção / preparação
# -------------------------------------------------------------------
def _coerce_numeric(val: Any) -> float:
    """
    Converte strings numéricas de forma robusta,
    trocando vírgula por ponto e tratando lixo comum.
    """
    if val is None:
        return 0.0
    if isinstance(val, (int, float, np.number)):
        if pd.isna(val):
            return 0.0
        return float(val)
    s = str(val).strip().replace('"', '')
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return 0.0
    # tentativa simples de normalização de separador decimal
    try:
        # se tiver uma vírgula e mais de um ponto, troca pontos de milhar
        if s.count(",") == 1 and s.count(".") > 1:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", ".")
        return float(s)
    except Exception:
        return 0.0

def _rows_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Recebe uma lista de dicionários e devolve um DataFrame com:
    - colunas padronizadas para EXPECTED
    - ordem de colunas == EXPECTED
    - numéricos coeridos (todas as colunas de EXPECTED são numéricas no modelo final)
    """
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=EXPECTED)

    out = []
    for r in rows:
        dst = {}
        for col in EXPECTED:
            val = r.get(col, 0)
            dst[col] = _coerce_numeric(val)
        out.append(dst)

    df = pd.DataFrame(out, columns=EXPECTED)
    # garante tipos numéricos (evita dtype=object)
    for c in df.columns:
        if df[c].dtype == "bool":
            df[c] = df[c].astype(int)
        elif not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = df[c].astype(float)
    return df

def _predict_targets(dfX: pd.DataFrame) -> Dict[str, List[Optional[float]]]:
    preds = {"target1": [], "target2": [], "target3": []}
    if MODEL_T1 is None or MODEL_T2 is None or MODEL_T3 is None:
        # artefatos ausentes
        return preds

    X = dfX.copy()
    # garante ordem e colunas
    missing = [c for c in EXPECTED if c not in X.columns]
    for m in missing:
        X[m] = 0.0
    X = X[EXPECTED]

    # predições
    try:
        p1 = MODEL_T1.predict(X).tolist()
    except Exception:
        p1 = [None] * len(X)
    try:
        p2 = MODEL_T2.predict(X).tolist()
    except Exception:
        p2 = [None] * len(X)
    try:
        p3 = MODEL_T3.predict(X).tolist()
    except Exception:
        p3 = [None] * len(X)

    preds["target1"] = p1
    preds["target2"] = p2
    preds["target3"] = p3
    return preds

def _predict_cluster(raw_rows: List[Dict[str, Any]]) -> Optional[List[int]]:
    """
    Prediz cluster usando CLUSTER_PIPE com as features numéricas do treino (NUM_FOR_CLUSTER).
    Se não houver pipeline ou colunas, retorna None.
    """
    if CLUSTER_PIPE is None or not NUM_FOR_CLUSTER:
        return None

    out = []
    for r in raw_rows:
        d = {}
        for c in NUM_FOR_CLUSTER:
            d[c] = _coerce_numeric(r.get(c, 0))
        out.append(d)
    df = pd.DataFrame(out, columns=NUM_FOR_CLUSTER)
    if df.empty:
        return None

    try:
        labels = CLUSTER_PIPE.predict(df).tolist()
        return labels
    except Exception:
        return None

def _inject_cluster_ohe(dfX: pd.DataFrame, clusters: Optional[List[int]]) -> pd.DataFrame:
    """
    Se o treinamento criou dummies de cluster (ex.: Cluster_0, Cluster_1),
    injeta as colunas correspondentes com base na predição de cluster.
    """
    # quais colunas de cluster eram esperadas no treino?
    expected_cluster_cols = [col for col in EXPECTED if col.startswith("Cluster_")]

    if not expected_cluster_cols:
        # treino não usou dummies de cluster; nada a fazer
        return dfX

    if clusters is None:
        # sem predição de cluster: apenas garanta colunas com zeros
        for c in expected_cluster_cols:
            if c not in dfX.columns:
                dfX[c] = 0
        return dfX

    # gera dummies a partir dos rótulos previstos
    cluster_series = pd.Series(clusters, name="cluster")
    dummies = pd.get_dummies(cluster_series, prefix="Cluster")

    # garanta que todas as colunas esperadas existam; se faltar, cria com 0
    for col in expected_cluster_cols:
        if col not in dummies.columns:
            dummies[col] = 0

    # mantenha apenas as colunas esperadas e alinhe índices
    dummies = dummies[expected_cluster_cols]
    dummies.index = dfX.index

    # injete no dataframe base (não sobrescreva colunas já existentes)
    for col in expected_cluster_cols:
        if col not in dfX.columns:
            dfX[col] = dummies[col].astype(int)

    return dfX

# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/predict/schema")
async def predict_schema():
    """
    Retorna um ARRAY de objetos para montar o formulário do frontend:
      [{ name, type, label, default }]
    Todas as colunas de EXPECTED são numéricas no modelo final -> type="number", default=0.
    """
    fields = []
    for col in EXPECTED:
        fields.append({"name": col, "type": "number", "label": col, "default": 0})
    return fields

@app.post("/predict")
async def predict_endpoint(
    payload: Any = Body(...),
    x_api_key_1: Optional[str] = Header(None, alias="X-API-Key"),
    x_api_key_2: Optional[str] = Header(None, alias="X-API-KEY"),
):
    # auth (tolerante a ambos cabeçalhos)
    given_key = x_api_key_1 or x_api_key_2 or ""
    if API_KEY and given_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # aceita tanto uma lista "crua" quanto { "data": [...] }
    rows: Union[List[Dict[str, Any]], None] = None
    if isinstance(payload, dict) and "data" in payload:
        rows = payload["data"]
    elif isinstance(payload, list):
        rows = payload

    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=422, detail="O corpo deve ser uma lista de objetos ou {data: [...]}.")

    # 1) Predição de cluster (antes de montar DF final) — usa NUM_FOR_CLUSTER
    cluster_labels = _predict_cluster(rows)

    # 2) Monta DF padronizado nas EXPECTED
    dfX = _rows_to_dataframe(rows)

    # 3) Injeta dummies de cluster se o treino gerou Cluster_*
    dfX = _inject_cluster_ohe(dfX, cluster_labels)

    # 4) Garante ordem final exata
    missing = [c for c in EXPECTED if c not in dfX.columns]
    for m in missing:
        dfX[m] = 0
    dfX = dfX[EXPECTED]

    # 5) Predições
    preds = _predict_targets(dfX)

    return {
        "predictions": {
            "target1": preds["target1"],
            "target2": preds["target2"],
            "target3": preds["target3"],
            "cluster": cluster_labels,
        },
        "count": len(rows),
    }

# --- NOVO ENDPOINT: perfil de clusters ---------------------------------------
@app.get("/clusters/profile")
async def clusters_profile(limit: int = Query(5, ge=1, le=30)):
    """
    Devolve perfis médios por cluster (com base em backend/model_artifacts/perfil_clusters.csv)
    e as TOP diferenças absolutas entre clusters.

    Retorno:
    {
      "clusters": [0,1],
      "n_features": 123,
      "means": { "0": {"featA": 1.2, ...}, "1": {...} },
      "top_diffs": [
         {"feature": "featX", "diff_abs": 12.3, "by_cluster": {"0": 9.1, "1": 21.4}},
         ...
      ],
      "used_features": ["featX","featY",...],     # features usadas no radar (top 'limit')
    }
    """
    if PERFIL_CLUSTERS_DF is None or PERFIL_CLUSTERS_DF.empty:
        raise HTTPException(status_code=404, detail="perfil_clusters.csv não encontrado ou vazio.")

    # Faz uma cópia defensiva
    df = PERFIL_CLUSTERS_DF.copy()

    # Tenta configurar o índice como 'cluster'
    if "cluster" in df.columns:
        df = df.set_index("cluster")
    else:
        # Se foi salvo com índice sem nome, a primeira coluna costuma ser o índice
        first_col = df.columns[0]
        # Heurística: se a primeira coluna for de inteiros 0/1, assuma que é o índice
        if pd.api.types.is_integer_dtype(df[first_col]) and set(df[first_col].unique()).issubset({0, 1}):
            df = df.set_index(first_col)
        # Se nada disso der certo, seguimos assim mesmo (clusters serão 0..N-1)
        # e torcemos para a importação já ter vindo com índice ok.

    # Mantém apenas numéricas
    df = df.select_dtypes(include=[np.number])

    if df.empty or df.shape[0] < 1:
        raise HTTPException(status_code=400, detail="Perfil de clusters sem colunas numéricas.")

    # Lista dos clusters (índice)
    clusters_idx = [int(i) for i in df.index.tolist()]
    # Médias por cluster (já são as médias)
    means = {str(int(i)): {str(c): float(df.loc[i, c]) for c in df.columns} for i in df.index}

    # Diferenças absolutas entre clusters (para k>2 usamos (max-min) por coluna)
    diffs = []
    for col in df.columns:
        col_vals = df[col].to_dict()  # chave: cluster idx
        # diff absoluto entre maior e menor média daquele feature
        max_v = float(np.nanmax(list(col_vals.values())))
        min_v = float(np.nanmin(list(col_vals.values())))
        diff_abs = abs(max_v - min_v)
        diffs.append({
            "feature": str(col),
            "diff_abs": float(diff_abs),
            "by_cluster": {str(int(k)): float(v) for k, v in col_vals.items()}
        })

    # Ordena por diferença e pega top 'limit'
    diffs.sort(key=lambda x: x["diff_abs"], reverse=True)
    top_diffs = diffs[:limit]
    used_features = [d["feature"] for d in top_diffs]

    return {
        "clusters": clusters_idx,
        "n_features": int(df.shape[1]),
        "means": means,
        "top_diffs": top_diffs,
        "used_features": used_features,
    }


@app.get("/feature_importance")
async def feature_importance():
    """
    Retorna importâncias se o modelo as expõe.
    Como temos um StackingRegressor, é comum não termos importâncias globais.
    Devolvemos um fallback simples com as N primeiras colunas, apenas para UI.
    """
    top = EXPECTED[:20]
    return {
        "available": False,
        "note": "StackingRegressor não expõe importâncias diretas; exibindo colunas principais como fallback.",
        "top_features": top,
    }

@app.get("/overview")
async def overview():
    """
    Estatísticas de alto nível para cards / gráficos.
    Se houver perfil de clusters, retorna agregados simples.
    """
    result = {
        "n_expected_features": len(EXPECTED),
        "has_cluster_pipeline": CLUSTER_PIPE is not None,
        "numerical_for_cluster": len(NUM_FOR_CLUSTER),
        "categorical_features": len(MODEL_CONFIG.get("categorical_features", [])),
    }
    if PERFIL_CLUSTERS_DF is not None and "cluster" in PERFIL_CLUSTERS_DF.columns:
        counts = PERFIL_CLUSTERS_DF["cluster"].value_counts().to_dict()
        result["cluster_counts"] = {str(k): int(v) for k, v in counts.items()}
    return result
