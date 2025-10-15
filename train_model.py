# train_model.py
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, Lasso
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.base import clone

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
DATA_CSV = "JogadoresV1.csv"  # ajuste se necessário
BACKEND_DIR = Path("backend")
ARTIFACTS_DIR = BACKEND_DIR / "model_artifacts"
PUBLIC_DIR = Path("frontend") / "public"  # onde o Vite serve estáticos
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["Target1", "Target2", "Target3"]

print("Iniciando o processo de treinamento...")

# ----------------------------------------------------------
# 1) Carregar e limpar dados
# ----------------------------------------------------------
print("1) Carregando dados...")
df = pd.read_csv(DATA_CSV, encoding="utf-8", na_values=['N/A', '', '#DIV/0!'])

df_processed = df.copy()
if "Código de Acesso" in df_processed.columns:
    codigo_acesso = df_processed["Código de Acesso"].copy()
else:
    codigo_acesso = pd.Series(index=df_processed.index, dtype=object)

# padroniza nomes
df_processed.columns = (
    df_processed.columns
    .str.strip()
    .str.replace(' ', '_', regex=False)
    .str.replace('[^a-zA-Z0-9_]', '', regex=True)
)

def force_clean_and_convert_string(series: pd.Series) -> pd.Series:
    series = (
        series.astype(str)
        .str.replace('"', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    series = series.replace(['-1.0', 'nan', 'N/A', 'NaN'], np.nan)
    return pd.to_numeric(series, errors='coerce')

# quais colunas são texto?
object_cols = df_processed.select_dtypes(include='object').columns.tolist()

# cores e familia F0207 NÃO viram número bruto
cols_categoricas_cor = [c for c in object_cols if c.startswith("Cor")]
cols_categoricas_f0207 = [c for c in object_cols if c.startswith("F0207")]
cols_nao_num = set(["Cdigo_de_Acesso"] + cols_categoricas_cor + cols_categoricas_f0207)

for col in object_cols:
    if col not in cols_nao_num:
        df_processed[col] = force_clean_and_convert_string(df_processed[col])

# drop de campos textuais de explicação
drop_cols = [
    'T1205Expl', 'T1199Expl', 'F0299__Explicao_Tempo', 'TempoTotalExpl',
    'PTempoTotalExpl', 'T1210Expl', 'T0499__Explicao_Tempo', 'DataHora_ltimo'
]
df_processed.drop(columns=drop_cols, errors='ignore', inplace=True)

# ----------------------------------------------------------
# 2) One-Hot das categóricas (cores + F0207)
# ----------------------------------------------------------
print("2) One-Hot Encoding das cores e F0207...")
categorical_cols = []
ohe_bases = []

color_cols = [c for c in df_processed.columns if c.startswith("Cor")]
if color_cols:
    ohe_color = pd.get_dummies(df_processed[color_cols].astype(str), prefix=color_cols, dummy_na=False)
    ohe_bases.append(ohe_color)
    categorical_cols.extend(ohe_color.columns.tolist())

f0207_cols = [c for c in df_processed.columns if c.startswith("F0207")]
if f0207_cols:
    ohe_f0207 = pd.get_dummies(df_processed[f0207_cols].astype(str), prefix=f0207_cols, dummy_na=False)
    ohe_bases.append(ohe_f0207)
    categorical_cols.extend(ohe_f0207.columns.tolist())

if ohe_bases:
    ohe_all = pd.concat(ohe_bases, axis=1)
    df_processed = pd.concat([df_processed.drop(columns=color_cols + f0207_cols, errors='ignore'), ohe_all], axis=1)

# ----------------------------------------------------------
# 3) Engenharia de features
# ----------------------------------------------------------
print("3) Engenharia de features...")

def add_safe_ratio(df, num_col, den_col, out_name):
    if num_col in df.columns and den_col in df.columns:
        df[out_name] = df[num_col] / df[den_col].replace(0, 1e-6)

Sono_col = next((c for c in df_processed.columns if 'QtdHorasSono' in c), None)
if Sono_col and 'Acordar' in df_processed.columns:
    max_acordar = df_processed['Acordar'].max()
    df_processed['Acordar_Invertido'] = max_acordar - df_processed['Acordar']
    df_processed['Indice_Sono_T1'] = df_processed[Sono_col] * df_processed['Acordar_Invertido']

if 'F1103' in df_processed.columns and 'F0713' in df_processed.columns:
    df_processed['F_Oposto_T3'] = df_processed['F1103'] - df_processed['F0713']

F07_cols = [c for c in df_processed.columns if c.startswith('F07') and len(c) == 5]
if F07_cols:
    df_processed['F07_Media'] = df_processed[F07_cols].mean(axis=1)

F11_cols = [c for c in df_processed.columns if c.startswith('F11') and len(c) == 5]
if F11_cols:
    df_processed['F11_Media'] = df_processed[F11_cols].mean(axis=1)

if 'P09' in df_processed.columns and 'T09' in df_processed.columns:
    add_safe_ratio(df_processed, 'P09', 'T09', 'Eficiencia_P09_T09')

p_cols = [c for c in df_processed.columns if c.startswith('P') and len(c) == 3]
t_cols = [c for c in df_processed.columns if c.startswith('T') and len(c) == 3]
if p_cols and t_cols:
    p_sum = df_processed[p_cols].sum(axis=1, skipna=True)
    t_sum = df_processed[t_cols].sum(axis=1, skipna=True).replace(0, 1e-6)
    df_processed['Eficiencia_Total'] = p_sum / t_sum

if 'F11_Media' in df_processed.columns and 'F07_Media' in df_processed.columns:
    df_processed['Gap_F11_F07'] = df_processed['F11_Media'] - df_processed['F07_Media']

if 'QtdHorasDormi' in df_processed.columns and Sono_col:
    df_processed['Sono_Ineficiencia'] = df_processed['QtdHorasDormi'] - df_processed[Sono_col]

if 'PTempoTotal' in df_processed.columns and 'TempoTotal' in df_processed.columns:
    add_safe_ratio(df_processed, 'PTempoTotal', 'TempoTotal', 'Indice_Final_PT_T')

all_p_cols = [c for c in df_processed.columns if c.startswith('P') and (len(c) == 3 or len(c) == 4)]
for p_col in all_p_cols:
    t_col = 'T' + p_col[1:]
    new_col = f'Eficiencia_{p_col}_{t_col}'
    if t_col in df_processed.columns and 'Expl' not in t_col and new_col not in df_processed.columns:
        add_safe_ratio(df_processed, p_col, t_col, new_col)

# limpeza final de resíduos
df_processed.drop(columns=[c for c in df_processed.columns if '_Limpo' in c or c.startswith('Soma_') or 'Acordar_Invertido' in c], errors='ignore', inplace=True)

# valores negativos -> NaN (serão imputados)
for col in df_processed.select_dtypes(include=np.number).columns:
    if col not in TARGETS:
        df_processed.loc[df_processed[col] < 0, col] = np.nan

# ----------------------------------------------------------
# 4) Imputação e clusterização
# ----------------------------------------------------------
print("4) Imputação & Clusterização...")
# mantém apenas linhas com targets definidos
df_processed = df_processed.dropna(subset=TARGETS)

# imputação simples
for col in df_processed.columns:
    if df_processed[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df_processed[col]):
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

# separa X completo (sem targets e sem Código)
X_full = df_processed.drop(columns=TARGETS + ['Cdigo_de_Acesso'], errors='ignore')
y = df_processed[TARGETS].copy()

# booleans -> int
for c in X_full.select_dtypes(include=['bool']).columns:
    X_full[c] = X_full[c].astype(int)

numerical_features_for_clustering = X_full.select_dtypes(include=np.number).columns.tolist()

cluster_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50, random_state=42)),
    ('kmeans', KMeans(n_clusters=2, random_state=42, n_init=10))
])
cluster_labels = cluster_pipeline.fit_predict(X_full[numerical_features_for_clustering])
df_processed['cluster'] = cluster_labels

# ----------------------------------------------------------
# 5) Preparação final (dummies de cluster, split e treino)
# ----------------------------------------------------------
print("5) Dummies de cluster e split...")
df_processed_ohe = pd.get_dummies(df_processed, columns=['cluster'], prefix='Cluster')

X_final = df_processed_ohe.drop(columns=TARGETS + ['Cdigo_de_Acesso'], errors='ignore')

for c in X_final.select_dtypes(include=['bool']).columns:
    X_final[c] = X_final[c].astype(int)

final_model_columns = X_final.columns.tolist()

X_train_val, X_test, y_train_val, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# modelos base
estimators = [
    ('rf', RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)),
    ('ridge', RidgeCV())
]
stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Lasso(alpha=0.001, random_state=42),
    cv=5
)

# gera CSV para o frontend com o conjunto de teste
print("6) Gerando CSV de teste para o frontend...")
df_for_front = df_processed.loc[y_test.index].copy()
if len(codigo_acesso) == len(df_processed):
    df_for_front['Código de Acesso'] = codigo_acesso.loc[y_test.index].values

for target in TARGETS:
    m_eval = clone(stacking)
    m_eval.fit(X_train, y_train[target])
    preds = m_eval.predict(X_test)
    df_for_front[f'{target}_Previsto'] = preds

csv_out = PUBLIC_DIR / "jogadores_com_clusters.csv"
df_for_front.to_csv(csv_out, index=False, encoding="utf-8")
print(f"   - CSV de avaliação salvo em: {csv_out.as_posix()}")

# ----------------------------------------------------------
# 6) Treino final 100% dados e salvamento de artefatos
# ----------------------------------------------------------
print("7) Treinando modelos finais e salvando artefatos...")
final_models = {}
for target in TARGETS:
    m = clone(stacking)
    m.fit(X_final, y[target])
    final_models[target] = m

# perfil de clusters (médias numéricas por cluster)
perfil_clusters = df_processed.groupby('cluster')[numerical_features_for_clustering].mean()

# salvar modelos e configs
joblib.dump(final_models['Target1'], ARTIFACTS_DIR / "stacking_model_target1.joblib")
joblib.dump(final_models['Target2'], ARTIFACTS_DIR / "stacking_model_target2.joblib")
joblib.dump(final_models['Target3'], ARTIFACTS_DIR / "stacking_model_target3.joblib")
joblib.dump(cluster_pipeline, ARTIFACTS_DIR / "cluster_pipeline.joblib")
perfil_clusters.to_csv(ARTIFACTS_DIR / "perfil_clusters.csv", index=True)

model_config = {
    "final_model_columns": final_model_columns,
    "numerical_features_for_clustering": numerical_features_for_clustering,
    "categorical_features": categorical_cols
}
with open(ARTIFACTS_DIR / "model_config.json", "w", encoding="utf-8") as f:
    json.dump(model_config, f, indent=2, ensure_ascii=False)

# salvar expected_features.json (ordem final que o backend vai exigir)
with open(ARTIFACTS_DIR / "expected_features.json", "w", encoding="utf-8") as f:
    json.dump(final_model_columns, f, indent=2, ensure_ascii=False)

print("\nTreinamento concluído com sucesso!")
