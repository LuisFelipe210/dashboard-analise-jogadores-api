# /train_model.py

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings

# Modelos e Ferramentas de Pré-processamento do Scikit-learn
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

print("Iniciando o processo de treinamento (com divisão Treino-Validação-Teste)...")

# --- 1. Carregamento e Pré-processamento ---
print("1. Carregando e processando dados...")
df = pd.read_csv('JogadoresV1.csv', encoding='utf-8', na_values=['N/A', '', '#DIV/0!'])
df_processed = df.copy()
cod_acesso = df_processed['Código de Acesso'].copy()
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

original_cols = df_processed.columns.tolist()

color_cols = [col for col in df_processed.columns if col.startswith('Cor') or col == 'F0207']
categorical_features = []
if color_cols:
    df_color_ohe = pd.get_dummies(df_processed[color_cols].astype(str), prefix=color_cols, dummy_na=False)
    categorical_features = df_color_ohe.columns.tolist()
    df_processed = pd.concat([df_processed.drop(columns=color_cols, errors='ignore'), df_color_ohe], axis=1)

# --- 2. Engenharia de Features ---
print("2. Criando features de engenharia...")
Sono_col = [col for col in df_processed.columns if 'QtdHorasSono' in col][0]
if 'Acordar' in df_processed.columns:
    max_acordar = df_processed['Acordar'].max()
    df_processed['Acordar_Invertido'] = max_acordar - df_processed['Acordar']
    df_processed['Indice_Sono_T1'] = df_processed[Sono_col] * df_processed['Acordar_Invertido']
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

# --- 3. Imputação e Clusterização ---
print("3. Imputando dados e executando a clusterização...")
TARGETS = ['Target1', 'Target2', 'Target3']
cod_acesso = cod_acesso.loc[df_processed.index.intersection(df_processed.dropna(subset=TARGETS).index)]
df_processed.dropna(subset=TARGETS, inplace=True)
for col in df_processed.columns:
    if df_processed[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df_processed[col]):
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
X_full = df_processed.drop(columns=TARGETS + ['Cdigo_de_Acesso'], errors='ignore')
y = df_processed[TARGETS]
for col in X_full.select_dtypes(include='bool').columns:
    X_full[col] = X_full[col].astype(int)
numerical_features_for_clustering = X_full.select_dtypes(include=np.number).columns.tolist()
cluster_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50, random_state=42)),
    ('kmeans', KMeans(n_clusters=2, random_state=42, n_init=10))
])
cluster_labels = cluster_pipeline.fit_predict(X_full[numerical_features_for_clustering])
df_processed['cluster'] = cluster_labels
perfil_clusters = df_processed.groupby('cluster')[numerical_features_for_clustering].mean()

# --- 4. Preparação Final e Divisão Treino-Validação-Teste ---
print("4. Dividindo dados em treino, validação e teste...")
df_processed_ohe = pd.get_dummies(df_processed, columns=['cluster'], prefix='Cluster')
X_final = df_processed_ohe.drop(columns=TARGETS + ['Cdigo_de_Acesso'], errors='ignore')
for col in X_final.select_dtypes(include='bool').columns:
    X_final[col] = X_final[col].astype(int)
final_model_columns = X_final.columns.tolist()

# Primeira divisão: 80% para treino+validação, 20% para teste
X_train_val, X_test, y_train_val, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
# Segunda divisão: 80% de 80% para treino (64% do total), 20% de 80% para validação (16% do total)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# --- 5. Treinamento para Avaliação e Geração do CSV ---
print("5. Treinando modelo na partição de treino e avaliando na de teste...")
estimators = [('rf', RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)), ('ridge', RidgeCV())]
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=Lasso(alpha=0.001, random_state=42), cv=5)
df_for_frontend = df_processed.loc[y_test.index].copy()
df_for_frontend['Código de Acesso'] = cod_acesso.loc[y_test.index]
for target_name in TARGETS:
    model_clone_for_eval = clone(stacking_regressor)
    model_clone_for_eval.fit(X_train, y_train[target_name])
    predictions_test = model_clone_for_eval.predict(X_test)
    df_for_frontend[f'{target_name}_Previsto'] = predictions_test
df_for_frontend.to_csv('frontend/jogadores_com_clusters.csv', index=False)
print("   - Arquivo 'jogadores_com_clusters.csv' gerado com dados de teste.")

# --- 6. Treinamento Final e Salvamento de Artefatos para API ---
print("6. Retreinando o modelo com 100% dos dados para a API...")
final_models_for_api = {}
for target_name in TARGETS:
    model_clone_for_api = clone(stacking_regressor)
    model_clone_for_api.fit(X_final, y[target_name])
    final_models_for_api[target_name] = model_clone_for_api

# --- 7. Salvando Artefatos da API ---
print("7. Salvando todos os artefatos para a API...")
output_dir = "backend/model_artifacts"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(final_models_for_api['Target1'], os.path.join(output_dir, "stacking_model_target1.joblib"))
joblib.dump(final_models_for_api['Target2'], os.path.join(output_dir, "stacking_model_target2.joblib"))
joblib.dump(final_models_for_api['Target3'], os.path.join(output_dir, "stacking_model_target3.joblib"))
joblib.dump(cluster_pipeline, os.path.join(output_dir, "cluster_pipeline.joblib"))
perfil_clusters.to_csv(os.path.join(output_dir, "perfil_clusters.csv"))
model_config = {
    "final_model_columns": final_model_columns,
    "numerical_features_for_clustering": numerical_features_for_clustering,
    "categorical_features": categorical_features
}
with open(os.path.join(output_dir, "model_config.json"), 'w') as f:
    json.dump(model_config, f, indent=4)

print("\nProcesso concluído com sucesso!")