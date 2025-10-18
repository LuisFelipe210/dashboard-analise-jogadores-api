# train_model.py
import os
import json
import warnings

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score # <--- CORREÇÃO AQUI
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pandas.api.types as pdt

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ===================================================================
# CONFIGURAÇÕES
# ===================================================================
DATA_FILE = "JogadoresV2.csv"
ARTIFACTS_DIR = Path("backend/artifacts")
PUBLIC_DIR = Path("dashboard-react-frontend/public")


ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5
N_JOBS = -1

print("--- INICIANDO PROCESSO DE TREINAMENTO E GERAÇÃO DE ARTEFATOS ---")

# ===================================================================
# FUNÇÕES DO NOTEBOOK ADAPTADAS
# ===================================================================

def load_and_clean_data(file_path, null_threshold=0.5):
    print("\n1. Carregando e limpando dados brutos...")
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except Exception as e:
        print(f"Erro ao carregar CSV: {e}. Certifique-se que o arquivo '{file_path}' está na raiz.")
        return None

    df.replace("N/A", np.nan, inplace=True)
    unwanted_columns = ['Código de Acesso', 'Data/Hora Último']
    existing_unwanted_columns = [col for col in unwanted_columns if col in df.columns]
    
    null_fraction = df.isna().mean()
    columns_with_many_nulls = null_fraction[null_fraction > null_threshold].index.tolist()
    
    columns_to_remove = list(set(existing_unwanted_columns + columns_with_many_nulls))
    
    if columns_to_remove:
        df = df.drop(columns=columns_to_remove)
    
    target_columns = [col for col in df.columns if 'Target' in col]
    if target_columns:
        df = df.dropna(subset=target_columns)
    
    print(f"Dados limpos: {df.shape}")
    return df

def process_data_imputation(df):
    print("\n2. Processando e imputando valores ausentes...")
    targets = [col for col in df.columns if 'Target' in col or 'target' in col.lower()]
    df_targets = df[targets].copy() if targets else pd.DataFrame()

    categorical_manual = [
        'QtdComida', 'QtdPessoas', 'QtdSom', 'QtdHorasDormi', 'QtdHorasSono',
        'Acordar', 'F0705', 'F0706', 'F0707', 'F0708', 'F0709', 'F0710', 'F0711', 'F0712', 'F0713',
        'F1101', 'F1103', 'F1105', 'F1107', 'F1109', 'F1111', 'P04', 'P08', 'P10', 'P12', 'P02', 'P03',
        'P07', 'P09', 'P13', 'F0104', 'F0201', 'F0203', 'F0205', 'QtdDormir', 'F0101', 'F0102', 'L0210 (não likert)'
    ]
    
    numeric_columns = [c for c in df.columns if c not in targets and c not in categorical_manual and pd.api.types.is_numeric_dtype(df[c])]
    categorical_columns = [c for c in df.columns if c not in targets and c not in numeric_columns]

    # Imputação
    numeric_imputer = SimpleImputer(strategy="median")
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    categorical_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    print("Imputação concluída.")
    return df, numeric_columns, categorical_columns, numeric_imputer, categorical_imputer

def handle_out_of_range_values(df, valid_ranges):
    print("\n3. Tratando valores fora do intervalo...")
    imputer = SimpleImputer(strategy='most_frequent')
    for column, value_range in valid_ranges.items():
        if column in df.columns:
            lower_bound, upper_bound = value_range
            df[column] = pd.to_numeric(df[column], errors='coerce')
            out_of_range_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            if out_of_range_mask.sum() > 0:
                df.loc[out_of_range_mask, column] = np.nan
                df[[column]] = imputer.fit_transform(df[[column]])
    print("Tratamento de outliers concluído.")
    return df

def handle_negative_values(df, categorical_features):
    print("\n4. Tratando valores negativos...")
    for column in df.columns:
        if column not in categorical_features: # Apenas numéricas
            col_num = pd.to_numeric(df[column], errors='coerce')
            neg_mask = col_num < 0
            if neg_mask.sum() > 0:
                median_value = col_num[col_num >= 0].median()
                df.loc[neg_mask, column] = median_value
    print("Tratamento de negativos concluído.")
    return df

def apply_one_hot_encoding(X_train, X_test, ohe_cols):
    print("\n5. Aplicando One-Hot Encoding...")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None)
    
    X_train_enc = encoder.fit_transform(X_train[ohe_cols])
    X_test_enc = encoder.transform(X_test[ohe_cols])

    enc_cols = encoder.get_feature_names_out(ohe_cols)
    X_train_enc_df = pd.DataFrame(X_train_enc, columns=enc_cols, index=X_train.index)
    X_test_enc_df = pd.DataFrame(X_test_enc, columns=enc_cols, index=X_test.index)

    X_train_final = pd.concat([X_train.drop(columns=ohe_cols), X_train_enc_df], axis=1)
    X_test_final = pd.concat([X_test.drop(columns=ohe_cols), X_test_enc_df], axis=1)
    
    print(f"Features após OHE: {X_train_final.shape[1]}")
    return X_train_final, X_test_final, encoder

# ===================================================================
# SCRIPT PRINCIPAL
# ===================================================================

# -- CARGA E LIMPEZA INICIAL --
df_raw = load_and_clean_data(DATA_FILE)
if df_raw is None:
    exit()

codigo_acesso = df_raw["Código de Acesso"].copy() if "Código de Acesso" in df_raw.columns else pd.Series(index=df_raw.index)

# -- IMPUTAÇÃO --
df_imputed, numeric_cols, categ_cols, num_imputer, cat_imputer = process_data_imputation(df_raw.copy())

# -- TRATAMENTO DE VALORES --
valid_ranges = {
    'QtdComida': (0, 3), 'QtdPessoas': (0, 2), 'QtdSom': (0, 3), 'QtdHorasDormi': (0, 3),
    'QtdHorasSono': (0, 2), 'Acordar': (1, 5), 'F0705': (1, 5), 'F0706': (1, 5),
    'F0707': (1, 5), 'F0708': (1, 5), 'F0709': (1, 5), 'F0710': (1, 5), 'F0711': (1, 5),
    'F0712': (1, 5), 'F0713': (1, 5), 'F1101': (0, 4), 'F1103': (0, 4), 'F1105': (0, 4),
    'F1107': (0, 4), 'F1109': (0, 4), 'F1111': (0, 4), 'P04': (1, 5), 'P08': (1, 5),
    'P10': (1, 5), 'P12': (1, 5), 'P02': (1, 5), 'P03': (1, 5), 'P07': (1, 5),
    'P09': (1, 5), 'P13': (1, 5), 'F0104': (0, 5), 'F0201': (0, 9), 'F0203': (0, 9),
    'F0205': (0, 2), 'L0210': (1, 8), 'QtdDormir': (0, 2), 'F0101': (0, 5), 'F0102': (0, 3)
}
df_ranged = handle_out_of_range_values(df_imputed.copy(), valid_ranges)
df_final = handle_negative_values(df_ranged.copy(), categ_cols)

# -- SEPARAÇÃO DE DADOS --
TARGETS = ['Target1', 'Target2', 'Target3']
X = df_final.drop(columns=TARGETS)
y = df_final[TARGETS]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# -- PRÉ-PROCESSAMENTO (SCALER E OHE) --
numerical_features = [c for c in X.columns if c not in categ_cols]
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

ohe_cols = ['Cor0202', 'Cor0204', 'Cor0206', 'F0207', 'Cor0208', 'Cor0209Outro']
X_train_final, X_test_final, encoder = apply_one_hot_encoding(X_train, X_test, ohe_cols)

# -- CLUSTERIZAÇÃO (PARA ANÁLISE E CSV DO FRONTEND) --
print("\n6. Realizando clusterização para análise...")
kmeans_model = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE)
# Usaremos os dados pré-processados finais para o cluster, para consistência
cluster_labels_train = kmeans_model.fit_predict(X_train_final)
cluster_labels_test = kmeans_model.predict(X_test_final)

# -- TREINAMENTO E SELEÇÃO DE MODELOS --
print("\n7. Treinando e selecionando os melhores modelos para cada Target...")
all_selected_features = {}
best_models = {}

def get_regression_models():
    base_learners = [
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=400, n_jobs=N_JOBS, random_state=RANDOM_STATE)),
        ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=600, n_jobs=N_JOBS, random_state=RANDOM_STATE)),
    ]
    stacking_model = StackingRegressor(estimators=base_learners, final_estimator=RandomForestRegressor(n_estimators=100, n_jobs=N_JOBS, random_state=RANDOM_STATE))
    return [
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=400, n_jobs=N_JOBS, random_state=RANDOM_STATE)),
        ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=600, n_jobs=N_JOBS, random_state=RANDOM_STATE)),
        ("XGBoost", XGBRegressor(random_state=RANDOM_STATE)),
        ("StackingRegressor", stacking_model),
    ]

for target in TARGETS:
    print(f"\n--- Treinando para: {target} ---")
    correlations = X_train_final.corrwith(y_train[target]).abs()
    selected_features = correlations[correlations > 0.29].index.tolist()
    all_selected_features[target] = selected_features
    print(f"   - Features selecionadas: {len(selected_features)}")
    
    X_train_sel = X_train_final[selected_features]
    
    best_r2_score_test = float('-inf')  # Para armazenar o melhor R² no conjunto de teste
    best_model_instance = None
    best_model_name = ""

    for name, model in get_regression_models():
        # Treinamento do modelo
        model.fit(X_train_sel, y_train[target])
        
        # Previsões no conjunto de teste
        X_test_sel = X_test_final[selected_features]
        y_pred_test = model.predict(X_test_sel)
        
        # Calculando o R² no conjunto de teste
        r2_test = r2_score(y_test[target], y_pred_test)
        print(f"   - Modelo: {name} (R² no Conjunto de Teste: {r2_test:.4f})")
        
        # Selecionando o melhor modelo baseado no R² no conjunto de teste
        if r2_test > best_r2_score_test:
            best_r2_score_test = r2_test
            best_model_instance = model
            best_model_name = name

    print(f"   - Melhor modelo no conjunto de teste: {best_model_name} (R² Teste: {best_r2_score_test:.4f})")
    best_models[target] = best_model_instance

# -- GERAÇÃO DE ARQUIVO CSV PARA FRONTEND --
print("\n8. Gerando CSV de teste para o frontend...")
df_front = df_raw.loc[y_test.index].copy()
df_front.rename(columns={'L0210 (não likert)': 'L0210_no_likert', 'Código de Acesso': 'Cdigo_de_Acesso'}, inplace=True)
df_front['cluster'] = cluster_labels_test

for target in TARGETS:
    selected_cols = all_selected_features[target]
    preds = best_models[target].predict(X_test_final[selected_cols])
    df_front[f'{target}_Previsto'] = preds

# O frontend espera as colunas one-hot-encoded, então vamos juntá-las
df_front_final = pd.concat([
    df_front.reset_index(drop=True),
    pd.DataFrame(X_test_final.reset_index(drop=True), columns=X_train_final.columns)
], axis=1)

# Remover colunas duplicadas que possam surgir
df_front_final = df_front_final.loc[:, ~df_front_final.columns.duplicated()]

csv_out_path = PUBLIC_DIR / "jogadores_com_clusters.csv"
df_front_final.to_csv(csv_out_path, index=False)
print(f"   - CSV para frontend salvo em: {csv_out_path}")

# -- SALVANDO ARTEFATOS FINAIS --
print("\n9. Salvando todos os artefatos do modelo...")
joblib.dump(num_imputer, ARTIFACTS_DIR / 'numeric_imputer.joblib')
joblib.dump(cat_imputer, ARTIFACTS_DIR / 'categorical_imputer.joblib')
joblib.dump(scaler, ARTIFACTS_DIR / 'scaler.joblib')
joblib.dump(encoder, ARTIFACTS_DIR / 'encoder_ohe.joblib')
joblib.dump(kmeans_model, ARTIFACTS_DIR / 'kmeans_model.joblib') # Salva o modelo KMeans

for target, model in best_models.items():
    joblib.dump(model, ARTIFACTS_DIR / f'best_model_{target}.joblib')

column_info = {
    'all_features_imputed': df_final.drop(columns=TARGETS).columns.tolist(),
    'numerical_features': numerical_features,
    'categorical_features': categ_cols,
    'ohe_cols': ohe_cols,
    'final_model_columns': X_train_final.columns.tolist(),
    'selected_features_by_target': all_selected_features,
    'valid_ranges': valid_ranges
}
with open(ARTIFACTS_DIR / "column_info.json", "w") as f:
    json.dump(column_info, f, indent=2)

print("\n--- Processo concluído com sucesso! ---")