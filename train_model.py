# Desafio-final/train_model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

os.environ['OMP_NUM_THREADS'] = '1'


print("Iniciando o processo de treinamento e preparação...")

# --- Carregamento e Limpeza ---
print("1. Carregando e limpando os dados...")
try:
    df = pd.read_csv('JogadoresV1.csv')
except FileNotFoundError:
    print("Erro: 'JogadoresV1.csv' não encontrado. Certifique-se de que o arquivo está na pasta raiz do projeto.")
    exit()

# Remover colunas desnecessárias
df.drop(columns=['TempoTotalExpl' , 'F0299 - Explicação Tempo' , 'T0499 - Explicação Tempo'] , inplace=True ,
        errors='ignore')
df.dropna(subset=['Target1' , 'Target2' , 'Target3'] , inplace=True)

# --- Tratamento de Nulos e Salvamento dos Valores de Imputação ---
print("2. Tratando valores nulos e salvando valores de imputação...")
imputation_values = {}
colunas_numericas = df.select_dtypes(include=np.number).columns.tolist()
colunas_categoricas = df.select_dtypes(include='object').columns.tolist()

for col in colunas_numericas:
    if df[col].isnull().any():
        median_val = df[col].median()
        fill_val = 0 if pd.isna(median_val) else median_val
        df[col] = df[col].fillna(fill_val)
        imputation_values[col] = fill_val

for col in colunas_categoricas:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        imputation_values[col] = mode_val

# --- Clusterização (K-Means) ---
print("3. Executando a clusterização...")
features_para_cluster = df.select_dtypes(include=np.number).drop(columns=['Target1' , 'Target2' , 'Target3'])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_para_cluster)

kmeans = KMeans(n_clusters=4 , random_state=42 , n_init='auto')
df ['cluster'] = kmeans.fit_predict(features_scaled)

perfil_clusters = df.groupby('cluster')[features_para_cluster.columns].mean().round(2)

# --- Preparação para Regressão ---
print("4. Preparando dados para o modelo de regressão...")
codigos_acesso = df ['Código de Acesso']
targets = df [['Target1' , 'Target2' , 'Target3' , 'cluster']]

features = df.drop(columns=['Código de Acesso' , 'Data/Hora Último' , 'Target1' , 'Target2' , 'Target3'])
features_encoded = pd.get_dummies(features , drop_first=True)
model_columns = features_encoded.columns.tolist()

top_features = features_para_cluster.columns.tolist() + ['cluster']
top_features_existentes = [f for f in top_features if f in model_columns]

X = features_encoded.reindex(columns=model_columns, fill_value=0)

# --- Treinamento dos Modelos de Regressão ---
print("5. Treinando modelos de regressão para cada Target...")
modelos = {}
for target_name in ['Target1' , 'Target2' , 'Target3']:
    print(f"   - Treinando para {target_name}...")
    y = targets [target_name]

    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X[top_features_existentes] , y)
    modelos[target_name] = model
    df[f'{target_name}_Previsto'] = model.predict(X[top_features_existentes])


# --- Salvando Artefatos ---
print("6. Salvando todos os artefatos (modelos, scaler, etc.)...")
output_dir = "backend/model_artifacts"
os.makedirs(output_dir , exist_ok=True)

joblib.dump(scaler , os.path.join(output_dir , "scaler.joblib"))
joblib.dump(kmeans , os.path.join(output_dir , "kmeans.joblib"))
joblib.dump(modelos['Target1'] , os.path.join(output_dir , "lgbm_target1.joblib"))
joblib.dump(modelos['Target2'] , os.path.join(output_dir , "lgbm_target2.joblib"))
joblib.dump(modelos['Target3'] , os.path.join(output_dir , "lgbm_target3.joblib"))

model_config = {
    "model_columns": model_columns ,
    "features_para_cluster": features_para_cluster.columns.tolist() ,
    "top_features_regressao": top_features_existentes,
    "categorical_features": [col for col in colunas_categoricas if col in features.columns]
}
with open(os.path.join(output_dir , "model_config.json") , 'w') as f:
    json.dump(model_config , f, indent=4)

with open(os.path.join(output_dir, "imputation_values.json"), 'w') as f:
    json.dump(imputation_values, f, indent=4)

perfil_clusters.to_csv(os.path.join(output_dir , "perfil_clusters.csv"))
df.to_csv('frontend/jogadores_com_clusters.csv' , index=False)

print("\nProcesso concluído com sucesso!")
print(f"Todos os artefatos foram salvos na pasta '{output_dir}'.")