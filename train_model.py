import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

print("Iniciando o processo de treinamento e preparação...")

# --- Carregamento e Limpeza ---
print("1. Carregando e limpando os dados...")
try:
    df = pd.read_csv('notebook/JogadoresV1.csv')
except FileNotFoundError:
    print("Erro: 'JogadoresV1.csv' não encontrado. Certifique-se de que o arquivo está na mesma pasta.")
    exit()

# Remover colunas desnecessárias
df.drop(columns=['TempoTotalExpl' , 'F0299 - Explicação Tempo' , 'T0499 - Explicação Tempo'] , inplace=True ,
        errors='ignore')
df.dropna(subset=['Target1' , 'Target2' , 'Target3'] , inplace=True)

# --- Tratamento de Nulos (Imputação) ---
print("2. Tratando valores nulos...")
colunas_numericas = df.select_dtypes(include=np.number).columns.tolist()
colunas_categoricas = df.select_dtypes(include='object').columns.tolist()

for col in colunas_numericas:
    if df [col].isnull().any():
        if df [col].isnull().all():
            df [col].fillna(0 , inplace=True)
        else:
            df [col].fillna(df [col].median() , inplace=True)

for col in colunas_categoricas:
    if df [col].isnull().any():
        df [col].fillna(df [col].mode() [0] , inplace=True)

# --- Clusterização (K-Means) ---
print("3. Executando a clusterização...")
features_para_cluster = df.select_dtypes(include=np.number).drop(columns=['Target1' , 'Target2' , 'Target3'])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_para_cluster)

kmeans = KMeans(n_clusters=4 , random_state=42 , n_init=10)
df ['cluster'] = kmeans.fit_predict(features_scaled)

# Calcular e salvar perfis dos clusters para o dashboard
perfil_clusters = df.groupby('cluster') [features_para_cluster.columns].mean().round(2)

# --- Preparação para Regressão ---
print("4. Preparando dados para o modelo de regressão...")
# Guardar identificadores para o final
codigos_acesso = df ['Código de Acesso']
targets = df [['Target1' , 'Target2' , 'Target3' , 'cluster']]

# One-Hot Encoding
features = df.drop(columns=['Código de Acesso' , 'Data/Hora Último' , 'Target1' , 'Target2' , 'Target3'])
features_encoded = pd.get_dummies(features , drop_first=True)
model_columns = features_encoded.columns.tolist()
top_features = features_para_cluster.columns [:25].tolist() + ['cluster']
# Garantir que as top features existem nas colunas codificadas
top_features_existentes = [f for f in top_features if f in model_columns]

X = features_encoded [top_features_existentes]

# --- Treinamento dos Modelos de Regressão ---
print("5. Treinando modelos de regressão para cada Target...")
modelos = {}
for target_name in ['Target1' , 'Target2' , 'Target3']:
    print(f"   - Treinando para {target_name}...")
    y = targets [target_name]

    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X , y)
    modelos [target_name] = model

# --- Salvando Artefatos ---
print("6. Salvando todos os artefatos (modelos, scaler, etc.)...")
output_dir = "backend/model_artifacts"
os.makedirs(output_dir , exist_ok=True)

joblib.dump(scaler , os.path.join(output_dir , "scaler.joblib"))
joblib.dump(kmeans , os.path.join(output_dir , "kmeans.joblib"))
joblib.dump(modelos ['Target1'] , os.path.join(output_dir , "lgbm_target1.joblib"))
joblib.dump(modelos ['Target2'] , os.path.join(output_dir , "lgbm_target2.joblib"))
joblib.dump(modelos ['Target3'] , os.path.join(output_dir , "lgbm_target3.joblib"))

# Salvar informações essenciais para a API
model_config = {
    "model_columns": model_columns ,
    "features_para_cluster": features_para_cluster.columns.tolist() ,
    "top_features_regressao": top_features_existentes
}
with open(os.path.join(output_dir , "model_config.json") , 'w') as f:
    json.dump(model_config , f)

perfil_clusters.to_csv(os.path.join(output_dir , "perfil_clusters.csv"))

# Salvar dados de exemplo para o dashboard
df.to_csv('frontend/jogadores_com_clusters.csv' , index=False)

print("\nProcesso concluído com sucesso!")
print(f"Todos os artefatos foram salvos na pasta '{output_dir}'.")
