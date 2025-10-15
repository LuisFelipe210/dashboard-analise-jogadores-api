# backend/main.py - VERSÃO FINAL E CORRIGIDA (Flask adaptado para compatibilidade)
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import os
from sklearn.impute import SimpleImputer
import json

# --- 1. Inicializar o aplicativo Flask ---
app = Flask(__name__)
CORS(app)  # Permite requisições de outras origens

# --- 2. Carregar os artefatos na inicialização ---
ARTIFACTS_DIR = 'artifacts'
column_info, selected_features_by_target = {}, {}

try:
    # Modelos
    model_t1 = joblib.load(os.path.join(ARTIFACTS_DIR, 'best_model_Target1.joblib'))
    model_t2 = joblib.load(os.path.join(ARTIFACTS_DIR, 'best_model_Target2.joblib'))
    model_t3 = joblib.load(os.path.join(ARTIFACTS_DIR, 'best_model_Target3.joblib'))

    # Pré-processadores
    numeric_imputer = joblib.load(os.path.join(ARTIFACTS_DIR, 'numeric_imputer.joblib'))
    categorical_imputer = joblib.load(os.path.join(ARTIFACTS_DIR, 'categorical_imputer.joblib'))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))
    encoder_ohe = joblib.load(os.path.join(ARTIFACTS_DIR, 'encoder_ohe.joblib'))

    # Metadados de colunas
    column_info = joblib.load(os.path.join(ARTIFACTS_DIR, 'column_info.joblib'))
    selected_features_by_target = joblib.load(os.path.join(ARTIFACTS_DIR, 'selected_features_by_target.joblib'))

    print("Todos os artefatos foram carregados com sucesso.")
except FileNotFoundError as e:
    print(f"AVISO: Arquivo de artefato não encontrado: {e}. A API pode não funcionar corretamente.")
except Exception as e:
    print(f"ERRO GERAL ao carregar artefatos: {e}")


# --- 3. Função de Pré-processamento para Novos Dados ---
def preprocess_new_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas as etapas de pré-processamento do notebook em novos dados."""
    
    df_copy = df.copy()

    # a. Limpeza inicial
    df_copy.drop(columns=column_info.get('unwanted_columns', []), errors='ignore', inplace=True)
    df_copy.replace("N/A", np.nan, inplace=True)
    
    # b. Identificar colunas
    all_numeric_cols = column_info.get('numerical_feature_columns', [])
    all_categorical_cols = column_info.get('categorical_manual', [])
    
    for col in all_numeric_cols + all_categorical_cols:
        if col not in df_copy.columns:
            df_copy[col] = np.nan

    # c. Imputação
    df_copy[all_numeric_cols] = numeric_imputer.transform(df_copy[all_numeric_cols])
    df_copy[all_categorical_cols] = categorical_imputer.transform(df_copy[all_categorical_cols])

    # d. Tratamento de valores fora do intervalo
    valid_ranges = column_info.get('valid_ranges', {})
    for column, value_range in valid_ranges.items():
        if column in df_copy.columns:
            lower, upper = value_range
            df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
            out_of_range_mask = (df_copy[column] < lower) | (df_copy[column] > upper)
            if out_of_range_mask.any():
                df_copy.loc[out_of_range_mask, column] = np.nan
                imputer_temp = SimpleImputer(strategy='most_frequent')
                df_copy[[column]] = imputer_temp.fit_transform(df_copy[[column]])
    
    # e. Tratamento de negativos
    for column in all_categorical_cols + all_numeric_cols:
        col_num = pd.to_numeric(df_copy[column], errors='coerce')
        neg_mask = (col_num < 0).fillna(False)
        if neg_mask.any():
            if column in all_numeric_cols:
                median_val = pd.to_numeric(df_copy[column][col_num >= 0]).median()
                df_copy.loc[neg_mask, column] = median_val if not pd.isna(median_val) else 0
            else:
                df_copy.loc[neg_mask, column] = np.nan
                imputer_temp = SimpleImputer(strategy='most_frequent')
                df_copy[[column]] = imputer_temp.fit_transform(df_copy[[column]])
    
    # f. Scaling
    df_num_scaled = pd.DataFrame(scaler.transform(df_copy[all_numeric_cols]), columns=all_numeric_cols, index=df_copy.index)

    # g. One-Hot Encoding
    categorical_to_encode = column_info.get('categorical_features_ohe', [])
    df_cat_encoded = pd.DataFrame(
        encoder_ohe.transform(df_copy[categorical_to_encode]),
        columns=encoder_ohe.get_feature_names_out(categorical_to_encode),
        index=df_copy.index
    )

    # h. Manter outras categóricas
    categorical_to_keep = [col for col in all_categorical_cols if col not in categorical_to_encode]
    df_cat_keep = df_copy[categorical_to_keep]

    # i. Combinar
    df_processed = pd.concat([df_num_scaled, df_cat_keep, df_cat_encoded], axis=1)

    return df_processed


# --- 4. Endpoints de Compatibilidade (Mocks) ---
@app.route('/predict/schema', methods=['GET'])
def predict_schema():
    if not column_info:
        return jsonify({"error": "Metadados de colunas não carregados."}), 500
    expected_cols = column_info.get('numerical_feature_columns', []) + column_info.get('categorical_manual', [])
    schema = [{'name': col, 'type': 'number', 'label': col, 'default': 0} for col in expected_cols]
    return jsonify(schema)

@app.route('/clusters/profile', methods=['GET'])
def clusters_profile():
    return jsonify({
        "clusters": [], "n_features": 0, "means": {}, "top_diffs": [], "used_features": [],
    })

@app.route('/overview', methods=['GET'])
def overview():
    num_cols = len(column_info.get('numerical_feature_columns', []))
    cat_cols = len(column_info.get('categorical_manual', []))
    return jsonify({"n_expected_features": num_cols + cat_cols, "has_cluster_pipeline": False})
    
@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    return jsonify({"available": False, "note": "O modelo atual não expõe importâncias de features."})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# --- 5. Endpoint de Predição Principal ---
@app.route('/predict', methods=['POST'])
def predict():
    API_KEY = os.getenv("API_KEY", "")
    given_key = request.headers.get("X-API-Key") or request.headers.get("X-API-KEY")
    if API_KEY and given_key != API_KEY:
        return jsonify({'error': 'Chave de API inválida'}), 401

    json_data = request.get_json()
    if isinstance(json_data, dict) and 'data' in json_data:
        json_data = json_data['data']

    if not isinstance(json_data, list):
        return jsonify({'error': 'O corpo da requisição deve ser uma lista de objetos.'}), 400

    try:
        df_raw = pd.DataFrame(json_data)
        if df_raw.empty:
            return jsonify({'error': 'Nenhum dado válido para processar.'}), 400
    except Exception as e:
        return jsonify({'error': f'Erro ao converter JSON para DataFrame: {e}'}), 400
        
    try:
        df_processed = preprocess_new_data(df_raw)
    except Exception as e:
        return jsonify({'error': f'Erro durante o pré-processamento: {e}'}), 500

    predictions = {}
    try:
        for target_name_api, model in [('Target1', model_t1), ('Target2', model_t2), ('Target3', model_t3)]:
            features_for_target = selected_features_by_target[target_name_api]
            
            missing_cols = set(features_for_target) - set(df_processed.columns)
            if missing_cols:
                for c in missing_cols: df_processed[c] = 0
            
            df_for_prediction = df_processed[features_for_target]
            predictions[target_name_api] = model.predict(df_for_prediction).tolist()

    except Exception as e:
         return jsonify({'error': f'Erro durante a predição: {e}'}), 500
    
    # ADAPTAÇÃO: Formatar a resposta para o frontend React
    n_rows = len(df_raw)
    response_data = {
        "predictions": {
            "target1": predictions.get('Target1', [None] * n_rows),
            "target2": predictions.get('Target2', [None] * n_rows),
            "target3": predictions.get('Target3', [None] * n_rows),
            "cluster": [None] * n_rows  # O novo modelo não gera clusters
        }
    }
    return jsonify(response_data)

# --- 6. Executar o Servidor ---
if __name__ == '__main__':
    # O comando do Docker Compose irá sobrescrever host e port
    app.run(host='0.0.0.0', port=8000, debug=True)