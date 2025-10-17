import json
import re
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
import joblib

import numpy as np
import pandas as pd
import pandas.api.types as pdt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.base import BaseEstimator

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score, r2_score, mean_absolute_error, accuracy_score, f1_score

from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

import joblib

# ===================================================================
# CONFIGURAÇÕES
# ===================================================================
DATA_FILE = "JogadoresV2.xlsx"  # Caminho do arquivo de dados
ARTIFACTS_DIR = Path("backend/artifacts")
PUBLIC_DIR = Path("dashboard-react-frontend/public")

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

CV_FOLDS = 5
RANDOM_STATE = 42
N_JOBS = -1
CORR_THRESHOLD = 0.29

# =========================
# HELPERS
# =========================
def is_classification_target(y: pd.Series) -> bool:
    """Define se o target é classificação ou regressão."""
    y = pd.Series(y)
    if isinstance(y.dtype, pdt.CategoricalDtype) or pdt.is_bool_dtype(y) or pdt.is_object_dtype(y):
        return True

    n = len(y)
    n_unique = y.nunique(dropna=True)
    if pdt.is_integer_dtype(y):
        return n_unique <= min(20, int(0.05 * n))
    if pdt.is_float_dtype(y):
        if n_unique <= 10 and np.allclose(y.dropna() % 1, 0, atol=1e-12):
            return True
        return False
    return False


def preprocess_data(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.select_dtypes(include=["object"]).columns:
        try:
            X[col] = pd.to_numeric(X[col], errors="raise")
        except ValueError:
            X[col] = X[col].astype("category")
    for col in X.select_dtypes(include=["category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    return X


def classification_models() -> List[Tuple[str, object]]:
    base = [
        ("LogReg", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)),
        ("RFClf", RandomForestClassifier(n_estimators=300, n_jobs=N_JOBS, random_state=RANDOM_STATE)),
    ]
    stack = StackingClassifier(estimators=base, final_estimator=LogisticRegression())
    return [
        ("LogReg", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)),
        ("RFClf", RandomForestClassifier(n_estimators=300, n_jobs=N_JOBS, random_state=RANDOM_STATE)),
        ("XGBClf", XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss")),
        ("StackingClf", stack),
    ]


def regression_models() -> List[Tuple[str, object]]:
    base = [
        ("RFReg", RandomForestRegressor(n_estimators=400, n_jobs=N_JOBS, random_state=RANDOM_STATE)),
        ("ETReg", ExtraTreesRegressor(n_estimators=600, n_jobs=N_JOBS, random_state=RANDOM_STATE)),
        ("GBReg", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ]
    stack = StackingRegressor(
        estimators=base,
        final_estimator=RandomForestRegressor(n_estimators=300, n_jobs=N_JOBS, random_state=RANDOM_STATE),
    )
    return [
        ("DTReg", DecisionTreeRegressor(random_state=RANDOM_STATE)),
        ("RFReg", RandomForestRegressor(n_estimators=400, n_jobs=N_JOBS, random_state=RANDOM_STATE)),
        ("ETReg", ExtraTreesRegressor(n_estimators=600, n_jobs=N_JOBS, random_state=RANDOM_STATE)),
        ("GBReg", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ("HGBReg", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
        ("XGBReg", XGBRegressor(random_state=RANDOM_STATE)),
        ("StackingReg", stack),
    ]

def safe_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def reg_metrics(y_true, y_pred) -> Dict[str, float]:
    return {"RMSE": safe_rmse(y_true, y_pred), "MAE": mean_absolute_error(y_true, y_pred), "R2": r2_score(y_true, y_pred)}


def clf_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),  # O retorno já é um float
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

def stratified_kfold_safe(y: pd.Series, n_splits: int):
    vc = y.value_counts(dropna=True)
    if len(vc) >= 2 and vc.min() >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def select_highly_correlated_features(
    X_train: pd.DataFrame, y_train: pd.Series, corr_threshold: float = CORR_THRESHOLD
) -> Tuple[pd.DataFrame, List[str]]:
    correlations = X_train.corrwith(y_train).abs()
    selected = correlations[correlations > corr_threshold].index.tolist()
    if not selected:
        selected = correlations.sort_values(ascending=False).index.tolist()[: min(20, X_train.shape[1])]
    print(f"Features selecionadas: {selected}")
    return X_train[selected], selected


def sanitize_filename(text: str) -> str:
    text = re.sub(r"\s+", "_", str(text))
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)

# ===================================================================
# FUNÇÕES DO PIPELINE
# ===================================================================

def load_and_clean_data(file_path, null_threshold=0.5):
    """
    Carrega e limpa os dados de um arquivo CSV ou Excel, com remoção de colunas e linhas com dados nulos.
    """
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)  # Carrega arquivo Excel
        else:
            df = pd.read_csv(file_path)  # Carrega arquivo CSV
        df.replace("N/A", np.nan, inplace=True)  # Substitui valores "N/A" por NaN
        print(f"✅ Dados carregados: {df.shape}")

    except Exception as e:
        print(f"❌ Erro ao carregar os dados: {e}")
        return None

    # Backup do formato original
    original_shape = df.shape

    # Remoção de colunas indesejadas e com muitos nulos
    unwanted_columns = ['Código de Acesso', 'Data/Hora Último']
    existing_unwanted_columns = [col for col in unwanted_columns if col in df.columns]

    null_fraction = df.isna().mean()
    columns_with_many_nulls = null_fraction[null_fraction > null_threshold].index.tolist()
    columns_to_remove = list(set(existing_unwanted_columns + columns_with_many_nulls))

    if columns_to_remove:
        df = df.drop(columns=columns_to_remove)
        print(f"🗑️ Colunas removidas: {columns_to_remove[:10]}")

    # Remover linhas onde os valores-alvo são nulos
    target_columns = [col for col in df.columns if 'Target' in col]
    if target_columns:
        df = df.dropna(subset=target_columns)
        print(f"🎯 Linhas com targets nulos removidas.")

    print(f"✅ Limpeza concluída: {original_shape} → {df.shape}")
    return df

def can_be_numeric(value):
    """
    Função auxiliar que verifica se um valor pode ser convertido em número.
    """
    if isinstance(value, str):
        try:
            float(value.replace(",", "."))
            return True
        except ValueError:
            return False
    return isinstance(value, (int, float))

def clean_and_convert(series):
    """
    Limpa strings desnecessárias e converte para numérico (valores inválidos viram NaN).
    """
    series = series.apply(lambda x: str(x).replace("'", "").strip() if isinstance(x, str) else x)
    return pd.to_numeric(series, errors="coerce")

def classify_column(series, categorical_manual):
    """
    Classifica as colunas como Numéricas ou Categóricas com base na lista manual e análise do conteúdo.
    """
    # Primeiro, verifica se a coluna está na lista manual de categóricas
    if series.name in categorical_manual:
        return "Categórica"
    
    # Se a coluna não estiver na lista manual, verifica se algum valor não pode ser numérico
    if series.dropna().apply(lambda x: not can_be_numeric(x)).any():
        return "Categórica"
    
    # Converte para numérico e verifica se contém valores válidos
    numeric_series = clean_and_convert(series)
    if not numeric_series.isna().all():
        return "Numérica"
    return "Categórica"

def process_data_imputation(df, categorical_manual):
    """
    Imputa valores ausentes para colunas numéricas e categóricas.
    
    Args:
        df (pd.DataFrame): DataFrame a ser processado
        categorical_manual (list): Lista de variáveis categóricas definidas manualmente

    Returns:
        pd.DataFrame: DataFrame processado
    """
    print("\n🔄 Iniciando o processamento dos dados...")

    # Identifica colunas de target (alvo)
    targets = [col for col in df.columns if 'Target' in col or 'target' in col.lower()]
    print(f"🎯 Colunas Target: {targets}")

    # Se houver colunas alvo, mantém uma cópia
    df_targets = df[targets] if targets else pd.DataFrame()

    numeric_columns = []       # Colunas numéricas detectadas
    categorical_columns = []   # Colunas categóricas detectadas

    # Classifica cada coluna como numérica ou categórica (ignorando os targets)
    for col in df.columns:
        if col not in targets:
            col_type = classify_column(df[col], categorical_manual)
            if col_type == "Numérica":
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)
    
    # Exibe resumo das colunas identificadas
    print(f"\n📊 Resumo dos tipos: Numéricas: {len(numeric_columns)} | Categóricas: {len(categorical_columns)}")

    # Tratamento de valores ausentes
    print("\n 3️⃣ Tratando valores ausentes...")
    missing_before = df.isna().sum().sum()
    print(f"   🛠 Valores ausentes antes da imputação: {missing_before}")

    # Imputa valores numéricos com mediana
    numeric_imputer = SimpleImputer(strategy="median")
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    # Imputa valores categóricos com o valor mais frequente
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    # Conta novamente valores ausentes após imputação
    missing_after = df.isna().sum().sum()
    print(f"   ✅ Valores ausentes após a imputação: {missing_after}")

    print("✅ Imputação concluída.")
    
    return df, numeric_columns, categorical_columns, numeric_imputer, categorical_imputer

def handle_out_of_range_values(df, valid_ranges):
    """
    Verifica e trata valores fora dos intervalos definidos nas colunas, substituindo-os pelo valor mais frequente.
    Exibe prints apenas para colunas onde houve substituição real.
    """
    imputer = SimpleImputer(strategy='most_frequent')

    for column, value_range in valid_ranges.items():
        if column in df.columns:
            lower_bound, upper_bound = value_range

            # Converte para numérico, colocando NaN para valores inválidos
            df[column] = pd.to_numeric(df[column], errors='coerce')

            # Verifica quantos valores estão fora do intervalo
            out_of_range_before = df[column].apply(lambda x: x < lower_bound or x > upper_bound).sum()

            # Se houver valores fora do intervalo, trata
            if out_of_range_before > 0:
                print(f"\n🔍 Tratando a coluna '{column}' com intervalo ({lower_bound}, {upper_bound})...")

                # Substitui valores fora do intervalo por NaN
                df[column] = df[column].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

                # Imputa valores ausentes com o valor mais frequente
                df[[column]] = imputer.fit_transform(df[[column]])

                # Verifica quantos valores ainda estão fora do intervalo após imputação
                out_of_range_after = df[column].apply(lambda x: x < lower_bound or x > upper_bound).sum()

                # Exibe informações sobre a substituição
                print(f"   🛠️ Valores fora do intervalo antes da substituição: {out_of_range_before}")
                print(f"   ✅ Valores fora do intervalo após a substituição: {out_of_range_after}")
                print(f"   💡 Valores fora do intervalo foram corrigidos com o valor mais frequente.")

    return df

def handle_negative_values(df, categorical_features):
    """
    Verifica e trata valores negativos:
      - Numéricas: substitui negativos pela mediana (considerando somente valores >= 0).
      - Categóricas: substitui negativos (numérico-like) pela moda.
    Exibe prints apenas para colunas onde houve substituição real.
    """
    df = df.copy()  # Evita modificar o DataFrame original

    for column in df.columns:
        # CATEGÓRICA
        if column in categorical_features:
            # Verifica se a coluna categórica possui valores negativos
            numeric_like = pd.to_numeric(df[column], errors='coerce')
            neg_mask = numeric_like < 0
            neg_count = int(neg_mask.fillna(False).sum())

            if neg_count > 0:
                print(f"\n🔍 Tratando a coluna categórica '{column}' (negativos → moda)...")
                # Substitui valores negativos por NaN
                df.loc[neg_mask, column] = np.nan

                # Imputa os valores negativos com a moda da coluna
                imputer = SimpleImputer(strategy='most_frequent')
                df[[column]] = imputer.fit_transform(df[[column]])

                print(f"   🛠️ Valores negativos substituídos: {neg_count}")
                print(f"   💡 Negativos foram corrigidos com a moda da coluna.")

        # NUMÉRICA
        else:
            # Verifica se a coluna numérica possui valores negativos
            col_num = pd.to_numeric(df[column], errors='coerce')
            neg_mask = col_num < 0
            neg_count = int(neg_mask.fillna(False).sum())

            if neg_count > 0:
                # Utiliza valores >= 0 para calcular a mediana
                base = col_num[(col_num >= 0)]
                if base.empty:
                    continue  # Nada a fazer se não houver base válida

                median_value = float(base.median())
                df.loc[neg_mask, column] = median_value

                print(f"\n🔍 Tratando a coluna numérica '{column}' (negativos → mediana)...")
                print(f"   🛠️ Valores negativos substituídos: {neg_count}")
                print(f"   💡 Negativos foram corrigidos com a mediana ({median_value}).")

    return df

def apply_scaler_and_ohe(X_train, X_test, numeric_columns, categorical_columns):
    """
    Aplica a normalização (scaling) nas colunas numéricas e One-Hot Encoding nas categóricas.
    """
    print("\nAplicando normalização e One-Hot Encoding...")

    # Normalização (scaling) para variáveis numéricas
    scaler = StandardScaler()
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])  # Para o conjunto de teste

    # One-Hot Encoding utilizando pd.get_dummies
    X_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

    # Garantir que X_test tenha as mesmas colunas de X_train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Exibe o print com os shapes finais de X_train e X_test
    print(f"✅ Shapes finais: X_train={X_train.shape}, X_test={X_test.shape}")

    return X_train, X_test, scaler

def determine_optimal_clusters(X):
    """
    Determina o número ótimo de clusters usando o método do cotovelo e o coeficiente de silhouette.
    """
    print("\nDeterminando o número ótimo de clusters...")

    k_min = 2  # Mínimo de clusters
    k_max = 12  # Máximo de clusters
    inertias = []  # Lista para armazenar a inércia (Método do Cotovelo)
    silhouettes = []  # Lista para armazenar o coeficiente de Silhouette

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X)
        labels = kmeans.predict(X)
        
        # Calculando a inércia (Método do Cotovelo)
        inertias.append(kmeans.inertia_)

        # Calculando o coeficiente de Silhouette
        silhouette = silhouette_score(X, labels)
        silhouettes.append(silhouette)

        # Printando os valores
        print(f"\n🔍 Para k={k}:")
        print(f"   Inércia: {kmeans.inertia_:.2f}")
        print(f"   Coeficiente de Silhouette: {silhouette:.4f}")
    
    # Escolhendo o melhor número de clusters (k) com o maior coeficiente de Silhouette
    best_k = range(k_min, k_max + 1)[int(np.argmax(silhouettes))]
    print(f"\n🔑 O número ótimo de clusters sugerido é k={best_k} (maior Silhouette)")

    print("\nResumo por k:")
    for k, inertia, sil in zip(range(k_min, k_max + 1), inertias, silhouettes):
        print(f"k={k:2d} | Inércia={inertia:.2f} | Silhouette={sil:.4f}")

    return best_k, inertias, silhouettes

def cluster_data(X_train, X_test, n_clusters, random_state=42):
    """
    Realiza a clusterização utilizando KMeans.
    Exibe prints detalhados sobre a distribuição dos clusters e centróides.
    """
    print("\nRealizando a clusterização com KMeans...")

    # Treinando o modelo KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    
    # Fit e predição para os dados de treino
    train_labels = kmeans.fit_predict(X_train)
    test_labels = kmeans.predict(X_test)

    # Criação de DataFrames com a coluna 'Cluster'
    X_train_clustered = X_train.copy()
    X_test_clustered = X_test.copy()
    X_train_clustered['Cluster'] = train_labels
    X_test_clustered['Cluster'] = test_labels

    # Exibindo distribuição dos clusters no conjunto de treino
    print("\nDistribuição de clusters (treino):")
    print(X_train_clustered['Cluster'].value_counts().sort_index())

    # Exibindo os centróides
    centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=X_train.columns)
    print("\nCentróides (primeiras colunas):")
    print(centroids_df.iloc[:, :min(10, centroids_df.shape[1])].round(3))

    return X_train_clustered, X_test_clustered, kmeans, centroids_df

def train_for_target(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    is_clf = is_classification_target(y_train)
    models = classification_models() if is_clf else regression_models()

    X_train_sel, selected_features = select_highly_correlated_features(X_train, y_train)
    X_test_sel = X_test[selected_features]

    best = {"name": None, "est": None, "pred": None, "metrics": None, "score": -np.inf}
    cv = stratified_kfold_safe(y_train, CV_FOLDS) if is_clf else KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models:
        # Define o tipo de model explicitamente como BaseEstimator
        if not isinstance(model, BaseEstimator):
            raise ValueError(f"O modelo {name} não é do tipo BaseEstimator")
        
        if is_clf:
            cv_scores = cross_val_score(model, X_train_sel, y_train, cv=cv, scoring="f1_macro", n_jobs=N_JOBS)
        else:
            mse_scores = cross_val_score(model, X_train_sel, y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=N_JOBS)
            cv_scores = np.sqrt(-mse_scores)

        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)
        metrics = clf_metrics(y_test, y_pred) if is_clf else reg_metrics(y_test, y_pred)
        score = metrics["f1_macro"] if is_clf else -metrics["RMSE"]

        if score > best["score"]:
            best.update({"name": name, "est": model, "pred": y_pred, "metrics": metrics, "score": score})

    return {
        "best_model_name": best["name"],
        "best_estimator": best["est"],
        "best_pred": best["pred"],
        "best_metrics": best["metrics"],
        "selected_features": selected_features,
        "is_classification": is_clf,
    }

def train_all(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    artifacts_dir: str = "backend/artifacts",  # Alterado para backend/artifacts
) -> Path:
    ARTIFACTS_DIR = Path(artifacts_dir)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)  # Cria a pasta 'backend/artifacts' caso não exista

    ts = time.strftime("%Y%m%d-%H%M%S")
    best_models: Dict[str, Dict[str, Any]] = {}
    predictions: Dict[str, np.ndarray] = {}

    for target in y_train.columns:
        print(f"\nTreinando target: {target}")
        out = train_for_target(X_train, y_train[target], X_test, y_test[target])
        best_models[target] = {
            "model": out["best_estimator"],
            "name": out["best_model_name"],
            "metrics": out["best_metrics"],
            "features": out["selected_features"],
            "is_classification": out["is_classification"],
        }
        predictions[target] = out["best_pred"]

    # salva modelos
    for target, model_data in best_models.items():
        best_model = model_data["model"]
        safe_target = sanitize_filename(target)
        joblib.dump(best_model, ARTIFACTS_DIR / f"{safe_target}_best_model.joblib")
        print(f"✅ Modelo para {target} salvo com sucesso.")

    # salva previsões no formato solicitado
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_csv(ARTIFACTS_DIR / "previsoes_modelos.csv", index=False)
    print(f"📄 Previsões salvas em: {ARTIFACTS_DIR / 'previsoes_modelos.csv'}")

    # salva resumo geral
    summary = {
        "timestamp": ts,
        "artifacts_dir": str(ARTIFACTS_DIR.resolve()),
        "predictions_file": "previsoes_modelos.csv",
        "targets": {
            t: {
                "model_name": m["name"],
                "metrics": m["metrics"],
                "features": m["features"],
                "is_classification": m["is_classification"],
                "model_file": f"{sanitize_filename(t)}_best_model.joblib",
            }
            for t, m in best_models.items()
        },
    }

    summary_path = ARTIFACTS_DIR / f"summary_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"📄 Summary salvo em: {summary_path}")
    return summary_path

def save_artifacts(imputers, scaler):
    """
    Salva os artefatos necessários para reutilização futura, incluindo os melhores modelos para cada variável alvo.
    """
    print("\nSalvando artefatos...")

    # Salvando imputadores e o scaler
    joblib.dump(imputers[0], ARTIFACTS_DIR / 'numeric_imputer.joblib')
    joblib.dump(imputers[1], ARTIFACTS_DIR / 'categorical_imputer.joblib')
    joblib.dump(scaler, ARTIFACTS_DIR / 'scaler.joblib')

    print("Artefatos salvos com sucesso.")

def check_for_nans(df, step_name=""):
    """
    Função auxiliar para verificar se existe NaN no dataframe após cada etapa.
    """
    nan_exists = df.isna().sum().sum() > 0
    if nan_exists:
        print(f"\n🔍 NaN detectado após a etapa: {step_name}")
        print(f"   Total de NaNs: {df.isna().sum().sum()}")
    else:
        print(f"\n✅ Nenhum NaN encontrado após a etapa: {step_name}")

# ===================================================================
# SCRIPT PRINCIPAL
# ===================================================================

# -- CARREGANDO E LIMPEZA DE DADOS --
df_raw = load_and_clean_data(DATA_FILE)
if df_raw is None:
    exit()
check_for_nans(df_raw, "Após carga e limpeza de dados")

categorical_manual = [
    'QtdComida', 'QtdPessoas', 'QtdSom', 'QtdHorasDormi', 'QtdHorasSono',
    'Acordar', 'F0705', 'F0706', 'F0707', 'F0708', 'F0709', 'F0710', 'F0711', 'F0712', 'F0713',
    'F1101', 'F1103', 'F1105', 'F1107', 'F1109', 'F1111', 'P04', 'P08', 'P10', 'P12', 'P02', 'P03',
    'P07', 'P09', 'P13', 'F0104', 'F0201', 'F0203', 'F0205', 'QtdDormir', 'F0101', 'F0102', 'L0210 (não likert)'
]

# -- TRATAMENTO DE VALORES AUSENTES --
df_imputed, numeric_cols, categorical_cols, num_imputer, cat_imputer = process_data_imputation(df_raw.copy(), categorical_manual)

# -- TRATAMENTO DE DADOS FORA DO INTERVALO --
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
check_for_nans(df_ranged, "Após tratamento de dados fora do intervalo")

# -- TRATAMENTO DE VALORES NEGATIVOS --
df_final = handle_negative_values(df_ranged.copy(), categorical_cols)
check_for_nans(df_final, "Após tratamento de valores negativos")

# -- DIVIDINDO O DATASET EM TREINAMENTO E TESTE --
TARGETS = ['Target1', 'Target2', 'Target3']
X = df_final.drop(columns=TARGETS)
y = df_final[TARGETS]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
check_for_nans(X_train, "Após divisão de treino")
check_for_nans(X_test, "Após divisão de teste")

# Exibe o tamanho dos conjuntos de treino e teste
print(f"\nConjunto de Treinamento: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Conjunto de Teste: X_test = {X_test.shape}, y_test = {y_test.shape}")

categorical_columns = ['Cor0202', 'Cor0204', 'Cor0206', 'F0207', 'Cor0208', 'Cor0209Outro']

# -- PADRONIZAÇÃO E OHE --
# Chamada da função ajustada
X_train, X_test, scaler = apply_scaler_and_ohe(X_train, X_test, numeric_cols, categorical_columns)
check_for_nans(X_train, "Após padronização e OHE - X_train")
check_for_nans(X_test, "Após padronização e OHE - X_test")

# -- CLUSTERIZAÇÃO (APÓS PADRONIZAÇÃO E OHE) - SOMENTE NO CONJUNTO DE TREINO --
best_k, _, _ = determine_optimal_clusters(X_train)  # Determina o número ótimo de clusters com o conjunto de treino
# Clusteriza somente com o conjunto de treino
X_train_clustered, X_test_clustered, kmeans_model, centroids_df = cluster_data(X_train, X_test, n_clusters=best_k) # Desempacotando corretamente
check_for_nans(X_train_clustered, "Após clusterização - X_train")
check_for_nans(X_test_clustered, "Após clusterização - X_test")

# -- AVALIAÇÃO DO MODELO DE REGRESSÃO --
X_train_clustered = preprocess_data(X_train_clustered)
X_test_clustered = preprocess_data(X_test_clustered)
summary_path = train_all(X_train_clustered, y_train, X_test_clustered, y_test)

# -- SALVANDO ARTEFATOS --
save_artifacts((num_imputer, cat_imputer), scaler)

print("\n--- Processo concluído com sucesso! ---")
