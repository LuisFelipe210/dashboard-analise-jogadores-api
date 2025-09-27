# Desafio-final/frontend/dashboard.py
import numpy as np
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error , r2_score
import json
import os

# --- Configuração da Página ---
st.set_page_config(
    page_title="Dashboard de Análise de Jogadores" ,
    layout="wide"
)

# --- Configurações da API ---
# Melhoria: Usa o nome do serviço 'backend' do Docker Compose. Fallback para localhost.
API_URL = os.getenv("API_URL" , "http://backend:8000/predict")
# Melhoria: Carrega a chave da API dos segredos do Streamlit ou de variável de ambiente.
API_KEY = st.secrets.get("API_KEY", os.getenv("API_KEY", "default-secret-key"))


# --- Funções Auxiliares ---
@st.cache_data
def load_data():
    """Carrega os dados de jogadores com clusters e previsões reais."""
    try:
        return pd.read_csv("jogadores_com_clusters.csv")
    except FileNotFoundError:
        st.error("Arquivo 'jogadores_com_clusters.csv' não encontrado. Execute o script de treinamento primeiro.")
        return None


def clean_data_for_json(df):
    """Limpa os dados para torná-los compatíveis com JSON."""
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf , -np.inf] , np.nan)
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            fill_val = 0 if pd.isna(median_val) else median_val
            df_clean[col] = df_clean[col].fillna(fill_val)
    for col in df_clean.select_dtypes(include=['object']).columns:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna('')
    return df_clean.where(pd.notna(df_clean), None)


# --- Lógica de Previsão ---
def run_prediction():
    """Pega o arquivo do estado da sessão, envia para a API e salva os resultados."""
    uploaded_file = st.session_state.get('file_uploader')
    if uploaded_file:
        try:
            if 'predictions' in st.session_state:
                del st.session_state['predictions']
            if 'df_preview' in st.session_state:
                del st.session_state['df_preview']

            new_data_df = pd.read_excel(uploaded_file)
            st.session_state.df_preview = new_data_df

            with st.spinner("Realizando previsões..."):
                clean_df = clean_data_for_json(new_data_df)
                data_dict = {"data": clean_df.to_dict(orient='records')}
                headers = {"X-API-KEY": API_KEY} # Melhoria: Adiciona o header de autenticação

                try:
                    # Melhoria: URL e Headers atualizados
                    response = requests.post(API_URL, json=data_dict, headers=headers, timeout=60)

                    if response.status_code == 200:
                        predictions = response.json().get('predictions', [])
                        if predictions:
                            st.session_state['predictions'] = pd.DataFrame(predictions)
                        else:
                            st.warning("A API retornou uma resposta vazia.")
                    elif response.status_code == 403:
                        st.error("Erro de Autenticação: A Chave de API é inválida. Verifique as configurações.")
                    else:
                        st.error(f"Erro na API: {response.status_code} - {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Não foi possível conectar à API em '{API_URL}'. Verifique se o backend está rodando e acessível. Erro: {e}")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo Excel: {e}")

# (O resto do arquivo `dashboard.py` pode permanecer o mesmo, pois as funções de plotagem não precisam de alterações)
def plot_real_vs_previsto(df , target_real , target_previsto):
    """Cria um gráfico de dispersão comparativo."""
    fig = px.scatter(
        df , x=target_real , y=target_previsto ,
        hover_data=['Código de Acesso'] ,
        trendline="ols" , trendline_color_override="red" ,
        title=f"Real vs. Previsto para {target_real}"
    )
    fig.update_layout(xaxis_title="Valor Real" , yaxis_title="Valor Previsto" , height=400)
    return fig


def plot_radar_chart(player_profile , cluster_profile , player_id):
    """Cria um gráfico de radar comparando um jogador com a média do seu cluster."""
    features = list(player_profile.keys()) [:8]
    player_values = [player_profile.get(f , 0) for f in features]
    cluster_values = [cluster_profile.get(f , 0) for f in features]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=cluster_values , theta=features , fill='toself' , name='Média do Cluster'))
    fig.add_trace(go.Scatterpolar(r=player_values , theta=features , fill='toself' , name=f'Jogador {player_id}'))
    max_val = max(max(player_values , default=0) , max(cluster_values , default=0)) * 1.2
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True , range=[0 , max_val if max_val > 0 else 1])) ,
        showlegend=True ,
        title=f"Comparativo: Jogador {player_id} vs. Média do Cluster"
    )
    return fig


def show_player_analysis(player_details):
    """Mostra análise detalhada do jogador em um container expansível."""
    with st.expander(f"📊 Análise do Perfil: {player_details.get('identifier' , 'N/A')}" , expanded=True):
        st.write(f"**Cluster Previsto:** {player_details.get('predicted_cluster' , 'N/A')}")
        if 'player_profile' in player_details and 'cluster_average_profile' in player_details:
            radar_fig = plot_radar_chart(
                player_details ['player_profile'] ,
                player_details ['cluster_average_profile'] ,
                player_details ['identifier']
            )
            st.plotly_chart(radar_fig , use_container_width=True)
        else:
            st.info("Dados detalhados do perfil não estão disponíveis para este jogador.")


# --- Carregamento dos Dados ---
data = load_data()

# --- Interface Principal ---
st.title("Dashboard de Análise e Previsão de Jogadores")

cluster_descriptions = {
    0: "**CLUSTER 0: Estrategistas Cautelosos** - Jogadores com tempo de jogo moderado, mas que demonstram alta eficiência e bom desempenho nos targets." ,
    1: "**CLUSTER 1: Jogadores Casuais** - Apresentam menor tempo de jogo e engajamento. Seus valores de target são geralmente mais baixos." ,
    2: "**CLUSTER 2: Exploradores Intensivos** - Grupo com o maior tempo de jogo e exploração. Podem não ter os maiores targets, mas são os mais engajados." ,
    3: "**CLUSTER 3: Performers de Elite** - Embora não joguem tanto quanto o Cluster 2, atingem os valores mais altos nos targets, indicando grande habilidade."
}
with st.expander("Guia dos Clusters - Entenda cada perfil de jogador" , expanded=False):
    st.markdown("### Descrição dos Clusters de Jogadores")
    for cluster_id , description in cluster_descriptions.items():
        st.markdown(f"**{cluster_id}** - {description}")
    st.info("💡 **Dica:** Use esses perfis para entender melhor as características de cada jogador nas análises abaixo.")

if data is not None:
    st.sidebar.header("Filtros do Dashboard")
    selected_clusters = st.sidebar.multiselect(
        "Selecione os Clusters" ,
        options=sorted(data ['cluster'].unique()) ,
        default=sorted(data ['cluster'].unique()) ,
        key="cluster_filter"
    )
else:
    selected_clusters = []

# --- Abas ---
tab1 , tab2 = st.tabs(["Análise de Desempenho" , "Previsão para Novos Jogadores"])

with tab1:
    if data is not None:
        st.header("Análise do Desempenho do Modelo nos Dados de Treino")
        filtered_data = data [data ['cluster'].isin(selected_clusters)] if selected_clusters else data
        if not all(f'Target{i}_Previsto' in filtered_data.columns for i in [1 , 2 , 3]):
            st.warning("Colunas de previsão não encontradas. Por favor, execute o script 'train_model.py' atualizado.")
        else:
            st.subheader("Métricas de Avaliação do Modelo")
            col1 , col2 , col3 = st.columns(3)
            for i , target in enumerate(['Target1' , 'Target2' , 'Target3']):
                rmse = np.sqrt(mean_squared_error(filtered_data [target] , filtered_data [f'{target}_Previsto']))
                r2 = r2_score(filtered_data [target] , filtered_data [f'{target}_Previsto'])
                with locals() [f"col{i + 1}"]:
                    st.metric(label=f"RMSE {target}" , value=f"{rmse:.2f}")
                    st.metric(label=f"R² {target}" , value=f"{r2:.2f}")
            st.subheader("Gráficos Comparativos: Real vs. Previsto")
            c1 , c2 , c3 = st.columns(3)
            with c1:
                st.plotly_chart(plot_real_vs_previsto(filtered_data , "Target1" , "Target1_Previsto") ,
                                use_container_width=True)
            with c2:
                st.plotly_chart(plot_real_vs_previsto(filtered_data , "Target2" , "Target2_Previsto") ,
                                use_container_width=True)
            with c3:
                st.plotly_chart(plot_real_vs_previsto(filtered_data , "Target3" , "Target3_Previsto") ,
                                use_container_width=True)
        st.subheader("Distribuição de Jogadores por Cluster")
        cluster_counts = filtered_data ['cluster'].value_counts().reset_index()
        cluster_counts.columns = ['cluster' , 'count']
        fig_dist = px.bar(cluster_counts , x='cluster' , y='count' , color='cluster' , text='count' ,
                          title=f"Distribuição nos Clusters Selecionados ({len(filtered_data)} jogadores)")
        st.plotly_chart(fig_dist , use_container_width=True)

# ==============================================================================
# ABA 2: PREVISÃO
# ==============================================================================
with tab2:
    st.header("Calcular Targets para Novos Jogadores")

    st.file_uploader(
        "Carregue um arquivo Excel (.xlsx) para iniciar a previsão" ,
        type="xlsx" ,
        key="file_uploader" ,
        on_change=run_prediction
    )

    if 'df_preview' in st.session_state:
        st.subheader("Amostra dos Dados Carregados")
        st.dataframe(st.session_state.df_preview.head())

    if 'predictions' in st.session_state:
        df_predictions = st.session_state ['predictions']
        st.subheader("Resultados das Previsões")
        display_cols = ['identifier' , 'predicted_cluster' , 'predicted_target1' , 'predicted_target2' ,
                        'predicted_target3']
        st.dataframe(df_predictions [display_cols])

        st.subheader("Análise Detalhada por Jogador")
        if not df_predictions.empty:
            selected_player_id = st.selectbox("Selecione um jogador:" , options=df_predictions ['identifier'].tolist())
            if selected_player_id:
                player_details = df_predictions [df_predictions ['identifier'] == selected_player_id].iloc [0].to_dict()
                show_player_analysis(player_details)
        else:
            st.info("Nenhuma previsão foi gerada.")