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
API_URL = os.getenv("API_URL" , "http://backend:8000/predict")
API_KEY = st.secrets.get("API_KEY" , os.getenv("API_KEY" , "default-secret-key"))


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
        if df_clean [col].isnull().any():
            median_val = df_clean [col].median()
            fill_val = 0 if pd.isna(median_val) else median_val
            df_clean [col] = df_clean [col].fillna(fill_val)
    for col in df_clean.select_dtypes(include=['object']).columns:
        if df_clean [col].isnull().any():
            df_clean [col] = df_clean [col].fillna('')
    return df_clean.where(pd.notna(df_clean) , None)


# --- Lógica de Previsão ---
def run_prediction():
    """Pega o arquivo do estado da sessão, envia para a API, salva os resultados e define uma mensagem de status."""
    uploaded_file = st.session_state.get('file_uploader')

    st.session_state.status_message = None
    st.session_state.status_type = None

    if uploaded_file:
        try:
            if 'predictions' in st.session_state:
                del st.session_state ['predictions']
            if 'df_preview' in st.session_state:
                del st.session_state ['df_preview']

            new_data_df = pd.read_excel(uploaded_file)
            st.session_state.df_preview = new_data_df

            with st.spinner("Realizando previsões..."):
                clean_df = clean_data_for_json(new_data_df)
                data_dict = {"data": clean_df.to_dict(orient='records')}
                headers = {"X-API-KEY": API_KEY}

                try:
                    response = requests.post(API_URL , json=data_dict , headers=headers , timeout=60)

                    if response.status_code == 200:
                        predictions = response.json().get('predictions' , [])
                        if predictions:
                            st.session_state ['predictions'] = pd.DataFrame(predictions)
                            st.session_state.active_tab = "Previsão para Novos Jogadores"
                            st.session_state.status_message = f"✅ Análise concluída com sucesso para {len(predictions)} jogadores!"
                            st.session_state.status_type = 'success'
                        else:
                            st.session_state.status_message = "⚠️ A análise foi concluída, mas a API retornou uma resposta vazia."
                            st.session_state.status_type = 'error'
                    elif response.status_code == 403:
                        st.session_state.status_message = "❌ Erro de Autenticação: A Chave de API é inválida. Verifique as configurações."
                        st.session_state.status_type = 'error'
                    else:
                        st.session_state.status_message = f"❌ Erro na API: {response.status_code} - {response.text}"
                        st.session_state.status_type = 'error'

                except requests.exceptions.RequestException as e:
                    st.session_state.status_message = f"❌ Não foi possível conectar à API em '{API_URL}'. Verifique se o backend está rodando. Erro: {e}"
                    st.session_state.status_type = 'error'
        except Exception as e:
            st.session_state.status_message = f"❌ Erro ao processar o arquivo Excel: {e}"
            st.session_state.status_type = 'error'
    else:
        st.session_state.status_message = "❌ Por favor, carregue um arquivo primeiro."
        st.session_state.status_type = 'error'


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
    fig.add_trace(go.Scatterpolar(
        r=cluster_values , theta=features , fill='toself' , name='Média do Cluster'
    ))
    fig.add_trace(go.Scatterpolar(
        r=player_values , theta=features , fill='toself' , name=f'Jogador {player_id}'
    ))
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

# --- NOVA ANÁLISE DE CLUSTERS ---
cluster_analysis_data = {
    0: {
        "title": "Especialistas em Performance (T1 & T2)" ,
        "emoji": "🎯" ,
        "description": "Jogadores altamente focados em maximizar os resultados dos Targets 1 e 2, demonstrando grande eficiência nessas áreas específicas." ,
        "evidence": "Este grupo lidera com os maiores valores médios para **Target1 (50.88)** e **Target2 (63.98)**, mas apresenta o menor desempenho no Target3. Isso indica um estilo de jogo especializado."
    } ,
    1: {
        "title": "Exploradores Engajados (T3)" ,
        "emoji": "🗺️" ,
        "description": "Este grupo se destaca por seu alto engajamento com as funcionalidades do jogo e por dominar o Target 3." ,
        "evidence": "Atingem o pico no **Target3 (74.35)** e apresentam os maiores índices de interação (ex: `QtdPessoas`, `QtdSom`), mostrando um perfil de jogador mais social e explorador."
    } ,
    2: {
        "title": "Jogadores Versáteis" ,
        "emoji": "⚖️" ,
        "description": "Jogadores com um perfil equilibrado, mantendo um desempenho consistente em todos os três targets sem se especializar em nenhum." ,
        "evidence": "Suas métricas para **Target1, 2 e 3** são moderadas e balanceadas, posicionando-os como um grupo generalista, capaz de se adaptar a diferentes objetivos do jogo."
    }
}

with st.expander("Guia dos Clusters - Clique para ver a análise detalhada de cada perfil 📊" , expanded=False):
    col1 , col2 , col3 = st.columns(3)

    with col1:
        c_id_0 = 0
        c_data_0 = cluster_analysis_data [c_id_0]
        with st.expander(f"{c_data_0 ['emoji']} CLUSTER {c_id_0}: {c_data_0 ['title']}" , expanded=False):
            st.write(c_data_0 ['description'])
            st.info(f"**Evidência nos Dados:** {c_data_0 ['evidence']}")

    with col2:
        c_id_1 = 1
        c_data_1 = cluster_analysis_data [c_id_1]
        with st.expander(f"{c_data_1 ['emoji']} CLUSTER {c_id_1}: {c_data_1 ['title']}" , expanded=False):
            st.write(c_data_1 ['description'])
            st.info(f"**Evidência nos Dados:** {c_data_1 ['evidence']}")

    with col3:
        c_id_2 = 2
        c_data_2 = cluster_analysis_data [c_id_2]
        with st.expander(f"{c_data_2 ['emoji']} CLUSTER {c_id_2}: {c_data_2 ['title']}" , expanded=False):
            st.write(c_data_2 ['description'])
            st.info(f"**Evidência nos Dados:** {c_data_2 ['evidence']}")

if data is not None:
    st.sidebar.header("Filtros do Dashboard")
    # Atualiza as opções para refletir o número correto de clusters
    cluster_options = sorted(data ['cluster'].unique())
    selected_clusters = st.sidebar.multiselect(
        "Selecione os Clusters" ,
        options=cluster_options ,
        default=cluster_options ,
        key="cluster_filter"
    )
else:
    selected_clusters = []

# --- Lógica de Navegação ---
tab_options = ["Análise de Desempenho" , "Previsão para Novos Jogadores"]

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Análise de Desempenho"

active_tab = st.radio(
    "Navegação" ,
    tab_options ,
    key="navigation_radio" ,
    horizontal=True ,
    index=tab_options.index(st.session_state.active_tab) ,
    label_visibility="collapsed"
)
st.session_state.active_tab = active_tab

# --- Conteúdo das Abas ---
if active_tab == "Análise de Desempenho":
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

elif active_tab == "Previsão para Novos Jogadores":
    st.header("Calcular Targets para Novos Jogadores")

    if 'status_message' not in st.session_state:
        st.session_state.status_message = None
    if 'status_type' not in st.session_state:
        st.session_state.status_type = None

    col1 , col2 = st.columns([3 , 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Carregue um arquivo Excel (.xlsx) para análise" ,
            type="xlsx" ,
            key="file_uploader" ,
            label_visibility="collapsed"
        )

    with col2:
        st.write("")
        st.write("")
        if uploaded_file is not None:
            if st.button("🚀 Realizar Análise"):
                run_prediction()
        else:
            st.button("Realizar Análise" , disabled=True , help="Por favor, carregue um arquivo primeiro.")

    st.divider()

    if st.session_state.status_message:
        if st.session_state.status_type == 'success':
            st.success(st.session_state.status_message)
        elif st.session_state.status_type == 'error':
            st.error(st.session_state.status_message)
        st.session_state.status_message = None
        st.session_state.status_type = None

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

