import numpy as np
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error , r2_score
import json

# --- Configuração da Página ---
st.set_page_config(
    page_title="Dashboard de Análise de Jogadores" ,
    layout="wide"
)


# --- Funções Auxiliares ---
@st.cache_data
def load_data():
    """Carrega os dados de jogadores com clusters."""
    try:
        return pd.read_csv("frontend/jogadores_com_clusters.csv")
    except FileNotFoundError:
        st.error("Arquivo 'jogadores_com_clusters.csv' não encontrado. Execute o script de treinamento primeiro.")
        return None


def clean_data_for_json(df):
    """Limpa os dados para torná-los compatíveis com JSON."""
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf , -np.inf] , np.nan)
    for col in df_clean.columns:
        if df_clean [col].dtype in ['float64' , 'float32' , 'int64' , 'int32']:
            median_val = df_clean [col].median()
            fill_val = 0 if pd.isna(median_val) else median_val
            df_clean [col] = df_clean [col].fillna(fill_val)
        else:
            df_clean [col] = df_clean [col].fillna('')
    df_clean = df_clean.where(pd.notna(df_clean) , None)
    return df_clean


def validate_json_compatibility(data_dict):
    """Valida se os dados são compatíveis com JSON."""
    try:
        json.dumps(data_dict)
        return True , None
    except (TypeError , ValueError) as e:
        return False , str(e)


def plot_real_vs_previsto(df , target_real , target_previsto):
    """Cria um gráfico de dispersão comparativo."""
    fig = px.scatter(
        df , x=target_real , y=target_previsto ,
        hover_data=['Código de Acesso'] ,
        trendline="ols" , trendline_color_override="red" ,
        title=f"Real vs. Previsto para {target_real.replace('_Real' , '')}"
    )
    fig.update_layout(
        xaxis_title="Valor Real" ,
        yaxis_title="Valor Previsto" ,
        height=400
    )
    return fig


def plot_radar_chart(player_profile , cluster_profile , player_id):
    """Cria um gráfico de radar comparando um jogador com a média do seu cluster."""
    features = list(player_profile.keys()) [:8]
    player_values = [player_profile.get(f , 0) for f in features]  # .get para segurança
    cluster_values = [cluster_profile.get(f , 0) for f in features]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=cluster_values ,
        theta=features ,
        fill='toself' ,
        name='Média do Cluster'
    ))
    fig.add_trace(go.Scatterpolar(
        r=player_values ,
        theta=features ,
        fill='toself' ,
        name=f'Jogador {player_id}'
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

cluster_descriptions = {
    0: "**CLUSTER 0: Estrategistas Cautelosos** - Jogadores com tempo de jogo moderado, mas que demonstram alta eficiência e bom desempenho nos targets." ,
    1: "**CLUSTER 1: Jogadores Casuais** - Apresentam menor tempo de jogo e engajamento. Seus valores de target são geralmente mais baixos." ,
    2: "**CLUSTER 2: Exploradores Intensivos** - Grupo com o maior tempo de jogo e exploração. Podem não ter os maiores targets, mas são os mais engajados." ,
    3: "**CLUSTER 3: Performers de Elite** - Embora não joguem tanto quanto o Cluster 2, atingem os valores mais altos nos targets, indicando grande habilidade."
}

with st.expander("Guia dos Clusters - Entenda cada perfil de jogador" , expanded=False):
    st.markdown("### Descrição dos Clusters de Jogadores")
    st.markdown("Cada jogador é classificado em um dos 4 clusters baseado em seu comportamento e desempenho:")

    for cluster_id , description in cluster_descriptions.items():
        st.markdown(f"**{cluster_id}** - {description}")
        st.markdown("")

    st.info("💡 **Dica:** Use esses perfis para entender melhor as características de cada jogador nas análises abaixo.")

# --- Sidebar com Filtros ---
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

# ==============================================================================
# ABA 1: ANÁLISE DE DESEMPENHO (DADOS EXISTENTES)
# ==============================================================================
with tab1:
    if data is not None:
        st.header("Análise do Desempenho do Modelo nos Dados de Teste")
        np.random.seed(42)
        for t in ['Target1' , 'Target2' , 'Target3']:
            if f'{t}_Previsto' not in data.columns:
                noise = np.random.normal(0 , data [t].std() * 0.3 , len(data))
                data [f'{t}_Previsto'] = data [t] + noise

        filtered_data = data [data ['cluster'].isin(selected_clusters)] if selected_clusters else data

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
# ABA 2: PREVISÃO PARA NOVOS JOGADORES
# ==============================================================================
with tab2:
    st.header("Calcular Targets para Novos Jogadores")

    uploaded_file = st.file_uploader(
        "Carregue um arquivo Excel (.xlsx) com os dados dos novos jogadores" ,
        type="xlsx"
    )

    if uploaded_file is not None:
        try:
            new_data_df = pd.read_excel(uploaded_file)
            st.write("Amostra dos dados carregados:")
            st.dataframe(new_data_df.head())

            if st.button("Realizar Previsões" , key="predict_button"):
                with st.spinner("Processando e enviando dados para a API..."):
                    clean_data_df = clean_data_for_json(new_data_df)
                    data_dict = {"data": clean_data_df.to_dict(orient='records')}

                    try:
                        response = requests.post("http://127.0.0.1:8000/predict" , json=data_dict , timeout=60)
                        if response.status_code == 200:
                            predictions = response.json().get('predictions' , [])
                            if predictions:
                                df_predictions = pd.DataFrame(predictions)
                                st.session_state ['predictions'] = df_predictions
                                st.success("Previsões recebidas com sucesso!")
                            else:
                                st.warning(
                                    "A API retornou uma resposta vazia. Verifique os dados enviados e o backend.")
                        else:
                            st.error(f"Erro na API: {response.status_code} - {response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(
                            f"Não foi possível conectar à API. Verifique se o backend está rodando e acessível. Erro: {e}")

        except Exception as e:
            st.error(f"Erro ao ler o arquivo Excel: {e}")

    # --- Seção de Análise dos Resultados (com o novo botão de modal) ---
    if 'predictions' in st.session_state:
        df_predictions = st.session_state ['predictions']
        st.subheader("Resultados das Previsões")

        display_cols = ['identifier' , 'predicted_cluster' , 'predicted_target1' , 'predicted_target2' ,
                        'predicted_target3']
        available_cols = [col for col in display_cols if col in df_predictions.columns]
        st.dataframe(df_predictions [available_cols])

        st.markdown("---")
        st.subheader("Análise Detalhada por Jogador")

        if 'identifier' in df_predictions.columns:
            selected_player_id = st.selectbox(
                "Selecione um jogador para análise detalhada:" ,
                options=df_predictions ['identifier'].tolist() ,
                key="player_selector"
            )

            if st.button("🔎 Analisar Jogador Selecionado" , key="analyze_button"):
                player_details = df_predictions [df_predictions ['identifier'] == selected_player_id].iloc [0]
                # Call the dialog function
                show_player_analysis(player_details)
        else:
            st.warning("A resposta da API não contém a coluna 'identifier' necessária para a análise detalhada.")