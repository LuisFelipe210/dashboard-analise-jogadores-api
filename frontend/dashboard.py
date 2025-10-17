import numpy as np
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
import os

# Configuração da página do Streamlit para usar um layout amplo.
st.set_page_config(page_title="Dashboard de Análise de Jogadores", layout="wide")

# URLs da API e chaves, com fallback para desenvolvimento local.
API_URL = os.getenv("API_URL", "http://backend:8000/predict")
API_KEY = st.secrets.get("API_KEY", os.getenv("API_KEY", "default-secret-key"))

@st.cache_data
def load_data():
    """
    Carrega os dados do arquivo CSV gerado pelo script de treinamento.
    Utiliza cache para otimizar o carregamento em execuções repetidas.
    """
    try:
        # O arquivo agora está dentro do diretório do frontend
        return pd.read_csv("jogadores_com_clusters.csv")
    except FileNotFoundError:
        st.error("Arquivo 'jogadores_com_clusters.csv' não encontrado. Execute o script de treinamento primeiro.")
        return None

def clean_data_for_json(df):
    """
    Prepara o DataFrame para ser enviado como JSON para a API, tratando
    valores infinitos, nulos e garantindo a correta tipagem.
    """
    df_clean = df.copy().replace([np.inf, -np.inf], np.nan)
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            fill_val = 0 if pd.isna(median_val) else median_val
            df_clean[col] = df_clean[col].fillna(fill_val)
    for col in df_clean.select_dtypes(include=['object']).columns:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna('')
    return df_clean.where(pd.notna(df_clean), None)

def run_prediction():
    """
    Orquestra o processo de envio de dados para a API e o recebimento das previsões.
    Gerencia o estado da aplicação para exibir status e resultados.
    """
    uploaded_file = st.session_state.get('file_uploader')
    st.session_state.status_message, st.session_state.status_type = None, None
    if uploaded_file:
        try:
            if 'predictions' in st.session_state: del st.session_state['predictions']
            if 'df_preview' in st.session_state: del st.session_state['df_preview']
            new_data_df = pd.read_excel(uploaded_file)
            st.session_state.df_preview = new_data_df
            with st.spinner("Realizando previsões..."):
                clean_df = clean_data_for_json(new_data_df)
                data_dict = {"data": clean_df.to_dict(orient='records')}
                headers = {"X-API-KEY": API_KEY}
                try:
                    response = requests.post(API_URL, json=data_dict, headers=headers, timeout=60)
                    if response.status_code == 200:
                        predictions = response.json().get('predictions', [])
                        if predictions:
                            st.session_state['predictions'] = pd.DataFrame(predictions)
                            st.session_state.active_tab = "Previsão para Novos Jogadores"
                            st.session_state.status_message = f"✅ Análise concluída para {len(predictions)} jogadores!"
                            st.session_state.status_type = 'success'
                        else:
                            st.session_state.status_message, st.session_state.status_type = "⚠️ A API retornou uma resposta vazia.", 'error'
                    elif response.status_code == 403:
                        st.session_state.status_message, st.session_state.status_type = "❌ Erro de Autenticação: Chave de API inválida.", 'error'
                    else:
                        st.session_state.status_message, st.session_state.status_type = f"❌ Erro na API: {response.status_code} - {response.text}", 'error'
                except requests.exceptions.RequestException as e:
                    st.session_state.status_message, st.session_state.status_type = f"❌ Não foi possível conectar à API. Erro: {e}", 'error'
        except Exception as e:
            st.session_state.status_message, st.session_state.status_type = f"❌ Erro ao processar o arquivo: {e}", 'error'
    else:
        st.session_state.status_message, st.session_state.status_type = "❌ Por favor, carregue um arquivo primeiro.", 'error'

def plot_error_distribution(df, target_real, target_previsto):
    """
    Cria um histograma para visualizar a distribuição dos erros de previsão.
    Ajuda a identificar o viés do modelo (tendência a superestimar ou subestimar).
    """
    error = df[target_previsto] - df[target_real]
    fig = px.histogram(
        error, nbins=30,
        title=f"Distribuição dos Erros"
    )
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
    fig.update_layout(
        xaxis_title="Erro de Previsão (Previsto - Real)",
        yaxis_title="Contagem",
        height=350,
        showlegend=False
    )
    return fig

def plot_residuals_vs_predicted(df, target_real, target_previsto):
    """
    Gera um gráfico de resíduos vs. valores previstos.
    Ideal para diagnosticar heterocedasticidade (se a variância do erro muda com o valor da previsão).
    """
    df['residuals'] = df[target_real] - df[target_previsto]
    fig = px.scatter(
        df, x=target_previsto, y='residuals',
        title=f"Resíduos vs. Valores Previstos",
        labels={'residuals': 'Resíduos (Real - Previsto)', target_previsto: 'Valor Previsto'},
        opacity=0.7
    )
    fig.add_hline(y=0, line_width=2, line_dash="dash", line_color="red")
    fig.update_layout(height=350)
    return fig


def plot_radar_chart(player_profile, cluster_profile, player_id):
    """
    Cria um gráfico de radar para comparar o perfil de um jogador
    com a média do cluster ao qual ele foi atribuído.
    """
    features = list(player_profile.keys())[:8]
    player_values = [player_profile.get(f, 0) for f in features]
    cluster_values = [cluster_profile.get(f, 0) for f in features]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=cluster_values, theta=features, fill='toself', name='Média do Cluster'))
    fig.add_trace(go.Scatterpolar(r=player_values, theta=features, fill='toself', name=f'Jogador {player_id}'))
    max_val = max(max(player_values, default=0), max(cluster_values, default=0)) * 1.2
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max_val if max_val > 0 else 1])), showlegend=True, title=f"Comparativo: Jogador {player_id} vs. Média do Cluster")
    return fig

def show_player_analysis(player_details):
    """
    Exibe uma análise detalhada de um jogador individual, incluindo o gráfico de radar.
    """
    with st.expander(f"📊 Análise do Perfil: {player_details.get('identifier', 'N/A')}", expanded=True):
        st.write(f"**Cluster Previsto:** {player_details.get('predicted_cluster', 'N/A')}")
        if 'player_profile' in player_details and 'cluster_average_profile' in player_details:
            radar_fig = plot_radar_chart(player_details['player_profile'], player_details['cluster_average_profile'], player_details['identifier'])
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Dados detalhados do perfil não disponíveis.")

# --- INÍCIO DA RENDERIZAÇÃO DO DASHBOARD ---
data = load_data()
st.title("Dashboard de Análise e Previsão de Jogadores")

# Seção expansível para descrever os perfis dos clusters
cluster_analysis_data = {
    0: {"title": "Jogadores de Alta Performance", "emoji": "🚀", "description": "Jogadores com alto desempenho geral, destacando-se em múltiplos targets e features de engajamento.", "evidence": "Apresentam valores consistentemente altos para Target1, Target2 e Target3."},
    1: {"title": "Jogadores com Desempenho Moderado", "emoji": "⚖️", "description": "Jogadores com um perfil equilibrado, mas com desempenho geral inferior ao cluster de alta performance.", "evidence": "Suas métricas são moderadas em todos os âmbitos, indicando um perfil mais casual ou em desenvolvimento."}
}
with st.expander("Guia dos Clusters - Clique para ver a análise de cada perfil 📊"):
    cols = st.columns(len(cluster_analysis_data))
    for i, col in enumerate(cols):
        with col:
            c_data = cluster_analysis_data[i]
            with st.container(border=True):
                 st.markdown(f"<h5>{c_data['emoji']} CLUSTER {i}: {c_data['title']}</h5>", unsafe_allow_html=True)
                 st.write(c_data['description'])
                 st.info(f"**Evidência:** {c_data['evidence']}")

# Sidebar para filtros
if data is not None:
    st.sidebar.header("Filtros do Dashboard")
    cluster_options = sorted(data['cluster'].unique())
    selected_clusters = st.sidebar.multiselect("Selecione os Clusters", options=cluster_options, default=cluster_options, key="cluster_filter")
else:
    selected_clusters = []

# Sistema de navegação por abas
tab_options = ["Análise de Desempenho", "Previsão para Novos Jogadores"]
if 'active_tab' not in st.session_state: st.session_state.active_tab = "Análise de Desempenho"
active_tab = st.radio("Navegação", tab_options, key="navigation_radio", horizontal=True, index=tab_options.index(st.session_state.active_tab), label_visibility="collapsed")
st.session_state.active_tab = active_tab

# Conteúdo da Aba "Análise de Desempenho"
if active_tab == "Análise de Desempenho":
    if data is not None:
        st.header("Análise do Desempenho do Modelo nos Dados de Teste")
        filtered_data = data[data['cluster'].isin(selected_clusters)] if selected_clusters else data
        if not all(f'Target{i}_Previsto' in filtered_data.columns for i in [1, 2, 3]):
            st.warning("Colunas de previsão não encontradas. Execute o script 'train_model.py' atualizado.")
        else:
            # --- SEÇÃO DE MÉTRICAS GERAIS ---
            st.subheader("Métricas de Avaliação (Geral)")
            col1, col2, col3 = st.columns(3)
            targets = ['Target1', 'Target2', 'Target3']
            for i, target in enumerate(targets):
                rmse = np.sqrt(mean_squared_error(filtered_data[target], filtered_data[f'{target}_Previsto']))
                r2 = r2_score(filtered_data[target], filtered_data[f'{target}_Previsto'])
                with locals()[f"col{i+1}"]:
                    st.metric(label=f"RMSE {target}", value=f"{rmse:.2f}")
                    st.metric(label=f"R² {target}", value=f"{r2:.2f}")

            st.divider()
            
            # --- SEÇÃO DE ANÁLISE DE ERROS (DESCRIÇÃO MELHORADA E MOVIDA) ---
            st.subheader("Análise Visual do Erro do Modelo")
            st.markdown("""
            <div style="background-color: #29384B; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h5 style='margin-top: 0;'>🔍 Como Interpretar os Gráficos de Análise de Erro</h5>
            <p>Esses gráficos nos ajudam a entender <strong>onde</strong> e <strong>como</strong> o modelo está errando, em vez de apenas saber o tamanho do erro.</p>
            
            <strong>1. Gráfico de Resíduos (o de dispersão, acima):</strong>
            <ul>
                <li><strong>O que é?</strong> Cada ponto é um jogador. A altura do ponto no gráfico mostra o tamanho do erro da previsão para ele.</li>
                <li><strong>Linha Vermelha:</strong> Representa o <strong>Erro Zero</strong>, ou seja, uma previsão perfeita.</li>
                <li>✅ <strong>Cenário Ideal:</strong> Os pontos devem se espalhar como uma "nuvem" aleatória em torno da linha vermelha, sem formar padrões (como um funil ou uma curva). Isso mostra que os erros do modelo são aleatórios e não sistemáticos.</li>
                <li>⚠️ <strong>Sinal de Alerta:</strong> Se os pontos formam um padrão (ex: os erros aumentam para previsões maiores), o modelo pode ter dificuldades com certos tipos de jogadores.</li>
            </ul>

            <strong>2. Gráfico de Distribuição de Erros (o de barras, abaixo):</strong>
            <ul>
                <li><strong>O que é?</strong> Mostra a frequência de cada "tamanho" de erro.</li>
                <li><strong>Linha Vermelha:</strong> Novamente, o <strong>Erro Zero</strong>.</li>
                <li>✅ <strong>Cenário Ideal:</strong> A barra mais alta deve estar exatamente no centro (sobre a linha vermelha), com as outras barras diminuindo de forma simétrica para os lados. Isso significa que a maioria dos erros é muito pequena e o modelo não tem um viés claro para superestimar ou subestimar.</li>
                <li>⚠️ <strong>Sinal de Alerta:</strong> Se o "pico" do gráfico estiver deslocado para um dos lados, o modelo tem um viés (uma tendência a sempre chutar para mais ou para menos).</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("<h6 style='text-align: center;'>Target 1</h6>", unsafe_allow_html=True)
                st.plotly_chart(plot_residuals_vs_predicted(filtered_data, "Target1", "Target1_Previsto"), use_container_width=True)
                st.plotly_chart(plot_error_distribution(filtered_data, "Target1", "Target1_Previsto"), use_container_width=True)
            with c2:
                st.markdown("<h6 style='text-align: center;'>Target 2</h6>", unsafe_allow_html=True)
                st.plotly_chart(plot_residuals_vs_predicted(filtered_data, "Target2", "Target2_Previsto"), use_container_width=True)
                st.plotly_chart(plot_error_distribution(filtered_data, "Target2", "Target2_Previsto"), use_container_width=True)
            with c3:
                st.markdown("<h6 style='text-align: center;'>Target 3</h6>", unsafe_allow_html=True)
                st.plotly_chart(plot_residuals_vs_predicted(filtered_data, "Target3", "Target3_Previsto"), use_container_width=True)
                st.plotly_chart(plot_error_distribution(filtered_data, "Target3", "Target3_Previsto"), use_container_width=True)

            st.divider()

            # --- SEÇÃO DE DESEMPENHO COMPARATIVO POR CLUSTER ---
            st.subheader("Análise Comparativa de Desempenho por Cluster")
            st.markdown("""
            <div style="background-color: #29384B; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h5>Como Interpretar o Gráfico</h5>
            <ul>
                <li>Este gráfico compara o <strong>Coeficiente de Determinação (R²)</strong> para cada Target, agrupado por Cluster.</li>
                <li>O R² varia de 0 a 1 e indica a proporção da variância no target que é previsível a partir das features.</li>
                <li>🎯 <strong>O Objetivo:</strong> Barras mais altas indicam um melhor desempenho do modelo para aquele target e cluster específicos. Uma grande diferença na altura das barras para o mesmo target entre clusters diferentes pode indicar que o modelo tem dificuldade em generalizar para um perfil de jogador específico.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            cluster_metrics = []
            clusters_to_analyze = sorted(filtered_data['cluster'].unique())

            for cluster_id in clusters_to_analyze:
                cluster_data = filtered_data[filtered_data['cluster'] == cluster_id]
                if not cluster_data.empty:
                    for target in targets:
                        r2 = r2_score(cluster_data[target], cluster_data[f'{target}_Previsto'])
                        cluster_metrics.append({
                            "Cluster": f"Cluster {cluster_id}",
                            "Target": target,
                            "R² Score": r2
                        })

            if cluster_metrics:
                metrics_df = pd.DataFrame(cluster_metrics)
                fig_cluster_perf = px.bar(
                    metrics_df,
                    x="Target",
                    y="R² Score",
                    color="Cluster",
                    barmode="group",
                    text_auto='.2f',
                    title="Comparativo do R² Score por Target e Cluster"
                )
                fig_cluster_perf.update_layout(
                    yaxis_title="R² Score",
                    xaxis_title="Targets",
                    legend_title="Clusters",
                    uniformtext_minsize=8, 
                    uniformtext_mode='hide'
                )
                fig_cluster_perf.update_traces(textposition='outside')
                st.plotly_chart(fig_cluster_perf, use_container_width=True)
            else:
                st.info("Nenhum dado disponível para a análise por cluster com os filtros selecionados.")
            
            st.divider()
            
            # --- SEÇÃO DE DISTRIBUIÇÃO GERAL POR CLUSTER ---
            st.subheader("Distribuição de Jogadores por Cluster")
            cluster_counts = filtered_data['cluster'].value_counts().reset_index()
            cluster_counts.columns = ['cluster', 'count']
            fig_dist = px.bar(cluster_counts, x='cluster', y='count', color='cluster', text='count', title=f"Distribuição nos Clusters ({len(filtered_data)} jogadores)")
            st.plotly_chart(fig_dist, use_container_width=True)

# Conteúdo da Aba "Previsão para Novos Jogadores"
elif active_tab == "Previsão para Novos Jogadores":
    st.header("Calcular Targets para Novos Jogadores")
    if 'status_message' not in st.session_state: st.session_state.status_message = None
    if 'status_type' not in st.session_state: st.session_state.status_type = None

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Carregue um arquivo Excel (.xlsx) para análise", type="xlsx", key="file_uploader", label_visibility="collapsed")
    with col2:
        st.write(""); st.write("")
        if uploaded_file is not None:
            if st.button("🚀 Realizar Análise"): run_prediction()
        else:
            st.button("Realizar Análise", disabled=True)
    st.divider()
    if st.session_state.status_message:
        if st.session_state.status_type == 'success': st.success(st.session_state.status_message)
        elif st.session_state.status_type == 'error': st.error(st.session_state.status_message)
        st.session_state.status_message, st.session_state.status_type = None, None
    if 'df_preview' in st.session_state:
        st.subheader("Amostra dos Dados Carregados")
        st.dataframe(st.session_state.df_preview.head())
    if 'predictions' in st.session_state:
        df_predictions = st.session_state['predictions']
        st.subheader("Resultados das Previsões")
        display_cols = ['identifier', 'predicted_cluster', 'predicted_target1', 'predicted_target2', 'predicted_target3']
        st.dataframe(df_predictions[display_cols])
        st.subheader("Análise Detalhada por Jogador")
        if not df_predictions.empty:
            selected_player_id = st.selectbox("Selecione um jogador:", options=df_predictions['identifier'].tolist())
            if selected_player_id:
                player_details = df_predictions[df_predictions['identifier'] == selected_player_id].iloc[0].to_dict()
                show_player_analysis(player_details)
        else:
            st.info("Nenhuma previsão foi gerada.")