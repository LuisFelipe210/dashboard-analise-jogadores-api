# Dashboard de Análise e Previsão de Jogadores

Este projeto consiste em uma solução completa de Machine Learning, incluindo uma API de backend (FastAPI) para previsões e um dashboard interativo (React) para análise de dados e performance dos modelos.

## Arquitetura do Sistema

O sistema é composto por três componentes principais: um script de treinamento, um backend para servir o modelo e um frontend em React para interação do usuário.

```mermaid
graph TD
    subgraph "Fase de Treinamento (Executado localmente, 1 vez)"
        A[Dados Brutos: JogadoresV1.csv] --> B(train_model.py)
        B --> C[Artefatos do Modelo Salvos em ./backend/artifacts/]
    end

    subgraph "Aplicação em Produção (Docker)"
        E[Frontend React] -- Requisição HTTP com Planilha Bruta --> F[Backend FastAPI]
        F -- Carrega Artefatos --> C
        F -- Realiza Processamento e Previsão --> G[Dados Enriquecidos com Previsões (JSON)]
        G -- Retorna Dados --> E
        E -- Renderiza Dashboard --> H[Usuário]
    end

    I[Usuário] -- Upload de Planilha --> E