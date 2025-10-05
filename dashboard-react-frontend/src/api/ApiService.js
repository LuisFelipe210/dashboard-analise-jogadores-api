// src/api/ApiService.js
import axios from 'axios';

// Pega a URL e a Chave da API das variáveis de ambiente
const API_URL = import.meta.env.VITE_API_URL;
const API_KEY = import.meta.env.VITE_API_KEY;

export const runPrediction = async (playerData) => {
  if (!API_URL || !API_KEY) {
    throw new Error("Variáveis de ambiente VITE_API_URL ou VITE_API_KEY não estão configuradas.");
  }

  try {
    const response = await axios.post(
      API_URL,
      { data: playerData }, // O corpo da requisição deve ser um objeto com a chave "data"
      {
        headers: {
          'X-API-KEY': API_KEY,
          'Content-Type': 'application/json',
        },
        timeout: 60000 // Timeout de 60 segundos
      }
    );
    return response.data.predictions; // Retorna apenas a lista de previsões
  } catch (error) {
    if (error.response) {
      // A requisição foi feita e o servidor respondeu com um status de erro
      const detail = error.response.data.detail || JSON.stringify(error.response.data);
      throw new Error(`Erro da API: ${error.response.status} - ${detail}`);
    } else if (error.request) {
      // A requisição foi feita mas não houve resposta
      throw new Error("Não foi possível conectar à API. Verifique se o backend está rodando e acessível.");
    } else {
      // Algo aconteceu ao configurar a requisição
      throw new Error(`Erro ao configurar a requisição: ${error.message}`);
    }
  }
};