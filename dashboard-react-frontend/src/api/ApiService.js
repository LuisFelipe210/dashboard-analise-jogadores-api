// src/api/ApiService.js
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL;   // ex.: http://localhost:8000
const API_KEY = import.meta.env.VITE_API_KEY;   // mesma do backend

if (!API_URL || !API_KEY) {
  throw new Error("VITE_API_URL ou VITE_API_KEY não configuradas no .env do frontend.");
}

const headers = {
  'X-API-Key': API_KEY,           // backend aceita X-API-Key e X-API-KEY
  'Content-Type': 'application/json',
};

export const runPrediction = async (rows) => {
  // rows deve ser: [{ col1: val, col2: val, ... }, { ... }]
  try {
    const { data } = await axios.post(`${API_URL}/predict`, rows, { headers, timeout: 60000 });
    if (!data || !data.predictions) {
      console.warn('Resposta inesperada da API:', data);
      throw new Error('Resposta inesperada da API. Campo "predictions" não encontrado.');
    }
    return data.predictions;
  } catch (error) {
    if (error.response) {
      const detail = error.response.data?.detail || JSON.stringify(error.response.data);
      throw new Error(`Erro da API: ${error.response.status} - ${detail}`);
    }
    if (error.request) {
      throw new Error("Não foi possível conectar à API. Verifique se o backend está rodando e acessível.");
    }
    throw new Error(`Erro ao configurar a requisição: ${error.message}`);
  }
};

// (opcionais – úteis para os cards, gráfico e form dinâmico)
export const getOverview = () => axios.get(`${API_URL}/overview`, { headers }).then(r => r.data);
export const getFeatureImportance = () => axios.get(`${API_URL}/feature_importance`, { headers }).then(r => r.data);
export const getSchema = () => axios.get(`${API_URL}/predict/schema`, { headers }).then(r => r.data);
export const getClusterProfiles = () =>
  axios.get(`${API_URL}/clusters/profile`, { headers }).then(r => r.data);

