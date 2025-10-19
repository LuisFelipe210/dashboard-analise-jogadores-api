// src/api/ApiService.js
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL;   // ex.: http://localhost:8000
const API_KEY = import.meta.env.VITE_API_KEY;   // mesma do backend

if (!API_URL || !API_KEY) {
  throw new Error("VITE_API_URL ou VITE_API_KEY não configuradas no .env do frontend.");
}

// axios base
const api = axios.create({
  baseURL: API_URL,
  timeout: 60_000,
  headers: {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
  },
});

// -----------------------------
// Helpers
// -----------------------------
function unwrapApiError(error) {
  if (error.response) {
    const detail = error.response.data?.detail || JSON.stringify(error.response.data);
    return new Error(`Erro da API: ${error.response.status} - ${detail}`);
  }
  if (error.request) {
    return new Error("Não foi possível conectar à API. Verifique se o backend está rodando e acessível.");
  }
  return new Error(`Erro ao configurar a requisição: ${error.message}`);
}

// -----------------------------
// Predição via JSON
// -----------------------------
/**
 * rows: Array<{ [col: string]: any }>
 * Retorna: { predictions, bucket_summary, count }
 */
export const runPrediction = async (rows) => {
  // ✅ sempre envie { data: rows }
  const { data } = await axios.post(`${API_URL}/predict`, { data: rows });
  if (!data || !data.predictions) {
    throw new Error("Resposta inesperada da API (sem 'predictions').");
  }
  return data.predictions; // <- { target1: [...], target2: [...], target3: [...], cluster: [...] }
};

// -----------------------------
// Predição via CSV (upload)
// -----------------------------
/**
 * file: File (CSV)
 * Retorna: { filename, predictions, bucket_summary, count }
 */
export async function runPredictionFromCsv(file) {
  try {
    const formData = new FormData();
    formData.append("file", file);

    // importante: NÃO setar Content-Type manualmente aqui
    const { data } = await api.post("/predict/file", formData, {
      headers: { "X-API-Key": API_KEY }, // o navegador seta o multipart boundary
    });

    if (!data || !data.predictions) {
      console.warn("Resposta inesperada da API:", data);
      throw new Error('Resposta inesperada da API. Campo "predictions" não encontrado.');
    }
    return data; // { filename, predictions, bucket_summary, count }
  } catch (err) {
    throw unwrapApiError(err);
  }
}

// -----------------------------
// Apenas resumo por faixas
// -----------------------------
/**
 * thresholds opcionais: { low?: number, high?: number }
 * Modo A: informar linhas (data) para o pipeline rodar.
 * Modo B: informar predictions já calculadas.
 */
export async function getBucketSummary({ rows, predictions, thresholds } = {}) {
  try {
    const payload = {};
    if (rows) payload.data = rows;
    if (predictions) payload.predictions = predictions;
    if (thresholds?.low != null) payload.low = thresholds.low;
    if (thresholds?.high != null) payload.high = thresholds.high;

    const { data } = await api.post("/targets/buckets", payload);
    return data; // { bucket_summary, count? }
  } catch (err) {
    throw unwrapApiError(err);
  }
}

// -----------------------------
// Utilidades do backend
// -----------------------------
export async function getOverview() {
  try {
    const { data } = await api.get("/overview");
    return data; // { n_expected_features }
  } catch (err) {
    throw unwrapApiError(err);
  }
}

export async function getSchema() {
  try {
    const { data } = await api.get("/predict/schema");
    return data; // string[] (colunas esperadas antes do OHE)
  } catch (err) {
    throw unwrapApiError(err);
  }
}

// Mantido por compatibilidade, mas o backend apenas informa substituição
export async function getClusterProfiles() {
  try {
    const { data } = await api.get("/clusters/profile");
    return data; // { message: ... }
  } catch (err) {
    throw unwrapApiError(err);
  }
}
