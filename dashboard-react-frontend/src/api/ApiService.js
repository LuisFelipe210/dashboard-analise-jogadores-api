// src/api/ApiService.js
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL;   // ex.: http://127.0.0.1:8000

if (!API_URL) {
  throw new Error("VITE_API_URL não configuradas no .env do frontend.");
}

// axios base
const api = axios.create({
  baseURL: API_URL,
  timeout: 60_000,
  headers: {
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
  try {
    const { data } = await api.post("/predict", { data: rows });
    if (!data || !data.predictions) {
      throw new Error("Resposta inesperada da API (sem 'predictions').");
    }
    return data.predictions; // { target1: [...], target2: [...], target3: [...], cluster: [...] }
  } catch (err) {
    throw unwrapApiError(err);
  }
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
    const { data } = await api.post("/predict/file", formData);

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

// -----------------------------
// 🔥 Radar por target
// -----------------------------
/**
 * player: objeto com as colunas pré-OHE (linha normalizada do Excel, sem __identifier)
 * target: "Target1" | "Target2" | "Target3"
 * Retorna: { target, cluster_id, labels, player_profile, cluster_average_profile, ... }
 */
export async function getRadar(body, target) {
  // body deve ser: { player, context_rows }  ← IMPORTANTE
  try {
    const { data } = await api.post("/radar", body, { params: { target } });
    // data.labels: 5 eixos; profiles em 0–5 (Likert)
    return data;
  } catch (err) {
    throw unwrapApiError(err);
  }
}
