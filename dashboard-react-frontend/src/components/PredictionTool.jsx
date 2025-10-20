// src/components/PredictionTool.jsx
import React, { useCallback, useMemo, useState, useEffect } from "react";
import {
  Box,
  Button,
  Typography,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Backdrop,
  Card,
} from "@mui/material";
// Removido read/utils do xlsx. A leitura acontece no Worker.
import { runPrediction, getSchema } from "../api/ApiService";

/* ===================== Overlay de loading controlado por etapas ===================== */
function useLatch(active, delayMs = 300) {
  const [latched, setLatched] = useState(false);
  useEffect(() => {
    let t;
    if (active) setLatched(true);
    else if (latched) t = setTimeout(() => setLatched(false), delayMs);
    return () => t && clearTimeout(t);
  }, [active, latched, delayMs]);
  return active || latched;
}

const LoadingOverlay = ({ open, stepText }) => {
  const visible = useLatch(open, 300);
  if (!visible) return null;

  return (
    <Backdrop
      open
      sx={{
        zIndex: (t) => t.zIndex.modal + 1,
        backdropFilter: "blur(6px)",
        background:
          "radial-gradient(1200px 600px at 50% -10%, rgba(127,86,217,0.24), transparent), rgba(8,8,12,0.55)",
      }}
      aria-live="polite"
      aria-busy="true"
    >
      <Card
        elevation={0}
        sx={{
          p: 4,
          width: 420,
          borderRadius: 4,
          bgcolor: "rgba(18,18,24,0.7)",
          border: "1px solid rgba(255,255,255,0.08)",
          boxShadow: "0 10px 30px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06)",
        }}
      >
        <Box sx={{ display: "flex", justifyContent: "center", mb: 2 }}>
          <CircularProgress />
        </Box>
        <Typography variant="h6" align="center" sx={{ mb: 1, color: "white", fontWeight: 600 }}>
          Processando análise
        </Typography>
        <Typography variant="body2" align="center" sx={{ color: "rgba(255,255,255,0.85)" }}>
          {stepText || "Preparando..."}
        </Typography>
      </Card>
    </Backdrop>
  );
};
/* ===================== Fim overlay ===================== */

/* ===================== Helpers e schema ===================== */
const removeAccents = (str) => {
  if (typeof str !== "string") return str;
  return str.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
};

const normalizeName = (s) => {
  if (s == null) return s;
  let out = String(s).trim();
  out = removeAccents(out).replace(/\s+/g, "_").replace(/[^a-zA-Z0-9_]/g, "");
  return out;
};

const SPECIAL_RENAMES = new Map([
  ["L0210_nao_likert", "L0210_no_likert"],
  ["Codigo_de_Acesso", "Cdigo_de_Acesso"],
]);

const applySpecialRename = (name) => {
  const key = normalizeName(name);
  return SPECIAL_RENAMES.get(key) || key;
};

const adaptSchema = (schemaRaw) => {
  if (!Array.isArray(schemaRaw)) throw new Error("Schema inválido: não é array.");
  if (schemaRaw.length === 0) return [];
  if (typeof schemaRaw[0] === "string") {
    return schemaRaw.map((name) => ({ name, type: "number", default: 0, label: name }));
  }
  if (typeof schemaRaw[0] === "object" && schemaRaw[0].name) {
    return schemaRaw.map((f) => ({
      name: f.name,
      type: f.type || "number",
      default: f.default != null ? f.default : f.type === "string" ? "" : 0,
      label: f.label || f.name,
    }));
  }
  throw new Error("Schema inválido: formato não reconhecido.");
};

const buildRowsFromSchema = (jsonData, schema) => {
  const normalizedRows = jsonData.map((orig) => {
    const norm = {};
    Object.keys(orig).forEach((k) => {
      norm[applySpecialRename(k)] = orig[k];
    });
    return norm;
  });

  return normalizedRows.map((orig) => {
    const row = {};
    let matches = 0;

    for (const f of schema) {
      const fname = f.name;
      if (Object.prototype.hasOwnProperty.call(orig, fname)) {
        const v = orig[fname];
        if (f.type === "number") {
          if (typeof v === "number") row[fname] = Number.isFinite(v) ? v : f.default || 0;
          else {
            const num = parseFloat(String(v).replace(",", "."));
            row[fname] = Number.isFinite(num) ? num : f.default || 0;
          }
        } else {
          row[fname] = v != null ? v : f.default || "";
        }
        matches++;
      } else {
        row[fname] = f.default != null ? f.default : f.type === "number" ? 0 : "";
      }
    }

    const idCandidates = [
      "Código de Acesso",
      "Codigo de Acesso",
      "Codigo_de_Acesso",
      "Cdigo_de_Acesso",
      "id",
      "ID",
      "player_id",
      "identifier",
    ];
    let identifier = null;
    for (const c of idCandidates) {
      const cNorm = applySpecialRename(c);
      if (Object.prototype.hasOwnProperty.call(orig, cNorm) && orig[cNorm] != null && orig[cNorm] !== "") {
        identifier = orig[cNorm];
        break;
      }
      if (Object.prototype.hasOwnProperty.call(orig, c) && orig[c] != null && orig[c] !== "") {
        identifier = orig[c];
        break;
      }
    }
    row.__identifier = identifier != null ? String(identifier) : "";

    if (matches < schema.length * 0.5) {
      // eslint-disable-next-line no-console
      console.warn(`Poucas colunas casaram: ${matches}/${schema.length}`, {
        origSample: Object.keys(orig).slice(0, 10),
      });
    }
    return row;
  });
};

const fmt = (v) => {
  if (v == null) return "-";
  if (typeof v === "number" && Number.isFinite(v)) return v.toFixed(2);
  if (typeof v === "string") {
    const n = Number(v);
    return Number.isFinite(n) ? n.toFixed(2) : v;
  }
  return String(v);
};
/* ===================== Fim helpers ===================== */

function PredictionTool() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [selectedPlayerId, setSelectedPlayerId] = useState("");

  const selectedPlayerDetails = useMemo(
    () => predictions.find((p) => p.identifier === selectedPlayerId),
    [predictions, selectedPlayerId]
  );

  const handleFileChange = useCallback((event) => {
    const selectedFile = event.target.files?.[0] || null;
    if (selectedFile) {
      setFile(selectedFile);
      setPredictions([]);
      setError("");
      setSuccess("");
      setSelectedPlayerId("");
    }
  }, []);

  const readSheetInWorker = useCallback(async (excelFile) => {
    const worker = new Worker(new URL("../workers/excelWorker.js", import.meta.url), { type: "module" });
    try {
      const buf = await excelFile.arrayBuffer();
      const result = await new Promise((resolve) => {
        worker.onmessage = (e) => resolve(e.data);
        worker.postMessage(buf);
      });
      if (!result.success) throw new Error(result.error || "Falha ao processar planilha");
      return result.data; // jsonData
    } finally {
      worker.terminate();
    }
  }, []);

  const handlePredict = useCallback(async () => {
    try {
      if (!file) {
        setError("Por favor, carregue um arquivo primeiro.");
        return;
      }

      setIsLoading(true);
      setError("");
      setSuccess("");
      setPredictions([]);
      setSelectedPlayerId("");

      // Etapa 1: leitura no Worker
      setLoadingStep("Lendo a planilha");
      await Promise.resolve(); // permite renderizar o overlay
      const jsonData = await readSheetInWorker(file);
      if (!Array.isArray(jsonData) || jsonData.length === 0) {
        throw new Error("O arquivo está vazio ou em um formato inválido.");
      }

      // Etapa 2: alinhamento com schema
      setLoadingStep("Alinhando com schema");
      const schemaRaw = await getSchema();
      const schema = adaptSchema(schemaRaw);
      const rows = buildRowsFromSchema(jsonData, schema);

      // Etapa 3: modelo
      setLoadingStep("Rodando o modelo");
      const predsDict = await runPrediction(rows);

      // Etapa 4: preparar resultados
      setLoadingStep("Gerando resultados");
      const n = Math.max(
        predsDict.target1?.length || 0,
        predsDict.target2?.length || 0,
        predsDict.target3?.length || 0,
        predsDict.cluster?.length || 0
      );
      if (n === 0) throw new Error("A API retornou zero previsões. Verifique os logs e o arquivo.");

      const safeCluster = predsDict.cluster || Array(n).fill(null);
      const safeT1 = predsDict.target1 || Array(n).fill(null);
      const safeT2 = predsDict.target2 || Array(n).fill(null);
      const safeT3 = predsDict.target3 || Array(n).fill(null);

      const table = Array.from({ length: n }).map((_, i) => {
        const id = String(rows[i]?.__identifier || "").trim() || `lin_${i + 1}`;
        return {
          identifier: id,
          predicted_cluster: safeCluster[i] != null ? safeCluster[i] : null,
          predicted_target1: safeT1[i] != null ? safeT1[i] : null,
          predicted_target2: safeT2[i] != null ? safeT2[i] : null,
          predicted_target3: safeT3[i] != null ? safeT3[i] : null,
        };
      });

      setPredictions(table);
      setSuccess(`Análise concluída com sucesso para ${table.length} jogadores.`);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(err);
      const msg =
        err && typeof err.message === "string"
          ? err.message
          : "Ocorreu um erro desconhecido ao processar o arquivo.";
      setError(msg);
    } finally {
      setIsLoading(false);
      setLoadingStep("");
    }
  }, [file, readSheetInWorker]);

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Calcular Targets para Novos Jogadores
      </Typography>

      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 4 }}>
        <Button variant="contained" component="label" aria-label="Carregar Excel" disabled={isLoading}>
          Carregar Arquivo Excel
          <input
            type="file"
            accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            hidden
            onChange={handleFileChange}
            disabled={isLoading}
          />
        </Button>

        {file && <Typography>{file.name}</Typography>}

        <Button
          onClick={handlePredict}
          variant="contained"
          color="primary"
          disabled={!file || isLoading}
          sx={{ ml: "auto" }}
          aria-label="Realizar análise"
        >
          {isLoading ? <CircularProgress size={24} /> : "🚀 Realizar Análise"}
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert severity="success" sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      {predictions.length > 0 && (
        <>
          <Typography variant="h6" gutterBottom>
            Resultados das Previsões
          </Typography>

          <TableContainer component={Paper} sx={{ mb: 4 }}>
            <Table size="small" aria-label="Tabela de previsões">
              <TableHead>
                <TableRow>
                  <TableCell>Identificador</TableCell>
                  <TableCell>Cluster Previsto</TableCell>
                  <TableCell>Target 1 Previsto</TableCell>
                  <TableCell>Target 2 Previsto</TableCell>
                  <TableCell>Target 3 Previsto</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {predictions.map((p) => (
                  <TableRow key={p.identifier} hover>
                    <TableCell>{p.identifier}</TableCell>
                    <TableCell>{p.predicted_cluster != null ? p.predicted_cluster : "-"}</TableCell>
                    <TableCell>{fmt(p.predicted_target1)}</TableCell>
                    <TableCell>{fmt(p.predicted_target2)}</TableCell>
                    <TableCell>{fmt(p.predicted_target3)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          <Typography variant="h6" gutterBottom>
            Análise Detalhada por Jogador
          </Typography>

          <FormControl fullWidth sx={{ mb: 4 }}>
            <InputLabel id="player-select-label">Selecione um jogador</InputLabel>
            <Select
              labelId="player-select-label"
              value={selectedPlayerId}
              label="Selecione um jogador"
              onChange={(e) => setSelectedPlayerId(e.target.value)}
            >
              {predictions.map((p) => (
                <MenuItem key={p.identifier} value={p.identifier}>
                  {p.identifier}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {selectedPlayerDetails ? (
            <Alert severity="info">
              O gráfico radar depende de perfis detalhados (player_profile e cluster_average_profile).
              Podemos reintroduzir isso adicionando um endpoint <code>/predict/legacy</code> no backend.
            </Alert>
          ) : null}
        </>
      )}

      <LoadingOverlay open={isLoading} stepText={loadingStep} />
    </Box>
  );
}

export default PredictionTool;
