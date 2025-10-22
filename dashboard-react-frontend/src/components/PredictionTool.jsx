import React, { useCallback, useMemo, useState, useEffect } from "react";
import {
  Box,
  AppBar,
  Toolbar,
  Container,
  Grid,
  Paper,
  Card,
  CardContent,
  Typography,
  Stack,
  Button,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  TableSortLabel,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Backdrop,
  Card,
  CardContent,
  Stack,
  TextField,
  Tooltip,
  Chip,
  Divider,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import Plot from "react-plotly.js";
import { runPrediction, getSchema, getRadar } from "../api/ApiService";

/* ===================== Overlay de loading ===================== */
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
  const visible = useLatch(open, 350);
  if (!visible) return null;
  return (
    <Backdrop
      open
      sx={{
        zIndex: (t) => t.zIndex.modal + 2,
        backdropFilter: "blur(8px)",
        background:
          "radial-gradient(1200px 600px at 50% -10%, rgba(36,99,235,0.10), transparent), rgba(8,8,18,0.72)",
      }}
      aria-live="polite"
      aria-busy="true"
    >
      <Card
        elevation={0}
        sx={{
          p: 4,
          width: 380,
          borderRadius: 4,
          bgcolor: "rgba(18,21,30,0.85)",
          border: "1.5px solid rgba(59,130,246,0.12)",
          boxShadow: "0 10px 36px 0 rgba(36,99,235,0.16)",
        }}
      >
        <Box sx={{ display: "flex", justifyContent: "center", mb: 2 }}>
          <CircularProgress />
        </Box>
        <Typography
          variant="h6"
          align="center"
          sx={{ mb: 1, color: "#F8FAFC", fontWeight: 700, letterSpacing: 0.2 }}
        >
          Processando análise
        </Typography>
        <Typography
          variant="body2"
          align="center"
          sx={{ color: "rgba(203,213,225,0.95)", fontWeight: 400 }}
        >
          {stepText || "Preparando..."}
        </Typography>
      </Card>
    </Backdrop>
  );
};

// Helpers e schema (NÃO MODIFICAR lógica)
const removeAccents = (str) => {
  if (typeof str !== "string") return str;
  return str.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
};
const normalizeName = (s) => {
  if (s == null) return s;
  let out = String(s).trim();
  out = removeAccents(out)
    .replace(/\s+/g, "_")
    .replace(/[^a-zA-Z0-9_]/g, "");
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
  if (!Array.isArray(schemaRaw))
    throw new Error("Schema inválido: não é array.");
  if (schemaRaw.length === 0) return [];
  if (typeof schemaRaw[0] === "string") {
    return schemaRaw.map((name) => ({
      name,
      type: "number",
      default: 0,
      label: name,
    }));
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
          if (typeof v === "number")
            row[fname] = Number.isFinite(v) ? v : f.default || 0;
          else {
            const num = parseFloat(String(v).replace(",", "."));
            row[fname] = Number.isFinite(num) ? num : f.default || 0;
          }
        } else {
          row[fname] = v != null ? v : f.default || "";
        }
        matches++;
      } else {
        row[fname] =
          f.default != null ? f.default : f.type === "number" ? 0 : "";
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
      if (
        Object.prototype.hasOwnProperty.call(orig, cNorm) &&
        orig[cNorm] != null &&
        orig[cNorm] !== ""
      ) {
        identifier = orig[cNorm];
        break;
      }
      if (
        Object.prototype.hasOwnProperty.call(orig, c) &&
        orig[c] != null &&
        orig[c] !== ""
      ) {
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

/* ===== Helpers do gráfico de Faixas ===== */
const toPercent = (arr) => {
  const xs = arr.filter((v) => Number.isFinite(v));
  if (!xs.length) return arr;
  const maxv = Math.max(...xs);
  return maxv <= 1 ? arr.map((v) => (Number.isFinite(v) ? v * 100 : v)) : arr;
};
const bucketize = (vals, low = 30, high = 60) => {
  let lt = 0,
    mid = 0,
    gt = 0;
  for (const v of vals) {
    if (!Number.isFinite(v)) continue;
    if (v < low) lt++;
    else if (v <= high) mid++;
    else gt++;
  }
  return { "<30": lt, "30-60": mid, ">60": gt };
};

const TARGETS = [
  { key: "predicted_target1", label: "Target1" },
  { key: "predicted_target2", label: "Target2" },
  { key: "predicted_target3", label: "Target3" },
];

const TT = {
  BUCKETS:
    "Contagem de pessoas por faixa de percentuais (<30 | 30–60 | >60) em cada target, a partir dos valores PREVISTOS.",
};
/* ===================== Fim helpers ===================== */

/* ========= Radar fetch ========= */
async function fetchRadar(playerRow, rowsData, target) {
  if (!playerRow) return null;

  const { __identifier, ...rest } = playerRow;
  const player = Object.fromEntries(
    Object.entries(rest).filter(
      ([k]) =>
        k !== "predicted_cluster" &&
        k !== "predicted_target1" &&
        k !== "predicted_target2" &&
        k !== "predicted_target3"
    )
  );

  // envia player + TODA a planilha normalizada para a MODA do backend
  return getRadar({ player, context_rows: rowsData }, target);
}

/* ========= Componente ========= */
function PredictionTool() {
  // Estado e lógica originais
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [rowsData, setRowsData] = useState([]); // <-- mantém as linhas normalizadas (para montar player)
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [selectedPlayerId, setSelectedPlayerId] = useState("");
  const [thresholds, setThresholds] = useState({ low: 30, high: 60 });
  const [orderBy, setOrderBy] = useState("identifier");
  const [order, setOrder] = useState("asc");

  // estado dos 3 radares
  const [radarLoading, setRadarLoading] = useState(false);
  const [radarError, setRadarError] = useState("");
  const [radarData, setRadarData] = useState({
    Target1: null,
    Target2: null,
    Target3: null,
  });

  const selectedPlayerDetails = useMemo(
    () => predictions.find((p) => p.identifier === selectedPlayerId),
    [predictions, selectedPlayerId]
  );

  const selectedPlayerRow = useMemo(() => {
    if (!selectedPlayerId) return null;
    const p = predictions.find((x) => x.identifier === selectedPlayerId);
    if (!p || typeof p.rowIndex !== "number") return null;
    return rowsData[p.rowIndex] || null;
  }, [rowsData, predictions, selectedPlayerId]);

  const handleFileChange = useCallback((event) => {
    const selectedFile = event.target.files?.[0] || null;
    if (selectedFile) {
      setFile(selectedFile);
      setPredictions([]);
      setRowsData([]);
      setError("");
      setSuccess("");
      setSelectedPlayerId("");
      setRadarData({ Target1: null, Target2: null, Target3: null });
      setRadarError("");
    }
  }, []);

  const handleSort = (key) => {
    if (orderBy === key) {
      setOrder((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setOrderBy(key);
      setOrder("asc");
    }
  };

  const sortedPredictions = useMemo(() => {
    const arr = [...predictions];
    const dir = order === "asc" ? 1 : -1;

    return arr.sort((a, b) => {
      const va = a[orderBy];
      const vb = b[orderBy];

      // null/undefined sempre ao final
      const aNull = va == null;
      const bNull = vb == null;
      if (aNull && bNull) return 0;
      if (aNull) return 1;
      if (bNull) return -1;

      // se for número, compara como número
      const na = Number(va);
      const nb = Number(vb);
      const aNum = Number.isFinite(na);
      const bNum = Number.isFinite(nb);

      if (aNum && bNum) return (na - nb) * dir;

      // fallback string
      return String(va).localeCompare(String(vb)) * dir;
    });
  }, [predictions, orderBy, order]);

  const readSheetInWorker = useCallback(async (excelFile) => {
    const worker = new Worker(
      new URL("../workers/excelWorker.js", import.meta.url),
      { type: "module" }
    );
    try {
      const buf = await excelFile.arrayBuffer();
      const result = await new Promise((resolve) => {
        worker.onmessage = (e) => resolve(e.data);
        worker.postMessage(buf);
      });
      if (!result.success)
        throw new Error(result.error || "Falha ao processar planilha");
      return result.data;
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
      setRowsData([]);
      setSelectedPlayerId("");
      setRadarData({ Target1: null, Target2: null, Target3: null });
      setRadarError("");

      setLoadingStep("Lendo a planilha");
      await Promise.resolve();
      const jsonData = await readSheetInWorker(file);
      if (!Array.isArray(jsonData) || jsonData.length === 0) {
        throw new Error("O arquivo está vazio ou em um formato inválido.");
      }
      setLoadingStep("Alinhando com schema");
      const schemaRaw = await getSchema();
      const schema = adaptSchema(schemaRaw);
      const rows = buildRowsFromSchema(jsonData, schema);
      setRowsData(rows); // <-- guardamos as linhas para montar 'player' depois

      setLoadingStep("Rodando o modelo");
      const predsDict = await runPrediction(rows);
      setLoadingStep("Gerando resultados");
      const n = Math.max(
        predsDict.target1?.length || 0,
        predsDict.target2?.length || 0,
        predsDict.target3?.length || 0
      );
      if (n === 0)
        throw new Error(
          "A API retornou zero previsões. Verifique os logs e o arquivo."
        );

      const safeT1 = predsDict.target1 || Array(n).fill(null);
      const safeT2 = predsDict.target2 || Array(n).fill(null);
      const safeT3 = predsDict.target3 || Array(n).fill(null);
      const table = Array.from({ length: n }).map((_, i) => {
        const id = String(rows[i]?.__identifier || "").trim() || `lin_${i + 1}`;
        return {
          identifier: id,
          rowIndex: i,
          predicted_target1: safeT1[i] != null ? safeT1[i] : null,
          predicted_target2: safeT2[i] != null ? safeT2[i] : null,
          predicted_target3: safeT3[i] != null ? safeT3[i] : null,
        };
      });
      setPredictions(table);
      setSuccess(
        `Análise concluída com sucesso para ${table.length} jogadores.`
      );
    } catch (err) {
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

  // Buckets
  const buckets = useMemo(() => {
    const out = {};
    for (const t of TARGETS) {
      const preds = predictions
        .map((r) => Number(r[t.key]))
        .filter((v) => Number.isFinite(v));
      const predsPercent = toPercent(preds);
      out[t.label] = bucketize(predsPercent, thresholds.low, thresholds.high);
    }
    return out;
  }, [predictions, thresholds]);

  // Quando um jogador é selecionado, buscar os 3 radares
  useEffect(() => {
    let cancel = false;
    async function loadRadars() {
      if (!selectedPlayerId || !selectedPlayerRow) {
        setRadarData({ Target1: null, Target2: null, Target3: null });
        setRadarError("");
        return;
      }
      setRadarLoading(true);
      setRadarError("");
      try {
        const [r1, r2, r3] = await Promise.all([
          fetchRadar(selectedPlayerRow, rowsData, "Target1"),
          fetchRadar(selectedPlayerRow, rowsData, "Target2"),
          fetchRadar(selectedPlayerRow, rowsData, "Target3"),
        ]);
        if (!cancel) {
          setRadarData({ Target1: r1, Target2: r2, Target3: r3 });
        }
      } catch (e) {
        console.error(e);
        if (!cancel) setRadarError(e?.message || "Erro ao carregar radares.");
      } finally {
        if (!cancel) setRadarLoading(false);
      }
    }
    loadRadars();
    return () => {
      cancel = true;
    };
  }, [selectedPlayerId, selectedPlayerRow, rowsData]);

  // Componente Radar (Plotly)
  const Radar = ({ title, data }) => {
    if (!data)
      return (
        <Card
          variant="outlined"
          sx={{
            p: 2,
            minHeight: 320,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Typography variant="body2" color="text.secondary">
            Sem dados para {title}
          </Typography>
        </Card>
      );

    const labels = data.labels || [];
    const player = data.player_profile || [];
    const cluster = data.cluster_average_profile || [];

    const playerRaw = (data.player_raw_values ?? player).map((v, i) => [
      v,
      labels[i],
    ]);
    const referenceRaw = (data.reference_raw_values ?? cluster).map((v, i) => [
      v,
      labels[i],
    ]);

    return (
      <Card variant="outlined" sx={{ p: 1.5 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 1 }}>
          {title}
        </Typography>
        <Plot
          data={[
            {
              type: "scatterpolar",
              r: player,
              theta: labels,
              fill: "toself",
              name: "Jogador",
              customdata: playerRaw, // nunca vazio
              hovertemplate:
                "%{customdata[1]}<br>" +
                "Likert do Jogador: %{customdata[0]}<br>" +
                "Valor: %{r:.1f}/5" +
                "<extra>Jogador</extra>",
            },
            {
              type: "scatterpolar",
              r: cluster,
              theta: labels,
              fill: "toself",
              name: "Moda Geral",
              customdata: referenceRaw, // nunca vazio
              hovertemplate:
                "%{customdata[1]}<br>" +
                "Referência (moda): %{customdata[0]}<br>" +
                "Valor: %{r:.1f}/5" +
                "<extra>Moda Geral</extra>",
            },
          ]}
          layout={{
            polar: {
              radialaxis: {
                visible: true,
                range: [0, 5],
                tick0: 0,
                dtick: 1,
                tickfont: { color: "#000" },
              },
            },
            margin: { l: 40, r: 40, t: 20, b: 20 },
            paper_bgcolor: "#29384B",
            plot_bgcolor: "#29384B",
            font: { color: "#FFFFFF" },
            legend: { orientation: "h", x: 0, y: 1.1 },
          }}
          useResizeHandler
          style={{ width: "100%", height: 320 }}
        />
      </Card>
    );
  };

  return (
    <Box
      sx={{ minHeight: "100vh", width: "100%", bgcolor: "background.default" }}
    >
      {/* Header fixo */}
      <AppBar
        position="static"
        elevation={0}
        color="transparent"
        sx={{
          background: "linear-gradient(90deg, #172554 0%, #2563EB 40%, #3B82F6 100%)",
          boxShadow: "0 2px 18px 0 rgba(36,99,235,0.12)",
          px: { xs: 1.5, md: 4 },
        }}
      >
        <Toolbar
          disableGutters
          sx={{
            minHeight: 64,
            px: { xs: 1.5, md: 0 },
            justifyContent: "space-between",
            alignItems: "center",
            width: "100%",
            maxWidth: 1800,
            mx: "auto",
          }}
        >
          <Typography
            variant="h6"
            sx={{
              fontWeight: 900,
              color: "#F8FAFC",
              letterSpacing: 0.8,
              textShadow: "0 2px 12px #1e293b55",
              fontSize: { xs: "1.18rem", md: "1.32rem" },
              flexGrow: 1,
              userSelect: "none",
            }}
          >
            Calcular Targets para Novos Jogadores
          </Typography>
          <Button
            onClick={handlePredict}
            variant="contained"
            aria-label="Realizar análise"
            disabled={!file || isLoading}
            sx={{
              px: 3.2,
              py: 1.25,
              fontWeight: 700,
              borderRadius: 4,
              background: "linear-gradient(90deg, #2563EB 0%, #3B82F6 100%)",
              color: "#F8FAFC",
              boxShadow: "0 2px 14px 0 rgba(36, 99, 235, 0.12)",
              fontSize: "1rem",
              transition: "0.17s",
              "&:hover": {
                background: "linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%)",
                boxShadow: "0 4px 22px 0 rgba(36, 99, 235, 0.15)",
              },
              "&:disabled": {
                background: "linear-gradient(90deg, #334155 0%, #64748B 100%)",
                color: "#CBD5E1",
                opacity: 0.8,
              },
            }}
          >
            {isLoading ? (
              <CircularProgress size={22} sx={{ color: "#F8FAFC" }} />
            ) : (
              "Realizar Análise"
            )}
          </Button>
        </Toolbar>
      </AppBar>

      <Container
        maxWidth={false}
        disableGutters
        sx={{ py: 3, px: { xs: 2, md: 4 } }}
      >
        <Stack spacing={2} sx={{ mb: 3 }}>
          {error && <Alert severity="error">{error}</Alert>}
          {success && <Alert severity="success">{success}</Alert>}
        </Stack>

        {/* Layout principal */}
        <Grid
          container
          spacing={2}
          alignItems="flex-start"
          justifyContent="space-between"
          sx={{ px: 3, py: 3 }}
        >
          {/* ESQUERDA: Fonte + Filtro + Detalhes + 3 Radares */}
          <Grid item xs={12} md={5} width="49%">
            {/* Fonte dos dados */}
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  Fonte dos dados
                </Typography>
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ mb: 2 }}
                >
                  Carregue o arquivo Excel com os jogadores para iniciar a
                  análise.
                </Typography>

                <Stack direction="row" spacing={2} alignItems="center">
                  <Button
                    variant="contained"
                    component="label"
                    aria-label="Carregar Excel"
                    disabled={isLoading}
                    sx={{
                      px: 3.2,
                      py: 1.3,
                      mt: 1,
                      width: "100%",
                      maxWidth: 290,
                      fontWeight: 700,
                      fontSize: "1rem",
                      borderRadius: 3.5,
                      background: "linear-gradient(90deg, #2563EB 0%, #3B82F6 100%)",
                      color: "#F8FAFC",
                      boxShadow: "0 2px 12px 0 rgba(36,99,235,0.13)",
                      transition: "0.15s",
                      "&:hover": {
                        background: "linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%)",
                        boxShadow: "0 6px 16px 0 rgba(36, 99, 235, 0.18)",
                      },
                      "&:disabled": {
                        background: "linear-gradient(90deg, #334155 0%, #64748B 100%)",
                        color: "#CBD5E1",
                        opacity: 0.8,
                      },
                      alignSelf: "center",
                      whiteSpace: "nowrap",
                    }}
                  >
                    Carregar Excel
                    <input
                      type="file"
                      accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                      hidden
                      onChange={handleFileChange}
                      disabled={isLoading}
                    />
                  </Button>
                  <Typography
                    variant="body2"
                    sx={{
                      color: "#F1F5F9",
                      opacity: 0.85,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      maxWidth: 260,
                      fontWeight: 500,
                      fontSize: "1rem",
                      mt: 1,
                      textAlign: "center",
                      width: "100%",
                    }}
                  >
                    {file ? file.name : "Nenhum arquivo selecionado"}
                  </Typography>
                </Stack>
              </CardContent>
            </Card>

            {/* Filtro e detalhes */}
            <Card variant="outlined" sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                  Filtro e detalhes
                </Typography>

                {predictions.length > 0 ? (
                  <FormControl fullWidth>
                    <InputLabel id="player-select-label">
                      Selecione um jogador
                    </InputLabel>
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
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Após carregar o Excel e executar a análise, você poderá
                    selecionar um jogador aqui para ver mais detalhes.
                  </Typography>
                )}

                {/* Card com as MESMAS infos mostradas na lista (mas do jogador selecionado) */}
                {selectedPlayerDetails && (
                  <Card
                    variant="outlined"
                    sx={{ mt: 2, bgcolor: "rgba(255,255,255,0.02)" }}
                  >
                    <CardContent>
                      <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                        Informações do Jogador Selecionado
                      </Typography>
                      <Divider sx={{ my: 1 }} />
                      <Stack spacing={0.5}>
                        <Typography variant="body2">
                          <strong>Identificador:</strong>{" "}
                          {selectedPlayerDetails.identifier}
                        </Typography>
                        <Typography variant="body2">
                          <strong>Target 1 Previsto:</strong>{" "}
                          {fmt(selectedPlayerDetails.predicted_target1)}
                        </Typography>
                        <Typography variant="body2">
                          <strong>Target 2 Previsto:</strong>{" "}
                          {fmt(selectedPlayerDetails.predicted_target2)}
                        </Typography>
                        <Typography variant="body2">
                          <strong>Target 3 Previsto:</strong>{" "}
                          {fmt(selectedPlayerDetails.predicted_target3)}
                        </Typography>
                      </Stack>
                    </CardContent>
                  </Card>
                )}

                {/* Status dos radares */}
                {selectedPlayerId && (
                  <>
                    {radarLoading && (
                      <Alert severity="info" sx={{ mt: 2 }}>
                        Carregando perfis de radar…
                      </Alert>
                    )}
                    {radarError && (
                      <Alert severity="error" sx={{ mt: 2 }}>
                        {radarError}
                      </Alert>
                    )}
                  </>
                )}
              </CardContent>
            </Card>

            {/* 3 Radares */}
            {selectedPlayerId && (
              <Stack
                direction="row"
                spacing={2}
                sx={{ mt: 3 }}
                useFlexGap
                flexWrap="wrap" // permite quebrar em telas menores
              >
                <Box sx={{ flex: "1 1 31%", minWidth: 280 }}>
                  <Radar title="Radar — Target1" data={radarData.Target1} />
                </Box>
                <Box sx={{ flex: "1 1 31%", minWidth: 280 }}>
                  <Radar title="Radar — Target2" data={radarData.Target2} />
                </Box>
                <Box sx={{ flex: "1 1 31%", minWidth: 280 }}>
                  <Radar title="Radar — Target3" data={radarData.Target3} />
                </Box>
              </Stack>
            )}
          </Grid>

          {/* DIREITA: Resultados + Gráfico de Faixas */}
          <Grid item xs={12} md={5} width="49%">
            <Box
              sx={{ height: "100%", display: "flex", flexDirection: "column" }}
            >
              <Typography variant="h6" sx={{ mb: 1 }}>
                Resultados das Previsões
              </Typography>

              {!predictions.length && (
                <Paper
                  variant="outlined"
                  sx={{ flex: 1, minHeight: 420, opacity: 0.3 }}
                />
              )}

              {!!predictions.length && (
                <TableContainer
                  component={Paper}
                  sx={{
                    flex: 1,
                    minHeight: 420,
                    maxHeight: 400,
                    overflowY: "auto",
                  }}
                >
                  <Table
                    size="small"
                    stickyHeader
                    aria-label="Tabela de previsões"
                  >
                    <TableHead>
                      <TableRow>
                        <TableCell
                          sortDirection={
                            orderBy === "identifier" ? order : false
                          }
                        >
                          <TableSortLabel
                            active={orderBy === "identifier"}
                            direction={orderBy === "identifier" ? order : "asc"}
                            onClick={() => handleSort("identifier")}
                          >
                            Identificador
                          </TableSortLabel>
                        </TableCell>

                        <TableCell
                          sortDirection={
                            orderBy === "predicted_target1" ? order : false
                          }
                        >
                          <TableSortLabel
                            active={orderBy === "predicted_target1"}
                            direction={
                              orderBy === "predicted_target1" ? order : "asc"
                            }
                            onClick={() => handleSort("predicted_target1")}
                          >
                            Target 1 Previsto
                          </TableSortLabel>
                        </TableCell>

                        <TableCell
                          sortDirection={
                            orderBy === "predicted_target2" ? order : false
                          }
                        >
                          <TableSortLabel
                            active={orderBy === "predicted_target2"}
                            direction={
                              orderBy === "predicted_target2" ? order : "asc"
                            }
                            onClick={() => handleSort("predicted_target2")}
                          >
                            Target 2 Previsto
                          </TableSortLabel>
                        </TableCell>

                        <TableCell
                          sortDirection={
                            orderBy === "predicted_target3" ? order : false
                          }
                        >
                          <TableSortLabel
                            active={orderBy === "predicted_target3"}
                            direction={
                              orderBy === "predicted_target3" ? order : "asc"
                            }
                            onClick={() => handleSort("predicted_target3")}
                          >
                            Target 3 Previsto
                          </TableSortLabel>
                        </TableCell>
                      </TableRow>
                    </TableHead>

                    <TableBody>
                      {sortedPredictions.map((p) => (
                        <TableRow
                          key={p.identifier}
                          hover
                          selected={selectedPlayerId === p.identifier}
                          onClick={() => setSelectedPlayerId(p.identifier)}
                          sx={{ cursor: "pointer" }}
                        >
                          <TableCell>{p.identifier}</TableCell>
                          <TableCell>{fmt(p.predicted_target1)}</TableCell>
                          <TableCell>{fmt(p.predicted_target2)}</TableCell>
                          <TableCell>{fmt(p.predicted_target3)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}

              {/* === Gráfico de Agrupamento por Faixas (Previsto) + OVERLAY === */}
              {!!predictions.length && (
                <>
                  <Typography variant="h6" sx={{ mt: 3, mb: 1 }}>
                    Agrupamento por Faixas (%) — por Target (Previsto)
                  </Typography>

                  <Tooltip title={TT.BUCKETS} arrow>
                    <Box sx={{ width: "100%", mb: 2, position: "relative" }}>
                      <Plot
                        data={[
                          {
                            x: TARGETS.map((t) => t.label),
                            y: TARGETS.map(
                              (t) => buckets[t.label]?.["<30"] ?? 0
                            ),
                            type: "bar",
                            name: "< 30",
                            hovertemplate: "%{y} abaixo de 30<extra></extra>",
                          },
                          {
                            x: TARGETS.map((t) => t.label),
                            y: TARGETS.map(
                              (t) => buckets[t.label]?.["30-60"] ?? 0
                            ),
                            type: "bar",
                            name: "30 – 60",
                            hovertemplate: "%{y} entre 30 e 60<extra></extra>",
                          },
                          {
                            x: TARGETS.map((t) => t.label),
                            y: TARGETS.map(
                              (t) => buckets[t.label]?.[">60"] ?? 0
                            ),
                            type: "bar",
                            name: "> 60",
                            hovertemplate: "%{y} acima de 60<extra></extra>",
                          },
                        ]}
                        layout={{
                          barmode: "stack",
                          title: `Distribuição por Faixas`,
                          xaxis: { title: "Targets" },
                          yaxis: { title: "Quantidade de pessoas" },
                          autosize: true,
                          paper_bgcolor: "#29384B",
                          plot_bgcolor: "#29384B",
                          font: { color: "#FFFFFF" },
                          legend: { orientation: "h" },
                          margin: { t: 50, r: 20, b: 50, l: 50 },
                          annotations: [
                            {
                              text: `Limiar baixo: ${thresholds.low} | alto: ${thresholds.high}`,
                              xref: "paper",
                              yref: "paper",
                              x: 0,
                              y: 1.12,
                              showarrow: false,
                              align: "left",
                              font: { size: 12, color: "#ddd" },
                            },
                          ],
                        }}
                        useResizeHandler
                        style={{ width: "100%", height: 360 }}
                      />
                    </Box>
                  </Tooltip>
                </>
              )}
            </Box>
          </Grid>
        </Grid>
      </Container>
      {/* Overlay de loading */}
      <LoadingOverlay open={isLoading} stepText={loadingStep} />
    </Box>
  );
}

export default PredictionTool;