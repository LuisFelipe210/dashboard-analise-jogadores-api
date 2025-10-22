import React, { useCallback, useMemo, useState, useEffect } from "react";
import {
  Box,
  AppBar,
  Toolbar,
  Container,
  Grid,
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
  TableSortLabel,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Backdrop,
  Tooltip,
  Divider,
  Paper, // Adicionado Paper para a tabela
} from "@mui/material";
import Plot from "react-plotly.js";
import { runPrediction, getSchema, getRadar } from "../api/ApiService";

// =========================
// helpers e schema (LÓGICA NÃO MODIFICADA)
// =========================
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
const toPercent = (arr) => {
  const xs = arr.filter((v) => Number.isFinite(v));
  if (!xs.length) return arr;
  const maxv = Math.max(...xs);
  return maxv <= 1 ? arr.map((v) => (Number.isFinite(v) ? v * 100 : v)) : arr;
};
const bucketize = (vals, low = 30, high = 60) => {
  let lt = 0, mid = 0, gt = 0;
  for (const v of vals) {
    if (!Number.isFinite(v)) continue;
    if (v < low) lt++;
    else if (v <= high) mid++;
    else gt++;
  }
  return { "<30": lt, "30-60": mid, ">60": gt };
};

// =========================
// Overlay de Loading (Estilizado)
// =========================
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
    <Backdrop open sx={{ zIndex: (t) => t.zIndex.modal + 2, backdropFilter: "blur(8px)", background: "rgba(8,8,18,0.72)" }}>
      <Card
        elevation={0}
        sx={{
          p: 4, width: 380, borderRadius: 4,
          background: "rgba(30,41,59,0.9)",
          border: "1px solid rgba(148,163,184,0.1)",
          boxShadow: "0 10px 36px 0 rgba(0,0,0,0.25)",
        }}
      >
        <Box sx={{ display: "flex", justifyContent: "center", mb: 2 }}>
          <CircularProgress sx={{ color: "#A5B4FC" }} />
        </Box>
        <Typography variant="h6" align="center" sx={{ mb: 1, color: "#F8FAFC", fontWeight: 700 }}>
          Processando análise
        </Typography>
        <Typography variant="body2" align="center" sx={{ color: "#CBD5E1" }}>
          {stepText || "Preparando..."}
        </Typography>
      </Card>
    </Backdrop>
  );
};

const TARGETS = [
  { key: "predicted_target1", label: "Target1" },
  { key: "predicted_target2", label: "Target2" },
  { key: "predicted_target3", label: "Target3" },
];
const TT = { BUCKETS: "Contagem de pessoas por faixa de percentuais (<30 | 30–60 | >60) em cada target, a partir dos valores PREVISTOS." };

async function fetchRadar(playerRow, rowsData, target) {
  if (!playerRow) return null;
  const { __identifier, ...rest } = playerRow;
  const player = Object.fromEntries(
    Object.entries(rest).filter(([k]) => k !== "predicted_cluster" && !k.startsWith("predicted_target"))
  );
  return getRadar({ player, context_rows: rowsData }, target);
}

// =========================
// Componente Principal
// =========================
function PredictionTool() {
  // Lógica de estado original (NÃO MODIFICADA)
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [rowsData, setRowsData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [selectedPlayerId, setSelectedPlayerId] = useState("");
  const [thresholds, setThresholds] = useState({ low: 30, high: 60 });
  const [orderBy, setOrderBy] = useState("identifier");
  const [order, setOrder] = useState("asc");
  const [radarLoading, setRadarLoading] = useState(false);
  const [radarError, setRadarError] = useState("");
  const [radarData, setRadarData] = useState({ Target1: null, Target2: null, Target3: null });

  // Memos e Callbacks originais (NÃO MODIFICADOS)
  const selectedPlayerDetails = useMemo(() => predictions.find((p) => p.identifier === selectedPlayerId), [predictions, selectedPlayerId]);
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
    if (orderBy === key) setOrder((prev) => (prev === "asc" ? "desc" : "asc"));
    else { setOrderBy(key); setOrder("asc"); }
  };
  const sortedPredictions = useMemo(() => {
    const arr = [...predictions];
    const dir = order === "asc" ? 1 : -1;
    return arr.sort((a, b) => {
      const va = a[orderBy]; const vb = b[orderBy];
      if (va == null) return 1; if (vb == null) return -1;
      const na = Number(va); const nb = Number(vb);
      if (Number.isFinite(na) && Number.isFinite(nb)) return (na - nb) * dir;
      return String(va).localeCompare(String(vb)) * dir;
    });
  }, [predictions, orderBy, order]);
  const readSheetInWorker = useCallback(async (excelFile) => {
    const worker = new Worker(new URL("../workers/excelWorker.js", import.meta.url), { type: "module" });
    try {
      const buf = await excelFile.arrayBuffer();
      const result = await new Promise((resolve) => {
        worker.onmessage = (e) => resolve(e.data);
        worker.postMessage(buf);
      });
      if (!result.success) throw new Error(result.error || "Falha ao processar planilha");
      return result.data;
    } finally { worker.terminate(); }
  }, []);
  const handlePredict = useCallback(async () => {
    try {
      if (!file) { setError("Por favor, carregue um arquivo primeiro."); return; }
      setIsLoading(true); setError(""); setSuccess(""); setPredictions([]); setRowsData([]);
      setSelectedPlayerId(""); setRadarData({ Target1: null, Target2: null, Target3: null }); setRadarError("");
      setLoadingStep("Lendo a planilha"); await Promise.resolve();
      const jsonData = await readSheetInWorker(file);
      if (!Array.isArray(jsonData) || jsonData.length === 0) throw new Error("O arquivo está vazio ou em um formato inválido.");
      setLoadingStep("Alinhando com schema");
      const schemaRaw = await getSchema();
      const schema = adaptSchema(schemaRaw);
      const rows = buildRowsFromSchema(jsonData, schema);
      setRowsData(rows);
      setLoadingStep("Rodando o modelo");
      const predsDict = await runPrediction(rows);
      setLoadingStep("Gerando resultados");
      const n = Math.max(predsDict.target1?.length || 0, predsDict.target2?.length || 0, predsDict.target3?.length || 0);
      if (n === 0) throw new Error("A API retornou zero previsões. Verifique os logs e o arquivo.");
      const safeT1 = predsDict.target1 || Array(n).fill(null);
      const safeT2 = predsDict.target2 || Array(n).fill(null);
      const safeT3 = predsDict.target3 || Array(n).fill(null);
      const table = Array.from({ length: n }).map((_, i) => ({
        identifier: String(rows[i]?.__identifier || "").trim() || `lin_${i + 1}`,
        rowIndex: i,
        predicted_target1: safeT1[i] != null ? safeT1[i] : null,
        predicted_target2: safeT2[i] != null ? safeT2[i] : null,
        predicted_target3: safeT3[i] != null ? safeT3[i] : null,
      }));
      setPredictions(table);
      setSuccess(`Análise concluída com sucesso para ${table.length} jogadores.`);
    } catch (err) {
      console.error(err);
      const msg = err?.message || "Ocorreu um erro desconhecido ao processar o arquivo.";
      setError(msg);
    } finally { setIsLoading(false); setLoadingStep(""); }
  }, [file, readSheetInWorker]);
  const buckets = useMemo(() => {
    const out = {};
    for (const t of TARGETS) {
      const preds = predictions.map((r) => Number(r[t.key])).filter(Number.isFinite);
      const predsPercent = toPercent(preds);
      out[t.label] = bucketize(predsPercent, thresholds.low, thresholds.high);
    }
    return out;
  }, [predictions, thresholds]);
  useEffect(() => {
    let cancel = false;
    async function loadRadars() {
      if (!selectedPlayerId || !selectedPlayerRow) {
        setRadarData({ Target1: null, Target2: null, Target3: null }); setRadarError(""); return;
      }
      setRadarLoading(true); setRadarError("");
      try {
        const [r1, r2, r3] = await Promise.all([
          fetchRadar(selectedPlayerRow, rowsData, "Target1"),
          fetchRadar(selectedPlayerRow, rowsData, "Target2"),
          fetchRadar(selectedPlayerRow, rowsData, "Target3"),
        ]);
        if (!cancel) setRadarData({ Target1: r1, Target2: r2, Target3: r3 });
      } catch (e) {
        console.error(e);
        if (!cancel) setRadarError(e?.message || "Erro ao carregar radares.");
      } finally { if (!cancel) setRadarLoading(false); }
    }
    loadRadars();
    return () => { cancel = true; };
  }, [selectedPlayerId, selectedPlayerRow, rowsData]);

  // Estilo de Card reutilizado do AnalysisDashboard
  const cardStyle = {
    background: "rgba(30,41,59,0.82)",
    backdropFilter: "blur(10px)",
    border: "1px solid rgba(148,163,184,0.08)",
    borderRadius: 4,
    boxShadow: "0 2px 16px 0 rgba(30,41,59,0.22)",
    color: "#E2E8F0",
  };

  const Radar = ({ title, data }) => {
    if (!data) return (
        <Card variant="outlined" sx={{ ...cardStyle, p: 2, minHeight: 320, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(51,65,85,0.13)" }}>
          <Typography variant="body2" color="#94A3B8"> Sem dados para {title} </Typography>
        </Card>
      );
    const labels = data.labels || []; const player = data.player_profile || []; const cluster = data.cluster_average_profile || [];
    const playerRaw = (data.player_raw_values ?? player).map((v, i) => [v, labels[i]]);
    const referenceRaw = (data.reference_raw_values ?? cluster).map((v, i) => [v, labels[i]]);

    return (
      <Card variant="outlined" sx={{ ...cardStyle, p: 1.5, background: "rgba(51,65,85,0.13)" }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 1, color: "#F1F5F9" }}> {title} </Typography>
        <Plot
          data={[
            { type: "scatterpolar", r: player, theta: labels, fill: "toself", name: "Jogador", customdata: playerRaw, hovertemplate: "%{customdata[1]}<br>Valor: %{r:.1f}/5<extra>Jogador</extra>", marker: { color: "#3B82F6" } },
            { type: "scatterpolar", r: cluster, theta: labels, fill: "toself", name: "Moda Geral", customdata: referenceRaw, hovertemplate: "%{customdata[1]}<br>Valor: %{r:.1f}/5<extra>Moda Geral</extra>", marker: { color: "#10B981" } },
          ]}
          layout={{
            polar: { radialaxis: { visible: true, range: [0, 5], tickfont: { color: "#94A3B8" } }, angularaxis: { tickfont: { color: "#CBD5E1" } } },
            margin: { l: 40, r: 40, t: 20, b: 20 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#E2E8F0" }, legend: { orientation: "h", x: 0.5, y: -0.1, xanchor: 'center' },
          }}
          useResizeHandler style={{ width: "100%", height: 320 }} config={{ displayModeBar: false }}
        />
      </Card>
    );
  };

  return (
    <Box sx={{ minHeight: "100vh", width: "100%", background: "linear-gradient(180deg, #0F172A 0%, #1E2D3B 100%)", color: "#E2E8F0" }}>
      <AppBar position="static" elevation={0} color="transparent" sx={{ px: { xs: 1.5, md: 4 } }}>
        <Toolbar disableGutters sx={{ minHeight: 64, mx: "auto", width: "100%", maxWidth: 1800, justifyContent: "space-between" }}>
          <Typography variant="h6" sx={{ fontWeight: 800, color: "#F8FAFC" }}>
            Calcular Targets para Novos Jogadores
          </Typography>
          <Button
            onClick={handlePredict} variant="contained" aria-label="Realizar análise" disabled={!file || isLoading}
            sx={{
              background: "linear-gradient(90deg, #2563EB 0%, #3B82F6 100%)", color: "#fff", textTransform: "none", fontWeight: 600, borderRadius: 2, px: 2.5,
              boxShadow: "0 2px 8px 0 rgba(30,41,59,0.18)",
              "&:hover": { background: "linear-gradient(90deg, #1E40AF 0%, #2563EB 100%)" },
              "&:disabled": { background: "#334155", color: "#94A3B8" }
            }}
          >
            {isLoading ? <CircularProgress size={22} color="inherit" /> : "Realizar Análise"}
          </Button>
        </Toolbar>
      </AppBar>

      <Container maxWidth={false} disableGutters sx={{ py: 3, px: { xs: 2, md: 4 } }}>
        <Stack spacing={2} sx={{ mb: 3 }}>
          {error && <Alert severity="error" sx={{ borderRadius: 2 }}>{error}</Alert>}
          {success && <Alert severity="success" sx={{ borderRadius: 2 }}>{success}</Alert>}
        </Stack>

        {/* LAYOUT ORIGINAL PRESERVADO */}
        <Grid container spacing={2} alignItems="flex-start" justifyContent="space-between" sx={{ px: 3, py: 3 }}>
          {/* ESQUERDA: Layout original */}
          <Grid item xs={12} md={5} width="49%">
            <Card variant="outlined" sx={{...cardStyle, mb: 3}}>
              <CardContent>
                <Typography variant="subtitle1" sx={{ fontWeight: 700, color: "#F1F5F9", mb: 1 }}> Fonte dos dados </Typography>
                <Typography variant="body2" color="#CBD5E1" sx={{ mb: 2 }}> Carregue o arquivo Excel para iniciar a análise. </Typography>
                <Stack direction="row" spacing={2} alignItems="center">
                  <Button
                    variant="contained" component="label" aria-label="Carregar Excel" disabled={isLoading}
                    sx={{
                      background: "linear-gradient(90deg, #2563EB 0%, #3B82F6 100%)", color: "#fff", textTransform: "none", fontWeight: 600, borderRadius: 2, px: 2.5,
                      "&:hover": { background: "linear-gradient(90deg, #1E40AF 0%, #2563EB 100%)" },
                    }}
                  >
                    Carregar Excel
                    <input type="file" accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" hidden onChange={handleFileChange} disabled={isLoading} />
                  </Button>
                  <Typography variant="body2" sx={{ color: "#CBD5E1", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 260 }}>
                    {file ? file.name : "Nenhum arquivo selecionado"}
                  </Typography>
                </Stack>
              </CardContent>
            </Card>

            <Card variant="outlined" sx={{ ...cardStyle, mb: 3 }}>
              <CardContent>
                <Typography variant="subtitle1" sx={{ fontWeight: 700, color: "#F1F5F9", mb: 2 }}> Filtro e detalhes </Typography>
                {predictions.length > 0 ? (
                  <FormControl fullWidth variant="filled" sx={{ "& .MuiFilledInput-root": { bgcolor: "rgba(51,65,85,0.5)" }, "& label": { color: "#94A3B8" }}}>
                    <InputLabel id="player-select-label">Selecione um jogador</InputLabel>
                    <Select labelId="player-select-label" value={selectedPlayerId} label="Selecione um jogador" onChange={(e) => setSelectedPlayerId(e.target.value)} sx={{color: "#F1F5F9"}}>
                      {predictions.map((p) => (<MenuItem key={p.identifier} value={p.identifier}>{p.identifier}</MenuItem>))}
                    </Select>
                  </FormControl>
                ) : (
                  <Typography variant="body2" color="#94A3B8">Após a análise, selecione um jogador aqui.</Typography>
                )}
                {selectedPlayerDetails && (
                  <Card variant="outlined" sx={{ mt: 2, background: "rgba(51,65,85,0.2)", borderRadius: 2, borderColor: "rgba(148,163,184,0.1)" }}>
                    <CardContent>
                      <Typography variant="subtitle2" sx={{ fontWeight: 700, color: "#A5B4FC" }}> Detalhes do Jogador </Typography>
                      <Divider sx={{ my: 1, borderColor: "rgba(148,163,184,0.1)" }} />
                      <Stack spacing={0.5}>
                        <Typography variant="body2"><strong>Identificador:</strong> {selectedPlayerDetails.identifier}</Typography>
                        <Typography variant="body2"><strong>Target 1 Previsto:</strong> {fmt(selectedPlayerDetails.predicted_target1)}</Typography>
                        <Typography variant="body2"><strong>Target 2 Previsto:</strong> {fmt(selectedPlayerDetails.predicted_target2)}</Typography>
                        <Typography variant="body2"><strong>Target 3 Previsto:</strong> {fmt(selectedPlayerDetails.predicted_target3)}</Typography>
                      </Stack>
                    </CardContent>
                  </Card>
                )}
                {selectedPlayerId && (
                  <Box sx={{mt: 2}}>
                    {radarLoading && <Alert severity="info" sx={{ bgcolor: 'rgba(59,130,246,0.1)', color: '#A5B4FC' }}>Carregando perfis...</Alert>}
                    {radarError && <Alert severity="error">{radarError}</Alert>}
                  </Box>
                )}
              </CardContent>
            </Card>

            {selectedPlayerId && (
              <Stack direction="row" spacing={2} useFlexGap flexWrap="wrap">
                <Box sx={{ flex: "1 1 31%", minWidth: 280 }}><Radar title="Radar — Target1" data={radarData.Target1} /></Box>
                <Box sx={{ flex: "1 1 31%", minWidth: 280 }}><Radar title="Radar — Target2" data={radarData.Target2} /></Box>
                <Box sx={{ flex: "1 1 31%", minWidth: 280 }}><Radar title="Radar — Target3" data={radarData.Target3} /></Box>
              </Stack>
            )}
          </Grid>

          {/* DIREITA: Layout original */}
          <Grid item xs={12} md={5} width="49%">
            <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
              <Typography variant="h6" sx={{ mb: 1, fontWeight: 700, color: "#F1F5F9" }}> Resultados das Previsões </Typography>
              {!predictions.length ? (
                <Paper variant="outlined" sx={{ ...cardStyle, flex: 1, minHeight: 420, display: 'flex', alignItems: 'center', justifyContent: 'center', background: "rgba(51,65,85,0.13)" }}>
                  <Typography color="#94A3B8">Os resultados aparecerão aqui</Typography>
                </Paper>
              ) : (
                <TableContainer component={Paper} sx={{ ...cardStyle, flex: 1, minHeight: 420, maxHeight: 400, overflowY: "auto", background: "rgba(30,41,59,0.82)" }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow sx={{ "& .MuiTableCell-root": { bgcolor: "rgba(15,23,42)", color: "#A5B4FC", fontWeight: 700, borderBottom: "1px solid rgba(148,163,184,0.15)" } }}>
                        <TableCell><TableSortLabel active={orderBy === "identifier"} direction={orderBy === "identifier" ? order : "asc"} onClick={() => handleSort("identifier")}>Identificador</TableSortLabel></TableCell>
                        <TableCell><TableSortLabel active={orderBy === "predicted_target1"} direction={orderBy === "predicted_target1" ? order : "asc"} onClick={() => handleSort("predicted_target1")}>Target 1 Previsto</TableSortLabel></TableCell>
                        <TableCell><TableSortLabel active={orderBy === "predicted_target2"} direction={orderBy === "predicted_target2" ? order : "asc"} onClick={() => handleSort("predicted_target2")}>Target 2 Previsto</TableSortLabel></TableCell>
                        <TableCell><TableSortLabel active={orderBy === "predicted_target3"} direction={orderBy === "predicted_target3" ? order : "asc"} onClick={() => handleSort("predicted_target3")}>Target 3 Previsto</TableSortLabel></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {sortedPredictions.map((p) => (
                        <TableRow key={p.identifier} hover selected={selectedPlayerId === p.identifier} onClick={() => setSelectedPlayerId(p.identifier)}
                                  sx={{ cursor: "pointer", "& .MuiTableCell-root": { borderBottomColor: "rgba(148,163,184,0.1)" }, "&.Mui-selected": { bgcolor: "rgba(59,130,246,0.15)" }, "&:hover": { bgcolor: "rgba(59,130,246,0.08)" } }}>
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

              {!!predictions.length && (
                <Box sx={{...cardStyle, mt: 3, p: 2}}>
                  <Typography variant="h6" sx={{ fontWeight: 700, color: "#F1F5F9", mb: 1 }}> Agrupamento por Faixas (%) </Typography>
                  <Tooltip title={TT.BUCKETS} arrow>
                    <Box sx={{ background: "rgba(51,65,85,0.13)", borderRadius: 2, p: 1 }}>
                      <Plot
                        data={[
                          { x: TARGETS.map((t) => t.label), y: TARGETS.map((t) => buckets[t.label]?.["<30"] ?? 0), type: "bar", name: "< 30", marker: {color: '#60A5FA'} },
                          { x: TARGETS.map((t) => t.label), y: TARGETS.map((t) => buckets[t.label]?.["30-60"] ?? 0), type: "bar", name: "30 – 60", marker: {color: '#34D399'} },
                          { x: TARGETS.map((t) => t.label), y: TARGETS.map((t) => buckets[t.label]?.[">60"] ?? 0), type: "bar", name: "> 60", marker: {color: '#FBBF24'} },
                        ]}
                        layout={{
                          barmode: "stack", autosize: true, paper_bgcolor: "transparent", plot_bgcolor: "transparent",
                          font: { color: "#E2E8F0" }, legend: { orientation: "h", y: -0.2, x: 0.5, xanchor: 'center' }, margin: { t: 20, r: 20, b: 60, l: 50 },
                          xaxis: { gridcolor: 'rgba(148,163,184,0.1)' }, yaxis: { gridcolor: 'rgba(148,163,184,0.1)' }
                        }}
                        useResizeHandler style={{ width: "100%", height: 360 }} config={{ displayModeBar: false }}
                      />
                    </Box>
                  </Tooltip>
                </Box>
              )}
            </Box>
          </Grid>
        </Grid>
      </Container>
      <LoadingOverlay open={isLoading} stepText={loadingStep} />
    </Box>
  );
}

export default PredictionTool;