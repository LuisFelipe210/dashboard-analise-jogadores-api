// src/components/AnalysisDashboard.jsx
import React, { useState, useEffect, useMemo } from "react";
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  OutlinedInput,
  Checkbox,
  ListItemText,
  Tooltip,
  Chip,
  Button,
} from "@mui/material";
import ClusterProfiles from "./ClusterProfiles";
import HelpOutline from "@mui/icons-material/HelpOutline";
import Download from "@mui/icons-material/Download";
import Plot from "react-plotly.js";

// ------------------------
// helpers
// ------------------------
const calcRMSE = (a, p) => {
  const n = a.length || 0;
  if (!n) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const e = (p[i] ?? 0) - (a[i] ?? 0);
    s += e * e;
  }
  return Math.sqrt(s / n);
};
const calcMAE = (a, p) => {
  const n = a.length || 0;
  if (!n) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.abs((p[i] ?? 0) - (a[i] ?? 0));
  return s / n;
};
const calcBias = (a, p) => {
  const n = a.length || 0;
  if (!n) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += (p[i] ?? 0) - (a[i] ?? 0);
  return s / n;
};
const calcR2 = (a, p) => {
  const n = a.length || 0;
  if (!n) return 0;
  const mean = a.reduce((x, y) => x + y, 0) / n;
  let ssT = 0,
    ssR = 0;
  for (let i = 0; i < n; i++) {
    ssT += (a[i] - mean) ** 2;
    ssR += (a[i] - p[i]) ** 2;
  }
  if (ssT === 0) return 1;
  return 1 - ssR / ssT;
};
const badge = (rmse, r2) => {
  if (r2 >= 0.5 && rmse <= 15) return { label: "Bom", color: "success" };
  if (r2 >= 0.3) return { label: "Ok", color: "warning" };
  return { label: "Atenção", color: "error" };
};

// tooltips (iguais)
const TT = {
  FILTER: "Escolha quais clusters ficam ativos na tela.",
  CARDS_PLAYERS: "Quantidade de jogadores após o filtro.",
  CARDS_CLUSTERS: "Clusters distintos no recorte.",
  CARDS_FEAT: "Número de variáveis de entrada esperadas pelo modelo.",
  CARDS_PIPE: "Se o pipeline de cluster está disponível no backend.",
  RMSE: "Erro médio em unidades do alvo. Menor é melhor.",
  R2: "Proporção de variação explicada. Pode ser negativa.",
  MAE: "Erro Absoluto Médio. Menor é melhor.",
  BIAS: "Média (Previsto − Real): positivo superestima; negativo subestima.",
  SCATTER: "Cada ponto é um jogador. Linha tracejada é o ideal.",
  RESIDUALS: "Histograma de resíduos: ideal é centrado em 0.",
  DISTRIB: "Quantos jogadores há em cada cluster.",
};

// download CSV
const downloadCSV = (rows) => {
  try {
    if (!rows?.length) return;
    const header = Object.keys(rows[0]);
    const csv = [header.join(",")]
      .concat(rows.map((r) => header.map((h) => r[h]).join(",")))
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "jogadores_filtrado.csv";
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    console.error(e);
  }
};

// CSV -> JSON robusto
const parseCSV = (text) => {
  try {
    const cleaned = text.trim();
    const lines = cleaned.split("\n").filter(Boolean);
    if (lines.length < 2) return [];
    const header = lines[0].split(",").map((h) => h.trim());
    const data = lines.slice(1).map((line) => {
      const vals = line.split(",");
      const obj = {};
      header.forEach((k, i) => {
        const raw = (vals[i] ?? "").trim();
        const num = Number(raw);
        obj[k] = Number.isFinite(num) && raw !== "" ? num : raw;
      });
      return obj;
    });
    return data.filter((r) => r.cluster !== undefined && r.cluster !== "");
  } catch (e) {
    console.error("parseCSV error", e);
    return [];
  }
};

function AnalysisDashboard() {
  const [data, setData] = useState([]);
  const [selectedClusters, setSelectedClusters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch("/jogadores_com_clusters.csv");
        if (!resp.ok)
          throw new Error("Falha ao carregar /jogadores_com_clusters.csv");
        const text = await resp.text();
        const parsed = parseCSV(text);
        const clusters = [...new Set(parsed.map((r) => r.cluster))].filter(
          (v) => v !== "" && v !== null && v !== undefined
        );
        clusters.sort((a, b) => Number(a) - Number(b));
        setData(parsed);
        setSelectedClusters(clusters);
      } catch (e) {
        console.error(e);
        setError(e.message || "Erro ao carregar CSV.");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const clusterOptions = useMemo(() => {
    const cs = [...new Set(data.map((r) => r.cluster))].filter(
      (v) => v !== "" && v !== null && v !== undefined
    );
    cs.sort((a, b) => Number(a) - Number(b));
    return cs;
  }, [data]);

  const filtered = useMemo(() => {
    if (!Array.isArray(selectedClusters)) return [];
    return data.filter((r) => selectedClusters.includes(r.cluster));
  }, [data, selectedClusters]);

  const targets = ["Target1", "Target2", "Target3"];

  const globalMetrics = useMemo(() => {
    const m = {};
    for (const t of targets) {
      const rows = filtered.filter(
        (r) => Number.isFinite(r[t]) && Number.isFinite(r[`${t}_Previsto`])
      );
      const A = rows.map((r) => r[t]);
      const P = rows.map((r) => r[`${t}_Previsto`]);
      m[t] = {
        rmse: +calcRMSE(A, P).toFixed(2),
        r2: +calcR2(A, P).toFixed(2),
        mae: +calcMAE(A, P).toFixed(2),
        bias: +calcBias(A, P).toFixed(2),
        n: rows.length,
      };
    }
    return m;
  }, [filtered]);

  // ranges seguros (evita min==max)
  const axesByTarget = useMemo(() => {
    const out = {};
    for (const t of targets) {
      const rows = filtered.filter(
        (r) => Number.isFinite(r[t]) && Number.isFinite(r[`${t}_Previsto`])
      );
      const A = rows.map((r) => r[t]);
      const P = rows.map((r) => r[`${t}_Previsto`]);
      const all = A.concat(P);
      let min = all.length ? Math.min(...all) : 0;
      let max = all.length ? Math.max(...all) : 1;
      if (min === max) {
        min -= 1;
        max += 1;
      }
      out[t] = { min: Math.floor(min), max: Math.ceil(max) };
    }
    return out;
  }, [filtered]);

  const clusterCounts = useMemo(() => {
    const acc = {};
    for (const r of filtered) acc[r.cluster] = (acc[r.cluster] || 0) + 1;
    return acc;
  }, [filtered]);

  const handleClusterChange = (e) => {
    const val = e.target.value;
    // garante que sempre seja array
    setSelectedClusters(Array.isArray(val) ? val : []);
  };

  if (loading)
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      {/* filtro + download */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
        <Tooltip title={TT.FILTER} arrow>
          <FormControl sx={{ minWidth: 260 }}>
            <InputLabel id="cluster-label">Filtrar por Cluster</InputLabel>
            <Select
              labelId="cluster-label"
              multiple
              value={selectedClusters}
              onChange={handleClusterChange}
              input={<OutlinedInput label="Filtrar por Cluster" />}
              renderValue={(sel) =>
                Array.isArray(sel)
                  ? sel
                      .slice()
                      .sort((a, b) => Number(a) - Number(b))
                      .join(", ")
                  : ""
              }
            >
              {clusterOptions.map((c) => (
                <MenuItem key={String(c)} value={c}>
                  <Checkbox checked={selectedClusters.indexOf(c) > -1} />
                  <ListItemText primary={`Cluster ${c}`} />
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Tooltip>

        <Button
          variant="outlined"
          startIcon={<Download />}
          onClick={() => downloadCSV(filtered)}
          sx={{ ml: "auto" }}
        >
          Baixar CSV (filtrado)
        </Button>
      </Box>

      {/* cards topo */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Tooltip title="Jogadores no recorte atual." arrow>
            <Card>
              <CardContent>
                <Typography variant="subtitle2">
                  JOGADORES (CSV FILTRADO)
                </Typography>
                <Typography variant="h4">{filtered.length}</Typography>
              </CardContent>
            </Card>
          </Tooltip>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Tooltip title="Clusters ativos." arrow>
            <Card>
              <CardContent>
                <Typography variant="subtitle2">CLUSTERS (ATIVOS)</Typography>
                <Typography variant="h4">
                  {Object.keys(clusterCounts).length}
                </Typography>
              </CardContent>
            </Card>
          </Tooltip>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Tooltip title="Número de features esperadas pelo modelo." arrow>
            <Card>
              <CardContent>
                <Typography variant="subtitle2">FEATURES ESPERADAS</Typography>
                <Typography variant="h4">396</Typography>
              </CardContent>
            </Card>
          </Tooltip>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Tooltip title="Pipeline de cluster disponível." arrow>
            <Card>
              <CardContent>
                <Typography variant="subtitle2">PIPELINE DE CLUSTER</Typography>
                <Typography variant="h4">Sim</Typography>
              </CardContent>
            </Card>
          </Tooltip>
        </Grid>
      </Grid>

      {/* métricas globais */}
      <Typography variant="h6" sx={{ mb: 1 }}>
        Métricas Globais (área filtrada)
      </Typography>
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {["Target1", "Target2", "Target3"].map((t) => {
          const { rmse, r2, mae, bias, n } = globalMetrics[t] || {
            rmse: 0,
            r2: 0,
            mae: 0,
            bias: 0,
            n: 0,
          };
          const b = badge(rmse, r2);
          return (
            <Grid item xs={12} md={4} key={t}>
              <Card>
                <CardContent>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Typography variant="h6">{t}</Typography>
                    <Chip size="small" label={b.label} color={b.color} />
                    <Typography variant="caption" sx={{ ml: "auto" }}>
                      N={n}
                    </Typography>
                  </Box>

                  <Tooltip title={TT.RMSE} arrow>
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 0.5,
                        mt: 0.5,
                      }}
                    >
                      <Typography>RMSE: {rmse}</Typography>
                      <HelpOutline
                        fontSize="small"
                        sx={{ color: "grey.500" }}
                      />
                    </Box>
                  </Tooltip>
                  <Tooltip title={TT.R2} arrow>
                    <Box
                      sx={{ display: "flex", alignItems: "center", gap: 0.5 }}
                    >
                      <Typography>R²: {r2}</Typography>
                      <HelpOutline
                        fontSize="small"
                        sx={{ color: "grey.500" }}
                      />
                    </Box>
                  </Tooltip>
                  <Tooltip title={TT.MAE} arrow>
                    <Box
                      sx={{ display: "flex", alignItems: "center", gap: 0.5 }}
                    >
                      <Typography>MAE: {mae}</Typography>
                      <HelpOutline
                        fontSize="small"
                        sx={{ color: "grey.500" }}
                      />
                    </Box>
                  </Tooltip>
                  <Tooltip title={TT.BIAS} arrow>
                    <Box
                      sx={{ display: "flex", alignItems: "center", gap: 0.5 }}
                    >
                      <Typography>Viés: {bias}</Typography>
                      <HelpOutline
                        fontSize="small"
                        sx={{ color: "grey.500" }}
                      />
                    </Box>
                  </Tooltip>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* Dispersões */}
      <Typography variant="h6" sx={{ mb: 1 }}>
        Comparativo Real vs. Previsto
      </Typography>
      <Grid container spacing={2} sx={{ mb: 4 }}>
        {["Target1", "Target2", "Target3"].map((t) => {
          const rows = filtered.filter(
            (r) => Number.isFinite(r[t]) && Number.isFinite(r[`${t}_Previsto`])
          );
          if (rows.length === 0)
            return (
              <Grid item xs={12} md={4} key={t}>
                <Alert severity="info">Sem dados para {t}</Alert>
              </Grid>
            );

          const A = rows.map((r) => r[t]);
          const P = rows.map((r) => r[`${t}_Previsto`]);
          const { min, max } = axesByTarget[t];
          const hover = rows.map(
            (r, i) =>
              `Real: ${A[i]?.toFixed?.(2) ?? A[i]}<br>Previsto: ${
                P[i]?.toFixed?.(2) ?? P[i]
              }<br>Resíduo: ${(P[i] - A[i])?.toFixed?.(2)}`
          );

          return (
            <Grid item xs={12} md={4} key={t}>
              <Tooltip title={TT.SCATTER} arrow>
                <Box>
                  <Plot
                    data={[
                      {
                        x: A,
                        y: P,
                        mode: "markers",
                        type: "scatter",
                        name: "Previsto vs Real",
                        marker: { size: 8 },
                        text: hover,
                        hoverinfo: "text",
                      },
                      {
                        x: [min, max],
                        y: [min, max],
                        mode: "lines",
                        type: "scatter",
                        name: "Linha Ideal",
                        line: { dash: "dash" },
                      },
                    ]}
                    layout={{
                      title: `<b>${t}</b>`,
                      xaxis: { title: "Real (Gabarito)", range: [min, max] },
                      yaxis: { title: "Previsto (Modelo)", range: [min, max] },
                      autosize: true,
                      paper_bgcolor: "#29384B",
                      plot_bgcolor: "#29384B",
                      font: { color: "#FFFFFF" },
                      showlegend: true,
                      legend: { x: 0.02, y: 0.98 },
                    }}
                    useResizeHandler
                    style={{ width: "100%", height: 360 }}
                  />
                </Box>
              </Tooltip>
            </Grid>
          );
        })}
      </Grid>

      {/* Distribuição de jogadores por cluster */}
      <Typography variant="h6" sx={{ mb: 1 }}>
        Distribuição de Jogadores por Cluster
      </Typography>
      <ClusterProfiles limit={5} />
      <Tooltip title={TT.DISTRIB} arrow>
        <Box component="span" sx={{ display: "inline-block", width: "100%" }}>
          <Plot
            data={[
              {
                x: Object.keys(clusterCounts),
                y: Object.values(clusterCounts),
                type: "bar",
                text: Object.values(clusterCounts),
                textposition: "auto",
              },
            ]}
            layout={{
              title: `Distribuição (${filtered.length} jogadores)`,
              xaxis: { title: "Cluster" },
              yaxis: { title: "Quantidade" },
              autosize: true,
              paper_bgcolor: "#29384B",
              plot_bgcolor: "#29384B",
              font: { color: "#FFFFFF" },
            }}
            useResizeHandler
            style={{ width: "100%", height: 320 }}
          />
        </Box>
      </Tooltip>

      {/* Resíduos */}
      <Typography variant="h6" sx={{ mb: 1 }}>
        Qualidade das Previsões (Resíduos)
      </Typography>
      <Grid container spacing={2} sx={{ mb: 4 }}>
        {["Target1", "Target2", "Target3"].map((t) => {
          const rows = filtered.filter(
            (r) => Number.isFinite(r[t]) && Number.isFinite(r[`${t}_Previsto`])
          );
          const res = rows.map((r) => r[`${t}_Previsto`] - r[t]);
          const bins = Math.max(
            6,
            Math.ceil(Math.log2(Math.max(1, res.length)) + 1)
          );
          return (
            <Grid item xs={12} md={4} key={t}>
              <Tooltip title={TT.RESIDUALS} arrow>
                <Box
                  component="span"
                  sx={{ display: "inline-block", width: "100%" }}
                >
                  <Plot
                    data={[{ x: res, type: "histogram", nbinsx: bins }]}
                    layout={{
                      title: `${t} — Resíduos (Previsto − Real)`,
                      xaxis: { title: "Resíduo" },
                      yaxis: { title: "Frequência" },
                      autosize: true,
                      paper_bgcolor: "#29384B",
                      plot_bgcolor: "#29384B",
                      font: { color: "#FFFFFF" },
                    }}
                    useResizeHandler
                    style={{ width: "100%", height: 260 }}
                  />
                </Box>
              </Tooltip>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
}

export default AnalysisDashboard;
