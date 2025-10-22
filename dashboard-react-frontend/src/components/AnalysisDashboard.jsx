// src/components/AnalysisDashboard.jsx
import React, { useEffect, useMemo, useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  Tooltip,
  Chip,
  Button,
  TextField,
  Stack,
  Divider,
} from "@mui/material";
import HelpOutline from "@mui/icons-material/HelpOutline";
import Download from "@mui/icons-material/Download";
import RefreshIcon from "@mui/icons-material/Refresh";
import Plot from "react-plotly.js";
import { axisClasses, BarChart, ScatterChart } from "@mui/x-charts";

// =========================
// CONFIG
// =========================
const DATASET_URL = "/jogadores_com_clusters.csv"; // ajuste se necessário

// =========================
// helpers
// =========================
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

// Escala para % se necessário (0–1 -> 0–100)
const toPercent = (arr) => {
  const xs = arr.filter((v) => Number.isFinite(v));
  if (!xs.length) return arr;
  const maxv = Math.max(...xs);
  return maxv <= 1 ? arr.map((v) => (Number.isFinite(v) ? v * 100 : v)) : arr;
};

// Agrupa nas faixas <low, [low, high], >high
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

// CSV -> JSON genérico
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
    return data;
  } catch (e) {
    console.error("parseCSV error", e);
    return [];
  }
};

// download CSV (do que estiver na tela)
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
    a.download = "dataset_previsto.csv";
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    console.error(e);
  }
};

const TT = {
  SCATTER: "Cada ponto é um registro. Linha tracejada é a linha ideal (y=x).",
  RESIDUALS: "Histograma de resíduos (Previsto − Real). Ideal é centrado em 0.",
  BUCKETS:
    "Contagem de pessoas por faixa de percentuais (<30 | 30–60 | >60) em cada target, a partir dos valores PREVISTOS.",
};

const TARGETS = ["Target1", "Target2", "Target3"];

function AnalysisDashboard() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [thresholds, setThresholds] = useState({ low: 30, high: 60 });

  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch(DATASET_URL);
        if (!resp.ok) throw new Error(`Falha ao carregar ${DATASET_URL}`);
        const text = await resp.text();
        const parsed = parseCSV(text);
        setRows(parsed);
      } catch (e) {
        console.error(e);
        setError(e.message || "Erro ao carregar CSV.");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // Métricas globais (usa TargetX e TargetX_Previsto do CSV)
  const globalMetrics = useMemo(() => {
    const m = {};
    for (const t of TARGETS) {
      const realCol = t;
      const predCol = `${t}_Previsto`;
      const valid = rows.filter(
        (r) => Number.isFinite(r[realCol]) && Number.isFinite(r[predCol])
      );
      const A = valid.map((r) => Number(r[realCol]));
      const P = valid.map((r) => Number(r[predCol]));
      m[t] = {
        rmse: +calcRMSE(A, P).toFixed(2),
        r2: +calcR2(A, P).toFixed(2),
        mae: +calcMAE(A, P).toFixed(2),
        bias: +calcBias(A, P).toFixed(2),
        n: valid.length,
      };
    }
    return m;
  }, [rows]);

  // Eixos seguros pro comparativo
  const axesByTarget = useMemo(() => {
    const out = {};
    for (const t of TARGETS) {
      const realCol = t;
      const predCol = `${t}_Previsto`;
      const valid = rows.filter(
        (r) => Number.isFinite(r[realCol]) && Number.isFinite(r[predCol])
      );
      const arr = [];
      for (const r of valid) {
        arr.push(Number(r[realCol]), Number(r[predCol]));
      }
      let min = arr.length ? Math.min(...arr) : 0;
      let max = arr.length ? Math.max(...arr) : 1;
      if (min === max) {
        min -= 1;
        max += 1;
      }
      out[t] = { min: Math.floor(min), max: Math.ceil(max) };
    }
    return out;
  }, [rows]);

  // Buckets por faixas a partir dos PREVISTOS
  const buckets = useMemo(() => {
    const out = {};
    for (const t of TARGETS) {
      const predCol = `${t}_Previsto`;
      const preds = rows
        .map((r) => Number(r[predCol]))
        .filter((v) => Number.isFinite(v));
      const predsPercent = toPercent(preds);
      out[t] = bucketize(predsPercent, thresholds.low, thresholds.high);
    }
    return out;
  }, [rows, thresholds]);

  // Card style reused
  const cardStyle = {
    background: "rgba(30,41,59,0.82)",
    backdropFilter: "blur(10px)",
    border: "1px solid rgba(148,163,184,0.08)",
    borderRadius: 4,
    boxShadow: "0 2px 16px 0 rgba(30,41,59,0.22)",
    transition:
      "transform 0.18s cubic-bezier(.4,0,.2,1), box-shadow 0.18s cubic-bezier(.4,0,.2,1)",
    "&:hover": {
      transform: "translateY(-6px) scale(1.012)",
      boxShadow: "0 8px 32px 0 rgba(30,41,59,0.29)",
    },
  };

  // Chart style reused
  const chartStyle = {
    background: "rgba(51,65,85,0.13)",
    borderRadius: 4,
    p: 2,
    [`& .${axisClasses.root}`]: { stroke: "#475569", color: "#E2E8F0" },
  };

  if (loading)
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box
      sx={{
        background: "linear-gradient(180deg, #0F172A 0%, #1E293B 100%)",
        color: "#E2E8F0",
        minHeight: "100vh",
        p: { xs: 2, md: 4 },
        borderRadius: 2,
      }}
    >
      {/* Barra de título */}
      <Typography
        variant="h4"
        sx={{
          mb: 3,
          fontWeight: 800,
          color: "#F8FAFC",
          letterSpacing: 0.5,
          textShadow: "0 2px 12px rgba(30,41,59,0.22)",
        }}
      >
        Painel de Análise de Jogadores
      </Typography>

      {/* Ações rápidas */}
      <Stack
        direction="row"
        alignItems="center"
        justifyContent="flex-end"
        spacing={2}
        sx={{ mb: 3 }}
      >
        <Button
          variant="contained"
          color="primary"
          startIcon={<Download />}
          onClick={() => downloadCSV(rows)}
          sx={{
            background: "linear-gradient(90deg, #2563EB 0%, #3B82F6 100%)",
            color: "#fff",
            textTransform: "none",
            fontWeight: 600,
            borderRadius: 2,
            px: 2.5,
            boxShadow: "0 2px 8px 0 rgba(30,41,59,0.18)",
            "&:hover": {
              background: "linear-gradient(90deg, #1E40AF 0%, #2563EB 100%)",
            },
          }}
        >
          Baixar CSV
        </Button>
      </Stack>

      {/* Seção: Métricas e Thresholds */}
      <Stack
        direction={{ xs: "column", md: "row" }}
        spacing={3}
        sx={{ mb: 4 }}
        useFlexGap
      >
        {/* Cards métricas rápidas */}
        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={3}
          flex={2}
          useFlexGap
        >
          <Card
            sx={{ ...cardStyle, minWidth: 0, flex: 1, p: { xs: 2.5, md: 4 } }}
          >
            <CardContent sx={{ p: 0 }}>
              <Typography
                variant="subtitle2"
                sx={{ color: "#A5B4FC", letterSpacing: 1.1, fontWeight: 700 }}
              >
                REGISTROS NO CSV
              </Typography>
              <Typography
                variant="h4"
                sx={{
                  fontWeight: 800,
                  color: "#F8FAFC",
                  mt: 0.5,
                  letterSpacing: 0.2,
                  textShadow: "0 2px 8px rgba(30,41,59,0.20)",
                }}
              >
                {rows.length}
              </Typography>
            </CardContent>
          </Card>
          <Card
            sx={{ ...cardStyle, minWidth: 0, flex: 1, p: { xs: 2.5, md: 4 } }}
          >
            <CardContent sx={{ p: 0 }}>
              <Typography
                variant="subtitle2"
                sx={{ color: "#A5B4FC", letterSpacing: 1.1, fontWeight: 700 }}
              >
                TARGETS COM PREVISTO
              </Typography>
              <Typography
                variant="h4"
                sx={{
                  fontWeight: 800,
                  color: "#F8FAFC",
                  mt: 0.5,
                  letterSpacing: 0.2,
                  textShadow: "0 2px 8px rgba(30,41,59,0.20)",
                }}
              >
                {TARGETS.reduce(
                  (acc, t) =>
                    acc +
                    (rows.some((r) => Number.isFinite(r[`${t}_Previsto`]))
                      ? 1
                      : 0),
                  0
                )}
              </Typography>
            </CardContent>
          </Card>
        </Stack>
      </Stack>

      {/* Seção: Métricas Globais */}
      <Card sx={{ ...cardStyle, p: { xs: 2.5, md: 4 }, mb: 5 }}>
        <Typography
          variant="h6"
          sx={{
            mb: 2,
            fontWeight: 800,
            color: "#F1F5F9",
            letterSpacing: 0.5,
            textShadow: "0 2px 8px rgba(30,41,59,0.18)",
          }}
        >
          Métricas Globais (comparando Real vs Previsto)
        </Typography>
        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={3}
          divider={
            <Divider
              flexItem
              orientation="vertical"
              sx={{ borderColor: "rgba(148,163,184,0.10)" }}
            />
          }
        >
          {TARGETS.map((t) => {
            const { rmse, r2, mae, bias, n } = globalMetrics[t] || {
              rmse: 0,
              r2: 0,
              mae: 0,
              bias: 0,
              n: 0,
            };
            const b = badge(rmse, r2);
            return (
              <Box
                key={t}
                sx={{
                  flex: 1,
                  minWidth: 0,
                  p: { xs: 1, md: 2 },
                  borderRadius: 2,
                  background: "rgba(51,65,85,0.09)",
                  boxShadow: "0 2px 8px 0 rgba(30,41,59,0.10)",
                  transition: "box-shadow 0.17s cubic-bezier(.4,0,.2,1)",
                  "&:hover": {
                    boxShadow: "0 8px 24px 0 rgba(30,41,59,0.17)",
                  },
                }}
              >
                <Stack spacing={1.2}>
                  <Stack direction="row" alignItems="center" spacing={1}>
                    <Typography
                      variant="h6"
                      sx={{
                        fontWeight: 800,
                        color: "#A5B4FC",
                        letterSpacing: 0.5,
                        textShadow: "0 2px 8px rgba(30,41,59,0.08)",
                      }}
                    >
                      {t}
                    </Typography>
                    <Chip
                      size="small"
                      label={b.label}
                      color={b.color}
                      sx={{
                        fontWeight: 700,
                        letterSpacing: 0.5,
                        fontSize: "0.93em",
                      }}
                    />
                    <Typography
                      variant="caption"
                      sx={{ ml: "auto", color: "#CBD5E1", fontWeight: 600 }}
                    >
                      N={n}
                    </Typography>
                  </Stack>
                  <Tooltip title="Erro médio quadrático. Menor é melhor." arrow>
                    <Stack direction="row" alignItems="center" spacing={0.5}>
                      <Typography sx={{ color: "#F8FAFC", fontWeight: 600 }}>
                        RMSE: {rmse}
                      </Typography>
                      <HelpOutline fontSize="small" sx={{ color: "#64748B" }} />
                    </Stack>
                  </Tooltip>
                  <Tooltip
                    title="Proporção da variação explicada. Pode ser negativa."
                    arrow
                  >
                    <Stack direction="row" alignItems="center" spacing={0.5}>
                      <Typography sx={{ color: "#F8FAFC", fontWeight: 600 }}>
                        R²: {r2}
                      </Typography>
                      <HelpOutline fontSize="small" sx={{ color: "#64748B" }} />
                    </Stack>
                  </Tooltip>
                  <Tooltip title="Erro absoluto médio. Menor é melhor." arrow>
                    <Stack direction="row" alignItems="center" spacing={0.5}>
                      <Typography sx={{ color: "#F8FAFC", fontWeight: 600 }}>
                        MAE: {mae}
                      </Typography>
                      <HelpOutline fontSize="small" sx={{ color: "#64748B" }} />
                    </Stack>
                  </Tooltip>
                  <Tooltip
                    title="Média (Previsto − Real): positivo superestima; negativo subestima."
                    arrow
                  >
                    <Stack direction="row" alignItems="center" spacing={0.5}>
                      <Typography sx={{ color: "#F8FAFC", fontWeight: 600 }}>
                        Viés: {bias}
                      </Typography>
                      <HelpOutline fontSize="small" sx={{ color: "#64748B" }} />
                    </Stack>
                  </Tooltip>
                </Stack>
              </Box>
            );
          })}
        </Stack>
      </Card>

      {/* Seção: Gráficos Comparativos */}
      <Card sx={{ ...cardStyle, p: { xs: 2.5, md: 4 }, mb: 5 }}>
        <Typography
          variant="h6"
          sx={{
            mb: 2,
            fontWeight: 800,
            color: "#F1F5F9",
            letterSpacing: 0.5,
            textShadow: "0 2px 8px rgba(30,41,59,0.18)",
          }}
        >
          Comparativo Real vs Previsto
        </Typography>
        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={3}
          useFlexGap
          flexWrap="wrap"
          alignItems="stretch"
          justifyContent="center"
          sx={{ mb: 0 }}
        >
          {TARGETS.map((t) => {
            const realCol = t;
            const predCol = `${t}_Previsto`;
            const valid = rows.filter(
              (r) => Number.isFinite(r[realCol]) && Number.isFinite(r[predCol])
            );
            if (valid.length === 0)
              return (
                <Box
                  key={t}
                  sx={{
                    flex: { xs: "1 1 100%", md: "1 1 45%" },
                    minWidth: 0,
                    mb: { xs: 2, md: 0 },
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Alert
                    severity="info"
                    sx={{
                      width: "100%",
                      maxWidth: 520,
                      mx: "auto",
                      borderRadius: 3,
                      background: "rgba(51,65,85,0.16)",
                      color: "#F1F5F9",
                      fontWeight: 700,
                      border: "1px solid rgba(148,163,184,0.16)",
                    }}
                  >
                    Sem dados comparáveis para {t}
                  </Alert>
                </Box>
              );

            const A = valid.map((r) => Number(r[realCol]));
            const P = valid.map((r) => Number(r[predCol]));
            const { min, max } = axesByTarget[t];
            return (
                <Box
                  key={t}
                  sx={{
                    flex: { xs: "1 1 100%", md: "1 1 45%" },
                    minWidth: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    mb: { xs: 2, md: 0 },
                  }}
                >
                  <Tooltip title={TT.SCATTER} arrow>
                    <Box
                      sx={{
                        width: "100%",
                        maxWidth: 520,
                        minHeight: 340,
                        mx: "auto",
                        background: "rgba(51,65,85,0.13)",
                        borderRadius: 4,
                        boxShadow: "0 2px 12px 0 rgba(30,41,59,0.14)",
                        p: 2,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <Plot
                        data={[
                          {
                            x: A,
                            y: P,
                            mode: "markers",
                            type: "scatter",
                            name: t,
                            marker: { size: 8 },
                            hovertemplate: `${t}<br>X: %{x}<br>Y: %{y}<extra></extra>`,
                          },
                          {
                            x: [min, max],
                            y: [min, max],
                            mode: "lines",
                            type: "scatter",
                            showlegend: false, // << CHANGED: não aparece na legenda
                            line: { width: 2, dash: "dot" },
                            hoverinfo: "skip",
                          },
                        ]}
                        layout={{
                          width: 480,
                          height: 320,
                          margin: { t: 30, b: 60, l: 50, r: 20 },
                          xaxis: {
                            title: { text: "Previsto", standoff: 10 },
                            range: [min, max],
                          },
                          yaxis: {
                            title: { text: "Real", standoff: 10 },
                            range: [min, max],
                          },
                          plot_bgcolor: "transparent",
                          paper_bgcolor: "transparent",
                          font: { color: "#E2E8F0", family: "Inter, sans-serif" },
                          legend: {
                            orientation: "h",
                            y: -0.18,
                            x: 0.5,
                            xanchor: "center",
                            font: { color: "#E2E8F0", size: 12 },
                          },
                        }}
                        config={{ displayModeBar: false }}
                      />
                    </Box>
                  </Tooltip>
                </Box>
            );
          })}
        </Stack>
      </Card>

      {/* Seção: Gráficos de Resíduos */}
      <Card sx={{ ...cardStyle, p: { xs: 2.5, md: 4 }, mb: 5 }}>
        <Typography
          variant="h6"
          sx={{
            mb: 2,
            fontWeight: 800,
            color: "#F1F5F9",
            letterSpacing: 0.5,
            textShadow: "0 2px 8px rgba(30,41,59,0.18)",
          }}
        >
          Ruído — Distribuição de Resíduos (Previsto − Real)
        </Typography>
        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={3}
          useFlexGap
          flexWrap="wrap"
          alignItems="stretch"
          justifyContent="center"
          sx={{ mb: 0 }}
        >
          {TARGETS.map((t) => {
            const realCol = t;
            const predCol = `${t}_Previsto`;
            const valid = rows.filter(
              (r) => Number.isFinite(r[realCol]) && Number.isFinite(r[predCol])
            );
            const res = valid.map(
              (r) => Number(r[predCol]) - Number(r[realCol])
            );
            return (
              <Box
                key={t}
                sx={{
                  flex: { xs: "1 1 100%", md: "1 1 45%" },
                  minWidth: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  mb: { xs: 2, md: 0 },
                }}
              >
                <Tooltip title={TT.RESIDUALS} arrow>
                  <Box
                    sx={{
                      width: "100%",
                      maxWidth: 520,
                      minHeight: 320,
                      mx: "auto",
                      background: "rgba(51,65,85,0.13)",
                      borderRadius: 4,
                      boxShadow: "0 2px 12px 0 rgba(30,41,59,0.14)",
                      p: 2,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <BarChart
                      height={240}
                      xAxis={[
                        {
                          label: "Resíduo",
                          scaleType: "band",
                          data: res.map((_, i) => i + 1),
                          tickLabelStyle: { fill: "#94A3B8", fontWeight: 600 },
                        },
                      ]}
                      yAxis={[
                        {
                          label: "Frequência",
                          tickLabelStyle: { fill: "#94A3B8", fontWeight: 600 },
                        },
                      ]}
                      series={[
                        {
                          data: res,
                          color: "#38BDF8",
                          label: t, // << CHANGED: mostra o nome do target na legenda
                        },
                      ]}
                      grid={{ horizontal: true }}
                      // << NOVO: legenda no topo (com a “bolinha” do target)
                      slotProps={{
                        legend: {
                          hidden: false,
                          direction: "row",
                          position: { vertical: "top", horizontal: "middle" },
                        },
                      }}
                      sx={{
                        ...chartStyle,
                        "& .MuiChartsAxis-root": { color: "#E2E8F0" },
                        background: "transparent",
                        borderRadius: 3,
                        // (opcional) dá um respiro extra pro topo por causa da legenda
                        "--ChartsLegend-rootOffset": "8px",
                      }}
                    />
                  </Box>
                </Tooltip>
              </Box>
            );
          })}
        </Stack>
      </Card>

      {/* Seção: Agrupamento por Faixas */}
      <Card sx={{ ...cardStyle, p: { xs: 2.5, md: 4 }, mb: 5 }}>
        <Typography
          variant="h6"
          sx={{
            mb: 2,
            fontWeight: 800,
            color: "#F1F5F9",
            letterSpacing: 0.5,
            textShadow: "0 2px 8px rgba(30,41,59,0.18)",
          }}
        >
          Agrupamento por Faixas (%) — por Target (Previsto)
        </Typography>
        <Tooltip title={TT.BUCKETS} arrow>
          <Box
            sx={{
              width: "100%",
              maxWidth: 1080,
              minHeight: 360,
              mx: "auto",
              background: "rgba(51,65,85,0.13)",
              borderRadius: 4,
              boxShadow: "0 2px 12px 0 rgba(30,41,59,0.14)",
              p: 2,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <BarChart
              height={320}
              xAxis={[
                {
                  scaleType: "band",
                  data: TARGETS,
                  label: "Targets",
                  tickLabelStyle: { fill: "#A5B4FC", fontWeight: 700 },
                  labelStyle: { fill: "#A5B4FC", fontWeight: 700 },
                },
              ]}
              yAxis={[
                {
                  label: "Quantidade de pessoas",
                  tickLabelStyle: { fill: "#F8FAFC", fontWeight: 700 },
                  labelStyle: { fill: "#F8FAFC", fontWeight: 700 },
                },
              ]}
              series={[
                {
                  data: TARGETS.map((t) => buckets[t]["<30"] ?? 0),
                  color: "#60A5FA",
                  label: "<30%",
                  stack: "stack1",
                },
                {
                  data: TARGETS.map((t) => buckets[t]["30-60"] ?? 0),
                  color: "#34D399",
                  label: "30–60%",
                  stack: "stack1",
                },
                {
                  data: TARGETS.map((t) => buckets[t][">60"] ?? 0),
                  color: "#FBBF24",
                  label: ">60%",
                  stack: "stack1",
                },
              ]}
              sx={{
                "& .MuiChartsAxis-root": { color: "#E2E8F0" },
                background: "transparent",
                borderRadius: 3,
              }}
            />
          </Box>
        </Tooltip>
      </Card>
    </Box>
  );
}

export default AnalysisDashboard;
