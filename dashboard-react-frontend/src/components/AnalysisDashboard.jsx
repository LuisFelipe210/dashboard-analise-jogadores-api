// src/components/AnalysisDashboard.jsx
import React, { useEffect, useMemo, useState } from "react";
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  Tooltip,
  Chip,
  Button,
  TextField,
} from "@mui/material";
import HelpOutline from "@mui/icons-material/HelpOutline";
import Download from "@mui/icons-material/Download";
import RefreshIcon from "@mui/icons-material/Refresh";
import Plot from "react-plotly.js";

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
  SCATTER:
    "Cada ponto é um registro. Linha tracejada é a linha ideal (y=x).",
  RESIDUALS:
    "Histograma de resíduos (Previsto − Real). Ideal é centrado em 0.",
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
      const preds = rows.map((r) => Number(r[predCol])).filter((v) => Number.isFinite(v));
      const predsPercent = toPercent(preds);
      out[t] = bucketize(predsPercent, thresholds.low, thresholds.high);
    }
    return out;
  }, [rows, thresholds]);

  if (loading)
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      {/* Topo: ações */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
        <Button
          variant="outlined"
          startIcon={<Download />}
          onClick={() => downloadCSV(rows)}
          sx={{ ml: "auto" }}
        >
          Baixar CSV
        </Button>
      </Box>

      {/* Cards topo */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2">REGISTROS NO CSV</Typography>
              <Typography variant="h4">{rows.length}</Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2">TARGETS COM PREVISTO</Typography>
              <Typography variant="h4">
                {
                  TARGETS.reduce(
                    (acc, t) =>
                      acc +
                      (rows.some((r) => Number.isFinite(r[`${t}_Previsto`])) ? 1 : 0),
                    0
                  )
                }
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Limiar ajustável (impacta o agrupamento por faixas) */}
        <Grid item xs={12} sm={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2">LIMIARES ATUAIS</Typography>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 1 }}>
                <TextField
                  label="Baixo (%)"
                  type="number"
                  size="small"
                  value={thresholds.low}
                  onChange={(e) =>
                    setThresholds((t) => ({ ...t, low: Number(e.target.value) }))
                  }
                  sx={{ width: 120 }}
                />
                <TextField
                  label="Alto (%)"
                  type="number"
                  size="small"
                  value={thresholds.high}
                  onChange={(e) =>
                    setThresholds((t) => ({ ...t, high: Number(e.target.value) }))
                  }
                  sx={{ width: 120 }}
                />
                <Tooltip title="O gráfico de faixas será recalculado automaticamente" arrow>
                  <RefreshIcon sx={{ ml: 0.5, color: "text.secondary" }} />
                </Tooltip>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Métricas globais */}
      <Typography variant="h6" sx={{ mb: 1 }}>
        Métricas Globais (comparando Real vs Previsto)
      </Typography>
      <Grid container spacing={2} sx={{ mb: 3 }}>
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

                  <Tooltip title="Erro médio quadrático. Menor é melhor." arrow>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mt: 0.5 }}>
                      <Typography>RMSE: {rmse}</Typography>
                      <HelpOutline fontSize="small" sx={{ color: "grey.500" }} />
                    </Box>
                  </Tooltip>
                  <Tooltip title="Proporção da variação explicada. Pode ser negativa." arrow>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                      <Typography>R²: {r2}</Typography>
                      <HelpOutline fontSize="small" sx={{ color: "grey.500" }} />
                    </Box>
                  </Tooltip>
                  <Tooltip title="Erro absoluto médio. Menor é melhor." arrow>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                      <Typography>MAE: {mae}</Typography>
                      <HelpOutline fontSize="small" sx={{ color: "grey.500" }} />
                    </Box>
                  </Tooltip>
                  <Tooltip title="Média (Previsto − Real): positivo superestima; negativo subestima." arrow>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                      <Typography>Viés: {bias}</Typography>
                      <HelpOutline fontSize="small" sx={{ color: "grey.500" }} />
                    </Box>
                  </Tooltip>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* Comparativo Real vs Previsto */}
      <Typography variant="h6" sx={{ mb: 1 }}>
        Comparativo Real vs Previsto
      </Typography>
      <Grid container spacing={2} sx={{ mb: 4 }}>
        {TARGETS.map((t) => {
          const realCol = t;
          const predCol = `${t}_Previsto`;
          const valid = rows.filter(
            (r) => Number.isFinite(r[realCol]) && Number.isFinite(r[predCol])
          );
          if (valid.length === 0)
            return (
              <Grid item xs={12} md={4} key={t}>
                <Alert severity="info">Sem dados comparáveis para {t}</Alert>
              </Grid>
            );

          const A = valid.map((r) => Number(r[realCol]));
          const P = valid.map((r) => Number(r[predCol]));
          const { min, max } = axesByTarget[t];
          const hover = A.map(
            (a, i) =>
              `Real (X): ${a?.toFixed?.(2) ?? a}<br>` +
              `Previsto (Y): ${P[i]?.toFixed?.(2) ?? P[i]}<br>` +
              `Resíduo (Y−X): ${(P[i] - a)?.toFixed?.(2)}`
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
                        name: "Pontos (Previsto vs Real)",
                        marker: { size: 8 },
                        text: hover,
                        hoverinfo: "text",
                      },
                      {
                        x: [min, max],
                        y: [min, max],
                        mode: "lines",
                        type: "scatter",
                        name: "Linha Ideal (y = x)",
                        line: { dash: "dash" },
                      },
                    ]}
                    layout={{
                      title: `<b>${t} — Comparativo</b>`,
                      xaxis: { title: "Real", range: [min, max] },
                      yaxis: { title: "Previsto", range: [min, max] },
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

      {/* Ruído (Resíduos) */}
      <Typography variant="h6" sx={{ mb: 1 }}>
        Ruído — Distribuição de Resíduos (Previsto − Real)
      </Typography>
      <Grid container spacing={2} sx={{ mb: 4 }}>
        {TARGETS.map((t) => {
          const realCol = t;
          const predCol = `${t}_Previsto`;
          const valid = rows.filter(
            (r) => Number.isFinite(r[realCol]) && Number.isFinite(r[predCol])
          );
          const res = valid.map((r) => Number(r[predCol]) - Number(r[realCol]));
          const bins =
            res.length > 1
              ? Math.max(6, Math.ceil(Math.log2(Math.max(1, res.length)) + 1))
              : 6;

          return (
            <Grid item xs={12} md={4} key={t}>
              <Tooltip title={TT.RESIDUALS} arrow>
                <Box sx={{ width: "100%" }}>
                  <Plot
                    data={[{ x: res, type: "histogram", nbinsx: bins, name: "Resíduos" }]}
                    layout={{
                      title: `${t} — Resíduos (Y − X)`,
                      xaxis: { title: "Resíduo" },
                      yaxis: { title: "Frequência" },
                      autosize: true,
                      paper_bgcolor: "#29384B",
                      plot_bgcolor: "#29384B",
                      font: { color: "#FFFFFF" },
                      showlegend: false,
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

      {/* Agrupamento por faixas (POR ÚLTIMO) */}
      <Typography variant="h6" sx={{ mb: 1 }}>
        Agrupamento por Faixas (%) — por Target (Previsto)
      </Typography>
      <Tooltip title={TT.BUCKETS} arrow>
        <Box sx={{ width: "100%", mb: 4 }}>
          <Plot
            data={[
              {
                x: TARGETS,
                y: TARGETS.map((t) => buckets[t]["<30"] ?? 0),
                type: "bar",
                name: "< 30%",
                hovertemplate: "%{y} abaixo de 30%<extra></extra>",
              },
              {
                x: TARGETS,
                y: TARGETS.map((t) => buckets[t]["30-60"] ?? 0),
                type: "bar",
                name: "30% – 60%",
                hovertemplate: "%{y} entre 30% e 60%<extra></extra>",
              },
              {
                x: TARGETS,
                y: TARGETS.map((t) => buckets[t][">60"] ?? 0),
                type: "bar",
                name: "> 60%",
                hovertemplate: "%{y} acima de 60%<extra></extra>",
              },
            ]}
            layout={{
              barmode: "stack",
              title: `Distribuição por Faixas (limiares: ${thresholds.low}% / ${thresholds.high}%)`,
              xaxis: { title: "Targets" },
              yaxis: { title: "Quantidade de pessoas" },
              autosize: true,
              paper_bgcolor: "#29384B",
              plot_bgcolor: "#29384B",
              font: { color: "#FFFFFF" },
              legend: { orientation: "h" },
            }}
            useResizeHandler
            style={{ width: "100%", height: 360 }}
          />
        </Box>
      </Tooltip>
    </Box>
  );
}

export default AnalysisDashboard;
