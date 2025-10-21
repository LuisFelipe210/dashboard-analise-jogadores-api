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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Backdrop,
} from "@mui/material";
import { runPrediction, getSchema } from "../api/ApiService";

// Overlay de loading atualizado
function useLatch(active, delayMs = 350) {
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
          <CircularProgress size={38} sx={{ color: "#3B82F6" }} />
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

// =================== COMPONENT ===================
function PredictionTool() {
  // Estado e lógica originais
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
      setSelectedPlayerId("");
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
      setLoadingStep("Rodando o modelo");
      const predsDict = await runPrediction(rows);
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

  // LAYOUT MODERNO
  // Redesigned layout: Container + Stack, dark gradient background, cards, modern appbar, responsive columns
  return (
    <Box
      sx={{
        minHeight: "100vh",
        width: "100%",
        background: "linear-gradient(180deg, #10172a 0%, #1e293b 100%)",
        bgcolor: "transparent",
        overflowX: "hidden",
        maxWidth: "100vw",
      }}
    >
      {/* AppBar */}
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

      {/* Main content */}
      <Container
        maxWidth={1200}
        disableGutters
        sx={{
          py: { xs: 2, md: 4 },
          px: { xs: 1, sm: 2, md: 3 },
          width: "100%",
          mx: "auto",
          maxWidth: "100%",
          overflowX: "hidden",
          minWidth: 0,
        }}
      >
        <Stack spacing={2} sx={{ mb: 3, maxWidth: 700, mx: "auto", px: { xs: 0.5, md: 0 }, minWidth: 0 }}>
          {error && <Alert severity="error">{error}</Alert>}
          {success && <Alert severity="success">{success}</Alert>}
        </Stack>
        <Grid
          container
          spacing={4}
          alignItems="stretch"
          justifyContent="center"
          sx={{
            width: "100%",
            mx: "auto",
            maxWidth: 1700,
            boxSizing: "border-box",
            minWidth: 0,
            overflowX: "hidden",
          }}
        >
          {/* Left column: upload + filter */}
          <Grid
            item
            xs={12}
            md={5}
            sx={{
              minWidth: 0,
              maxWidth: { xs: "100%", md: 520 },
              width: "100%",
              alignSelf: "stretch",
              display: "flex",
              flexDirection: "column",
              gap: 4,
              boxSizing: "border-box",
            }}
          >
            {/* Upload Card */}
            <Card
              elevation={0}
              sx={{
                bgcolor: "rgba(23,30,48,0.88)",
                borderRadius: 6,
                border: "1.5px solid rgba(59,130,246,0.15)",
                boxShadow: "0 4px 18px 0 rgba(36,99,235,0.10)",
                transition: "box-shadow 0.18s, border 0.15s",
                "&:hover": {
                  boxShadow: "0 10px 32px 0 rgba(36,99,235,0.13)",
                  border: "1.5px solid #2563EB",
                },
                mx: { xs: 0, md: "auto" },
                width: "100%",
                maxWidth: "100%",
                minHeight: 190,
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                overflowX: "hidden",
              }}
            >
              <CardContent sx={{ px: { xs: 2, sm: 3 }, py: { xs: 2.5, sm: 3.5 }, width: "100%", minWidth: 0 }}>
                <Stack spacing={2.2} alignItems="center" justifyContent="center" sx={{ width: "100%", minWidth: 0, maxWidth: "100%" }}>
                  <Typography
                    variant="h6"
                    sx={{
                      fontWeight: 800,
                      color: "#F8FAFC",
                      letterSpacing: 0.17,
                      mb: 0.5,
                      textAlign: "center",
                    }}
                  >
                    Fonte dos Dados
                  </Typography>
                  <Typography
                    variant="body1"
                    sx={{
                      color: "rgba(203,213,225,0.80)",
                      textAlign: "center",
                      fontWeight: 400,
                      fontSize: "1.09rem",
                    }}
                  >
                    Carregue o arquivo Excel com os jogadores para iniciar a análise.
                  </Typography>
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
            {/* Filter/Player Card */}
            <Card
              elevation={0}
              sx={{
                bgcolor: "rgba(23,30,48,0.76)",
                borderRadius: 6,
                border: "1.5px solid rgba(59,130,246,0.10)",
                boxShadow: "0 2px 14px 0 rgba(36,99,235,0.08)",
                transition: "box-shadow 0.14s, border 0.13s",
                "&:hover": {
                  boxShadow: "0 6px 22px 0 rgba(36,99,235,0.13)",
                  border: "1.5px solid #3B82F6",
                },
                mx: { xs: 0, md: "auto" },
                width: "100%",
                maxWidth: "100%",
                minHeight: 150,
                display: "flex",
                alignItems: "center",
                overflowX: "hidden",
              }}
            >
              <CardContent sx={{ px: { xs: 2, sm: 3 }, py: { xs: 2.5, sm: 3.5 }, width: "100%", minWidth: 0 }}>
                <Stack spacing={2} sx={{ width: "100%", minWidth: 0, maxWidth: "100%" }}>
                  <Typography
                    variant="subtitle1"
                    sx={{
                      fontWeight: 800,
                      color: "#F8FAFC",
                      letterSpacing: 0.14,
                      mb: 0.5,
                    }}
                  >
                    Filtro e Detalhes
                  </Typography>
                  {predictions.length > 0 ? (
                    <FormControl fullWidth sx={{ mb: 0.5, minWidth: 0, maxWidth: "100%" }}>
                      <InputLabel
                        id="player-select-label"
                        sx={{
                          color: "#F1F5F9",
                          fontWeight: 600,
                          "&.Mui-focused": { color: "#3B82F6" },
                          background: "rgba(15,23,42,0.77)",
                          px: 0.5,
                        }}
                      >
                        Selecione um jogador
                      </InputLabel>
                      <Select
                        labelId="player-select-label"
                        value={selectedPlayerId}
                        label="Selecione um jogador"
                        onChange={(e) => setSelectedPlayerId(e.target.value)}
                        sx={{
                          color: "#F1F5F9",
                          ".MuiSelect-icon": { color: "#3B82F6" },
                          ".MuiOutlinedInput-notchedOutline": {
                            borderColor: "rgba(59,130,246,0.21)",
                          },
                          "&:hover .MuiOutlinedInput-notchedOutline": {
                            borderColor: "#3B82F6",
                          },
                          "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                            borderColor: "#2563EB",
                          },
                          background: "rgba(15,23,42,0.77)",
                          borderRadius: 2,
                          fontWeight: 600,
                          minWidth: 0,
                          maxWidth: "100%",
                          width: "100%",
                        }}
                        MenuProps={{
                          PaperProps: {
                            sx: {
                              bgcolor: "#1E293B",
                              color: "#F1F5F9",
                              borderRadius: 2,
                              maxWidth: "100%",
                            },
                          },
                        }}
                      >
                        {predictions.map((p) => (
                          <MenuItem key={p.identifier} value={p.identifier}>
                            {p.identifier}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  ) : (
                    <Typography
                      variant="body2"
                      sx={{
                        color: "rgba(203,213,225,0.68)",
                        fontWeight: 400,
                        fontSize: "1.06rem",
                      }}
                    >
                      Após carregar o Excel e executar a análise, você poderá selecionar um jogador aqui para ver mais detalhes.
                    </Typography>
                  )}
                  {selectedPlayerDetails ? (
                    <Alert
                      severity="info"
                      sx={{
                        mt: 1,
                        bgcolor: "rgba(30,41,59,0.90)",
                        color: "#F8FAFC",
                        border: "1px solid #2563EB",
                        fontWeight: 500,
                        "& .MuiAlert-icon": { color: "#3B82F6" },
                        borderRadius: 2,
                        px: 1,
                        fontSize: "1.01rem",
                        overflowX: "auto",
                        wordBreak: "break-word",
                      }}
                    >
                      O gráfico radar depende de perfis detalhados (player_profile e cluster_average_profile).
                      Podemos reintroduzir isso adicionando um endpoint <code>/predict/legacy</code> no backend.
                    </Alert>
                  ) : null}
                </Stack>
              </CardContent>
            </Card>
          </Grid>
          {/* Right column: Results */}
          <Grid
            item
            xs={12}
            md={7}
            sx={{
              minWidth: 0,
              width: "100%",
              maxWidth: { xs: "100%", md: 780 },
              alignSelf: "stretch",
              display: "flex",
              flexDirection: "column",
              boxSizing: "border-box",
            }}
          >
            <Card
              elevation={0}
              sx={{
                bgcolor: "rgba(23,30,48,0.96)",
                borderRadius: 7,
                border: "1.5px solid rgba(59,130,246,0.12)",
                boxShadow: "0 8px 36px 0 rgba(36,99,235,0.09)",
                height: "100%",
                minHeight: 420,
                maxHeight: 700,
                display: "flex",
                flexDirection: "column",
                px: { xs: 1, sm: 2.5 },
                py: { xs: 1.5, sm: 2.5 },
                mb: { xs: 0, md: 0 },
                width: "100%",
                maxWidth: "100%",
                overflowX: "hidden",
              }}
            >
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 800,
                  color: "#F8FAFC",
                  letterSpacing: 0.14,
                  mb: 2,
                  fontSize: "1.22rem",
                }}
              >
                Resultados das Previsões
              </Typography>
              {!predictions.length && (
                <Box
                  sx={{
                    flex: 1,
                    minHeight: 340,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    opacity: 0.33,
                    overflowX: "auto",
                    minWidth: 0,
                  }}
                >
                  <Typography
                    variant="body1"
                    sx={{
                      color: "#F8FAFC",
                      fontWeight: 500,
                      fontSize: "1.13rem",
                      textAlign: "center",
                      wordBreak: "break-word",
                    }}
                  >
                    Nenhum resultado disponível.<br />
                    Carregue um arquivo e execute a análise.
                  </Typography>
                </Box>
              )}
              {!!predictions.length && (
                <TableContainer
                  sx={{
                    flex: 1,
                    minHeight: 320,
                    maxHeight: 520,
                    overflowY: "auto",
                    overflowX: "auto",
                    borderRadius: 4,
                    width: "100%",
                    minWidth: 0,
                    boxSizing: "border-box",
                  }}
                >
                  <Table
                    size="small"
                    stickyHeader
                    aria-label="Tabela de previsões"
                    sx={{
                      borderRadius: 4,
                      width: "100%",
                      minWidth: 0,
                      tableLayout: "auto",
                      "& .MuiTableCell-root": {
                        color: "#F1F5F9",
                        fontWeight: 500,
                        borderBottom: "1px solid rgba(59,130,246,0.10)",
                        fontSize: "1rem",
                        background: "transparent",
                        py: 1,
                        px: { xs: 1, sm: 2 },
                        wordBreak: "break-word",
                        maxWidth: 210,
                      },
                      "& .MuiTableRow-hover:hover": {
                        background: "linear-gradient(90deg, #1e40af 0%, #2563eb22 100%)",
                        "& .MuiTableCell-root": { color: "#3B82F6" },
                      },
                    }}
                  >
                    <TableHead>
                      <TableRow
                        sx={{
                          background: "linear-gradient(90deg, #1e293b 0%, #2563eb1a 100%)",
                          "& .MuiTableCell-root": {
                            color: "#F8FAFC",
                            fontWeight: 800,
                            background: "transparent",
                            borderBottom: "2px solid #2563EB",
                            fontSize: "1.09rem",
                            px: { xs: 1, sm: 2 },
                          },
                        }}
                      >
                        <TableCell>Identificador</TableCell>
                        <TableCell>Cluster Previsto</TableCell>
                        <TableCell>Target 1 Previsto</TableCell>
                        <TableCell>Target 2 Previsto</TableCell>
                        <TableCell>Target 3 Previsto</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {predictions.map((p) => (
                        <TableRow
                          key={p.identifier}
                          hover
                          sx={{
                            transition: "background 0.13s",
                            "&:hover": {
                              background: "linear-gradient(90deg, #1e40af 0%, #2563eb22 100%)",
                            },
                          }}
                        >
                          <TableCell>{p.identifier}</TableCell>
                          <TableCell>{p.predicted_cluster ?? "-"}</TableCell>
                          <TableCell>{fmt(p.predicted_target1)}</TableCell>
                          <TableCell>{fmt(p.predicted_target2)}</TableCell>
                          <TableCell>{fmt(p.predicted_target3)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Card>
          </Grid>
        </Grid>
      </Container>
      {/* Overlay de loading */}
      <LoadingOverlay open={isLoading} stepText={loadingStep} />
    </Box>
  );
}

export default PredictionTool;