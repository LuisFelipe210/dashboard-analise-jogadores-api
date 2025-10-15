// src/components/PredictionTool.jsx
import React, { useState } from 'react';
import {
  Box, Button, Typography, CircularProgress, Alert,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper,
  FormControl, InputLabel, Select, MenuItem
} from '@mui/material';
import { read, utils } from 'xlsx';
import { runPrediction, getSchema } from '../api/ApiService';

// === normalização de nomes (espelha o train) ===
const removeAccents = (str) => {
  if (typeof str !== 'string') return str;
  return str.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
};

const normalizeName = (s) => {
  if (s == null) return s;
  let out = String(s);
  out = out.trim();
  out = removeAccents(out);
  out = out.replace(/\s+/g, '_');          // espaços -> _
  out = out.replace(/[^a-zA-Z0-9_]/g, ''); // remove símbolos
  return out;
};

// mapeamentos específicos vistos no dataset
const SPECIAL_RENAMES = new Map([
  ['L0210_nao_likert', 'L0210_no_likert'],
  ['Codigo_de_Acesso', 'Cdigo_de_Acesso'],
]);

const applySpecialRename = (name) => {
  const key = normalizeName(name);
  return SPECIAL_RENAMES.get(key) || key;
};

// schema pode vir ["F0101",...] ou [{name:"F0101",type:"number"},...]
const adaptSchema = (schemaRaw) => {
  if (!Array.isArray(schemaRaw)) throw new Error('Schema inválido: não é array.');
  if (schemaRaw.length === 0) return [];

  if (typeof schemaRaw[0] === 'string') {
    return schemaRaw.map((name) => ({ name, type: 'number', default: 0, label: name }));
  }
  if (typeof schemaRaw[0] === 'object' && schemaRaw[0].name) {
    return schemaRaw.map(f => ({
      name: f.name,
      type: f.type || 'number',
      default: f.default ?? (f.type === 'string' ? '' : 0),
      label: f.label || f.name
    }));
  }
  throw new Error('Schema inválido: formato não reconhecido.');
};

function PredictionTool() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]); // [{identifier, predicted_cluster, predicted_target1, ...}]
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [selectedPlayerId, setSelectedPlayerId] = useState('');

  const pickIdentifier = (row) => {
    const candidates = [
      'Código de Acesso', 'Codigo_de_Acesso', 'Codigo de Acesso',
      'Cdigo_de_Acesso', 'id', 'ID', 'player_id', 'identifier'
    ];
    for (const k of candidates) {
      if (k in row && row[k] !== undefined && row[k] !== null && row[k] !== '') return row[k];
    }
    return null;
  };

  // Alinha a linha do Excel ao schema; normaliza nomes e converte números.
  const buildRowsFromSchema = (jsonData, schema) => {
    // normaliza as chaves de cada linha do Excel
    const normalizedRows = jsonData.map(orig => {
      const norm = {};
      Object.keys(orig).forEach(k => {
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
          let v = orig[fname];
          if (f.type === 'number') {
            if (typeof v === 'number') {
              row[fname] = Number.isFinite(v) ? v : f.default ?? 0;
            } else {
              const num = parseFloat(String(v).replace(',', '.'));
              row[fname] = Number.isFinite(num) ? num : (f.default ?? 0);
            }
          } else {
            row[fname] = (v ?? f.default ?? '');
          }
          matches++;
        } else {
          row[fname] = f.default ?? (f.type === 'number' ? 0 : '');
        }
      }

      // Identifier amigável (procurando também a versão normalizada)
      const idCandidates = [
        'Código de Acesso', 'Codigo de Acesso', 'Codigo_de_Acesso', 'Cdigo_de_Acesso',
        'id', 'ID', 'player_id', 'identifier'
      ];
      let identifier = null;
      for (const c of idCandidates) {
        const cNorm = applySpecialRename(c);
        if (Object.prototype.hasOwnProperty.call(orig, cNorm) && orig[cNorm] != null && orig[cNorm] !== '') {
          identifier = orig[cNorm];
          break;
        }
        if (Object.prototype.hasOwnProperty.call(orig, c) && orig[c] != null && orig[c] !== '') {
          identifier = orig[c];
          break;
        }
      }
      row.__identifier = identifier;

      if (matches < schema.length * 0.5) {
        console.warn(`Poucas colunas casaram: ${matches}/${schema.length}`, { origSample: Object.keys(orig).slice(0, 10) });
      }
      return row;
    });
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPredictions([]);
      setError('');
      setSuccess('');
      setSelectedPlayerId('');
    }
  };

  const handlePredict = async () => {
    if (!file) {
      setError("Por favor, carregue um arquivo primeiro.");
      return;
    }
    setIsLoading(true);
    setError('');
    setSuccess('');
    setPredictions([]);
    setSelectedPlayerId('');

    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const data = new Uint8Array(event.target.result);
        const workbook = read(data, { type: 'array' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const jsonData = utils.sheet_to_json(worksheet, { raw: true });

        if (jsonData.length === 0) {
          throw new Error("O arquivo está vazio ou em um formato inválido.");
        }

        // 1) schema do backend
        const schemaRaw = await getSchema();
        const schema = adaptSchema(schemaRaw);

        // 2) alinhar linhas conforme schema + normalização
        const rows = buildRowsFromSchema(jsonData, schema);

        // 3) enviar para o backend (array cru)
        const predsDict = await runPrediction(rows);
        console.log('Predições (brutas):', predsDict);

        // 4) montar tabela para UI
        const n = predsDict.target1?.length ?? 0;
        const clusterArr = predsDict.cluster ?? Array(n).fill(null);

        if (n === 0) {
          throw new Error('A API retornou zero previsões. Verifique logs do backend e os nomes do arquivo.');
        }

        const tableRows = Array.from({ length: n }).map((_, i) => {
          const id = rows[i].__identifier ?? `lin_${i+1}`;
          return {
            identifier: id,
            predicted_cluster: clusterArr ? clusterArr[i] : null,
            predicted_target1: predsDict.target1?.[i] ?? null,
            predicted_target2: predsDict.target2?.[i] ?? null,
            predicted_target3: predsDict.target3?.[i] ?? null,
          };
        });

        setPredictions(tableRows);
        setSuccess(`Análise concluída com sucesso para ${tableRows.length} jogadores!`);
      } catch (err) {
        console.error(err);
        setError(err.message || "Ocorreu um erro desconhecido.");
      } finally {
        setIsLoading(false);
      }
    };
    reader.readAsArrayBuffer(file);
  };

  const selectedPlayerDetails = predictions.find(p => p.identifier === selectedPlayerId);
  const fmt = (v) => (typeof v === 'number' ? v.toFixed(2) : v ?? '-');

  return (
    <Box>
      <Typography variant="h5" gutterBottom>Calcular Targets para Novos Jogadores</Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 4 }}>
        <Button variant="contained" component="label">
          Carregar Arquivo Excel
          <input type="file" accept=".xlsx" hidden onChange={handleFileChange} />
        </Button>
        {file && <Typography>{file.name}</Typography>}
        <Button
          onClick={handlePredict}
          variant="contained"
          color="primary"
          disabled={!file || isLoading}
          sx={{ ml: 'auto' }}
        >
          {isLoading ? <CircularProgress size={24} /> : '🚀 Realizar Análise'}
        </Button>
      </Box>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}

      {predictions.length > 0 && (
        <>
          <Typography variant="h6" gutterBottom>Resultados das Previsões</Typography>
          <TableContainer component={Paper} sx={{ mb: 4 }}>
            <Table>
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
                  <TableRow key={p.identifier}>
                    <TableCell>{p.identifier}</TableCell>
                    <TableCell>{p.predicted_cluster ?? '-'}</TableCell>
                    <TableCell>{fmt(p.predicted_target1)}</TableCell>
                    <TableCell>{fmt(p.predicted_target2)}</TableCell>
                    <TableCell>{fmt(p.predicted_target3)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          <Typography variant="h6" gutterBottom>Análise Detalhada por Jogador</Typography>
          <FormControl fullWidth sx={{ mb: 4 }}>
            <InputLabel id="player-select-label">Selecione um jogador</InputLabel>
            <Select
              labelId="player-select-label"
              value={selectedPlayerId}
              label="Selecione um jogador"
              onChange={(e) => setSelectedPlayerId(e.target.value)}
            >
              {predictions.map(p => (
                <MenuItem key={p.identifier} value={p.identifier}>{p.identifier}</MenuItem>
              ))}
            </Select>
          </FormControl>

          {selectedPlayerDetails ? (
            <Alert severity="info">
              O gráfico radar depende de perfis detalhados (player_profile / cluster_average_profile).
              Podemos reintroduzir isso adicionando um endpoint <code>/predict/legacy</code> no backend, se quiser.
            </Alert>
          ) : null}
        </>
      )}
    </Box>
  );
}

export default PredictionTool;
