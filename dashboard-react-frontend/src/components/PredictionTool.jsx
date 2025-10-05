// src/components/PredictionTool.jsx
import React, { useState } from 'react';
import { Box, Button, Typography, CircularProgress, Alert, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import { read, utils } from 'xlsx';
import { runPrediction } from '../api/ApiService';
import Plot from 'react-plotly.js';

function PredictionTool() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [selectedPlayerId, setSelectedPlayerId] = useState('');

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPredictions([]); // Limpa previsões antigas
      setError('');
      setSuccess('');
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
        const jsonData = utils.sheet_to_json(worksheet);

        if (jsonData.length === 0) {
          throw new Error("O arquivo está vazio ou em um formato inválido.");
        }

        const apiResponse = await runPrediction(jsonData);
        setPredictions(apiResponse);
        setSuccess(`Análise concluída com sucesso para ${apiResponse.length} jogadores!`);
      } catch (err) {
        setError(err.message || "Ocorreu um erro desconhecido.");
      } finally {
        setIsLoading(false);
      }
    };
    reader.readAsArrayBuffer(file);
  };
  
  const selectedPlayerDetails = predictions.find(p => p.identifier === selectedPlayerId);

  return (
    <Box>
      <Typography variant="h5" gutterBottom>Calcular Targets para Novos Jogadores</Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 4 }}>
        <Button
          variant="contained"
          component="label"
        >
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
                    <TableCell>{p.predicted_cluster}</TableCell>
                    <TableCell>{p.predicted_target1}</TableCell>
                    <TableCell>{p.predicted_target2}</TableCell>
                    <TableCell>{p.predicted_target3}</TableCell>
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

            {selectedPlayerDetails && (
               <Plot
                data={[
                  {
                    type: 'scatterpolar',
                    r: Object.values(selectedPlayerDetails.cluster_average_profile).slice(0, 8),
                    theta: Object.keys(selectedPlayerDetails.cluster_average_profile).slice(0, 8),
                    fill: 'toself',
                    name: 'Média do Cluster'
                  },
                  {
                    type: 'scatterpolar',
                    r: Object.values(selectedPlayerDetails.player_profile).slice(0, 8),
                    theta: Object.keys(selectedPlayerDetails.player_profile).slice(0, 8),
                    fill: 'toself',
                    name: `Jogador ${selectedPlayerDetails.identifier}`
                  }
                ]}
                layout={{
                  title: `Comparativo: Jogador ${selectedPlayerDetails.identifier} vs. Média do Cluster`,
                  polar: { radialaxis: { visible: true, range: [0, 50] } },
                  autosize: true,
                  paper_bgcolor: '#29384B',
                  plot_bgcolor: '#29384B',
                  font: { color: '#FFFFFF' }
                }}
                useResizeHandler={true}
                style={{ width: '100%', height: '100%' }}
              />
            )}
        </>
      )}
    </Box>
  );
}

export default PredictionTool;