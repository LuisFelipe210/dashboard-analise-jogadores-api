// src/components/AnalysisDashboard.jsx
import React, { useState, useEffect } from 'react';
import { Box, Grid, Card, CardContent, Typography, CircularProgress, Alert, FormControl, InputLabel, Select, MenuItem, OutlinedInput, Checkbox, ListItemText, Tooltip } from '@mui/material';
import HelpOutline from '@mui/icons-material/HelpOutline';
import Plot from 'react-plotly.js';

// Função para converter texto CSV em array de objetos JSON (MAIS ROBUSTA)
const parseCSV = (text) => {
  const cleanedText = text.replace(/"/g, '').trim();
  const lines = cleanedText.split('\n');
  const header = lines[0].split(',').map(h => h.trim());
  const data = lines.slice(1).map(line => {
    const values = line.split(',');
    return header.reduce((obj, key, i) => {
      const value = values[i] ? values[i].trim() : '';
      obj[key] = !isNaN(parseFloat(value)) && isFinite(value) ? parseFloat(value) : value;
      return obj;
    }, {});
  });
  return data.filter(row => row.cluster !== undefined && Object.keys(row).length === header.length);
};

// Funções para cálculo de métricas (COMPLETAS E CORRIGIDAS)
const calculateRMSE = (actual, predicted) => {
  const n = actual.length;
  if (n === 0) return 0;
  const sumSquaredError = actual.reduce((sum, val, i) => sum + Math.pow(val - predicted[i], 2), 0);
  return Math.sqrt(sumSquaredError / n);
};

const calculateR2 = (actual, predicted) => {
  const n = actual.length;
  if (n === 0) return 0;
  const meanActual = actual.reduce((sum, val) => sum + val, 0) / n;
  const ssTotal = actual.reduce((sum, val) => sum + Math.pow(val - meanActual, 2), 0);
  const ssResidual = actual.reduce((sum, val, i) => sum + Math.pow(val - predicted[i], 2), 0);
  if (ssTotal === 0) return 1;
  return 1 - (ssResidual / ssTotal);
};

// Textos dos tooltips
const RMSE_TOOLTIP_TEXT = "Raiz do Erro Quadrático Médio (RMSE): Mede a distância média entre os valores previstos pelo modelo e os valores reais. Quanto menor o valor, mais preciso é o modelo.";
const R2_TOOLTIP_TEXT = "R-Quadrado (R²): Indica a porcentagem da variação dos dados reais que o modelo consegue explicar. Varia de 0 a 1, onde 1 é um ajuste perfeito. Quanto maior, melhor.";

function AnalysisDashboard() {
  const [data, setData] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [clusterOptions, setClusterOptions] = useState([]);
  const [selectedClusters, setSelectedClusters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('/jogadores_com_clusters.csv');
        if (!response.ok) {
          throw new Error('Falha ao carregar o arquivo CSV. Verifique se ele está na pasta /public.');
        }
        const text = await response.text();
        const parsedData = parseCSV(text);
        
        const clusters = [...new Set(parsedData.map(item => item.cluster))].sort((a,b) => a-b);
        
        setData(parsedData);
        setClusterOptions(clusters);
        setSelectedClusters(clusters);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    if (data.length > 0) {
      const newFilteredData = data.filter(row => selectedClusters.includes(row.cluster));
      setFilteredData(newFilteredData);

      if (newFilteredData.length > 0) {
        const newMetrics = {};
        ['Target1', 'Target2', 'Target3'].forEach(target => {
          const validPairs = newFilteredData
            .map(d => ({
              actual: d[target],
              predicted: d[`${target}_Previsto`]
            }))
            .filter(pair => typeof pair.actual === 'number' && typeof pair.predicted === 'number');

          if (validPairs.length > 0) {
            const actual = validPairs.map(p => p.actual);
            const predicted = validPairs.map(p => p.predicted);
            newMetrics[`rmse_${target}`] = calculateRMSE(actual, predicted).toFixed(2);
            newMetrics[`r2_${target}`] = calculateR2(actual, predicted).toFixed(2);
          } else {
             newMetrics[`rmse_${target}`] = 'N/A';
             newMetrics[`r2_${target}`] = 'N/A';
          }
        });
        setMetrics(newMetrics);
      } else {
        setMetrics({});
      }
    }
  }, [selectedClusters, data]);
  
  const handleClusterChange = (event) => {
    const { target: { value } } = event;
    setSelectedClusters(typeof value === 'string' ? value.split(',') : value);
  };
  
  if (loading) return <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}><CircularProgress /></Box>;
  if (error) return <Alert severity="error">{error}</Alert>;

  const clusterCounts = filteredData.reduce((acc, row) => {
    acc[row.cluster] = (acc[row.cluster] || 0) + 1;
    return acc;
  }, {});

  return (
    <Box>
      <FormControl fullWidth sx={{ mb: 4 }}>
        <InputLabel id="cluster-multiple-checkbox-label">Filtrar por Cluster</InputLabel>
        <Select
          labelId="cluster-multiple-checkbox-label"
          multiple
          value={selectedClusters}
          onChange={handleClusterChange}
          input={<OutlinedInput label="Filtrar por Cluster" />}
          renderValue={(selected) => selected.sort((a,b)=>a-b).join(', ')}
        >
          {clusterOptions.map((cluster) => (
            <MenuItem key={cluster} value={cluster}>
              <Checkbox checked={selectedClusters.indexOf(cluster) > -1} />
              <ListItemText primary={`Cluster ${cluster}`} />
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <Typography variant="h5" gutterBottom>Métricas de Avaliação do Modelo</Typography>
      <Grid container spacing={2} sx={{ mb: 4 }}>
        {['Target1', 'Target2', 'Target3'].map(target => (
          <Grid item xs={12} md={4} key={target}>
            <Card>
              <CardContent>
                <Typography variant="h6">{target}</Typography>
                <Tooltip title={RMSE_TOOLTIP_TEXT} arrow placement="top">
                  <Box sx={{ display: 'flex', alignItems: 'center', cursor: 'help' }}>
                    <Typography>RMSE: {metrics[`rmse_${target}`] || 'N/A'}</Typography>
                    <HelpOutline sx={{ ml: 0.5, fontSize: '1rem', color: 'grey.500' }} />
                  </Box>
                </Tooltip>
                <Tooltip title={R2_TOOLTIP_TEXT} arrow placement="top">
                  <Box sx={{ display: 'flex', alignItems: 'center', cursor: 'help' }}>
                    <Typography>R²: {metrics[`r2_${target}`] || 'N/A'}</Typography>
                    <HelpOutline sx={{ ml: 0.5, fontSize: '1rem', color: 'grey.500' }} />
                  </Box>
                </Tooltip>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
      
      <Typography variant="h5" gutterBottom>Gráficos Comparativos: Real vs. Previsto</Typography>
       <Grid container spacing={2} sx={{ mb: 4 }}>
        {['Target1', 'Target2', 'Target3'].map(target => {
          const actualValues = filteredData.map(d => d[target]);
          const predictedValues = filteredData.map(d => d[`${target}_Previsto`]);
          const allValues = [...actualValues, ...predictedValues].filter(v => typeof v === 'number');
          const minVal = allValues.length > 0 ? Math.min(...allValues) : 0;
          const maxVal = allValues.length > 0 ? Math.max(...allValues) : 100;

          return (
            <Grid item xs={12} md={4} key={target}>
              <Plot
                data={[
                  {
                    x: actualValues,
                    y: predictedValues,
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Previsões do Modelo',
                    marker: { color: '#00C49F' },
                  },
                  {
                    x: [minVal, maxVal],
                    y: [minVal, maxVal],
                    mode: 'lines',
                    type: 'scatter',
                    name: 'Linha de Previsão Ideal',
                    line: { color: '#FF6B6B', dash: 'dash' }
                  }
                ]}
                layout={{
                  title: `<b>${target}</b>`,
                  xaxis: { title: 'Valor Real (Gabarito)' },
                  yaxis: { title: 'Valor Previsto (Previsão do Modelo)' },
                  autosize: true,
                  paper_bgcolor: '#29384B',
                  plot_bgcolor: '#29384B',
                  font: { color: '#FFFFFF' },
                  showlegend: true,
                  legend: { x: 0.01, y: 0.98, bgcolor: 'rgba(0,0,0,0.3)' }
                }}
                useResizeHandler={true}
                style={{ width: '100%', height: '400px' }}
              />
            </Grid>
          );
        })}
      </Grid>
      
      <Typography variant="h5" gutterBottom>Distribuição de Jogadores por Cluster</Typography>
      <Plot
          data={[{
              x: Object.keys(clusterCounts),
              y: Object.values(clusterCounts),
              type: 'bar',
              text: Object.values(clusterCounts),
              textposition: 'auto',
              marker: {color: '#00C49F'}
          }]}
          layout={{
              title: `Distribuição nos Clusters (${filteredData.length} jogadores)`,
              xaxis: { title: 'Cluster' },
              yaxis: { title: 'Quantidade de Jogadores' },
              autosize: true,
              paper_bgcolor: '#29384B',
              plot_bgcolor: '#29384B',
              font: { color: '#FFFFFF' }
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
      />
    </Box>
  );
}

export default AnalysisDashboard;