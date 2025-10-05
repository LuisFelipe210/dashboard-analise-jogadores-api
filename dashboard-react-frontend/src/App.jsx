// src/App.jsx
import React, { useState } from 'react';
import { Box, Tab, Tabs, Typography, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import AnalysisDashboard from './components/AnalysisDashboard';
import PredictionTool from './components/PredictionTool';

// Definindo um tema escuro similar ao do Streamlit
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00C49F',
    },
    background: {
      default: '#1A1A2E',
      paper: '#29384B',
    },
  },
});

function App() {
  const [activeTab, setActiveTab] = useState(0);

  const handleChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ width: '100%', padding: '2rem' }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Dashboard de Análise e Previsão de Jogadores
        </Typography>
        
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleChange} aria-label="abas de navegação">
            <Tab label="Análise de Desempenho" />
            <Tab label="Previsão para Novos Jogadores" />
          </Tabs>
        </Box>

        {/* Conteúdo da Aba 1 */}
        {activeTab === 0 && (
          <Box sx={{ paddingTop: '2rem' }}>
            <AnalysisDashboard />
          </Box>
        )}

        {/* Conteúdo da Aba 2 */}
        {activeTab === 1 && (
          <Box sx={{ paddingTop: '2rem' }}>
            <PredictionTool />
          </Box>
        )}
      </Box>
    </ThemeProvider>
  );
}

export default App;