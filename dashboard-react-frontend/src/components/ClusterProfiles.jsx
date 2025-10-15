// src/components/ClusterProfiles.jsx
import React, { useEffect, useState, useMemo } from 'react';
import {
  Box, Grid, Card, CardContent, Typography, Alert, CircularProgress, List, ListItem, ListItemText,
} from '@mui/material';
import Plot from 'react-plotly.js';
import { getClusterProfiles } from '../api/ApiService';

/**
 * Componente que mostra:
 *  - Top diferenças absolutas entre clusters (lista)
 *  - Radar comparando os clusters nas TOP features
 *
 * Backend: GET /clusters/profile
 */
function ClusterProfiles({ limit = 5 }) {
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState('');
  const [data, setData] = useState(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setErr('');
    getClusterProfiles()
      .then((res) => {
        if (!mounted) return;
        setData(res);
      })
      .catch((e) => {
        if (!mounted) return;
        setErr(e?.response?.data?.detail || e.message || 'Falha ao carregar perfil de clusters.');
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => { mounted = false; };
  }, [limit]);

  const radar = useMemo(() => {
    if (!data) return null;

    const features = data.used_features || [];
    const clusters = data.clusters || [];

    // Monta matriz [cluster][featureValue]
    const series = clusters.map((cl) => {
      const mm = data.means?.[String(cl)] || {};
      return features.map((f) => mm[f] ?? 0);
    });

    // Normalização min-max por feature para visual mais estável no radar
    const normSeries = (() => {
      if (features.length === 0) return series;
      const mins = features.map((f, i) =>
        Math.min(...series.map((row) => row[i])));
      const maxs = features.map((f, i) =>
        Math.max(...series.map((row) => row[i])));
      return series.map((row) =>
        row.map((v, i) => {
          const min = mins[i], max = maxs[i];
          if (max === min) return 0.5; // evita divisão por zero
          return (v - min) / (max - min); // [0,1]
        })
      );
    })();

    // Para radar, repetimos o primeiro ponto no final
    const theta = [...features, features[0]];

    const traces = clusters.map((cl, idx) => {
      const r = [...normSeries[idx], normSeries[idx][0]];
      return {
        type: 'scatterpolar',
        r,
        theta,
        fill: 'toself',
        name: `Cluster ${cl}`,
      };
    });

    return {
      data: traces,
      layout: {
        title: `<b>Radar: diferenças por feature (normalizado)</b>`,
        polar: {
          radialaxis: { visible: true, range: [0, 1] },
        },
        paper_bgcolor: '#29384B',
        plot_bgcolor: '#29384B',
        font: { color: '#FFFFFF' },
        margin: { t: 50, r: 10, b: 10, l: 10 },
        showlegend: true,
      },
    };
  }, [data]);

  if (loading) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}><CircularProgress /></Box>;
  }
  if (err) {
    return <Alert severity="warning">{err}</Alert>;
  }
  if (!data) return null;

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom>Perfis dos Clusters (Top diferenças)</Typography>

      <Grid container spacing={2}>
        {/* Bloco da lista com as TOP diferenças */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Top diferenças absolutas</Typography>
              <List dense>
                {(data.top_diffs || []).map((d, i) => (
                  <ListItem key={d.feature}>
                    <ListItemText
                      primary={`${i + 1}. ${d.feature}`}
                      secondary={
                        <>
                          {Object.entries(d.by_cluster).map(([cl, val]) => (
                            <span key={cl} style={{ display: 'inline-block', marginRight: 12 }}>
                              <b>Cluster {cl}:</b> {Number(val).toFixed(2)}
                            </span>
                          ))}
                          <span style={{ display: 'inline-block' }}>
                            <b>Δ abs:</b> {Number(d.diff_abs).toFixed(2)}
                          </span>
                        </>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Bloco do radar */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              {radar ? (
                <Plot
                  data={radar.data}
                  layout={radar.layout}
                  useResizeHandler
                  style={{ width: '100%', height: 420 }}
                />
              ) : (
                <Alert severity="info">Sem dados suficientes para montar o radar.</Alert>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ mt: 2 }}>
        <Typography variant="body2" color="text.secondary">
          * Os valores no radar são <i>normalizados por feature</i> para comparação visual.
          A lista à esquerda mostra os valores absolutos originais por cluster e a diferença (Δ).
        </Typography>
      </Box>
    </Box>
  );
}

export default ClusterProfiles;
