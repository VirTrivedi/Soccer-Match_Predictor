import React from 'react';

interface TeamSeasonStats {
  // This is an assumed structure based on season_predictor.py output
  // The actual backend response for /api/predict/season needs to conform.
  Rank?: number; // Added by frontend if not present
  team_name: string;
  avg_pos: number;
  median_pos?: number; // Optional based on backend script
  avg_pts: number;
  p_win_league: number;
  p_top_4: number;
  p_relegation: number;
  avg_w?: number;
  avg_d?: number;
  avg_l?: number;
  avg_gf?: number;
  avg_ga?: number;
  avg_gd?: number;
  // message and note fields are for general backend communication
  message?: string; // This would be part of the parent object, not individual rows
  note?: string;    // This would be part of the parent object, not individual rows
  error?: string;   // This would be part of the parent object, not individual rows
}

interface SeasonPredictionResponse {
  // This is the expected structure for the 'data' prop
  // It can contain a 'table' (array of team stats), or top-level message/note/error
  table?: TeamSeasonStats[];
  predicted_standings?: TeamSeasonStats[]; // Alternative key from backend app.py
  message?: string;
  note?: string;
  error?: string;
}

interface SeasonPredictionVisualizerProps {
  data: SeasonPredictionResponse | null;
}

const SeasonPredictionVisualizer: React.FC<SeasonPredictionVisualizerProps> = ({ data }) => {
  if (!data) return <p>No season prediction data available.</p>;

  // Handle if the data itself is an error or note (e.g. from conceptual backend response)
  if (data.error) return <p style={{color: 'red'}}>Error displaying visualization: {data.error}</p>;
  
  // The backend might return the table under 'table' or 'predicted_standings'
  const tableData = data.table || data.predicted_standings;

  // If there's a note from the backend (e.g. "conceptual prediction") and no table data, display the note.
  if (data.note && !tableData) {
    return (
      <div style={{ marginTop: '20px' }}>
        <h4>Season Simulation Results</h4>
        <p>Note: {data.note}</p>
        {data.message && <p><em>Backend message: {data.message}</em></p>}
      </div>
    );
  }

  if (!tableData || tableData.length === 0) {
    if (data.message) return <p><em>Backend message: {data.message}</em></p>;
    return <p>No season table data to display. Backend might be stubbed, response structure differs, or simulation yielded no results.</p>;
  }
  
  // Add Rank if not present, assuming sorted by avg_pos or whatever the backend sorts by
   const rankedTableData = tableData.map((row, index) => ({
    ...row,
    Rank: row.Rank !== undefined ? row.Rank : index + 1,
  }));


  return (
    <div style={{ marginTop: '20px', overflowX: 'auto' }}>
      <h4>Season Simulation Results</h4>
      {data.message && !tableData && <p><em>Backend message: {data.message}</em></p>}
      <table style={{ width: '100%', minWidth:'900px', borderCollapse: 'collapse', fontSize: '0.85em' }}>
        <thead>
          <tr>
            <th style={thStyle}>Rank</th>
            <th style={{...thStyle, minWidth: '150px', textAlign: 'left'}}>Team</th>
            <th style={thStyle}>Avg. Pos</th>
            <th style={thStyle}>Avg. Pts</th>
            <th style={thStyle}>P(Win League)</th>
            <th style={thStyle}>P(Top 4)</th>
            <th style={thStyle}>P(Releg.)</th>
            <th style={thStyle}>Avg W</th>
            <th style={thStyle}>Avg D</th>
            <th style={thStyle}>Avg L</th>
            <th style={thStyle}>Avg GF</th>
            <th style={thStyle}>Avg GA</th>
            <th style={thStyle}>Avg GD</th>
          </tr>
        </thead>
        <tbody>
          {rankedTableData.map((team, index) => (
            <tr key={team.team_name || index} style={index % 2 === 0 ? {backgroundColor: '#f9f9f9'} : {}}>
              <td style={tdStyle}>{team.Rank}</td>
              <td style={{...tdStyle, textAlign: 'left'}}>{team.team_name}</td>
              <td style={tdStyle}>{team.avg_pos?.toFixed(2)}</td>
              <td style={tdStyle}>{team.avg_pts?.toFixed(2)}</td>
              <td style={tdStyle}>{(team.p_win_league * 100).toFixed(1)}%</td>
              <td style={tdStyle}>{(team.p_top_4 * 100).toFixed(1)}%</td>
              <td style={tdStyle}>{(team.p_relegation * 100).toFixed(1)}%</td>
              <td style={tdStyle}>{team.avg_w?.toFixed(1)}</td>
              <td style={tdStyle}>{team.avg_d?.toFixed(1)}</td>
              <td style={tdStyle}>{team.avg_l?.toFixed(1)}</td>
              <td style={tdStyle}>{team.avg_gf?.toFixed(1)}</td>
              <td style={tdStyle}>{team.avg_ga?.toFixed(1)}</td>
              <td style={tdStyle}>{team.avg_gd?.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {data.note && tableData && <p style={{marginTop: '10px'}}><small><em>Note: {data.note}</em></small></p>}
    </div>
  );
};

const thStyle: React.CSSProperties = {
  borderBottom: '2px solid #007bff', // Accent color for header bottom border
  padding: '10px 8px',
  backgroundColor: '#f0f8ff', // Light blue for header background
  textAlign: 'center',
  fontWeight: 'bold',
};

const tdStyle: React.CSSProperties = {
  borderBottom: '1px solid #ddd', // Lighter lines for row separation
  padding: '8px',
  textAlign: 'center',
};

export default SeasonPredictionVisualizer;