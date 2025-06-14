import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

interface ScoreProbability {
  score: string;
  probability: number;
}

interface OutcomeProbabilities {
  homeWin: number;
  draw: number;
  awayWin: number;
}

interface MatchPredictionData {
  // This is an assumed structure based on typical model output
  // The actual backend response for /api/predict/match needs to conform to this
  // or this component needs to be adapted.
  predicted_scoreline_probabilities?: ScoreProbability[]; // e.g., [{score: "1-0", probability: 0.15}, ...]
  score_grid?: number[][]; // e.g., a 6x6 grid of probabilities for scores 0-0 to 5-5
  outcome_probabilities?: OutcomeProbabilities; // e.g., {homeWin: 0.45, draw: 0.25, awayWin: 0.30}
  message?: string; // For messages from backend like "Full prediction logic integration requires..."
  note?: string;
  error?: string;
}

interface MatchPredictionVisualizerProps {
  data: MatchPredictionData | null;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82Ca9D'];

const MatchPredictionVisualizer: React.FC<MatchPredictionVisualizerProps> = ({ data }) => {
  if (!data) return <p>No prediction data available.</p>;
  if (data.error) return <p style={{color: 'red'}}>Error displaying visualization: {data.error}</p>;
  // If there's a note from the backend (e.g. "conceptual prediction"), display it and maybe don't render charts.
  if (data.note && (!data.outcome_probabilities && !data.predicted_scoreline_probabilities && !data.score_grid)) {
    return (
      <div style={{ marginTop: '20px' }}>
        <h4>Match Prediction Visualizations</h4>
        <p>Note: {data.note}</p>
        {data.message && <p><em>Backend message: {data.message}</em></p>}
      </div>
    );
  }


  const { predicted_scoreline_probabilities, score_grid, outcome_probabilities } = data;

  const outcomeData = outcome_probabilities ? [
    { name: 'Home Win', probability: parseFloat((outcome_probabilities.homeWin * 100).toFixed(1)) },
    { name: 'Draw', probability: parseFloat((outcome_probabilities.draw * 100).toFixed(1)) },
    { name: 'Away Win', probability: parseFloat((outcome_probabilities.awayWin * 100).toFixed(1)) },
  ] : [];

  return (
    <div style={{ marginTop: '20px' }}>
      <h4>Match Prediction Visualizations</h4>

      {outcome_probabilities && outcomeData.length > 0 && (
        <div style={{ marginBottom: '30px' }}>
          <h5>Overall Outcome Probabilities</h5>
          <ResponsiveContainer width="80%" height={300}>
            <BarChart data={outcomeData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft' }} unit="%" />
              <Tooltip formatter={(value) => `${value}%`} />
              <Legend />
              <Bar dataKey="probability" name="Probability" unit="%">
                {outcomeData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {predicted_scoreline_probabilities && predicted_scoreline_probabilities.length > 0 && (
        <div style={{ marginBottom: '30px' }}>
          <h5>Top Predicted Scorelines (Top 5)</h5>
          <table style={{ width: 'clamp(300px, 50%, 500px)', margin: 'auto', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{border: '1px solid #ddd', padding: '8px', backgroundColor: '#f9f9f9'}}>Score</th>
                <th style={{border: '1px solid #ddd', padding: '8px', backgroundColor: '#f9f9f9'}}>Probability</th>
              </tr>
            </thead>
            <tbody>
              {predicted_scoreline_probabilities.slice(0, 5).map((p, index) => ( 
                <tr key={index}>
                  <td style={{border: '1px solid #ddd', padding: '8px'}}>{p.score}</td>
                  <td style={{border: '1px solid #ddd', padding: '8px'}}>{(p.probability * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {score_grid && score_grid.length > 0 && score_grid.every(row => Array.isArray(row)) && (
        <div>
          <h5>Score Grid Probabilities (Home vs Away)</h5>
          <p style={{fontSize: '0.9em', color: '#555'}}>Scores from 0 (top/left) to {score_grid.length -1} (home) / {score_grid[0].length -1} (away)</p>
          <table style={{ margin: 'auto', borderCollapse: 'collapse', border: '1px solid #ddd' }}>
            <thead>
              <tr>
                <th style={{border: '1px solid #ddd', padding: '5px', width: '30px', height: '30px'}}>H\A</th>
                {score_grid[0].map((_, awayScore) => (
                  <th key={awayScore} style={{border: '1px solid #ddd', padding: '5px', backgroundColor: '#f0f0f0', width: '40px', height: '30px', textAlign: 'center'}}>{awayScore}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {score_grid.map((homeScores, homeScore) => (
                <tr key={homeScore}>
                  <td style={{border: '1px solid #ddd', padding: '5px', backgroundColor: '#f0f0f0', fontWeight: 'bold', width: '30px', height: '30px', textAlign: 'center'}}>{homeScore}</td>
                  {homeScores.map((prob, awayScore) => (
                    <td key={`${homeScore}-${awayScore}`} 
                        style={{
                          border: '1px solid #ddd', 
                          padding: '5px', 
                          width: '40px', height: '30px', textAlign: 'center',
                          backgroundColor: prob > 0.1 ? 'rgba(0, 136, 254, 0.5)' : (prob > 0.05 ? 'rgba(0, 136, 254, 0.3)' : (prob > 0.01 ? 'rgba(0, 136, 254, 0.1)' : 'transparent'))
                        }}>
                      {(prob * 100).toFixed(1)}%
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
           <small style={{display: 'block', marginTop: '10px'}}>Note: Background color intensity indicates higher probability. Assumes grid up to 5-5 or as provided.</small>
        </div>
      )}
      
      {(!outcome_probabilities && !predicted_scoreline_probabilities && !score_grid) && !data.note && (
        <p>Detailed visualization data not available in the current response. Backend might be stubbed or response structure differs from expected.</p>
      )}
      {data.message && <p><em>Backend message: {data.message}</em></p>}
    </div>
  );
};

export default MatchPredictionVisualizer;