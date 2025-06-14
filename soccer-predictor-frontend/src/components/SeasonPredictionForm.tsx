import React, { useState } from 'react';
import axios from 'axios';
import LeagueSelector from './LeagueSelector';
import ApiResponseDisplay from './ApiResponseDisplay';
import SeasonPredictionVisualizer from './SeasonPredictionVisualizer';

interface SeasonPredictionFormProps {
  leagues: { [key: string]: string };
}

const SeasonPredictionForm: React.FC<SeasonPredictionFormProps> = ({ leagues }) => {
  const [selectedLeague, setSelectedLeague] = useState('');
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Add this console.log for debugging
  console.log("SeasonPredictionForm render state:", {
    isLoading,
    selectedLeague,
    isLeagueSelected: !!selectedLeague, // Explicit boolean check
    buttonDisabled: isLoading || !selectedLeague // apiKey REMOVED from disabled logic
  });

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedLeague) {
      setError("Please select a league.");
      return;
    }
    setIsLoading(true);
    setPredictionResult(null);
    setError(null);
    try {
      const response = await axios.post('http://localhost:5001/api/predict/season', {
        league_code: selectedLeague,
      }
      // If API key is preferred in header by backend:
      //, { headers: { 'X-API-Key': apiKey } }
      );
      setPredictionResult(response.data);
    } catch (err: any) {
      console.error("Error making season prediction:", err);
      setError(err.response?.data?.error || "Failed to make season prediction.");
      setPredictionResult(err.response?.data || { error: "Season prediction request failed" });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="prediction-form">
      <h3>Full Season Simulation</h3>
      <form onSubmit={handleSubmit}>
        <LeagueSelector leagues={leagues} selectedLeague={selectedLeague} onLeagueChange={setSelectedLeague} />
        <button type="submit" disabled={isLoading || !selectedLeague}>
          {isLoading ? 'Simulating...' : 'Simulate Season'}
        </button>
      </form>
      {error && <p className="error-message">Error: {error}</p>}
      <ApiResponseDisplay data={predictionResult} title="Season Simulation Response" />
        <SeasonPredictionVisualizer data={predictionResult} />
    </div>
  );
};

export default SeasonPredictionForm;