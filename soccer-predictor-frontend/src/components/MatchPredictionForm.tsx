import React, { useState, useEffect } from 'react';
import axios from 'axios';
import LeagueSelector from './LeagueSelector';
import TeamSelector, { Team } from './TeamSelector'; // Import Team interface
import ApiResponseDisplay from './ApiResponseDisplay';
import MatchPredictionVisualizer from './MatchPredictionVisualizer';

interface MatchPredictionFormProps {
  leagues: { [key: string]: string };
}

const MatchPredictionForm: React.FC<MatchPredictionFormProps> = ({ leagues }) => {
  const [homeLeague, setHomeLeague] = useState('');
  const [awayLeague, setAwayLeague] = useState('');
  const [homeTeams, setHomeTeams] = useState<Team[]>([]);
  const [awayTeams, setAwayTeams] = useState<Team[]>([]);
  const [selectedHomeTeam, setSelectedHomeTeam] = useState('');
  const [selectedAwayTeam, setSelectedAwayTeam] = useState('');
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchTeams = async (leagueCode: string, setTeamsFunc: React.Dispatch<React.SetStateAction<Team[]>>) => {
    if (!leagueCode) {
      setTeamsFunc([]);
      return;
    }
    // Removed apiKey check here, as /api/teams doesn't need it from client
    try {
      setIsLoading(true);
      setError(null);
      const response = await axios.get(`http://localhost:5001/api/teams/${leagueCode}`);
      setTeamsFunc(Array.isArray(response.data) ? response.data : []);
    } catch (err: any) {
      console.error("Error fetching teams:", err);
      const message = err.response?.data?.error || `Failed to fetch teams for ${leagueCode}.`;
      setError(message);
      setTeamsFunc([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // Fetch teams if homeLeague is present; apiKey is not needed for this call.
    if (homeLeague) {
      fetchTeams(homeLeague, setHomeTeams);
    } else {
      setHomeTeams([]); // Clear teams if no league
    }
    setSelectedHomeTeam(''); // Reset team selection when league changes
  }, [homeLeague]); // Removed apiKey from dependency array

  useEffect(() => {
    // Fetch teams if awayLeague is present; apiKey is not needed for this call.
    if (awayLeague) {
      fetchTeams(awayLeague, setAwayTeams);
    } else {
      setAwayTeams([]); // Clear teams if no league
    }
    setSelectedAwayTeam(''); // Reset team selection when league changes
  }, [awayLeague]); // Removed apiKey from dependency array

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedHomeTeam || !selectedAwayTeam) {
      setError("Please select both home and away teams.");
      return;
    }
    setIsLoading(true);
    setPredictionResult(null);
    setError(null);
    try {
      const response = await axios.post('http://localhost:5001/api/predict/match', {
        home_team_id: parseInt(selectedHomeTeam),
        away_team_id: parseInt(selectedAwayTeam),
        home_league_code: homeLeague,
        away_league_code: awayLeague,
      }
      // If API key is preferred in header by backend:
      //
      );
      setPredictionResult(response.data);
    } catch (err: any) {
      console.error("Error making prediction:", err);
      setError(err.response?.data?.error || "Failed to make prediction.");
      setPredictionResult(err.response?.data || { error: "Prediction request failed" });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="prediction-form">
      <h3>Single Match Prediction</h3>
      <form onSubmit={handleSubmit}>
        <div>
          <LeagueSelector leagues={leagues} selectedLeague={homeLeague} onLeagueChange={setHomeLeague} label="Home League" />
          <TeamSelector teams={homeTeams} selectedTeam={selectedHomeTeam} onTeamChange={setSelectedHomeTeam} label="Home Team" disabled={!homeLeague || homeTeams.length === 0 || isLoading} />
        </div>
        <hr style={{margin: '10px 0'}}/>
        <div>
          <LeagueSelector leagues={leagues} selectedLeague={awayLeague} onLeagueChange={setAwayLeague} label="Away League" />
          <TeamSelector teams={awayTeams} selectedTeam={selectedAwayTeam} onTeamChange={setSelectedAwayTeam} label="Away Team" disabled={!awayLeague || awayTeams.length === 0 || isLoading} />
        </div>
        <button type="submit" disabled={isLoading || !selectedHomeTeam || !selectedAwayTeam}>
          {isLoading ? 'Predicting...' : 'Predict Match'}
        </button>
      </form>
      {error && <p className="error-message">Error: {error}</p>}
      <ApiResponseDisplay data={predictionResult} title="Match Prediction Response" />
        <MatchPredictionVisualizer data={predictionResult} />
    </div>
  );
};

export default MatchPredictionForm;