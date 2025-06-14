import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './App.css';
import Header from './components/Header';
import MatchPredictionForm from './components/MatchPredictionForm';
import SeasonPredictionForm from './components/SeasonPredictionForm';
// ApiResponseDisplay is used within the forms, not directly here unless for a global display

function App() {
  const [backendMessage, setBackendMessage] = useState('');
  const [leagues, setLeagues] = useState<{ [key: string]: string }>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    axios.get('http://localhost:5001/api/test')
      .then(response => {
        setBackendMessage((response.data as { message: string }).message);
      })
      .catch(err => {
        console.error('Error fetching test message:', err);
        setBackendMessage('Could not connect to backend. Ensure Flask is running on port 5001.');
        setError('Failed to connect to backend. Is it running on port 5001?');
      });

    axios.get('http://localhost:5001/api/leagues')
      .then(response => {
        setLeagues(response.data as { [key: string]: string } || {});
      })
      .catch(err => {
        console.error('Error fetching leagues:', err);
        // Keep specific error for leagues separate if needed, or combine
        setError(prevError => prevError ? prevError + " | Failed to fetch leagues." : 'Failed to fetch leagues. Backend might be down or leagues endpoint has an issue.');
        setLeagues({} as { [key: string]: string }); // Set to empty object on error
      });
  }, []);

  return (
    <div className="App">
      <Header />
      <main className="App-content">
        <h1>Soccer Predictor Dashboard</h1>
        <p><em>Backend status: {backendMessage}</em></p>
        {error && <p className="error-message">App Error: {error}</p>}

        {Object.keys(leagues).length > 0 ? (
          <>
            <MatchPredictionForm leagues={leagues} />
            <SeasonPredictionForm leagues={leagues} />
          </>
        ) : (
          !error && <p>Loading league data or no leagues available. Ensure backend is running and `leagues` endpoint is functional.</p>
        )}
        
      </main>
    </div>
  );
}

export default App;