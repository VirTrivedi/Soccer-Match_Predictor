import React from 'react';

interface LeagueSelectorProps {
  leagues: { [key: string]: string };
  selectedLeague: string;
  onLeagueChange: (leagueCode: string) => void;
  label?: string;
}

const LeagueSelector: React.FC<LeagueSelectorProps> = ({ leagues, selectedLeague, onLeagueChange, label="Select League" }) => {
  return (
    <div>
      <label htmlFor="league-select">{label}: </label>
      <select id="league-select" value={selectedLeague} onChange={(e) => onLeagueChange(e.target.value)}>
        <option value="">--Select a League--</option>
        {Object.entries(leagues).map(([code, name]) => (
          <option key={code} value={code}>{name} ({code})</option>
        ))}
      </select>
    </div>
  );
};

export default LeagueSelector;