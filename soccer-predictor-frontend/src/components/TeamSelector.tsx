import React from 'react';

export interface Team {
  id: number;
  name: string;
}

interface TeamSelectorProps {
  teams: Team[];
  selectedTeam: string; // Store team ID as string
  onTeamChange: (teamId: string) => void;
  label?: string;
  disabled?: boolean;
}

const TeamSelector: React.FC<TeamSelectorProps> = ({ teams, selectedTeam, onTeamChange, label = "Select Team", disabled = false }) => {
  return (
    <div>
      <label htmlFor="team-select">{label}: </label>
      <select id="team-select" value={selectedTeam} onChange={(e) => onTeamChange(e.target.value)} disabled={disabled || teams.length === 0}>
        <option value="">--Select a Team--</option>
        {teams.map((team) => (
          <option key={team.id} value={String(team.id)}>{team.name}</option>
        ))}
      </select>
    </div>
  );
};

export default TeamSelector;