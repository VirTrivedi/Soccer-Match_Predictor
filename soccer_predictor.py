import api_client
from datetime import datetime, timedelta
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from dotenv import load_dotenv

load_dotenv()

ALLOWED_LEAGUES = {
    "BL1": "Bundesliga",
    "DED": "Eredivisie",
    "BSA": "Campeonato Brasileiro Série A",
    "PD": "Primera Division",
    "FL1": "Ligue 1",
    "ELC": "Championship",
    "PPL": "Primeira Liga",
    "SA": "Serie A",
    "PL": "Premier League"
}

def get_match_outcome(match_data, team_id_of_interest):
    """
    Determines the outcome of a match for a specific team_id_of_interest
    based on football-data.org fixture data structure (new format).
    """
    home_team_data = match_data.get('homeTeam')
    away_team_data = match_data.get('awayTeam')
    score_data = match_data.get('score')

    if not home_team_data or not away_team_data or not score_data:
        return None

    home_id = home_team_data.get('id')
    away_id = away_team_data.get('id')
    winner_status = score_data.get('winner')
    
    full_time_score = score_data.get('fullTime')
    if full_time_score is None:
        return None
        
    home_goals = full_time_score.get('home')
    away_goals = full_time_score.get('away')

    if home_id is None or away_id is None:
        return None

    if winner_status == "HOME_TEAM":
        return 'WIN' if team_id_of_interest == home_id else 'LOSS'
    elif winner_status == "AWAY_TEAM":
        return 'WIN' if team_id_of_interest == away_id else 'LOSS'
    elif winner_status == "DRAW":
        return 'DRAW'
    else:
        # Fallback if winner_status is None or unexpected, use scores
        if home_goals is None or away_goals is None:
            return None
        if home_goals > away_goals:
            return 'WIN' if team_id_of_interest == home_id else 'LOSS'
        elif away_goals > home_goals:
            return 'WIN' if team_id_of_interest == away_id else 'LOSS'
        else:
            return 'DRAW'
        
def get_outcome_from_scoreline(scoreline_str):
    """Converts a 'H-A' scoreline string to 'WIN', 'LOSS', or 'DRAW' for the home team."""
    try:
        if pd.isna(scoreline_str) or not isinstance(scoreline_str, str) or '-' not in scoreline_str:
            return "UNKNOWN" 
        
        parts = scoreline_str.split('-', 1)
        if len(parts) != 2:
            return "UNKNOWN"
            
        home_score_str, away_score_str = parts
        home_score = int(home_score_str)
        away_score = int(away_score_str)

        if home_score > away_score:
            return "WIN"
        elif home_score < away_score:
            return "LOSS"
        else:
            return "DRAW"
    except ValueError:
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"

def calculate_h2h_features(h2h_matches: list, perspective_home_team_id: int, perspective_away_team_id: int):
    """
    Calculates head-to-head features from the perspective of perspective_home_team_id.
    Uses new match data structure.
    """
    features = {
        'h2h_home_wins': 0,
        'h2h_draws': 0,
        'h2h_away_wins': 0,
        'h2h_home_goals_sum': 0,
        'h2h_away_goals_sum': 0,
        'h2h_matches_played': 0,
        'h2h_avg_goals_scored_perspective': 0.0,
        'h2h_avg_goals_conceded_perspective': 0.0,
        'h2h_win_rate_perspective': 0.0,
        'h2h_draw_rate_perspective': 0.0,
        'h2h_loss_rate_perspective': 0.0,
    }
    
    for match in h2h_matches:
        home_team_data = match.get('homeTeam')
        away_team_data = match.get('awayTeam')
        score_data = match.get('score', {}).get('fullTime')

        if not home_team_data or not away_team_data or not score_data:
            continue

        match_home_id = home_team_data.get('id')
        match_away_id = away_team_data.get('id')
        
        if match_home_id is None or match_away_id is None:
            continue
            
        if not ({match_home_id, match_away_id} == {perspective_home_team_id, perspective_away_team_id}):
            continue 
            
        features['h2h_matches_played'] += 1
        
        outcome_for_perspective_home = get_match_outcome(match, perspective_home_team_id)

        if outcome_for_perspective_home == 'WIN':
            features['h2h_home_wins'] += 1
        elif outcome_for_perspective_home == 'DRAW':
            features['h2h_draws'] += 1
        elif outcome_for_perspective_home == 'LOSS':
            features['h2h_away_wins'] += 1

        home_goals_match = score_data.get('home')
        away_goals_match = score_data.get('away')

        if home_goals_match is not None and away_goals_match is not None:
            if match_home_id == perspective_home_team_id:
                features['h2h_home_goals_sum'] += home_goals_match
                features['h2h_away_goals_sum'] += away_goals_match
            elif match_away_id == perspective_home_team_id:
                features['h2h_home_goals_sum'] += away_goals_match
                features['h2h_away_goals_sum'] += home_goals_match
        
        if features['h2h_matches_played'] > 0:
            features['h2h_avg_goals_scored_perspective'] = features['h2h_home_goals_sum'] / features['h2h_matches_played']
            features['h2h_avg_goals_conceded_perspective'] = features['h2h_away_goals_sum'] / features['h2h_matches_played']
            features['h2h_win_rate_perspective'] = features['h2h_home_wins'] / features['h2h_matches_played']
            features['h2h_draw_rate_perspective'] = features['h2h_draws'] / features['h2h_matches_played']
            features['h2h_loss_rate_perspective'] = features['h2h_away_wins'] / features['h2h_matches_played']

    return features

def calculate_form_features(team_matches: list, team_id: int, num_games: int, team_league_code: str, api_client_ref, venue_filter: str | None = None):
    """
    Calculates form features for a team from their last N games, incorporating opponent strength.
    More recent matches within the window have higher weight.
    Can be filtered by venue ('HOME', 'AWAY', or None for overall).
    """
    prefix = ""
    if venue_filter == "HOME":
        prefix = "home_venue_"
    elif venue_filter == "AWAY":
        prefix = "away_venue_"

    features = {
        f'{prefix}form_wins_last_{num_games}': 0.0,
        f'{prefix}form_draws_last_{num_games}': 0.0,
        f'{prefix}form_losses_last_{num_games}': 0.0,
        f'{prefix}form_goals_scored_last_{num_games}': 0.0,
        f'{prefix}form_goals_conceded_last_{num_games}': 0.0,
        f'{prefix}form_goal_diff_last_{num_games}': 0.0,
        f'{prefix}form_matches_considered': 0,
        f'{prefix}form_avg_goals_scored_last_{num_games}': 0.0,
        f'{prefix}form_avg_goals_conceded_last_{num_games}': 0.0,
        f'{prefix}form_win_rate_last_{num_games}': 0.0,
        f'{prefix}form_draw_rate_last_{num_games}': 0.0,
        f'{prefix}form_loss_rate_last_{num_games}': 0.0,
        f'{prefix}form_weighted_performance_points_last_{num_games}': 0.0,
        f'{prefix}form_avg_opponent_strength_score_last_{num_games}': 0.0,
    }
    
    weighted_performance_points_sum = 0.0
    total_opponent_strength_score_weighted = 0.0
    default_opponent_strength_score = 0.05 

    filtered_by_venue_matches = []
    if venue_filter == "HOME":
        filtered_by_venue_matches = [m for m in team_matches if m.get('homeTeam', {}).get('id') == team_id]
    elif venue_filter == "AWAY":
        filtered_by_venue_matches = [m for m in team_matches if m.get('awayTeam', {}).get('id') == team_id]
    else:
        filtered_by_venue_matches = team_matches

    valid_matches = [
        m for m in filtered_by_venue_matches 
        if m.get('utcDate') and \
           m.get('homeTeam') and m.get('awayTeam') and \
           m.get('score', {}).get('fullTime')
    ]
    try:
        # Sort by date, descending (most recent first)
        sorted_matches = sorted(valid_matches, key=lambda x: x['utcDate'], reverse=True)
    except (TypeError, KeyError) as e:
        print(f"Error sorting matches for team {team_id} (form calculation): {e}")
        return features

    recent_n_games = sorted_matches[:num_games]
    features[f'{prefix}form_matches_considered'] = len(recent_n_games)

    if not recent_n_games:
        return features

    total_weight_applied = 0.0
    for i, match in enumerate(recent_n_games):
        # Gives higher weight for more recent matches
        match_weight = float(num_games - i)
        total_weight_applied += match_weight

        outcome = get_match_outcome(match, team_id)
        score_full_time = match.get('score', {}).get('fullTime', {})
        home_team_data = match.get('homeTeam', {})
        away_team_data = match.get('awayTeam', {})

        if outcome == 'WIN':
            features[f'{prefix}form_wins_last_{num_games}'] += match_weight
        elif outcome == 'DRAW':
            features[f'{prefix}form_draws_last_{num_games}'] += match_weight
        elif outcome == 'LOSS':
            features[f'{prefix}form_losses_last_{num_games}'] += match_weight

        match_home_goals = score_full_time.get('home')
        match_away_goals = score_full_time.get('away')

        if match_home_goals is not None and match_away_goals is not None:
            if home_team_data.get('id') == team_id:
                features[f'{prefix}form_goals_scored_last_{num_games}'] += match_home_goals * match_weight
                features[f'{prefix}form_goals_conceded_last_{num_games}'] += match_away_goals * match_weight
            elif away_team_data.get('id') == team_id:
                features[f'{prefix}form_goals_scored_last_{num_games}'] += match_away_goals * match_weight
                features[f'{prefix}form_goals_conceded_last_{num_games}'] += match_home_goals * match_weight
    
        opponent_id = None
        if home_team_data.get('id') == team_id:
            opponent_id = away_team_data.get('id')
        else:
            opponent_id = home_team_data.get('id')

        opponent_strength_score = default_opponent_strength_score
        if opponent_id and team_league_code and match.get('utcDate'):
            try:
                league_standings_snapshot = api_client_ref.get_league_standings(team_league_code)

                if league_standings_snapshot:
                    opponent_rank = -1
                    for standing_entry in league_standings_snapshot:
                        if standing_entry.get('team_id') == opponent_id:
                            opponent_rank = standing_entry.get('position', -1)
                            break
                    
                    if opponent_rank > 0:
                        opponent_strength_score = 1.0 / opponent_rank 
            except Exception as e:
                print(f"Error processing opponent strength for match {match.get('id')}, opponent {opponent_id}: {e}")
        
        match_points = 0
        if outcome == 'WIN':
            match_points = 3
        elif outcome == 'DRAW':
            match_points = 1
        
        current_match_weighted_performance = match_points * opponent_strength_score * match_weight
        weighted_performance_points_sum += current_match_weighted_performance
        total_opponent_strength_score_weighted += opponent_strength_score * match_weight

    features[f'{prefix}form_goal_diff_last_{num_games}'] = features[f'{prefix}form_goals_scored_last_{num_games}'] - features[f'{prefix}form_goals_conceded_last_{num_games}']
    features[f'{prefix}form_weighted_performance_points_last_{num_games}'] = weighted_performance_points_sum

    if total_weight_applied > 0:
        features[f'{prefix}form_avg_goals_scored_last_{num_games}'] = features[f'{prefix}form_goals_scored_last_{num_games}'] / total_weight_applied
        features[f'{prefix}form_avg_goals_conceded_last_{num_games}'] = features[f'{prefix}form_goals_conceded_last_{num_games}'] / total_weight_applied
        features[f'{prefix}form_win_rate_last_{num_games}'] = features[f'{prefix}form_wins_last_{num_games}'] / total_weight_applied
        features[f'{prefix}form_draw_rate_last_{num_games}'] = features[f'{prefix}form_draws_last_{num_games}'] / total_weight_applied
        features[f'{prefix}form_loss_rate_last_{num_games}'] = features[f'{prefix}form_losses_last_{num_games}'] / total_weight_applied
        features[f'{prefix}form_avg_opponent_strength_score_last_{num_games}'] = total_opponent_strength_score_weighted / total_weight_applied
    else:
        features[f'{prefix}form_avg_opponent_strength_score_last_{num_games}'] = default_opponent_strength_score
    
    return features

def get_most_recent_h2h_match_id(matches_pool: list, team1_id: int, team2_id: int) -> int | None:
    """
    Finds the ID of the most recent head-to-head match between two teams from a given pool of matches.
    Returns None if no such match is found.
    """
    h2h_matches_found = []
    for match in matches_pool:
        home_team_data = match.get('homeTeam')
        away_team_data = match.get('awayTeam')
        match_id = match.get('id')

        # Ensure essential data fields are present
        if not (home_team_data and away_team_data and match_id is not None):
            continue

        match_home_id = home_team_data.get('id')
        match_away_id = away_team_data.get('id')

        if match_home_id is None or match_away_id is None:
            continue

        # Check if the current match is between the two specified teams
        is_h2h = (match_home_id == team1_id and match_away_id == team2_id) or \
                   (match_home_id == team2_id and match_away_id == team1_id)

        if is_h2h:
            h2h_matches_found.append(match_id)

    if not h2h_matches_found:
        return None

    return h2h_matches_found[len(h2h_matches_found)-1]

def generate_score_grid_probabilities(model_classes, probabilities_array):
    '''
    Generates a 6x6 grid of score probabilities.
    '''
    if not isinstance(model_classes, list) or not isinstance(probabilities_array, np.ndarray):
        print("Warning: Invalid input types for generate_score_grid_probabilities.")
        return None
    if len(model_classes) != len(probabilities_array):
        print("Warning: model_classes and probabilities_array length mismatch.")
        return None

    score_grid = np.zeros((6, 6))

    for i, class_label in enumerate(model_classes):
        try:
            if not isinstance(class_label, str) or '-' not in class_label:
                continue

            parts = class_label.split('-')
            home_score_str, away_score_str = parts[0], parts[1]
            
            home_score = int(home_score_str)
            away_score = int(away_score_str)

            if 0 <= home_score <= 5 and 0 <= away_score <= 5:
                score_grid[home_score][away_score] = probabilities_array[i]

        except ValueError:
            continue
        except Exception as e:
            continue
            
    return score_grid

def create_dataset_from_matches(historical_h2h_matches: list,
                                home_team_all_matches_pool: list, 
                                away_team_all_matches_pool: list, 
                                num_form_games: int,
                                home_team_league_code_for_h2h_context: str,
                                away_team_league_code_for_h2h_context: str,
                                api_client_ref):
    """
    Creates a dataset from historical H2H matches. For each match, features are calculated
    based on data *prior* to that match. Perspective is always actual home team of that historical match.
    Uses new match data structure.
    Form features are derived from the provided broader match pools for each team.
    """
    dataset = []
    
    # Ensure matches have minimal required data and sort them chronologically
    valid_matches = [
        m for m in historical_h2h_matches 
        if m.get('utcDate') and \
           m.get('homeTeam', {}).get('id') is not None and \
           m.get('awayTeam', {}).get('id') is not None and \
           m.get('score', {}).get('fullTime', {}).get('home') is not None and \
           m.get('score', {}).get('fullTime', {}).get('away') is not None and \
           m.get('id') is not None
    ]
    try:
        sorted_historical_matches = sorted(valid_matches, key=lambda x: x['utcDate'])
    except (TypeError, KeyError) as e:
        print(f"Error sorting historical H2H matches for dataset creation: {e}")
        return pd.DataFrame()

    for i, current_match in enumerate(sorted_historical_matches):
        current_match_date_str = current_match['utcDate']
        current_match_id = current_match['id']
        
        actual_home_id = current_match['homeTeam']['id']
        actual_away_id = current_match['awayTeam']['id']
            
        score_full_time = current_match.get('score', {}).get('fullTime', {})
        home_goals = score_full_time.get('home')
        away_goals = score_full_time.get('away')

        if home_goals is None or away_goals is None:
            print(f"Skipping match {current_match_id} due to missing goals data (should have been filtered).")
            continue

        home_goals_capped = min(home_goals, 5)
        away_goals_capped = min(away_goals, 5)
        target_scoreline = f"{home_goals_capped}-{away_goals_capped}"

        # Features are calculated based on matches *before* the current one
        matches_before_current = sorted_historical_matches[:i]
        
        h2h_features = calculate_h2h_features(matches_before_current, actual_home_id, actual_away_id)
        
        # Form features: Need all matches of each team *before* current_match_date_str
        # Use the provided broader pools for form calculation.
        
        home_team_matches_for_form_calc = [
            m for m in home_team_all_matches_pool
            if m['utcDate'] < current_match_date_str and 
               (m.get('homeTeam', {}).get('id') == actual_home_id or m.get('awayTeam', {}).get('id') == actual_home_id)
        ]
        away_team_matches_for_form_calc = [
            m for m in away_team_all_matches_pool
            if m['utcDate'] < current_match_date_str and
               (m.get('homeTeam', {}).get('id') == actual_away_id or m.get('awayTeam', {}).get('id') == actual_away_id)
        ]

        current_home_team_league_code = home_team_league_code_for_h2h_context
        current_away_team_league_code = away_team_league_code_for_h2h_context
        
        home_form_overall = calculate_form_features(home_team_matches_for_form_calc, actual_home_id, num_form_games, current_home_team_league_code, api_client_ref, venue_filter=None)
        home_form_home_venue = calculate_form_features(home_team_matches_for_form_calc, actual_home_id, num_form_games, current_home_team_league_code, api_client_ref, venue_filter="HOME")
        home_form_away_venue = calculate_form_features(home_team_matches_for_form_calc, actual_home_id, num_form_games, current_home_team_league_code, api_client_ref, venue_filter="AWAY")

        away_form_overall = calculate_form_features(away_team_matches_for_form_calc, actual_away_id, num_form_games, current_away_team_league_code, api_client_ref, venue_filter=None)
        away_form_home_venue = calculate_form_features(away_team_matches_for_form_calc, actual_away_id, num_form_games, current_away_team_league_code, api_client_ref, venue_filter="HOME")
        away_form_away_venue = calculate_form_features(away_team_matches_for_form_calc, actual_away_id, num_form_games, current_away_team_league_code, api_client_ref, venue_filter="AWAY")

        row = {'match_id': current_match_id, 'date': current_match_date_str,
               'home_team_id_h2h_match': actual_home_id, 'away_team_id_h2h_match': actual_away_id}
        row.update(h2h_features)

        # Overall form
        for key, val in home_form_overall.items(): row[f'current_home_{key}'] = val
        for key, val in away_form_overall.items(): row[f'current_away_{key}'] = val
        # Home team at home venue
        for key, val in home_form_home_venue.items(): row[f'current_home_home_venue_{key}'] = val
        # Home team at away venue
        for key, val in home_form_away_venue.items(): row[f'current_home_away_venue_{key}'] = val
        # Away team at home venue
        for key, val in away_form_home_venue.items(): row[f'current_away_home_venue_{key}'] = val
        # Away team at away venue
        for key, val in away_form_away_venue.items(): row[f'current_away_away_venue_{key}'] = val
        
        row['target_scoreline'] = target_scoreline
        dataset.append(row)
        
    return pd.DataFrame(dataset)

def get_user_input():
    """
    Gets team names, leagues, and API key from the user.
    Allows league input by code or full name.
    """    
    valid_home_league = False
    home_team_league_code = ""
    while not valid_home_league:
        print("\nAvailable leagues (enter code or full name):")
        for code, name in ALLOWED_LEAGUES.items():
            print(f"  {code}: {name}")
        home_team_league_input = input(f"Enter the home team's league (e.g., PL or Premier League): ").strip()
        
        if home_team_league_input.upper() in ALLOWED_LEAGUES:
            home_team_league_code = home_team_league_input.upper()
            print(f"Selected home team league: {ALLOWED_LEAGUES[home_team_league_code]}")
            valid_home_league = True
        else:
            found_by_name = False
            for code, name in ALLOWED_LEAGUES.items():
                if home_team_league_input.lower() == name.lower():
                    home_team_league_code = code
                    print(f"Selected home team league: {name}")
                    valid_home_league = True
                    found_by_name = True
                    break
            if not found_by_name:
                print(f"Invalid league '{home_team_league_input}'. Please choose from the list above by code or full name.")

    home_team_list = api_client.get_teams_by_league(home_team_league_code)
    valid_home_team_name = False
    while not valid_home_team_name:
        print("\nAvailable home teams:")
        for team in home_team_list:
            print(f"  {team['name']}: {team['shortName']}: {team['tla']}")
    
        home_team_name = None
        while not home_team_name:
            home_team_name = input("Enter the home team name (full name, shortname, or tla): ").strip()
            
        found_team = False
        for team in home_team_list:
            if (home_team_name.lower() == team['name'].lower() or 
                home_team_name.lower() == team['shortName'].lower() or 
                home_team_name.lower() == team['tla'].lower()):

                home_team_name = team['name']
                home_team_id = team['id']
                print(f"Selected home team: {home_team_name}")
                valid_home_team_name = True
                found_team = True
                break
        if not found_team:
            print(f"Invalid home team '{home_team_name}'. Please choose from the list above by full name, shortname, or tla.")

    valid_away_league = False
    away_team_league_code = ""
    while not valid_away_league:
        print("\nAvailable leagues (enter code or full name):")
        for code, name in ALLOWED_LEAGUES.items():
            print(f"  {code}: {name}")
        away_team_league_input = input(f"Enter the away team's league (e.g., PL or Premier League): ").strip()

        if away_team_league_input.upper() in ALLOWED_LEAGUES:
            away_team_league_code = away_team_league_input.upper()
            print(f"Selected away team league: {ALLOWED_LEAGUES[away_team_league_code]}")
            valid_away_league = True
        else:
            found_by_name = False
            for code, name in ALLOWED_LEAGUES.items():
                if away_team_league_input.lower() == name.lower():
                    away_team_league_code = code
                    print(f"Selected away team league: {name}")
                    valid_away_league = True
                    found_by_name = True
                    break
            if not found_by_name:
                print(f"Invalid league '{away_team_league_input}'. Please choose from the list above by code or full name.")
    
    away_team_list = api_client.get_teams_by_league(away_team_league_code)
    valid_away_team_name = False
    while not valid_away_team_name:
        print("\nAvailable away teams:")
        for team in away_team_list:
            print(f"  {team['name']}: {team['shortName']}: {team['tla']}")
    
        away_team_name = None
        while not away_team_name:
            away_team_name = input("Enter the away team name (full name, shortname, or tla): ").strip()
            
        found_team = False
        for team in away_team_list:
            if (away_team_name.lower() == team['name'].lower() or 
                away_team_name.lower() == team['shortName'].lower() or 
                away_team_name.lower() == team['tla'].lower()):

                away_team_name = team['name']
                away_team_id = team['id']
                print(f"Selected away team: {away_team_name}")
                valid_away_team_name = True
                found_team = True
                break
        if not found_team:
            print(f"Invalid away team '{away_team_name}'. Please choose from the list above by full name, shortname, or tla.")
    
    return home_team_name, home_team_id, home_team_league_code, away_team_name, away_team_id, away_team_league_code

def main():
    print("--- Soccer Match Predictor (using football-data.org) ---")

    api_key = os.getenv("API_KEY")
    if not api_key:
        print("\nYour football-data.org API key is not set as an environment variable (API_KEY).")
        api_key = input("Please enter your football-data.org API key: ")
    else:
        print("\nUsing football-data.org API key from environment variable.")

    api_client.set_api_key(api_key)

    home_team_name, home_team_id, home_team_league_code, away_team_name, away_team_id, away_team_league_code = get_user_input()

    if not all([home_team_name, home_team_id, home_team_league_code, away_team_name, away_team_id, away_team_league_code]):
        print("Error: Home team/league and away team/league must be provided.")
        return

    print(f"\nFetching ID for home team (user specified): {home_team_name}...")
    print(f"Found ID for {home_team_name}: {home_team_id}")

    print(f"\nFetching ID for away team (user specified): {away_team_name}...")
    print(f"Found ID for {away_team_name}: {away_team_id}")
    
    print("\n--- Fetching Historical & Recent Match Data ---")
    num_form_games = 10
    num_years_history = 2

    date_to = datetime.now()
    date_from_history = date_to - timedelta(days=num_years_history*365)

    date_to_str = date_to.strftime('%Y-%m-%d')
    date_from_history_str = date_from_history.strftime('%Y-%m-%d')

    print(f"\nFetching recent matches for {home_team_name} (pool for current form)...")
    home_team_recent_matches_pool = api_client.get_matches_for_team(home_team_id, date_from_history_str, date_to_str)
    print(f"Found {len(home_team_recent_matches_pool)} matches in pool for {home_team_name}.")

    print(f"\nFetching recent matches for {away_team_name} (pool for current form)...")
    away_team_recent_matches_pool = api_client.get_matches_for_team(away_team_id, date_from_history_str, date_to_str)
    print(f"Found {len(away_team_recent_matches_pool)} matches in pool for {away_team_name}.")

    most_recent_h2h_match_id = get_most_recent_h2h_match_id(home_team_recent_matches_pool, home_team_id, away_team_id)
    if most_recent_h2h_match_id is not None:
        print(f"Fetching H2H matches between {home_team_name} and {away_team_name}...")
        historical_h2h_matches = api_client.get_head_to_head_matches(most_recent_h2h_match_id)
        if not historical_h2h_matches:
            print(f"Warning: No direct H2H matches found for {home_team_name} vs {away_team_name}. Model training might fail or be unreliable.")
        else:
            print(f"Found {len(historical_h2h_matches)} H2H matches for stats and dataset construction.")
    else:
        print(f"No recent H2H matches found between {home_team_name} and {away_team_name}. Cannot train model without H2H data. Exiting.")
        return

    print("\n--- Feature Engineering ---")
    historical_df = pd.DataFrame()
    if historical_h2h_matches:
        historical_df = create_dataset_from_matches(
            historical_h2h_matches,
            home_team_recent_matches_pool,
            away_team_recent_matches_pool,
            num_form_games=num_form_games,
            home_team_league_code_for_h2h_context=home_team_league_code,
            away_team_league_code_for_h2h_context=away_team_league_code,
            api_client_ref=api_client
        )
    if historical_df.empty:
        print("Could not create a training dataset (historical_df is empty). Model cannot be trained. Exiting.")
        return
        
    print(f"\nCreated historical dataset with {len(historical_df)} samples.")
    if len(historical_df) < 2:
        print("Historical dataset too small to train any model (less than 2 samples). Exiting.")
        return
    if len(historical_df) < 10:
        print("Warning: The historical dataset is very small. Model performance is likely to be poor.")

    print("\n--- Model Training ---")
    feature_columns = [col for col in historical_df.columns if col not in ['match_id', 'date', 'home_team_id_h2h_match', 'away_team_id_h2h_match', 'target_scoreline']]
    X = historical_df[feature_columns]
    y = historical_df['target_scoreline']
    X = X.fillna(0)

    if X.empty:
        print("Feature set X is empty after attempting to fill NaNs. Cannot train. Exiting.")
        return

    X_train = X
    y_train = y
    if X_train.empty:
        print("X_train is empty. Cannot train model. This might happen if dataset is too small. Exiting.")
        return

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample', 
                                   min_samples_leaf=max(1, int(len(X_train)*0.05)) if len(X_train) >= 20 else 1)
    try:
        model.fit(X_train, y_train)
        print("Model trained successfully.")
        if hasattr(model, 'feature_importances_') and not X_train.empty:
            importances = model.feature_importances_
            feature_names = X_train.columns
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
            print("\n--- Feature Importances (Top 15) ---")
            print(feature_importance_df.head(15))
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    print("\n--- Prediction for Upcoming Match ---")
    upcoming_h2h_features = calculate_h2h_features(historical_h2h_matches, home_team_id, away_team_id) 
    
    # Form features for the upcoming match
    # Overall form
    upcoming_home_form_overall = calculate_form_features(home_team_recent_matches_pool, home_team_id, num_form_games, home_team_league_code, api_client, venue_filter=None)
    upcoming_away_form_overall = calculate_form_features(away_team_recent_matches_pool, away_team_id, num_form_games, away_team_league_code, api_client, venue_filter=None)
    
    # Home venue specific form
    upcoming_home_form_home_venue = calculate_form_features(home_team_recent_matches_pool, home_team_id, num_form_games, home_team_league_code, api_client, venue_filter="HOME")
    upcoming_away_form_home_venue = calculate_form_features(away_team_recent_matches_pool, away_team_id, num_form_games, away_team_league_code, api_client, venue_filter="HOME")
    
    # Away venue specific form
    upcoming_home_form_away_venue = calculate_form_features(home_team_recent_matches_pool, home_team_id, num_form_games, home_team_league_code, api_client, venue_filter="AWAY")
    upcoming_away_form_away_venue = calculate_form_features(away_team_recent_matches_pool, away_team_id, num_form_games, away_team_league_code, api_client, venue_filter="AWAY")

    upcoming_match_features_dict = {}
    upcoming_match_features_dict.update(upcoming_h2h_features)
    
    # Add overall form features
    for key, val in upcoming_home_form_overall.items(): upcoming_match_features_dict[f'current_home_{key}'] = val
    for key, val in upcoming_away_form_overall.items(): upcoming_match_features_dict[f'current_away_{key}'] = val
    
    # Add home team's venue form features
    for key, val in upcoming_home_form_home_venue.items(): upcoming_match_features_dict[f'current_home_home_venue_{key}'] = val
    for key, val in upcoming_home_form_away_venue.items(): upcoming_match_features_dict[f'current_home_away_venue_{key}'] = val
    
    # Add away team's venue form features
    for key, val in upcoming_away_form_home_venue.items(): upcoming_match_features_dict[f'current_away_home_venue_{key}'] = val
    for key, val in upcoming_away_form_away_venue.items(): upcoming_match_features_dict[f'current_away_away_venue_{key}'] = val

    upcoming_features_df = pd.DataFrame([upcoming_match_features_dict])
    for col in X_train.columns:
        if col not in upcoming_features_df.columns:
            upcoming_features_df[col] = 0 
    upcoming_features_df = upcoming_features_df[X_train.columns] 
    upcoming_features_df = upcoming_features_df.fillna(0)

    try:
        probabilities = model.predict_proba(upcoming_features_df)
        print("\n--- Predicted Scoreline Probabilities (Top 5) ---")
        if hasattr(model, 'classes_') and probabilities.shape[1] == len(model.classes_):
            # Create a list of (scoreline, probability) tuples
            score_probabilities = []
            for i, class_label in enumerate(model.classes_):
                score_probabilities.append((class_label, probabilities[0][i]))
            
            sorted_score_probabilities = sorted(score_probabilities, key=lambda item: item[1], reverse=True)
            
            # Print the top 5 predicted scorelines with probabilities
            for i in range(min(5, len(sorted_score_probabilities))):
                label, prob = sorted_score_probabilities[i]
                print(f"  Score {label}: {prob:.2%}")
        else:
            print("Could not display score probabilities (model.classes_ or probabilities mismatch).")

        score_grid = None
        if hasattr(model, 'classes_') and probabilities.shape[1] == len(model.classes_):
            score_grid = generate_score_grid_probabilities(list(model.classes_), probabilities[0])

        if score_grid is not None:
            print("\n--- Predicted Score Grid (Home on Left, Away on Top) ---")
            header = "Away->|  0  |  1  |  2  |  3  |  4  |  5  |"
            print(header)
            print("-" * len(header))
            for i in range(6):
                row_str = f"Home {i} |"
                for j in range(6):
                    row_str += f" {score_grid[i][j]:.2%} |"
                print(row_str)
                print("-" * len(header))
            
            # Calculate and display overall Win/Draw/Loss from the grid
            home_win_prob_grid = 0
            draw_prob_grid = 0
            away_win_prob_grid = 0
            
            for r in range(6):
                for c in range(6):
                    if r > c:
                        home_win_prob_grid += score_grid[r][c]
                    elif r == c:
                        draw_prob_grid += score_grid[r][c]
                    else:
                        away_win_prob_grid += score_grid[r][c]
            
            print("\n--- Overall Outcome Probabilities (derived from score grid) ---")
            print(f"  {home_team_name} (Home) Win: {home_win_prob_grid:.2%}")
            print(f"  Draw: {draw_prob_grid:.2%}")
            print(f"  {away_team_name} (Away) Win: {away_win_prob_grid:.2%}")

        else:
            print("\nScore grid could not be generated.")
            
    except Exception as e:
        print(f"Error during prediction or evaluation: {e}")

if __name__ == "__main__":
    main()