import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_FOOTBALL_KEY')
API_HOST = "v3.football.api-sports.io"
LEAGUE_ID = 39  # Premier League
SEASON = 2023   # 2023 Season

def fetch_teams_from_api(api_key_env, api_host_env, league_id_env, season_env):
    """Fetches all teams for a given league and season."""
    if not api_key_env:
        print("Error: API_FOOTBALL_KEY not found for fetching teams.")
        return None
    
    headers = {
        'x-rapidapi-key': api_key_env,
        'x-rapidapi-host': api_host_env
    }
    url = f"https://{api_host_env}/teams"
    querystring = {"league": str(league_id_env), "season": str(season_env)}
    
    print("\nFetching teams from API-Football...")
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=15)
        response.raise_for_status()
        teams_api_data = response.json()
        
        teams_list = []
        if 'response' in teams_api_data and teams_api_data['response']:
            for team_entry in teams_api_data['response']:
                team_info = team_entry.get('team')
                if team_info and 'id' in team_info and 'name' in team_info:
                    teams_list.append({'id': team_info['id'], 'name': team_info['name']})
            print(f"Fetched {len(teams_list)} teams.")
            return teams_list
        else:
            print("No teams found in API response.")
            return None
    except requests.RequestException as e:
        print(f"Error fetching teams from API: {e}")
        return None

def get_user_team_selection(teams_list):
    """Allows user to select two teams from the provided list."""
    if not teams_list:
        print("No teams available for selection.")
        return None, None

    print("\nPlease select two teams for the matchup:")
    for i, team in enumerate(teams_list):
        print(f"{i + 1}. {team['name']}")
    
    while True:
        try:
            home_team_idx = int(input(f"Enter number for Home Team (1-{len(teams_list)}): ")) - 1
            if 0 <= home_team_idx < len(teams_list):
                break
            else:
                print("Invalid selection. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            away_team_idx = int(input(f"Enter number for Away Team (1-{len(teams_list)}): ")) - 1
            if 0 <= away_team_idx < len(teams_list) and away_team_idx != home_team_idx:
                break
            elif away_team_idx == home_team_idx:
                print("Away team cannot be the same as the home team. Please select a different team.")
            else:
                print("Invalid selection. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    home_team = teams_list[home_team_idx]
    away_team = teams_list[away_team_idx]
    print(f"\nSelected matchup: {home_team['name']} (Home) vs {away_team['name']} (Away)")
    return home_team, away_team

def predict_matchup_outcome(trained_model, home_team_name, away_team_name, feature_columns):
    """
    Predicts outcome probabilities for a given matchup.
    Uses placeholder stats for avg_goals and win_rate, similar to training data generation.
    """
    print(f"\nPredicting outcome for: {home_team_name} vs {away_team_name}")
    
    # Create a feature vector for the matchup
    # IMPORTANT: Using random placeholders for stats features, as per current data generation.
    # For a real prediction, these should be actual stats for the selected teams.
    match_features = {
        'home_team_goals_avg': np.random.rand() * 2,
        'away_team_goals_avg': np.random.rand() * 2,
        'home_team_win_rate': np.random.rand(),
        'away_team_win_rate': np.random.rand(),
        'home_advantage': 1  # Home team always has home advantage in this custom prediction
    }
    
    # Ensure the order of columns matches the training data
    match_df = pd.DataFrame([match_features], columns=feature_columns)
    
    probabilities = trained_model.predict_proba(match_df)[0] # Get probabilities for the first (and only) sample
    
    # Assuming 0: Home Win, 1: Draw, 2: Away Win from your encoding
    outcome_labels = ['Home Win', 'Draw', 'Away Win']
    
    print("\nPredicted Probabilities:")
    for label, prob in zip(outcome_labels, probabilities):
        print(f"{label} ({home_team_name if label == 'Home Win' else away_team_name if label == 'Away Win' else 'Draw'}): {prob*100:.2f}%")
    
    return probabilities


def fetch_data_from_api():
    """
    Fetches match data from api-football and preprocesses it.
    """
    if not API_KEY:
        print("Error: API_FOOTBALL_KEY not found in environment variables.")
        print("Please ensure it is set in your .env file and load_dotenv() is called.")
        # Fallback to random data if API key is missing
        return pd.DataFrame({
            'home_team_goals_avg': np.random.rand(100) * 2,
            'away_team_goals_avg': np.random.rand(100) * 2,
            'home_team_win_rate': np.random.rand(100),
            'away_team_win_rate': np.random.rand(100),
            'home_advantage': np.random.randint(0, 2, 100),
            'result': np.random.choice(['Home Win', 'Draw', 'Away Win'], 100)
        })

    headers = {
        'x-rapidapi-key': API_KEY,
        'x-rapidapi-host': API_HOST
    }
    url = f"https://{API_HOST}/fixtures"
    querystring = {"league": str(LEAGUE_ID), "season": str(SEASON)}

    print("Fetching data from API-Football...")
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=30) # Added timeout
        response.raise_for_status()
        api_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        # Fallback to random data on API error
        return pd.DataFrame({
            'home_team_goals_avg': np.random.rand(100) * 2,
            'away_team_goals_avg': np.random.rand(100) * 2,
            'home_team_win_rate': np.random.rand(100),
            'away_team_win_rate': np.random.rand(100),
            'home_advantage': np.random.randint(0, 2, 100),
            'result': np.random.choice(['Home Win', 'Draw', 'Away Win'], 100)
        })

    print(f"Fetched {len(api_data.get('response', []))} fixtures.")

    processed_data = []
    if 'response' in api_data and api_data['response']:
        for fixture_item in api_data['response']:
            if not fixture_item or 'teams' not in fixture_item or 'goals' not in fixture_item:
                print(f"Skipping fixture due to missing critical data: {fixture_item.get('fixture', {}).get('id')}")
                continue

            fixture_details = fixture_item.get('fixture', {})
            teams_data = fixture_item.get('teams', {}) # Renamed to avoid conflict
            goals = fixture_item.get('goals', {})

            if fixture_details.get('status', {}).get('short') == 'FT' and \
               goals.get('home') is not None and goals.get('away') is not None:

                home_goals = goals['home']
                away_goals = goals['away']
                result = ''
                if home_goals > away_goals:
                    result = 'Home Win'
                elif home_goals < away_goals:
                    result = 'Away Win'
                else:
                    result = 'Draw'

                home_team_winner = teams_data.get('home', {}).get('winner')
                home_advantage = 0
                if home_team_winner is True:
                    home_advantage = 1
                elif home_team_winner is False and teams_data.get('away', {}).get('winner') is not True :
                    home_advantage = 0
                
                processed_data.append({
                    'home_team_goals_avg': np.random.rand() * 2,
                    'away_team_goals_avg': np.random.rand() * 2,
                    'home_team_win_rate': np.random.rand(),
                    'away_team_win_rate': np.random.rand(),
                    'home_advantage': home_advantage,
                    'result': result
                })
            else:
                print(f"Skipping fixture {fixture_details.get('id')}: Not finished or goals missing.")
    
    if not processed_data:
        print("Warning: No data processed from API. Using dummy data for demonstration.")
        return pd.DataFrame({
            'home_team_goals_avg': np.random.rand(100) * 2,
            'away_team_goals_avg': np.random.rand(100) * 2,
            'home_team_win_rate': np.random.rand(100),
            'away_team_win_rate': np.random.rand(100),
            'home_advantage': np.random.randint(0, 2, 100),
            'result': np.random.choice(['Home Win', 'Draw', 'Away Win'], 100)
        })

    return pd.DataFrame(processed_data)

# --- Main script execution ---
if __name__ == "__main__":
    # Fetch and prepare training data
    df = fetch_data_from_api()

    if df.empty or 'result' not in df.columns or len(df) < 20:
        print("Not enough data fetched or data is malformed for training. Exiting.")
        exit()
    
    df['result_encoded'] = df['result'].map({'Home Win': 0, 'Draw': 1, 'Away Win': 2})
    df.dropna(subset=['result_encoded'], inplace=True)
    df['result_encoded'] = df['result_encoded'].astype(int)

    X = df.drop(columns=['result', 'result_encoded'])
    y = df['result_encoded']
    
    feature_cols = X.columns.tolist() # Store feature column names

    if len(X) < 5 or len(y) < 5 or len(np.unique(y)) < 2 :
        print("Not enough data to perform train-test split. Ensure API returns sufficient and varied data.")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Training Accuracy: {accuracy:.2f}")
    print("\nClassification Report (on test set):\n")
    target_names_ordered = ['Home Win', 'Draw', 'Away Win']
    print(classification_report(y_test, y_pred, target_names=target_names_ordered, zero_division=0))

    # --- New: Allow user to select teams and predict outcome ---
    print("\n--- Predict Custom Matchup ---")
    all_teams = fetch_teams_from_api(API_KEY, API_HOST, LEAGUE_ID, SEASON)
    
    if all_teams:
        home_team_selected, away_team_selected = get_user_team_selection(all_teams)
        if home_team_selected and away_team_selected:
            predict_matchup_outcome(model, home_team_selected['name'], away_team_selected['name'], feature_cols)
        else:
            print("Could not proceed with prediction due to team selection issues.")
    else:
        print("Could not fetch teams for custom prediction.")