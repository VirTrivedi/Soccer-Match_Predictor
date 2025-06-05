import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import api_client
from soccer_predictor import (
    predict_single_match_probabilities,
    ALLOWED_LEAGUES,
    train_league_model
)
from dotenv import load_dotenv

# Global Setup
load_dotenv()
NUM_FORM_GAMES = 10
NUM_YEARS_HISTORY = 2

def determine_match_outcome_from_probabilities(
    probabilities_array_single_pred: np.ndarray, 
    model_classes: list, 
    home_team_name: str, 
    away_team_name: str
) -> tuple[str, int, int, str] | None:
    """
    Determines the outcome of a match based on predicted probabilities for scorelines.
    """
    if probabilities_array_single_pred is None or not model_classes:
        print("Warning: Probabilities or model classes are missing.")
        return None
    
    if probabilities_array_single_pred.ndim != 1 or len(probabilities_array_single_pred) != len(model_classes):
        print(f"Warning: Probabilities array shape mismatch. Expected 1D array of length {len(model_classes)}, got shape {probabilities_array_single_pred.shape}")
        return None

    try:
        # Select scoreline based on probability distribution
        chosen_score_str = np.random.choice(model_classes, p=probabilities_array_single_pred)

        print(f"  Selected scoreline for {home_team_name} vs {away_team_name}: {chosen_score_str} (Probabilistically chosen)")

        score_parts = chosen_score_str.split('-')
        home_goals = int(score_parts[0])
        away_goals = int(score_parts[1])
        
        # Ensure goals are within a typical range
        home_goals = min(home_goals, 5) 
        away_goals = min(away_goals, 5)


        if home_goals > away_goals:
            outcome_str = 'HOME_WIN'
        elif away_goals > home_goals:
            outcome_str = 'AWAY_WIN'
        else:
            outcome_str = 'DRAW'
            
        return outcome_str, home_goals, away_goals, chosen_score_str

    except (ValueError, IndexError) as e:
        print(f"Error parsing scoreline '{chosen_score_str if 'chosen_score_str' in locals() else 'unknown score'}' or processing probabilities: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error in determine_match_outcome_from_probabilities: {e}")
        return None

def get_season_prediction_input() -> tuple[str, str] | None:
    """
    Gets and validates user input for league code and CSV file path for season fixtures.
    Allows user to quit by entering 'q'.
    """
    # League Input
    league_code_upper = None
    while True:
        print("\nAvailable leagues for season prediction:")
        for code, name in ALLOWED_LEAGUES.items():
            print(f"  {code}: {name}")
        league_code_input = input("Enter the league code (e.g., PL) or 'q' to quit: ").strip()

        if league_code_input.lower() == 'q':
            print("Exiting.")
            return None
        
        league_code_upper = league_code_input.upper()
        if league_code_upper in ALLOWED_LEAGUES:
            print(f"Selected league: {ALLOWED_LEAGUES[league_code_upper]}")
            break
        else:
            print(f"Invalid league code '{league_code_input}'. Please choose from the list or enter 'q' to quit.")

    # CSV File Path Input
    csv_file_path_str = None
    while True:
        csv_file_path_input = input("Enter the CSV file path for season fixtures (HomeTeamName,AwayTeamName columns) or 'q' to quit: ").strip()

        if csv_file_path_input.lower() == 'q':
            print("Exiting.")
            return None

        if os.path.exists(csv_file_path_input):
            print(f"Selected CSV file: {csv_file_path_input}")
            csv_file_path_str = csv_file_path_input
            break
        else:
            print(f"File not found: '{csv_file_path_input}'. Please enter a valid file path or 'q' to quit.")
            
    return league_code_upper, csv_file_path_str

def load_fixtures_from_csv_and_map_ids(csv_file_path: str, league_code: str, api_client_ref) -> list | None:
    """
    Loads fixtures from a CSV file, maps team names to IDs using the API,
    and structures them into a list of fixture dictionaries.
    """
    try:
        # Assume CSV has columns HomeTeamName, AwayTeamName, and potentially no header
        fixtures_df = pd.read_csv(csv_file_path, header=None, names=['HomeTeamName', 'AwayTeamName'])
        if fixtures_df.empty:
            print(f"Warning: CSV file '{csv_file_path}' is empty.")
            return []
    except FileNotFoundError:
        print(f"Error: Fixture CSV file not found at '{csv_file_path}'.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Fixture CSV file '{csv_file_path}' is empty or improperly formatted (no columns to parse).")
        return None
    except Exception as e:
        print(f"Error reading or parsing CSV file '{csv_file_path}': {e}")
        return None

    print(f"Successfully read {len(fixtures_df)} potential fixtures from CSV: {csv_file_path}")

    teams_in_league = api_client_ref.get_teams_by_league(league_code)
    if not teams_in_league:
        print(f"Error: Could not fetch teams for league '{league_code}' from API to map names. Cannot process CSV fixtures.")
        return None

    name_to_team_data_map = {}
    for team in teams_in_league:
        team_data = {'id': team['id'], 'canonical_name': team['name']}
        if team.get('name'):
            name_to_team_data_map[str(team['name']).strip().lower()] = team_data
        if team.get('shortName'):
            name_to_team_data_map[str(team['shortName']).strip().lower()] = team_data
        if team.get('tla'):
            name_to_team_data_map[str(team['tla']).strip().lower()] = team_data
    
    processed_fixtures = []
    unmapped_fixtures_count = 0

    for index, row in fixtures_df.iterrows():
        home_team_name_csv = str(row.get('HomeTeamName', '')).strip().lower()
        away_team_name_csv = str(row.get('AwayTeamName', '')).strip().lower()

        if not home_team_name_csv or not away_team_name_csv:
            print(f"Warning: Missing HomeTeamName or AwayTeamName in CSV row {index + 1}. Skipping.")
            unmapped_fixtures_count +=1
            continue

        home_team_data = name_to_team_data_map.get(home_team_name_csv)
        away_team_data = name_to_team_data_map.get(away_team_name_csv)

        if home_team_data and away_team_data:
            fixture = {
                'id': f"csv_fixture_{index + 1}",
                'homeTeam': {'id': home_team_data['id'], 'name': home_team_data['canonical_name']},
                'awayTeam': {'id': away_team_data['id'], 'name': away_team_data['canonical_name']},
                'utcDate': None,
                'status': 'SCHEDULED_FROM_CSV',
            }
            processed_fixtures.append(fixture)
        else:
            unmapped_fixtures_count += 1
            unmappable_home_msg = f"'{row.get('HomeTeamName', '')}'" if not home_team_data else "Mapped"
            unmappable_away_msg = f"'{row.get('AwayTeamName', '')}'" if not away_team_data else "Mapped"
            print(f"Warning: Could not map team names from CSV row {index + 1}. Home: {unmappable_home_msg}, Away: {unmappable_away_msg}. Skipping fixture.")

    if unmapped_fixtures_count > 0:
        print(f"\nSummary: Skipped {unmapped_fixtures_count} out of {len(fixtures_df)} fixtures from CSV due to unmappable team names.")
    
    if not processed_fixtures and not fixtures_df.empty:
        print("Warning: No fixtures could be processed from the CSV file after attempting to map team names.")
    elif not processed_fixtures and fixtures_df.empty:
         pass
    else:
        print(f"Successfully processed {len(processed_fixtures)} fixtures from CSV.")

    return processed_fixtures

def predict_entire_season(league_code: str, csv_file_path: str, api_key_val: str):
    """
    Predicts outcomes for all matches in a given season for a league using fixtures from a CSV file,
    and constructs a predicted league table.
    """
    api_client.set_api_key(api_key_val)
    print(f"\n--- Starting Season Prediction for {ALLOWED_LEAGUES.get(league_code, league_code)} using fixtures from: {csv_file_path} ---")

    # Load Fixtures from CSV
    processed_fixtures = load_fixtures_from_csv_and_map_ids(csv_file_path, league_code, api_client)
    if processed_fixtures is None or not processed_fixtures:
        print(f"Could not load or process fixtures from CSV '{csv_file_path}'. Exiting.")
        return
    print(f"Successfully loaded and mapped {len(processed_fixtures)} fixtures from CSV.")

    # Fetch League Teams (IDs and Names)
    print(f"\nFetching all team data for league: {league_code} for table construction...")
    teams_in_league = api_client.get_teams_by_league(league_code)
    if not teams_in_league:
        print(f"Could not fetch teams for league {league_code} for table construction. Exiting.")
        return
    team_id_to_name = {team['id']: team['name'] for team in teams_in_league}
    print(f"Found {len(teams_in_league)} teams for the league table.")

    current_date_str = datetime.now().strftime('%Y-%m-%d')
    date_until_history_for_training = current_date_str
    print(f"Model training data will be fetched up to: {date_until_history_for_training}")

    # Train League Model
    print("\n--- Training League-Wide Model ---")
    trained_model_data = train_league_model(
        league_code=league_code,
        api_client_ref=api_client,
        num_years_history=NUM_YEARS_HISTORY,
        num_form_games=NUM_FORM_GAMES,
        date_until_history=date_until_history_for_training
    )

    if trained_model_data is None:
        print("CRITICAL: Failed to train a league model. Cannot proceed with season prediction.")
        return
    
    model, X_train_columns_order = trained_model_data
    print(f"League model training complete. Model has {len(X_train_columns_order)} features.")

    # Historical Data Pool for Predictions
    general_pool_end_date_str = current_date_str
    general_pool_start_date_dt = datetime.strptime(current_date_str, '%Y-%m-%d') - timedelta(days=NUM_YEARS_HISTORY * 365)
    general_pool_start_date_str = general_pool_start_date_dt.strftime('%Y-%m-%d')
    
    all_teams_match_pool_for_prediction = {}
    print(f"\nFetching match pool for all teams for form/H2H context (from {general_pool_start_date_str} to {general_pool_end_date_str})...")
    for team_info in teams_in_league:
        team_id = team_info['id']
        team_name = team_info['name']
        team_matches_pool = api_client.get_matches_for_team(team_id, general_pool_start_date_str, general_pool_end_date_str)
        all_teams_match_pool_for_prediction[team_id] = team_matches_pool
    print("Match pool fetching complete.")

    # Initialize League Table
    league_table_data = []
    for team_id, team_name in team_id_to_name.items():
        league_table_data.append({
            'team_id': team_id, 'name': team_name, 'P': 0, 'W': 0, 'D': 0, 'L': 0, 
            'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0
        })
    league_table_df = pd.DataFrame(league_table_data).set_index('team_id')

    # Iterate Through Fixtures
    print("\n--- Predicting Season Matches from CSV ---")
    for i, match_data in enumerate(processed_fixtures):
        match_id = match_data.get('id')
        home_team_info = match_data.get('homeTeam')
        away_team_info = match_data.get('awayTeam')

        if not home_team_info or not away_team_info or home_team_info.get('id') is None or away_team_info.get('id') is None:
            print(f"Skipping match {match_id} (CSV fixture {i+1}) due to missing team data after mapping.")
            continue

        home_team_id = home_team_info['id']
        away_team_id = away_team_info['id']
        
        home_team_name = home_team_info['name'] 
        away_team_name = away_team_info['name']

        print(f"\nPredicting CSV Fixture ({i+1}/{len(processed_fixtures)}): {home_team_name} vs {away_team_name}")

        if not model or not X_train_columns_order:
            print(f"  Skipping prediction for {home_team_name} vs {away_team_name} due to no model available from training.")
            league_table_df.loc[home_team_id, 'P'] += 1
            league_table_df.loc[away_team_id, 'P'] += 1
            continue

        home_team_hist_matches_from_pool = all_teams_match_pool_for_prediction.get(home_team_id, [])
        away_team_hist_matches_from_pool = all_teams_match_pool_for_prediction.get(away_team_id, [])
        
        temp_h2h_pool = []
        for m in home_team_hist_matches_from_pool:
            if m.get('homeTeam',{}).get('id') == away_team_id or \
               m.get('awayTeam',{}).get('id') == away_team_id:
                temp_h2h_pool.append(m)
        for m in away_team_hist_matches_from_pool:
            if m.get('homeTeam',{}).get('id') == home_team_id or \
               m.get('awayTeam',{}).get('id') == home_team_id:
                if m.get('id') not in [x.get('id') for x in temp_h2h_pool]:
                    temp_h2h_pool.append(m)
        try:
            fixture_h2h_context_matches = sorted(temp_h2h_pool, key=lambda x: x.get('utcDate', ''))
        except (TypeError, KeyError) as e:
            print(f"  Warning: Could not sort H2H matches for {home_team_name} vs {away_team_name} from pool due to data issues: {e}")
            fixture_h2h_context_matches = temp_h2h_pool

        home_form_pool = home_team_hist_matches_from_pool
        away_form_pool = away_team_hist_matches_from_pool

        # Call the prediction function from soccer_predictor
        prediction_output = predict_single_match_probabilities(
            model=model,
            X_train_columns_order=X_train_columns_order,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_team_league_code=league_code, 
            away_team_league_code=league_code, 
            historical_h2h_matches_for_upcoming=fixture_h2h_context_matches,
            home_team_recent_matches_pool=home_form_pool,
            away_team_recent_matches_pool=away_form_pool,
            api_client_ref=api_client,
            num_form_games=NUM_FORM_GAMES
        )

        if prediction_output:
            probabilities_array, model_classes = prediction_output
            outcome_details = determine_match_outcome_from_probabilities(probabilities_array[0], model_classes, home_team_name, away_team_name)

            if outcome_details:
                outcome_str, pred_home_goals, pred_away_goals, _ = outcome_details
                
                league_table_df.loc[home_team_id, 'P'] += 1
                league_table_df.loc[away_team_id, 'P'] += 1
                league_table_df.loc[home_team_id, 'GF'] += pred_home_goals
                league_table_df.loc[home_team_id, 'GA'] += pred_away_goals
                league_table_df.loc[away_team_id, 'GF'] += pred_away_goals
                league_table_df.loc[away_team_id, 'GA'] += pred_home_goals

                if outcome_str == 'HOME_WIN':
                    league_table_df.loc[home_team_id, 'W'] += 1
                    league_table_df.loc[home_team_id, 'Pts'] += 3
                    league_table_df.loc[away_team_id, 'L'] += 1
                elif outcome_str == 'AWAY_WIN':
                    league_table_df.loc[away_team_id, 'W'] += 1
                    league_table_df.loc[away_team_id, 'Pts'] += 3
                    league_table_df.loc[home_team_id, 'L'] += 1
                else:
                    league_table_df.loc[home_team_id, 'D'] += 1
                    league_table_df.loc[home_team_id, 'Pts'] += 1
                    league_table_df.loc[away_team_id, 'D'] += 1
            else:
                print(f"  Could not determine outcome from probabilities for {home_team_name} vs {away_team_name}.")
                league_table_df.loc[home_team_id, 'P'] += 1
                league_table_df.loc[away_team_id, 'P'] += 1
        else:
            print(f"  Prediction failed for {home_team_name} vs {away_team_name} (predict_single_match_probabilities returned None).")
            league_table_df.loc[home_team_id, 'P'] += 1
            league_table_df.loc[away_team_id, 'P'] += 1
            
        league_table_df['GD'] = league_table_df['GF'] - league_table_df['GA']


    # Display Final Table
    print("\n--- Predicted Final League Table ---")
    league_table_df_sorted = league_table_df.sort_values(by=['Pts', 'GD', 'GF'], ascending=[False, False, False])
    
    league_table_df_sorted.insert(0, 'Pos', range(1, len(league_table_df_sorted) + 1))
    
    display_columns = ['Pos', 'name', 'P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']
    print(league_table_df_sorted[display_columns].to_string(index=False))


if __name__ == "__main__":
    API_KEY_MAIN = os.getenv("API_KEY")
    if not API_KEY_MAIN:
        print("\nYour football-data.org API key is not set as an environment variable (API_KEY).")
        API_KEY_MAIN = input("Please enter your football-data.org API key: ").strip()
    
    if not API_KEY_MAIN:
        print("API Key is required to run predictions. Exiting.")
    else:
        user_input = get_season_prediction_input()
        if user_input is None:
            pass
        else:
            league_to_predict, csv_path = user_input
            predict_entire_season(league_to_predict, csv_path, API_KEY_MAIN)