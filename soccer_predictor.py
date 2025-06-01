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

def get_match_outcome(match_data, team_id_of_interest):
    """
    Determines the outcome of a match for a specific team_id_of_interest
    based on api-football.com fixture data structure.
    """
    teams_data = match_data.get('teams', {})
    home_team_data = teams_data.get('home', {})
    away_team_data = teams_data.get('away', {})
    goals_data = match_data.get('goals', {})

    if not home_team_data or not away_team_data or not goals_data:
        return None

    home_id = home_team_data.get('id')
    away_id = away_team_data.get('id')
    home_winner = home_team_data.get('winner')
    away_winner = away_team_data.get('winner')
    home_goals = goals_data.get('home')
    away_goals = goals_data.get('away')

    if home_id is None or away_id is None or home_goals is None or away_goals is None:
        return None

    # Case 1: Explicit winner
    if home_winner is True and team_id_of_interest == home_id:
        return 'WIN'
    if away_winner is True and team_id_of_interest == away_id:
        return 'WIN'
    
    # Case 2: Explicit draw
    if home_winner is False and away_winner is False:
        return 'DRAW'
    
    # Case 3: Scores are equal -> draw
    if home_goals == away_goals:
        return 'DRAW'

    # Case 4: Loss for the team of interest
    if team_id_of_interest == home_id and home_winner is False:
        return 'LOSS'
    if team_id_of_interest == away_id and away_winner is False:
        return 'LOSS'
        
    # Case 5: Fallback if winner booleans are None but scores differ
    if team_id_of_interest == home_id:
        return 'WIN' if home_goals > away_goals else 'LOSS'
    if team_id_of_interest == away_id:
        return 'WIN' if away_goals > home_goals else 'LOSS'
    
    # If none of the above conditions matched, return None
    return None 

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
    """
    features = {
        'h2h_home_wins': 0,
        'h2h_draws': 0,
        'h2h_away_wins': 0,
        'h2h_home_goals_sum': 0,
        'h2h_away_goals_sum': 0,
        'h2h_matches_played': 0
    }
    
    # Filter matches that have necessary data
    for match in h2h_matches:
        teams = match.get('teams', {})
        goals = match.get('goals', {})
        if not teams or not goals or not teams.get('home') or not teams.get('away'):
            continue

        match_home_id = teams['home'].get('id')
        match_away_id = teams['away'].get('id')
        
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

        if match_home_id == perspective_home_team_id:
            if goals.get('home') is not None: features['h2h_home_goals_sum'] += goals['home']
            if goals.get('away') is not None: features['h2h_away_goals_sum'] += goals['away']
        elif match_away_id == perspective_home_team_id:
            if goals.get('away') is not None: features['h2h_home_goals_sum'] += goals['away']
            if goals.get('home') is not None: features['h2h_away_goals_sum'] += goals['home']
            
    return features

def calculate_form_features(team_matches: list, team_id: int, num_games: int = 5):
    """
    Calculates form features for a team from their last N games.
    """
    features = {
        f'form_wins_last_{num_games}': 0,
        f'form_draws_last_{num_games}': 0,
        f'form_losses_last_{num_games}': 0,
        f'form_goals_scored_last_{num_games}': 0,
        f'form_goals_conceded_last_{num_games}': 0,
        f'form_goal_diff_last_{num_games}': 0,
        f'form_matches_considered': 0
    }
    
    # Filter for matches with necessary data and sort by date
    valid_matches = [m for m in team_matches if m.get('fixture', {}).get('date') and m.get('teams') and m.get('goals')]
    try:
        sorted_matches = sorted(valid_matches, key=lambda x: x['fixture']['date'], reverse=True)
    except (TypeError, KeyError) as e:
        print(f"Error sorting matches for team {team_id} (form calculation): {e}")
        return features

    recent_n_games = sorted_matches[:num_games]
    features['form_matches_considered'] = len(recent_n_games)

    if not recent_n_games:
        return features

    for match in recent_n_games:
        outcome = get_match_outcome(match, team_id)
        goals = match['goals']
        teams = match['teams']

        if outcome == 'WIN':
            features[f'form_wins_last_{num_games}'] += 1
        elif outcome == 'DRAW':
            features[f'form_draws_last_{num_games}'] += 1
        elif outcome == 'LOSS':
            features[f'form_losses_last_{num_games}'] += 1

        # Accumulate goals based on whether team_id was home or away in this match
        if teams.get('home', {}).get('id') == team_id:
            if goals.get('home') is not None: features[f'form_goals_scored_last_{num_games}'] += goals['home']
            if goals.get('away') is not None: features[f'form_goals_conceded_last_{num_games}'] += goals['away']
        elif teams.get('away', {}).get('id') == team_id:
            if goals.get('away') is not None: features[f'form_goals_scored_last_{num_games}'] += goals['away']
            if goals.get('home') is not None: features[f'form_goals_conceded_last_{num_games}'] += goals['home']
            
    features[f'form_goal_diff_last_{num_games}'] = features[f'form_goals_scored_last_{num_games}'] - features[f'form_goals_conceded_last_{num_games}']
    return features

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

def create_dataset_from_matches(all_historical_h2h_matches: list, num_form_games: int = 5):
    """
    Creates a dataset from historical H2H matches. For each match, features are calculated
    based on data *prior* to that match. Perspective is always actual home team of that historical match.
    """
    dataset = []
    
    # Ensure matches have minimal required data and sort them chronologically (oldest first)
    valid_matches = [
        m for m in all_historical_h2h_matches 
        if m.get('fixture', {}).get('date') and \
           m.get('teams', {}).get('home', {}).get('id') is not None and \
           m.get('teams', {}).get('away', {}).get('id') is not None and \
           m.get('goals') is not None
    ]
    try:
        sorted_historical_matches = sorted(valid_matches, key=lambda x: x['fixture']['date'])
    except (TypeError, KeyError) as e:
        print(f"Error sorting historical H2H matches for dataset creation: {e}")
        return pd.DataFrame()

    for i, current_match in enumerate(sorted_historical_matches):
        fixture_data = current_match['fixture']
        teams_data = current_match['teams']
        current_match_date_str = fixture_data['date']
        
        actual_home_id = teams_data['home']['id']
        actual_away_id = teams_data['away']['id']
            
        goals_data = current_match.get('goals', {})
        home_goals = goals_data.get('home')
        away_goals = goals_data.get('away')

        # Check if goals are None, if so, skip this match for the dataset
        if home_goals is None or away_goals is None:
            print(f"Skipping match {current_match.get('fixture', {}).get('id')} due to missing goals data.")
            continue

        home_goals_capped = min(home_goals, 5)
        away_goals_capped = min(away_goals, 5)
        target_scoreline = f"{home_goals_capped}-{away_goals_capped}"

        matches_before_current = sorted_historical_matches[:i]
        
        h2h_features = calculate_h2h_features(matches_before_current, actual_home_id, actual_away_id)
        
        home_team_matches_for_form_calc = [
            m for m in all_historical_h2h_matches
            if m['fixture']['date'] < current_match_date_str and 
               (m['teams']['home']['id'] == actual_home_id or m['teams']['away']['id'] == actual_home_id)
        ]
        away_team_matches_for_form_calc = [
            m for m in all_historical_h2h_matches
            if m['fixture']['date'] < current_match_date_str and
               (m['teams']['home']['id'] == actual_away_id or m['teams']['away']['id'] == actual_away_id)
        ]

        home_form = calculate_form_features(home_team_matches_for_form_calc, actual_home_id, num_form_games)
        away_form = calculate_form_features(away_team_matches_for_form_calc, actual_away_id, num_form_games)

        row = {'match_id': fixture_data.get('id'), 'date': current_match_date_str,
               'home_team_id_h2h_match': actual_home_id, 'away_team_id_h2h_match': actual_away_id}
        row.update(h2h_features)
        for key, val in home_form.items(): row[f'current_home_{key}'] = val
        for key, val in away_form.items(): row[f'current_away_{key}'] = val
        row['target_scoreline'] = target_scoreline
        dataset.append(row)
        
    return pd.DataFrame(dataset)

def get_user_input():
    """Gets team names and API key from the user."""
    home_team_name = input("Enter the home team name: ")
    away_team_name = input("Enter the away team name: ")
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("\nYour api-football.com API key is not set as an environment variable (API_KEY).")
        api_key = input("Please enter your api-football.com API key: ")
    else:
        print("\nUsing api-football.com API key from environment variable.")
    return home_team_name, away_team_name, api_key

def main():
    print("--- Soccer Match Predictor (using api-football.com) ---")
    home_team_name, away_team_name, api_key = get_user_input()

    if not all([home_team_name, away_team_name, api_key]):
        print("Error: Home team, away team, and API key must be provided.")
        return
    api_client.set_api_key(api_key)

    print(f"\nFetching ID for home team (user specified): {home_team_name}...")
    user_home_team_id = api_client.get_team_id(home_team_name)
    if not user_home_team_id:
        print(f"Could not find team ID for {home_team_name}. Exiting.")
        return
    print(f"Found ID for {home_team_name}: {user_home_team_id}")

    print(f"\nFetching ID for away team (user specified): {away_team_name}...")
    user_away_team_id = api_client.get_team_id(away_team_name)
    if not user_away_team_id:
        print(f"Could not find team ID for {away_team_name}. Exiting.")
        return
    print(f"Found ID for {away_team_name}: {user_away_team_id}")

    print("\n--- Fetching Historical & Recent Match Data ---")
    num_form_games = 10 # For form calculation (last N games)
    num_years_history = 5 # For H2H history (last N years)

    date_to = datetime.now()
    date_from_h2h_history = date_to - timedelta(days=num_years_history*365)

    date_to_str = date_to.strftime('%Y-%m-%d')
    date_from_h2h_history_str = date_from_h2h_history.strftime('%Y-%m-%d')

    print(f"Fetching H2H matches between {home_team_name} and {away_team_name} (from {date_from_h2h_history_str} to {date_to_str})...")
    all_h2h_matches_for_stats_and_dataset = api_client.get_head_to_head_matches(
        user_home_team_id, user_away_team_id, 
        date_from=date_from_h2h_history_str, date_to=date_to_str
    )
    if not all_h2h_matches_for_stats_and_dataset:
        print(f"Warning: No direct H2H matches found for {home_team_name} vs {away_team_name} in the last {num_years_history} years. Model training might fail or be unreliable.")
    else:
        print(f"Found {len(all_h2h_matches_for_stats_and_dataset)} H2H matches for stats and dataset construction.")

    print(f"\nFetching recent matches for {home_team_name} (pool for current form)...")
    home_team_recent_matches_pool = api_client.get_matches_for_team(user_home_team_id, "2021-01-01", "2023-12-31")
    print(f"Found {len(home_team_recent_matches_pool)} matches in pool for {home_team_name}.")

    print(f"\nFetching recent matches for {away_team_name} (pool for current form)...")
    away_team_recent_matches_pool = api_client.get_matches_for_team(user_away_team_id, "2021-01-01", "2023-12-31")
    print(f"Found {len(away_team_recent_matches_pool)} matches in pool for {away_team_name}.")
    
    print("\n--- Feature Engineering ---")
    historical_df = pd.DataFrame()
    if all_h2h_matches_for_stats_and_dataset:
        historical_df = create_dataset_from_matches(all_h2h_matches_for_stats_and_dataset, num_form_games=num_form_games)
    
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

    min_samples_for_split = 10 
    min_samples_per_class_for_stratify = 2
    
    can_stratify = len(y.unique()) > 1 and all(y.value_counts() >= min_samples_per_class_for_stratify)

    X_train, X_test, y_train, y_test = pd.DataFrame(columns=X.columns), pd.Series(dtype='int'), pd.DataFrame(columns=X.columns), pd.Series(dtype='int')
    if len(X) >= min_samples_for_split and can_stratify:
        test_size_actual = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_actual, random_state=42, stratify=y)
    elif len(X) >= min_samples_for_split:
        print("Warning: Cannot stratify train/test split. Using non-stratified split.")
        test_size_actual = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_actual, random_state=42)
    else:
        print("Warning: Dataset too small for test split. Training on all available data.")
        X_train, y_train = X, y

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
    upcoming_h2h_features = calculate_h2h_features(all_h2h_matches_for_stats_and_dataset, user_home_team_id, user_away_team_id)
    upcoming_home_form = calculate_form_features(home_team_recent_matches_pool, user_home_team_id, num_form_games)
    upcoming_away_form = calculate_form_features(away_team_recent_matches_pool, user_away_team_id, num_form_games)
    
    upcoming_match_features_dict = {}
    upcoming_match_features_dict.update(upcoming_h2h_features)
    for key, val in upcoming_home_form.items(): upcoming_match_features_dict[f'current_home_{key}'] = val
    for key, val in upcoming_away_form.items(): upcoming_match_features_dict[f'current_away_{key}'] = val

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

        if not X_test.empty and not y_test.empty:
            print("\n--- Model Evaluation on Test Set ---")
            y_pred_test = model.predict(X_test)

            # Scoreline Accuracy
            scoreline_accuracy = accuracy_score(y_test, y_pred_test)
            print(f"Scoreline Accuracy (exact score): {scoreline_accuracy:.2f}")

            # Outcome Accuracy
            y_test_outcomes = y_test.apply(get_outcome_from_scoreline)
            y_pred_outcomes = pd.Series(y_pred_test, index=y_test.index).apply(get_outcome_from_scoreline)

            valid_indices = (y_test_outcomes != "UNKNOWN") & (y_pred_outcomes != "UNKNOWN")
            y_test_outcomes_valid = y_test_outcomes[valid_indices]
            y_pred_outcomes_valid = y_pred_outcomes[valid_indices]

            if not y_test_outcomes_valid.empty:
                outcome_accuracy = accuracy_score(y_test_outcomes_valid, y_pred_outcomes_valid)
                print(f"Outcome Accuracy (Win/Loss/Draw): {outcome_accuracy:.2f}")
            else:
                print("Outcome Accuracy (Win/Loss/Draw): Could not be calculated (no valid comparable outcomes).")

            print("\nClassification Report (Scorelines):")
            actual_present_scoreline_labels = np.unique(np.concatenate((y_test.unique(), y_pred_test)))
            report_labels_scoreline = sorted([l for l in model.classes_ if l in actual_present_scoreline_labels])
            
            if report_labels_scoreline:
                report_target_names_scoreline = [str(l) for l in report_labels_scoreline]
                print(classification_report(y_test, y_pred_test, 
                                            labels=report_labels_scoreline, 
                                            target_names=report_target_names_scoreline, 
                                            zero_division=0))
            else:
                print("Could not generate classification report for scorelines (no common/reportable labels).")

            # Classification Report for Outcomes
            if not y_test_outcomes_valid.empty:
                print("\nClassification Report (Outcomes - Win/Loss/Draw):")
                outcome_order = ["WIN", "DRAW", "LOSS"] 
                present_outcome_labels = sorted(
                    list(set(y_test_outcomes_valid.unique()) | set(y_pred_outcomes_valid.unique())),
                    key=lambda x: outcome_order.index(x) if x in outcome_order else float('inf')
                )
                present_outcome_labels = [l for l in present_outcome_labels if l in outcome_order]

                if present_outcome_labels:
                    print(classification_report(y_test_outcomes_valid, y_pred_outcomes_valid, 
                                                labels=present_outcome_labels, 
                                                zero_division=0))
                else:
                    print("Could not generate classification report for outcomes (no reportable outcome labels).")
            else:
                print("No valid outcomes to generate an outcome classification report.")

        elif not X_train.empty and y_train.size > 0:
            print("\nNote: Model was trained on all available historical H2H data, or test set was too small; no separate test set evaluation performed.")
            
    except Exception as e:
        print(f"Error during prediction or evaluation: {e}")

if __name__ == "__main__":
    main()