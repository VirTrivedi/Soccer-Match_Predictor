import requests
import os
from datetime import datetime

# API Configuration for football-data.org
API_BASE_URL = "https://api.football-data.org/v4/"
API_KEY = None
HEADERS = {}

def set_api_key(key: str):
    """
    Sets the API key for football-data.org.

    Args:
        key (str): The key for football-data.org.
    """
    global API_KEY
    global HEADERS
    API_KEY = key
    HEADERS = {
        "X-Auth-Token": API_KEY,
    }

def get_teams_by_league(league_code: str) -> list:
    """
    Fetches a simplified list of teams (name, shortName, tla) for a given league code 
    using the /competitions/{league_code}/teams endpoint of football-data.org.
    """
    if not API_KEY:
        print("Error: API key not set. Call set_api_key() first.")
        return []

    try:
        response = requests.get(f"{API_BASE_URL}competitions/{league_code}/teams", headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        teams_data = data.get("teams")
        if teams_data:
            simplified_teams = []
            for team_item in teams_data:
                simplified_teams.append({
                    "id": team_item.get("id"),
                    "name": team_item.get("name"),
                    "shortName": team_item.get("shortName"),
                    "tla": team_item.get("tla")
                })
            return simplified_teams
        else:
            print(f"No teams found for league code '{league_code}'. API Response: {data}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed for get_teams(league_code={league_code}): {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in get_teams(league_code={league_code}): {e}")
        return []

def get_matches_for_team(team_id: int, date_from: str, date_to: str) -> list:
    """
    Fetches finished matches for a specific team within a date range from football-data.org.
    """
    if not API_KEY:
        print("Error: API key not set. Call set_api_key() first.")
        return []
    
    params = {
        "status": "FINISHED",
        "dateFrom": date_from,
        "dateTo": date_to
    }

    try:
        response = requests.get(f"{API_BASE_URL}teams/{team_id}/matches", headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()

        result_set = data.get("resultSet")
        matches_list = data.get("matches")

        if result_set and result_set.get("count", 0) > 0 and matches_list:
            return matches_list
        else:
            print(f"No matches found for team {team_id} (from {date_from} to {date_to}). 'resultSet' missing or 'matches' list empty. API Response: {data}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"API request failed for get_matches_for_team(team_id={team_id}, from={date_from}, to={date_to}): {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}, Response content: {e.response.text}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in get_matches_for_team(team_id={team_id}, from={date_from}, to={date_to}): {e}")
        return []

def get_head_to_head_matches(match_id: int) -> list:
    """
    Fetches head-to-head finished matches between two teams from football-data.org.
    """
    if not API_KEY:
        print("Error: API key not set. Call set_api_key() first.")
        return []
    
    try:
        response = requests.get(f"{API_BASE_URL}matches/{match_id}/head2head", headers=HEADERS)
        response.raise_for_status()
        data = response.json()

        result_set = data.get("resultSet")
        matches_list = data.get("matches")

        if result_set and result_set.get("count", 0) > 0 and matches_list:
            return matches_list
        else:    
            print(f"No h2h matches found, 'resultSet' missing or 'matches' list empty. API Response: {data}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"API request failed for get_head_to_head_matches(match_id={match_id}): {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}, Response content: {e.response.text}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in get_head_to_head_matches(match_id={match_id}): {e}")
        return []