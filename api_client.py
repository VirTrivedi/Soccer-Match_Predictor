import requests
import os
from datetime import datetime

# API Configuration for football-data.org
API_BASE_URL = "https://api.football-data.org/v4/"
API_KEY = None
HEADERS = {}

# Cache for league teams
league_teams_cache = {}

# Cache for latest league standings
league_standings_cache = {}

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
    Uses a cache to minimize API calls.
    """
    global league_teams_cache

    if league_code in league_teams_cache:
        return league_teams_cache[league_code]

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
            league_teams_cache[league_code] = simplified_teams
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
    
def get_league_standings(league_code: str) -> list:
    """
    Fetches the latest available league standings for a given league code.
    Results are cached per league_code for the duration of the script's run.
    Subsequent calls for the same league_code will return the cached snapshot.
    """
    if league_code in league_standings_cache:
        return league_standings_cache[league_code]

    if not API_KEY:
        print("Error: API key not set. Call set_api_key() first.")
        return []

    url = f"{API_BASE_URL}competitions/{league_code}/standings"
            
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()

        standings_data = data.get("standings")
        if not standings_data:
            print(f"No 'standings' key found in API response for league {league_code}. Response: {data}")
            return []

        total_standings_table = None
        for standing_type_info in standings_data:
            if standing_type_info.get("type") == "TOTAL":
                total_standings_table = standing_type_info.get("table")
                break
        
        if not total_standings_table:
            print(f"No 'TOTAL' standings found for league {league_code}. Available types: {[s.get('type') for s in standings_data]}")
            return []

        processed_league_table = []
        for team_standing in total_standings_table:
            team_info = team_standing.get("team")
            if not team_info:
                print(f"Warning: Team info missing for an entry in standings table. Entry: {team_standing}")
                continue 

            processed_league_table.append({
                "team_id": team_info.get("id"),
                "team_name": team_info.get("name"),
                "position": team_standing.get("position"),
                "points": team_standing.get("points"),
                "played_games": team_standing.get("playedGames"),
            })
        
        if not processed_league_table and total_standings_table:
             print(f"Processed league table is empty, though 'total_standings_table' data was present for league {league_code}.")

        league_standings_cache[league_code] = processed_league_table
        return processed_league_table

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred while fetching standings for {league_code}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"API request failed for get_league_standings(league_code={league_code}): {e}")
        return []
    except ValueError as e:
        print(f"Failed to decode JSON response for {league_code}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in get_league_standings(league_code={league_code}): {e}")
        return []