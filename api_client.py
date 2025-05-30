import requests
import os
from datetime import datetime

# API Configuration for api-football.com
API_BASE_URL = "https://v3.football.api-sports.io/"
API_KEY = None
HEADERS = {}

def set_api_key(key: str):
    """
    Sets the API key for api-football.com requests via RapidAPI.

    Args:
        key (str): The RapidAPI key for api-football.com.
    """
    global API_KEY
    global HEADERS
    API_KEY = key
    HEADERS = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }

def get_team_id(team_name: str) -> int | None:
    """
    Fetches the ID of a team based on its name using the /teams endpoint of api-football.com.
    It prioritizes an exact match on the team's name. If no exact match is found,
    it returns the ID of the first team in the search results as a fallback.
    """
    if not API_KEY:
        print("Error: API key not set. Call set_api_key() first.")
        return None

    params = {"name": team_name}
    try:
        print(f"DEBUG: Requesting team ID for '{team_name}' from {API_BASE_URL}teams with params: {params}")
        response = requests.get(f"{API_BASE_URL}teams", headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()

        # Check if the API returned any results and if the 'response' array is present
        if data.get("results", 0) > 0 and data.get("response"):
            for team_info_item in data["response"]:
                team_data = team_info_item.get("team")
                if team_data and team_data.get("name", "").lower() == team_name.lower():
                    print(f"DEBUG: Exact match found for '{team_name}': ID {team_data['id']}")
                    return team_data["id"]
            
            # If no exact match, fall back to the first result
            first_team_data = data["response"][0].get("team")
            if first_team_data and first_team_data.get("id"):
                print(f"Warning: No exact name match for '{team_name}'. Using first result: {first_team_data.get('name')} (ID: {first_team_data['id']})")
                return first_team_data["id"]
            else:
                print(f"Error: First result for '{team_name}' is malformed or missing ID. Response: {data['response'][0]}")
                return None
        else:
            error_msg = data.get("errors", "Unknown error or empty response.")
            print(f"No teams found or error for '{team_name}'. API Response: {error_msg}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed during get_team_id for '{team_name}': {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in get_team_id for '{team_name}': {e}")
        return None

def get_matches_for_team(team_id: int, date_from: str, date_to: str) -> list:
    """
    Fetches finished matches for a specific team within a date range from api-football.com.
    """
    if not API_KEY:
        print("Error: API key not set. Call set_api_key() first.")
        return []
    
    try:
        season_year = datetime.strptime(date_to, '%Y-%m-%d').year
    except ValueError:
        print("Error: Invalid date_to format. Should be YYYY-MM-DD for season derivation.")
        return []

    params = {
        "team": team_id,
        "from": date_from,
        "to": date_to,
        "status": "FT-AET-PEN",
        "season": season_year
    }
    
    try:
        response = requests.get(f"{API_BASE_URL}fixtures", headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results", 0) > 0 and data.get("response"):
            return data["response"]
        else:
            errors = data.get("errors", "No matches found or unknown error.")
            print(f"No matches found for team {team_id} (season {season_year}, from {date_from} to {date_to}). API Errors: {errors}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"API request failed for get_matches_for_team(team_id={team_id}): {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in get_matches_for_team(team_id={team_id}): {e}")
        return []

def get_head_to_head_matches(team1_id: int, team2_id: int, date_from: str | None = None, date_to: str | None = None) -> list:
    """
    Fetches head-to-head finished matches between two teams from api-football.com.
    """
    if not API_KEY:
        print("Error: API key not set. Call set_api_key() first.")
        return []

    params = {
        "h2h": f"{team1_id}-{team2_id}",
        "status": "FT"
    }

    # Add date parameters if provided
    if date_from and date_to:
        params["from"] = date_from
        params["to"] = date_to
    else:
        print("Warning: Calling get_head_to_head_matches without date_from and date_to. API will use its default range/season for H2H.")
    
    try:
        response = requests.get(f"{API_BASE_URL}fixtures/headtohead", headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results", 0) > 0 and data.get("response"):
            return data["response"]
        else:
            errors = data.get("errors", "No H2H matches found or unknown error.")
            print(f"No H2H matches found for {team1_id} vs {team2_id} with params {params}. API Errors: {errors}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed for get_head_to_head_matches({team1_id} vs {team2_id}): {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in get_head_to_head_matches({team1_id} vs {team2_id}): {e}")
        return []