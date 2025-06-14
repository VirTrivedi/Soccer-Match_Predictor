import os
import sys
from flask import Flask, jsonify, request, send_from_directory # Added send_from_directory
from flask_cors import CORS
import pandas as pd

# --- Add backend directory to sys.path to allow imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Attempt to import predictor modules, handling potential issues
try:
    # These should be directly in the backend folder now
    import api_client
    import soccer_predictor
    import season_predictor
except ImportError as e:
    print(f"Initial Error importing predictor modules: {e}", file=sys.stderr)
    api_client = None
    soccer_predictor = None
    season_predictor = None
except Exception as e: # Catch other potential errors during import
    print(f"General Error importing predictor modules: {e}", file=sys.stderr)
    api_client = None
    soccer_predictor = None
    season_predictor = None


# Load environment variables if .env file exists (for API_KEY)
from dotenv import load_dotenv
# Explicitly load .env from the current (backend) directory
dotenv_path = os.path.join(current_dir, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    print(f".env file not found at {dotenv_path}", file=sys.stderr)


# --- Flask App Setup ---
# Serve static files from the React build directory
# Assumes app.py is in dashboard/backend/ and React build is in dashboard/frontend/build/
static_folder_path = os.path.abspath(os.path.join(current_dir, '../frontend/build'))
app = Flask(__name__, static_folder=static_folder_path, static_url_path='/')
CORS(app)  # Enable CORS for all routes - important for dev even if serving static files

# --- API Key Handling ---
FOOTBALL_DATA_API_KEY = os.getenv("API_KEY") # Get from .env or environment

def ensure_api_key(): # Removed provided_key parameter
    global FOOTBALL_DATA_API_KEY # This is loaded from .env at startup
    if FOOTBALL_DATA_API_KEY:
        if api_client:
            api_client.set_api_key(FOOTBALL_DATA_API_KEY)
            # print(f"API Key (from .env) set: {FOOTBALL_DATA_API_KEY[:4]}...", file=sys.stderr) # For debugging
            return True
        else:
            print("api_client module not loaded, cannot set key from .env.", file=sys.stderr)
            return False
    print("API Key is not configured in the backend environment (.env file).", file=sys.stderr)
    return False

# --- Refactored Predictor Logic (Stubs) ---
# The api_key_val parameter is kept in the signature of refactored_predict_single_match
# as the subtask is focused on ensure_api_key and how routes call it.
# The calls to refactored_predict_single_match will now pass None for api_key_val.
def refactored_predict_single_match(home_team_id, away_team_id, home_league, away_league): # Removed api_key_val
    if not soccer_predictor:
        return {"error": "Soccer predictor module not loaded correctly."}, 500
    if not api_client:
        return {"error": "API Client module not loaded correctly."}, 500

    # Removed ensure_api_key(api_key_val) check, as it's handled by the route
    try:
        return {
            "message": "Single match prediction called (conceptual - backend needs full predictor refactor)",
            "home_team_id": home_team_id, "away_team_id": away_team_id,
            "home_league": home_league, "away_league": away_league,
            "note": "Full prediction logic integration requires deeper refactoring of soccer_predictor.py to return structured data instead of printing."
        }, 200
    except Exception as e:
        app.logger.error(f"Error in single match prediction: {e}")
        return {"error": str(e)}, 500

def refactored_predict_season(league_code, csv_file_path_param, num_simulations=100): # Removed api_key_val, Default to 100
    if not season_predictor:
        return {"error": "Season predictor module not loaded correctly."}, 500
    if not api_client:
        return {"error": "API Client module not loaded correctly."}, 500

    # Removed ensure_api_key(api_key_val) check, as it's handled by the route
    try:
        league_code_map_for_csv = {"BL1": "bl1", "FL1": "fl1", "PD": "pd", "PL": "pl", "SA": "sa"}
        mapped_code = league_code_map_for_csv.get(league_code.upper())
        if not mapped_code:
            return {"error": f"No sample CSV mapping for league code: {league_code}"}, 400
        
        actual_csv_file_path = os.path.join(current_dir, f"sample-{mapped_code}-data.csv")

        if not os.path.exists(actual_csv_file_path):
            return {"error": f"Sample CSV file not found: {actual_csv_file_path}"}, 404
        
        return {
            "message": f"Season prediction called for {league_code} using {actual_csv_file_path} (conceptual)",
            "num_simulations": num_simulations,
            "note": "Full season prediction logic integration requires deeper refactoring of season_predictor.py to return structured data."
        }, 200
    except Exception as e:
        app.logger.error(f"Error in season prediction: {e}")
        return {"error": str(e)}, 500

# --- API Endpoints ---
@app.route('/api/test')
def test_endpoint():
    return jsonify({"message": "Flask backend is running!"})

@app.route('/api/leagues')
def get_leagues_route():
    if soccer_predictor and hasattr(soccer_predictor, 'ALLOWED_LEAGUES'):
        return jsonify(soccer_predictor.ALLOWED_LEAGUES)
    
    print("ALLOWED_LEAGUES not found in soccer_predictor module or module not loaded. Using fallback.", file=sys.stderr)
    fallback_leagues = { "PL": "Premier League", "BL1": "Bundesliga", "SA": "Serie A", "PD": "Primera Division", "FL1": "Ligue 1" }
    return jsonify(fallback_leagues)


@app.route('/api/teams/<league_code>')
def get_teams_route_api(league_code): 
    # API key is no longer fetched from header here
    if not ensure_api_key(): # Call without parameters
        return jsonify({"error": "API key for football-data.org is not configured on the server."}), 503 # Service Unavailable
    
    if not api_client:
         return jsonify({"error": "API client module not available."}), 500

    teams = api_client.get_teams_by_league(league_code.upper())
    if teams is not None: 
        return jsonify(teams)
    else: 
        return jsonify({"error": f"Could not fetch teams for league {league_code}. League might be invalid or API error."}), 404

@app.route('/api/predict/match', methods=['POST'])
def predict_match_route_api(): 
    data = request.get_json()
    if not data: return jsonify({"error": "Request body must be JSON"}), 400
    home_team_id = data.get('home_team_id')
    away_team_id = data.get('away_team_id')
    home_league = data.get('home_league_code')
    away_league = data.get('away_league_code')
    # api_key_from_req is no longer used from request for ensure_api_key

    if not all([home_team_id, away_team_id, home_league, away_league]):
        # Modified error message for consistency with original instruction for this subtask (though not strictly necessary)
        return jsonify({"error": "Missing required parameters"}), 400
    try:
        home_team_id, away_team_id = int(home_team_id), int(away_team_id)
    except ValueError:
        return jsonify({"error": "team_ids must be integers"}), 400
        
    # Call ensure_api_key without parameters
    if not ensure_api_key():
        return jsonify({"error": "API key for football-data.org is not configured on the server."}), 503

    result, status_code = refactored_predict_single_match(home_team_id, away_team_id, home_league, away_league) # Removed None for api_key_val
    return jsonify(result), status_code

@app.route('/api/predict/season', methods=['POST'])
def predict_season_route_api(): 
    data = request.get_json()
    if not data: return jsonify({"error": "Request body must be JSON"}), 400
    league_code = data.get('league_code')
    num_sims = data.get('num_simulations', 100) 
    # api_key_from_req is no longer used from request
    if not league_code: return jsonify({"error": "Missing league_code"}), 400 # Simplified: "Missing required parameter: league_code"

    # Call ensure_api_key without parameters
    if not ensure_api_key():
        return jsonify({"error": "API key for football-data.org is not configured on the server."}), 503

    result, status_code = refactored_predict_season(league_code.upper(), None, num_simulations=int(num_sims)) # Removed None for api_key_val
    return jsonify(result), status_code

# --- Serve React App ---
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print(f"Flask app starting. Static folder: {app.static_folder}", file=sys.stderr)
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)