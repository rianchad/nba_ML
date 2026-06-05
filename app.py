#!/opt/homebrew/bin/python3.11
"""
app.py — local web server for NBA win probability predictions.

Usage:
    pip install flask
    python app.py
    open http://localhost:5001
"""

import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

BUNDLE_PATH = os.environ.get('NBA_BUNDLE', 'nba_mac_export.pkl')

# ── Global state ──────────────────────────────────────────────────────────────
bundle = None
model = None
features = None
medians = None
featured_df = None

MATCHUP_FEATURES = [
    'mu_off_mismatch_home', 'mu_off_mismatch_away',
    'mu_best_mismatch_home', 'mu_best_mismatch_away',
    'mu_worst_mismatch_home',
    'mu_home_star_vs_def', 'mu_away_star_vs_def',
    'mu_home_best_def', 'mu_away_best_def',
    'mu_guard_edge_home', 'mu_wing_edge_home', 'mu_big_edge_home',
]

NBA_TEAM_NAMES = {
    'ATL': 'Atlanta Hawks',      'BKN': 'Brooklyn Nets',
    'BOS': 'Boston Celtics',     'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',      'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',   'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',    'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers',        'LAL': 'LA Lakers',
    'MEM': 'Memphis Grizzlies',  'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans','NYK': 'New York Knicks',
    'OKC': 'OKC Thunder',        'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers','SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',  'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',          'WAS': 'Washington Wizards',
}

NBA_TEAM_COLORS = {
    'ATL': '#C8102E', 'BKN': '#000000', 'BOS': '#007A33', 'CHA': '#1D1160',
    'CHI': '#CE1141', 'CLE': '#860038', 'DAL': '#00538C', 'DEN': '#0E2240',
    'DET': '#C8102E', 'GSW': '#1D428A', 'HOU': '#CE1141', 'IND': '#002D62',
    'LAC': '#C8102E', 'LAL': '#552583', 'MEM': '#5D76A9', 'MIA': '#98002E',
    'MIL': '#00471B', 'MIN': '#0C2340', 'NOP': '#0C2340', 'NYK': '#006BB6',
    'OKC': '#007AC1', 'ORL': '#0077C0', 'PHI': '#006BB6', 'PHX': '#1D1160',
    'POR': '#E03A3E', 'SAC': '#5A2D81', 'SAS': '#C4CED4', 'TOR': '#CE1141',
    'UTA': '#002B5C', 'WAS': '#002B5C',
}


def load_bundle():
    global bundle, model, features, medians, featured_df
    print(f"Loading bundle: {BUNDLE_PATH}")
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle['model']
    features = bundle['features']
    medians = bundle['medians']
    featured_df = bundle['featured_df']
    print("Bundle loaded.")


def predict_game(home, away, playoffs=False, adv_params=None):
    home, away = home.upper(), away.upper()
    anchor = ['PTS_roll10', 'WINRATE_roll10']

    def latest(ab):
        r = featured_df[featured_df['TEAM_ABBREVIATION'] == ab].dropna(subset=anchor)
        if len(r) == 0:
            raise ValueError(f"Team '{ab}' not found or missing data.")
        return r.iloc[-1]

    h, a = latest(home), latest(away)

    rec = {
        'IS_PLAYOFFS': int(playoffs),
        'H2H_HOME_WIN': 0.5,
        'H2H_MEETINGS_THIS_SEASON': 0,
    }
    for col in features:
        if col in rec:
            continue
        if col in MATCHUP_FEATURES or col.endswith('_gap'):
            rec[col] = 0
            continue
        base = col.replace('_home', '').replace('_away', '')
        src = h if '_home' in col else a
        rec[col] = src.get(base, np.nan)

    if adv_params:
        param_map = {
            # Rest & schedule
            'rest_days_home':        'REST_DAYS_home',
            'rest_days_away':        'REST_DAYS_away',
            'is_back_to_back_home':  'IS_BACK_TO_BACK_home',
            'is_back_to_back_away':  'IS_BACK_TO_BACK_away',
            # Travel & venue
            'travel_km_home':        'TRAVEL_KM_home',
            'travel_km_away':        'TRAVEL_KM_away',
            'tz_change_home':        'TZ_CHANGE_home',
            'tz_change_away':        'TZ_CHANGE_away',
            'is_altitude_home':      'IS_ALTITUDE_home',
            # Streaks
            'win_streak_home':       'WIN_STREAK_home',
            'win_streak_away':       'WIN_STREAK_away',
            'home_streak_home':      'HOME_STREAK_home',
            'home_streak_away':      'HOME_STREAK_away',
            'road_streak_home':      'ROAD_STREAK_home',
            'road_streak_away':      'ROAD_STREAK_away',
            # Trends
            'pts_trend_home':        'PTS_TREND_home',
            'pts_trend_away':        'PTS_TREND_away',
            'pm_trend_home':         'PM_TREND_home',
            'pm_trend_away':         'PM_TREND_away',
            'wr_trend_home':         'WINRATE_TREND_home',
            'wr_trend_away':         'WINRATE_TREND_away',
            # Strength of schedule
            'sos_home':              'SOS_roll10_home',
            'sos_away':              'SOS_roll10_away',
            # Head-to-head
            'h2h_home_win':          'H2H_HOME_WIN',
            'h2h_meetings':          'H2H_MEETINGS_THIS_SEASON',
            # Season context
            'games_into_season':     'GAMES_INTO_SEASON_home',
            'is_late_season':        'IS_LATE_SEASON_home',
        }
        for k, v in adv_params.items():
            feat = param_map.get(k)
            if feat and feat in rec:
                rec[feat] = v

        # Auto-compute gap features when both sides are provided
        def _gap(a_key, b_key, gap_feat):
            a_val = adv_params.get(a_key)
            b_val = adv_params.get(b_key)
            if a_val is not None and b_val is not None and gap_feat in features:
                rec[gap_feat] = float(a_val) - float(b_val)

        _gap('rest_days_home',  'rest_days_away',  'rest_days_gap')
        _gap('win_streak_home', 'win_streak_away',  'streak_gap')
        _gap('road_streak_home','road_streak_away', 'road_streak_gap')
        _gap('travel_km_home',  'travel_km_away',   'travel_gap')
        _gap('pts_trend_home',  'pts_trend_away',   'trend_gap')
        _gap('wr_trend_home',   'wr_trend_away',    'wr_trend_gap')

    X = pd.DataFrame([rec])[features].apply(pd.to_numeric, errors='coerce').fillna(medians)
    p = model.predict_proba(X)[0]
    home_prob, away_prob = float(p[1]), float(p[0])
    margin = abs(home_prob - away_prob)
    confidence = 'High' if margin > 0.15 else 'Medium' if margin > 0.07 else 'Low'
    return {
        'home': home,
        'away': away,
        'home_prob': round(home_prob * 100, 1),
        'away_prob': round(away_prob * 100, 1),
        'margin': round(margin * 100, 1),
        'confidence': confidence,
        'winner': home if home_prob > away_prob else away,
        'playoffs': playoffs,
    }


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route('/api/teams')
def get_teams():
    teams = sorted(featured_df['TEAM_ABBREVIATION'].unique().tolist())
    result = []
    for abbr in teams:
        result.append({
            'abbr': abbr,
            'name': NBA_TEAM_NAMES.get(abbr, abbr),
            'color': NBA_TEAM_COLORS.get(abbr, '#555555'),
        })
    return jsonify(result)


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    home = data.get('home', '').strip()
    away = data.get('away', '').strip()
    playoffs = bool(data.get('playoffs', False))
    adv_params = data.get('adv_params') or {}

    if not home or not away:
        return jsonify({'error': 'home and away are required'}), 400
    if home == away:
        return jsonify({'error': 'Teams must be different'}), 400

    try:
        result = predict_game(home, away, playoffs=playoffs,
                              adv_params=adv_params if adv_params else None)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


if __name__ == '__main__':
    try:
        load_bundle()
    except FileNotFoundError:
        print(f"ERROR: Bundle not found at '{BUNDLE_PATH}'")
        print("Set NBA_BUNDLE env var or place nba_mac_export.pkl in the same directory.")
        sys.exit(1)
    app.run(host='127.0.0.1', port=5001, debug=False)
