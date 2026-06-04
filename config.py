"""
config.py — all shared configuration, constants, and reference data.
"""

# ── Seasons ────────────────────────────────────────────
SEASONS = [
    '2015-16', '2016-17', '2017-18', '2018-19',
    # '2019-20' excluded — bubble season distorts home court
    '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26',
]
TEST_SEASON = '2024-25'        # completed season -> reliable accuracy

# ── Rolling windows ────────────────────────────────────
ROLLING_SHORT = 5
ROLLING_LONG  = 10
ROLLING_XLONG = 20
TOP_N_PLAYERS = 5

# ── Cache files (each data source caches separately so partial runs work) ──
CACHE_DIR        = 'cache'
CACHE_CORE       = 'cache/core.pkl'        # team + player game logs
CACHE_ADVANCED   = 'cache/advanced.pkl'    # advanced team stats
CACHE_SCORING    = 'cache/scoring.pkl'     # paint/fastbreak/etc
CACHE_MISC       = 'cache/misc.pkl'        # points off TO, second chance
CACHE_OPPONENT   = 'cache/opponent.pkl'    # opponent (defensive) stats
CACHE_CLUTCH_T   = 'cache/clutch_team.pkl'
CACHE_CLUTCH_P   = 'cache/clutch_player.pkl'
CACHE_HUSTLE_T   = 'cache/hustle_team.pkl'
CACHE_HUSTLE_P   = 'cache/hustle_player.pkl'
CACHE_SHOTZONE   = 'cache/shotzone.pkl'
CACHE_PLAYER_SHOTZONE = 'cache/player_shotzone.pkl'
CACHE_TRACKING   = 'cache/tracking.pkl'
CACHE_BIO        = 'cache/bio.pkl'         # player positions/height
CACHE_ONOFF      = 'cache/onoff.pkl'
CACHE_PBP        = 'cache/pbp_features.pkl'   # processed play-by-play
CACHE_PBP_RAW    = 'cache/pbp_progress.pkl'   # resumable raw fetch progress
CACHE_REF_MAP    = 'cache/ref_game_map.parquet'      # GAME_ID -> ref names
CACHE_COACH_MAP  = 'cache/coach_season_map.parquet'  # (TEAM_ID, SEASON) -> coach name
CACHE_REF_STATS  = 'cache/ref_career_stats.pkl'      # processed ref career data
CACHE_COACH_STATS= 'cache/coach_career_stats.pkl'    # processed coach career data
MODEL_FILE       = 'nba_model.pkl'

# ── Rate limiting ──────────────────────────────────────
SLEEP_MIN = 0.3
SLEEP_MAX = 0.7
PBP_SLEEP_MIN = 0.3   # play-by-play is many calls, go a bit faster
PBP_SLEEP_MAX = 0.7

# ── Altitude ───────────────────────────────────────────
ALTITUDE_TEAMS = {'DEN'}

# ── Plot theme ─────────────────────────────────────────
PLOT_RC = {
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27',
    'axes.edgecolor': '#3a3d4d',   'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',       'xtick.color': '#a0a0b0',
    'ytick.color': '#a0a0b0',      'grid.color': '#2a2d3d',
    'grid.linestyle': '--',        'grid.alpha': 0.6,
    'font.family': 'sans-serif',
}
BLUE='#00d4ff'; RED='#ff6b6b'; GREEN='#51cf66'; YELLOW='#ffd43b'
PURPLE='#cc5de8'; ORANGE='#ff922b'; TEAL='#20c997'

# ── Arena coordinates (lat, lon) for travel-distance features ──
# Approximate arena locations by team abbreviation.
ARENA_COORDS = {
    'ATL': (33.757, -84.396), 'BOS': (42.366, -71.062), 'BKN': (40.683, -73.975),
    'CHA': (35.225, -80.839), 'CHI': (41.881, -87.674), 'CLE': (41.497, -81.688),
    'DAL': (32.790, -96.810), 'DEN': (39.749, -105.008), 'DET': (42.341, -83.055),
    'GSW': (37.768, -122.388), 'HOU': (29.751, -95.362), 'IND': (39.764, -86.155),
    'LAC': (34.043, -118.267), 'LAL': (34.043, -118.267), 'MEM': (35.138, -90.051),
    'MIA': (25.781, -80.187), 'MIL': (43.045, -87.917), 'MIN': (44.979, -93.276),
    'NOP': (29.949, -90.082), 'NYK': (40.751, -73.993), 'OKC': (35.463, -97.515),
    'ORL': (28.539, -81.384), 'PHI': (39.901, -75.172), 'PHX': (33.446, -112.071),
    'POR': (45.532, -122.667), 'SAC': (38.580, -121.500), 'SAS': (29.427, -98.437),
    'TOR': (43.643, -79.379), 'UTA': (40.768, -111.901), 'WAS': (38.898, -77.021),
}

# US timezone approximations (UTC offset) for jet-lag features
ARENA_TZ = {
    'ATL':-5,'BOS':-5,'BKN':-5,'CHA':-5,'CHI':-6,'CLE':-5,'DAL':-6,'DEN':-7,
    'DET':-5,'GSW':-8,'HOU':-6,'IND':-5,'LAC':-8,'LAL':-8,'MEM':-6,'MIA':-5,
    'MIL':-6,'MIN':-6,'NOP':-6,'NYK':-5,'OKC':-6,'ORL':-5,'PHI':-5,'PHX':-7,
    'POR':-8,'SAC':-8,'SAS':-6,'TOR':-5,'UTA':-7,'WAS':-5,
}
