"""
fetch_player_pbp.py — extract per-player play-by-play stats during main PBP processing.

While parsing play-by-play events, we extract clutch contributions (final 2 min of close
games) and quarter-specific runs, then roll these into per-player features.

Endpoints used:
  - PlayByPlayV2 (already fetched in fetch_pbp.py): extract player IDs and event types
  - Team/player logs: merge to get player identities and basic stats

Output: DataFrame[GAME_ID, PLAYER_ID, SEASON, clutch_fgm, clutch_fga, clutch_to, etc.]
        and quarterly max-run features per team per game.
"""

import pandas as pd
import numpy as np


def extract_player_clutch_stats(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    From play-by-play DataFrame, extract per-player clutch contributions
    (final 2 minutes, score margin <= 5 points).

    NOTE: CACHE_PBP stores pre-processed game-level features, not raw PBP rows.
    Raw PBP rows would need PCTIMESTRING, EVENTTYPE, PLAYER_ID columns.
    This function returns empty if passed the processed cache (expected).

    Returns DataFrame[GAME_ID, PLAYER_ID, SEASON,
                      clutch_fgm, clutch_fga, clutch_to, clutch_fta, clutch_ftm].
    """
    _empty = pd.DataFrame(columns=[
        'GAME_ID', 'PLAYER_ID', 'SEASON',
        'clutch_fgm', 'clutch_fga', 'clutch_to', 'clutch_fta', 'clutch_ftm'
    ])
    if pbp_df.empty or 'PCTIMESTRING' not in pbp_df.columns:
        return _empty

    pbp = pbp_df.copy()

    # Standardize time format: "10:23" or "0:45" -> minutes int
    def parse_time(s):
        try:
            if pd.isna(s):
                return 999
            s = str(s).strip()
            if ':' not in s:
                return 999
            parts = s.split(':')
            return int(parts[0])
        except:
            return 999

    pbp['MINUTES'] = pbp['PCTIMESTRING'].apply(parse_time)
    pbp['IS_CLUTCH'] = (pbp['MINUTES'] <= 2) & (pbp['SCORE'].fillna('').str.len() > 0)

    # Score margin: extract from "HHH - AAA" format
    def parse_margin(score_str):
        try:
            if pd.isna(score_str) or not str(score_str).strip():
                return 999
            parts = str(score_str).strip().split('-')
            if len(parts) != 2:
                return 999
            h, a = int(parts[0].strip()), int(parts[1].strip())
            return abs(h - a)
        except:
            return 999

    pbp['MARGIN'] = pbp['SCORE'].apply(parse_margin)
    pbp['IS_CLOSE'] = pbp['MARGIN'] <= 5
    pbp['IS_CLUTCH'] = pbp['IS_CLUTCH'] & pbp['IS_CLOSE']

    # Event classification
    pbp['IS_FGM'] = pbp['EVENTTYPE'].isin([1, 2])  # FGM types
    pbp['IS_FGA'] = pbp['EVENTTYPE'].isin([1, 2, 3])  # FGM or FGA
    pbp['IS_TO']  = pbp['EVENTTYPE'] == 5
    pbp['IS_FTA'] = pbp['EVENTTYPE'].isin([3, 4])  # FT attempt
    pbp['IS_FTM'] = pbp['EVENTTYPE'] == 4  # FT made

    rows = []
    for (gid, pid), g in pbp[pbp['IS_CLUTCH']].groupby(['GAME_ID', 'PLAYER_ID']):
        season = g['SEASON'].iloc[0] if 'SEASON' in g.columns else None
        rows.append({
            'GAME_ID': gid,
            'PLAYER_ID': pid,
            'SEASON': season,
            'clutch_fgm': int(g['IS_FGM'].sum()),
            'clutch_fga': int(g['IS_FGA'].sum()),
            'clutch_to': int(g['IS_TO'].sum()),
            'clutch_fta': int(g['IS_FTA'].sum()),
            'clutch_ftm': int(g['IS_FTM'].sum()),
        })

    return pd.DataFrame(rows)


def extract_quarter_runs(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract quarter-specific max run features.

    CACHE_PBP stores pre-processed game-level features from fetch_pbp.py, NOT raw rows.
    Processed columns available: pbp_max_run_home, pbp_max_run_away, Q1_DIFF..Q4_DIFF,
    pbp_clutch_net_home.

    Returns a game-level DataFrame (no TEAM_ID) with columns:
      GAME_ID, q1_max_run_home, q2_max_run_home, q3_max_run_home, q4_max_run_home,
               q1_max_run_away, q2_max_run_away, q3_max_run_away, q4_max_run_away
    build_quarter_run_features handles merging this by GAME_ID only.
    """
    _empty_cols = [
        'GAME_ID',
        'q1_max_run_home', 'q2_max_run_home', 'q3_max_run_home', 'q4_max_run_home',
        'q1_max_run_away', 'q2_max_run_away', 'q3_max_run_away', 'q4_max_run_away',
    ]

    if pbp_df is None or pbp_df.empty or 'GAME_ID' not in pbp_df.columns:
        return pd.DataFrame(columns=_empty_cols)

    # ── Processed format (from fetch_pbp.py CACHE_PBP) ───────────────────────
    if 'pbp_max_run_home' in pbp_df.columns:
        result = pbp_df[['GAME_ID']].copy()

        # Quarter point differentials as proxy for per-quarter dominance
        for q in [1, 2, 3, 4]:
            diff_col = f'Q{q}_DIFF'
            diff = pd.to_numeric(pbp_df.get(diff_col, 0), errors='coerce').fillna(0)
            # Positive diff = home dominated; negative = away dominated
            result[f'q{q}_max_run_home'] = diff.clip(lower=0)
            result[f'q{q}_max_run_away'] = (-diff).clip(lower=0)

        # Override Q4 with actual max run data (more accurate)
        if 'pbp_max_run_home' in pbp_df.columns:
            result['q4_max_run_home'] = pd.to_numeric(
                pbp_df['pbp_max_run_home'], errors='coerce').fillna(0)
        if 'pbp_max_run_away' in pbp_df.columns:
            result['q4_max_run_away'] = pd.to_numeric(
                pbp_df['pbp_max_run_away'], errors='coerce').fillna(0)

        return result[_empty_cols]

    # ── Raw PBP format (per-play rows with scoreHome/scoreAway columns) ───────
    pbp = pbp_df.copy()
    if 'PERIOD' not in pbp.columns:
        return pd.DataFrame(columns=_empty_cols)

    pbp['QUARTER'] = pd.to_numeric(pbp['PERIOD'], errors='coerce').fillna(1).astype(int).clip(upper=4)
    rows = []
    for gid, game_pbp in pbp.groupby('GAME_ID'):
        row = {'GAME_ID': gid}
        for qtr in [1, 2, 3, 4]:
            q_data = game_pbp[game_pbp['QUARTER'] == qtr]
            if q_data.empty or 'scoreHome' not in q_data.columns:
                row[f'q{qtr}_max_run_home'] = 0.0
                row[f'q{qtr}_max_run_away'] = 0.0
                continue
            margins = (pd.to_numeric(q_data['scoreHome'], errors='coerce').fillna(0) -
                       pd.to_numeric(q_data['scoreAway'], errors='coerce').fillna(0)).values
            max_home = max_away = run_home = run_away = 0.0
            for d in np.diff(margins):
                if d > 0:
                    run_home += d; run_away = 0; max_home = max(max_home, run_home)
                elif d < 0:
                    run_away += -d; run_home = 0; max_away = max(max_away, run_away)
            row[f'q{qtr}_max_run_home'] = max_home
            row[f'q{qtr}_max_run_away'] = max_away
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_empty_cols)
    return pd.DataFrame(rows)[_empty_cols]

