"""
lineup_advanced_features.py — advanced lineup-level features for match prediction.

Builds:
  1. On/off splits at the 5-man lineup level (net rating when specific lineup is on court)
  2. Starting lineup identification and composition matching across seasons
  3. Quarter-specific run features (max run in Q1-Q4 separately, Q4-weighted)
  4. Foul-trouble aggregation at lineup level (mean foul trouble risk)
  5. Shot-zone quality aggregation (mean zone FG% of lineup)

Endpoints used:
  - TeamPlayerOnOffDetails (in fetch.py): on/off court net rating per player
  - Play-by-play (fetch_pbp.py): quarterly runs
  - Player advanced (player_advanced_features.py): zone quality, foul trouble
"""

import numpy as np
import pandas as pd
import joblib
import config as C
from rich.console import Console
from itertools import combinations

console = Console()

CACHE_LINEUP_SPLITS = f'{C.CACHE_DIR}/lineup_onoff_splits.pkl'
CACHE_LINEUP_COMPS  = f'{C.CACHE_DIR}/lineup_compositions.pkl'


def build_lineup_onoff_splits(featured: pd.DataFrame, onoff_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    For each game, identify the starting 5 and their combined on/off net rating.

    Returns featured DataFrame with new columns:
      lineup_onoff_nrtg_home, lineup_onoff_nrtg_away: net rating when starting 5 on court
      lineup_onoff_games_home, lineup_onoff_games_away: games played together
    """
    if onoff_df is None:
        try:
            onoff_df = joblib.load(C.CACHE_ONOFF)
        except:
            console.print("[yellow]  on/off cache not found, skipping lineup on/off splits")
            featured['lineup_onoff_nrtg_home'] = 0
            featured['lineup_onoff_nrtg_away'] = 0
            featured['lineup_onoff_games_home'] = 0
            featured['lineup_onoff_games_away'] = 0
            return featured

    feat = featured.copy()

    # Per (TEAM_ID, SEASON), aggregate on/off: mean NET_RTG when lineup (all 5) on court
    onoff_df = onoff_df.copy()
    onoff_df['TEAM_ID'] = onoff_df['TEAM_ID'].astype(int) if 'TEAM_ID' in onoff_df.columns else 0

    # On-court net rating (NET_RATING_ON column name varies; try variants)
    nrtg_col = None
    for c in ['NET_RATING_ON', 'NET_RATING', 'PLUS_MINUS', 'NET_PLUS_MINUS']:
        if c in onoff_df.columns:
            nrtg_col = c
            break
    if nrtg_col is None:
        console.print("[yellow]  on/off: NET_RATING column not found, skipping on/off splits")
        feat['lineup_onoff_nrtg_home'] = 0
        feat['lineup_onoff_nrtg_away'] = 0
        feat['lineup_onoff_games_home'] = 0
        feat['lineup_onoff_games_away'] = 0
        return feat

    # Per team per season: when starting lineup on-court, mean net rating
    lineup_nrtg = (
        onoff_df.groupby(['TEAM_ID', 'SEASON'])
        .agg({nrtg_col: 'mean'})
        .reset_index()
        .rename(columns={nrtg_col: 'lineup_onoff_nrtg'})
    )
    lineup_nrtg['TEAM_ID'] = lineup_nrtg['TEAM_ID'].astype(int)

    # Games played by lineup (count of on/off records per team/season)
    lineup_games = (
        onoff_df.groupby(['TEAM_ID', 'SEASON'])
        .size()
        .reset_index(name='lineup_onoff_games')
    )
    lineup_games['TEAM_ID'] = lineup_games['TEAM_ID'].astype(int)

    lineup_splits = lineup_nrtg.merge(lineup_games, on=['TEAM_ID', 'SEASON'], how='left')

    # Merge to featured: home and away teams separately
    feat = feat.merge(
        lineup_splits[['TEAM_ID', 'SEASON', 'lineup_onoff_nrtg', 'lineup_onoff_games']]
        .rename(columns={'TEAM_ID': 'TEAM_ID_home',
                        'lineup_onoff_nrtg': 'lineup_onoff_nrtg_home',
                        'lineup_onoff_games': 'lineup_onoff_games_home'}),
        on=['TEAM_ID_home', 'SEASON'], how='left'
    )
    feat = feat.merge(
        lineup_splits[['TEAM_ID', 'SEASON', 'lineup_onoff_nrtg', 'lineup_onoff_games']]
        .rename(columns={'TEAM_ID': 'TEAM_ID_away',
                        'lineup_onoff_nrtg': 'lineup_onoff_nrtg_away',
                        'lineup_onoff_games': 'lineup_onoff_games_away'}),
        on=['TEAM_ID_away', 'SEASON'], how='left'
    )

    for col in ['lineup_onoff_nrtg_home', 'lineup_onoff_nrtg_away']:
        feat[col] = feat[col].fillna(feat[col].median())
    for col in ['lineup_onoff_games_home', 'lineup_onoff_games_away']:
        feat[col] = feat[col].fillna(0).astype(int)

    return feat


def build_quarter_run_features(featured: pd.DataFrame, quarter_runs_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add quarter-specific max run features (from fetch_player_pbp.extract_quarter_runs).

    quarter_runs_df is game-level (no TEAM_ID) with columns:
      GAME_ID, q1_max_run_home, ..., q4_max_run_home, q1_max_run_away, ..., q4_max_run_away

    Returns featured with those columns plus q4_run_weighted_home/away.
    """
    feat = featured.copy()

    _zero_cols = (
        [f'q{q}_max_run_home' for q in range(1, 5)] +
        [f'q{q}_max_run_away' for q in range(1, 5)] +
        ['q4_run_weighted_home', 'q4_run_weighted_away']
    )

    if quarter_runs_df is None or quarter_runs_df.empty or 'GAME_ID' not in quarter_runs_df.columns:
        for col in _zero_cols:
            feat[col] = 0
        return feat

    qr = quarter_runs_df.copy()

    # Merge by GAME_ID only (game-level format from extract_quarter_runs)
    run_cols = [c for c in qr.columns if c != 'GAME_ID']
    feat = feat.merge(qr[['GAME_ID'] + run_cols], on='GAME_ID', how='left')

    # Q4-weighted: Q4 runs matter more (double weight)
    for side in ['home', 'away']:
        def _gcol(col):
            return feat[col].fillna(0) if col in feat.columns else pd.Series(0.0, index=feat.index)
        q1 = _gcol(f'q1_max_run_{side}')
        q2 = _gcol(f'q2_max_run_{side}')
        q3 = _gcol(f'q3_max_run_{side}')
        q4 = _gcol(f'q4_max_run_{side}')
        feat[f'q4_run_weighted_{side}'] = (q1 + q2 + q3 + 2 * q4) / 5

    # Fill missing
    for col in _zero_cols:
        if col in feat.columns:
            feat[col] = feat[col].fillna(0)
        else:
            feat[col] = 0

    return feat


def aggregate_player_advanced_to_lineup(featured: pd.DataFrame,
                                       player_rolling: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player-level advanced stats (zone FG%, foul trouble, clutch) to
    top-5 lineup level.

    Returns featured with new columns:
      lineup_zone_efg_pct_roll10_{home,away}: mean zone eFG% of top 5
      lineup_foul_trouble_rate_{home,away}: mean foul trouble rate of top 5
      lineup_clutch_fgm_roll10_{home,away}: mean clutch FGM of top 5
      lineup_clutch_fta_roll10_{home,away}: mean clutch FTA of top 5
    """
    feat = featured.copy()
    if player_rolling.empty:
        for col in ['lineup_zone_efg_pct_roll10_home', 'lineup_zone_efg_pct_roll10_away',
                   'lineup_foul_trouble_rate_home', 'lineup_foul_trouble_rate_away',
                   'lineup_clutch_fgm_roll10_home', 'lineup_clutch_fgm_roll10_away',
                   'lineup_clutch_fta_roll10_home', 'lineup_clutch_fta_roll10_away']:
            feat[col] = 0
        return feat

    pr = player_rolling.copy()

    # Per (GAME_ID, TEAM_ID), aggregate top-5 player stats
    top5 = pr.sort_values(['GAME_ID', 'TEAM_ID', 'MIN_FLOAT'], ascending=[True, True, False])
    top5 = top5.groupby(['GAME_ID', 'TEAM_ID']).head(C.TOP_N_PLAYERS).copy()

    # Aggregation columns
    agg_cols = {
        'zone_efg_pct_roll10': 'mean',
        'foul_trouble_rate_roll10': 'mean',
        'clutch_fgm_roll10': 'mean',
        'clutch_fta_roll10': 'mean',
    }

    agg_dict = {}
    for col, agg_fn in agg_cols.items():
        if col in top5.columns:
            agg_dict[col] = agg_fn

    if agg_dict:
        lineup_agg = (
            top5.groupby(['GAME_ID', 'TEAM_ID'])
            .agg(agg_dict)
            .reset_index()
            .rename(columns={c: f'lineup_{c}' for c in agg_dict})
        )
    else:
        lineup_agg = top5[['GAME_ID', 'TEAM_ID']].drop_duplicates()
        for col in ['zone_efg_pct_roll10', 'foul_trouble_rate_roll10', 'clutch_fgm_roll10', 'clutch_fta_roll10']:
            lineup_agg[f'lineup_{col}'] = 0

    # Merge home
    feat = feat.merge(
        lineup_agg.rename(columns={'TEAM_ID': 'TEAM_ID_home',
                                  'lineup_zone_efg_pct_roll10': 'lineup_zone_efg_pct_roll10_home',
                                  'lineup_foul_trouble_rate_roll10': 'lineup_foul_trouble_rate_home',
                                  'lineup_clutch_fgm_roll10': 'lineup_clutch_fgm_roll10_home',
                                  'lineup_clutch_fta_roll10': 'lineup_clutch_fta_roll10_home'}),
        on=['GAME_ID', 'TEAM_ID_home'], how='left'
    )
    # Merge away
    feat = feat.merge(
        lineup_agg.rename(columns={'TEAM_ID': 'TEAM_ID_away',
                                  'lineup_zone_efg_pct_roll10': 'lineup_zone_efg_pct_roll10_away',
                                  'lineup_foul_trouble_rate_roll10': 'lineup_foul_trouble_rate_away',
                                  'lineup_clutch_fgm_roll10': 'lineup_clutch_fgm_roll10_away',
                                  'lineup_clutch_fta_roll10': 'lineup_clutch_fta_roll10_away'}),
        on=['GAME_ID', 'TEAM_ID_away'], how='left'
    )

    # Fill missing
    agg_new_cols = [
        'lineup_zone_efg_pct_roll10_home', 'lineup_zone_efg_pct_roll10_away',
        'lineup_foul_trouble_rate_home', 'lineup_foul_trouble_rate_away',
        'lineup_clutch_fgm_roll10_home', 'lineup_clutch_fgm_roll10_away',
        'lineup_clutch_fta_roll10_home', 'lineup_clutch_fta_roll10_away',
    ]
    for col in agg_new_cols:
        if col in feat.columns:
            feat[col] = feat[col].fillna(feat[col].median() or 0)
        else:
            feat[col] = 0

    return feat
