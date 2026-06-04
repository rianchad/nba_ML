"""
player_advanced_features.py — advanced per-player stats for improved lineup features.

Builds:
  1. Shot-zone efficiency: per-player rolling zone-FG% (6 zones) weighted by usage
  2. Foul-trouble rate: proportion of games where player is in foul trouble
  3. Clutch contribution: per-player rolling clutch stats (FGM, FTA, TO in final 2 min close games)
  4. Season-to-date usage and opportunity metrics

Endpoints used:
  - LeagueDashPlayerShotLocations (in fetch.py as CACHE_SHOTZONE): zone-level FGA/FGM
  - Core game logs: PF (personal fouls) and minutes to identify foul trouble
  - Player PBP (via fetch_player_pbp): clutch_fgm, clutch_fta, etc.
"""

import numpy as np
import pandas as pd
import joblib
from rich.console import Console
import config as C

console = Console()

CACHE_PLAYER_ZONE = f'{C.CACHE_DIR}/player_zone_stats.pkl'
CACHE_PLAYER_FOUL = f'{C.CACHE_DIR}/player_foul_stats.pkl'
CACHE_PLAYER_CLUTCH = f'{C.CACHE_DIR}/player_clutch_roll.pkl'


def _zone_cat(col_name: str) -> str:
    """Map a zone label to paint / midrange / 3pt / other."""
    s = str(col_name).lower()
    # Paint: Restricted Area + In The Paint (Non-RA)
    if any(k in s for k in ('restricted', 'non-ra', 'non_ra', 'paint')):
        return 'paint'
    # 3-pointers: corners + above the break
    if any(k in s for k in ('corner 3', 'above the break', 'three', '3-pt', 'backcourt')):
        return '3pt'
    # Midrange
    if 'mid' in s or 'mid-range' in s:
        return 'midrange'
    return 'other'


def _extract_wide_zone(shotzone_df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle wide-format LeagueDashPlayerShotLocations By Zone data.
    Actual column format: '{Zone Name}_FGM', '{Zone Name}_FGA', '{Zone Name}_FG_PCT'
    e.g. 'Restricted Area_FGM', 'In The Paint (Non-RA)_FGA', 'Mid-Range_FGM'.
    Returns long-format DataFrame[PLAYER_ID, SEASON, ZONE_CAT, FGM, FGA, FG_PCT].
    """
    if 'PLAYER_ID' not in shotzone_df.columns:
        return pd.DataFrame()

    cols = list(shotzone_df.columns)
    # Columns ending with '_FGM' (not the base 'FGM' column)
    fgm_cols = [c for c in cols if str(c).endswith('_FGM')]
    if not fgm_cols:
        return pd.DataFrame()

    season_col = 'SEASON' if 'SEASON' in shotzone_df.columns else None
    rows = []

    for fgm_c in fgm_cols:
        zone_label = str(fgm_c)[:-4]          # strip trailing '_FGM'
        cat = _zone_cat(zone_label)
        fga_c = zone_label + '_FGA'
        fgpct_c = zone_label + '_FG_PCT'

        sub = pd.DataFrame()
        sub['PLAYER_ID'] = shotzone_df['PLAYER_ID']
        sub['ZONE_CAT'] = cat
        sub['FGM'] = pd.to_numeric(shotzone_df[fgm_c], errors='coerce').fillna(0)
        sub['FGA'] = (pd.to_numeric(shotzone_df[fga_c], errors='coerce').fillna(0)
                      if fga_c in shotzone_df.columns else sub['FGM'])
        if fgpct_c in shotzone_df.columns:
            sub['FG_PCT_RAW'] = pd.to_numeric(shotzone_df[fgpct_c], errors='coerce')
        sub['SEASON'] = shotzone_df[season_col].values if season_col else ''
        rows.append(sub)

    if not rows:
        return pd.DataFrame()

    long = pd.concat(rows, ignore_index=True)
    agg = (long.groupby(['PLAYER_ID', 'SEASON', 'ZONE_CAT'])
           .agg({'FGM': 'sum', 'FGA': 'sum'})
           .reset_index())
    agg['FG_PCT'] = agg['FGM'] / agg['FGA'].replace(0, np.nan)
    return agg


def add_shot_zone_quality(player_rolling: pd.DataFrame, shotzone_df=None) -> pd.DataFrame:
    """
    Add per-player shot-zone efficiency metrics to rolling stats.

    Handles both long format (SHOT_ZONE_BASIC column) and wide format
    (LeagueDashPlayerShotLocations By Zone with flattened MultiIndex columns).
    Returns player_rolling with zone_fg_pct_paint/mid/3pt_roll10 and zone_efg_pct_roll10.
    """
    pr = player_rolling.copy()

    # Add default columns so downstream code always finds them
    for col in ['zone_fg_pct_paint_roll10', 'zone_fg_pct_mid_roll10',
                'zone_fg_pct_3pt_roll10', 'zone_efg_pct_roll10']:
        pr[col] = 0.45

    if shotzone_df is None:
        try:
            shotzone_df = joblib.load(C.CACHE_PLAYER_SHOTZONE)
        except Exception:
            return pr

    if not isinstance(shotzone_df, pd.DataFrame) or shotzone_df.empty:
        return pr

    shotzone_df = shotzone_df.copy()

    # Guard: need PLAYER_ID
    if 'PLAYER_ID' not in shotzone_df.columns:
        return pr

    shotzone_df['PLAYER_ID'] = pd.to_numeric(shotzone_df['PLAYER_ID'], errors='coerce')
    shotzone_df = shotzone_df.dropna(subset=['PLAYER_ID'])
    shotzone_df['PLAYER_ID'] = shotzone_df['PLAYER_ID'].astype(int)

    if 'SEASON' not in shotzone_df.columns:
        shotzone_df['SEASON'] = ''

    # ── Long format (SHOT_ZONE_BASIC column) ──────────────────────────────────
    if 'SHOT_ZONE_BASIC' in shotzone_df.columns:
        shotzone_df['ZONE'] = shotzone_df['SHOT_ZONE_BASIC'].astype(str).fillna('Unknown')
        zone_stats = (
            shotzone_df.groupby(['PLAYER_ID', 'SEASON', 'ZONE'])
            .agg({'FGA': 'sum', 'FGM': 'sum'})
            .reset_index()
        )
        zone_stats['FG_PCT'] = zone_stats['FGM'] / zone_stats['FGA'].replace(0, np.nan)
        zone_stats['ZONE_CAT'] = zone_stats['ZONE'].apply(_zone_cat)
        zone_agg = (
            zone_stats.groupby(['PLAYER_ID', 'SEASON', 'ZONE_CAT'])
            .agg({'FG_PCT': 'mean'})
            .reset_index()
            .rename(columns={'FG_PCT': 'ZONE_FG_PCT'})
        )

    # ── Wide format (LeagueDashPlayerShotLocations By Zone) ───────────────────
    else:
        zone_agg = _extract_wide_zone(shotzone_df)
        if zone_agg.empty:
            return pr
        zone_agg = zone_agg.rename(columns={'FG_PCT': 'ZONE_FG_PCT'})

    # Merge per-zone category into player_rolling
    for cat, out_col in [('paint', 'zone_fg_pct_paint'),
                         ('midrange', 'zone_fg_pct_mid'),
                         ('3pt', 'zone_fg_pct_3pt')]:
        subset = (zone_agg[zone_agg['ZONE_CAT'] == cat]
                  [['PLAYER_ID', 'SEASON', 'ZONE_FG_PCT']]
                  .rename(columns={'ZONE_FG_PCT': out_col}))
        if subset.empty:
            pr[out_col] = np.nan
            continue
        pr = pr.merge(subset, on=['PLAYER_ID', 'SEASON'], how='left')

    # Roll 10-game window per player
    for col in ['zone_fg_pct_paint', 'zone_fg_pct_mid', 'zone_fg_pct_3pt']:
        roll_col = f'{col}_roll10'
        if col in pr.columns:
            pr[roll_col] = (
                pr.groupby('PLAYER_ID')[col]
                .transform(lambda x: x.shift(1).rolling(C.ROLLING_LONG, min_periods=1).mean())
            )
        # fill defaults (already set above, so only overwrite non-NaN)
        if roll_col in pr.columns:
            pr[roll_col] = pr[roll_col].fillna(0.45)

    # Weighted eFG%
    paint = pr['zone_fg_pct_paint_roll10'] if 'zone_fg_pct_paint_roll10' in pr.columns else pd.Series(0.5, index=pr.index)
    mid   = pr['zone_fg_pct_mid_roll10']   if 'zone_fg_pct_mid_roll10'   in pr.columns else pd.Series(0.4, index=pr.index)
    _3pt  = pr['zone_fg_pct_3pt_roll10']   if 'zone_fg_pct_3pt_roll10'   in pr.columns else pd.Series(0.35, index=pr.index)
    pr['zone_efg_pct_roll10'] = (0.5 * paint.fillna(0.5) +
                                 0.3 * mid.fillna(0.4) +
                                 0.2 * _3pt.fillna(0.35))
    pr['zone_efg_pct_roll10'] = pr['zone_efg_pct_roll10'].fillna(0.45)

    return pr


def add_foul_trouble_rate(player_rolling: pd.DataFrame, player_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add foul-trouble rate per player (rolling 10-game proportion of games where
    PF >= 5 or MIN <= 20 due to foul concerns).

    Returns player_rolling with new column: foul_trouble_rate_roll10.
    """
    if player_df is None:
        core = joblib.load(C.CACHE_CORE)
        player_df = core['player_df'].copy()

    pr = player_rolling.copy()

    # Per-player per-game: identify foul trouble
    pdf = player_df.copy()
    pdf['GAME_DATE'] = pd.to_datetime(pdf['GAME_DATE'])
    pdf = pdf.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
    pdf['IN_FOUL_TROUBLE'] = (
        ((pdf['PF'] >= 5) | (pdf['MIN'] <= 20)) &
        (pdf['MIN'] > 0)
    ).astype(float)

    # Rolling 10-game proportion
    foul_rolling = (
        pdf.groupby('PLAYER_ID')['IN_FOUL_TROUBLE']
        .transform(lambda x: x.shift(1).rolling(C.ROLLING_LONG, min_periods=1).mean())
    )
    pdf['foul_trouble_rate_roll10'] = foul_rolling.fillna(0)

    # Merge back: match by PLAYER_ID + GAME_ID
    foul_merge = pdf[['PLAYER_ID', 'GAME_ID', 'foul_trouble_rate_roll10']].drop_duplicates(['PLAYER_ID', 'GAME_ID'])
    pr = pr.merge(foul_merge, on=['PLAYER_ID', 'GAME_ID'], how='left')
    pr['foul_trouble_rate_roll10'] = pr['foul_trouble_rate_roll10'].fillna(0)

    return pr


def add_clutch_rolling(player_rolling: pd.DataFrame, player_clutch_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add per-player rolling clutch stats (final 2 min, close games, shift-1 for leakage prevention).

    Merges player_clutch_df (from fetch_player_pbp.extract_player_clutch_stats)
    and computes 10-game rolling: clutch_fgm_roll10, clutch_fta_roll10, clutch_to_roll10.
    """
    if player_clutch_df is None or player_clutch_df.empty:
        for col in ['clutch_fgm_roll10', 'clutch_fta_roll10', 'clutch_to_roll10']:
            player_rolling[col] = 0
        return player_rolling

    pr = player_rolling.copy()
    pc = player_clutch_df.copy()

    # Merge clutch stats by GAME_ID + PLAYER_ID
    pr = pr.merge(pc[['GAME_ID', 'PLAYER_ID', 'clutch_fgm', 'clutch_fta', 'clutch_to']],
                  on=['GAME_ID', 'PLAYER_ID'], how='left')
    pr['clutch_fgm'] = pr['clutch_fgm'].fillna(0)
    pr['clutch_fta'] = pr['clutch_fta'].fillna(0)
    pr['clutch_to']  = pr['clutch_to'].fillna(0)

    # Roll per PLAYER_ID
    pr['GAME_DATE'] = pd.to_datetime(pr['GAME_DATE'])
    pr = pr.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

    for stat in ['clutch_fgm', 'clutch_fta', 'clutch_to']:
        pr[f'{stat}_roll10'] = (
            pr.groupby('PLAYER_ID')[stat]
            .transform(lambda x: x.shift(1).rolling(C.ROLLING_LONG, min_periods=1).mean())
            .fillna(0)
        )

    return pr[pr.columns.difference(['clutch_fgm', 'clutch_fta', 'clutch_to'])]
