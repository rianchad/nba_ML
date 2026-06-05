"""
ref_coach_features.py — referee crew and head-coach feature engineering.

build_ref_features(matchup_df, team_df=None)
    Fetches per-game referee assignments via BoxScoreSummaryV2 (Officials table),
    then computes chronological pre-game crew tendencies using the core game log
    for home win outcomes, foul counts, and possession estimates.

build_coach_features(matchup_df, team_df=None)
    Fetches head-coach assignments per team/season via CommonTeamRoster (Coaches
    table), then computes chronological pre-game career splits (home/away/playoff/
    back-to-back win %, experience) from the core game log.

Both functions median-fill any rows where a lookup fails, matching the existing
missing-value handling used throughout the pipeline.
"""

import numpy as np
import pandas as pd
import joblib, os
from rich.console import Console

import config as C
import fetch_ref_coach as frc

console = Console()

CACHE_REF_STATS   = f'{C.CACHE_DIR}/ref_career_stats.pkl'
CACHE_COACH_STATS = f'{C.CACHE_DIR}/coach_career_stats.pkl'


# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_team_df(team_df):
    if team_df is not None:
        return team_df.copy()
    core = joblib.load(C.CACHE_CORE)
    return core['team_df'].copy()


def _prep_team_df(tdf: pd.DataFrame) -> pd.DataFrame:
    """Attach derived columns needed by both feature builders."""
    tdf = tdf.copy()
    tdf['GAME_DATE'] = pd.to_datetime(tdf['GAME_DATE'])
    tdf['WIN']       = (tdf['WL'] == 'W').astype(float)
    tdf['IS_HOME']   = tdf['MATCHUP'].str.contains(r'vs\.').astype(int)
    tdf['TEAM_ID']   = tdf['TEAM_ID'].astype(int)

    # Possessions: standard approximation (team perspective)
    for col in ['FGA', 'FTA', 'OREB', 'TOV']:
        if col not in tdf.columns:
            tdf[col] = np.nan
    tdf['EST_POSS'] = (tdf['FGA'] - tdf['OREB'] + tdf['TOV'] + 0.44 * tdf['FTA'])

    # IS_BACK_TO_BACK not present in raw core cache; derive from game-date gaps
    if 'IS_BACK_TO_BACK' not in tdf.columns:
        tdf = tdf.sort_values(['TEAM_ID', 'GAME_DATE'])
        tdf['IS_BACK_TO_BACK'] = (
            tdf.groupby('TEAM_ID')['GAME_DATE']
               .diff().dt.days.fillna(3) == 1
        ).astype(int)

    if 'IS_PLAYOFFS' not in tdf.columns:
        tdf['IS_PLAYOFFS'] = 0

    if 'PF' not in tdf.columns:
        tdf['PF'] = np.nan

    return tdf


def _per_game_stats(tdf: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot team-level rows into one row per game with home/away stats side by side.
    Returns DataFrame[GAME_ID, GAME_DATE, HOME_WIN, HOME_PF, AWAY_PF,
                      TOTAL_FOULS, FOUL_BIAS, AVG_POSS].
    """
    home_cols = {'WIN': 'HOME_WIN', 'PF': 'HOME_PF', 'EST_POSS': 'HOME_POSS'}
    away_cols = {'PF': 'AWAY_PF',  'EST_POSS': 'AWAY_POSS'}

    home = (tdf[tdf['IS_HOME'] == 1]
            [['GAME_ID', 'GAME_DATE'] + list(home_cols)]
            .rename(columns=home_cols))
    away = (tdf[tdf['IS_HOME'] == 0]
            [['GAME_ID'] + list(away_cols)]
            .rename(columns=away_cols))

    g = home.merge(away, on='GAME_ID', how='inner').drop_duplicates('GAME_ID')
    g['TOTAL_FOULS'] = g['HOME_PF']   + g['AWAY_PF']
    g['FOUL_BIAS']   = g['HOME_PF']   - g['AWAY_PF']   # positive -> more fouls on home
    g['AVG_POSS']    = (g['HOME_POSS'] + g['AWAY_POSS']) / 2
    return g.sort_values('GAME_DATE').reset_index(drop=True)


def _expanding_mean_shift(series: pd.Series) -> pd.Series:
    """Per-group expanding mean shifted by 1 (no leakage of current game)."""
    return series.expanding().mean().shift(1)


# ── Referee features ──────────────────────────────────────────────────────────

def build_ref_features(matchup_df: pd.DataFrame, team_df=None) -> pd.DataFrame:
    """
    Add referee crew historical tendency stats to each row of matchup_df.

    Endpoints used (via fetch_ref_coach):
      - BoxScoreSummaryV2 (index 2, Officials): FIRST_NAME, LAST_NAME per game
      - Core game logs (team_df): PF for foul counts, FGA/FTA/OREB/TOV for possessions

    New columns (single values per game row — referees apply to both teams):
      ref_home_win_pct   : crew career home-team win % prior to this game
      ref_foul_rate      : crew career avg total fouls per game
      ref_pace_effect    : crew career avg estimated possessions per game
      ref_home_foul_bias : crew career avg (home fouls - away fouls) per game
    """
    tdf       = _prep_team_df(_load_team_df(team_df))
    game_stats = _per_game_stats(tdf)
    game_ids   = matchup_df['GAME_ID'].unique().tolist()

    ref_map    = frc.fetch_ref_game_map(game_ids)
    ref_career = _build_ref_career(ref_map, game_stats)
    joblib.dump(ref_career, CACHE_REF_STATS)

    crew = _crew_stats(ref_map, ref_career)

    out = matchup_df.copy().merge(crew, on='GAME_ID', how='left')
    ref_cols = ['ref_home_win_pct', 'ref_foul_rate', 'ref_pace_effect', 'ref_home_foul_bias']
    for col in ref_cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = out[col].fillna(out[col].median())

    coverage = out[ref_cols].notna().all(axis=1).mean()
    console.print(f"  [cyan]Referee features added ({coverage:.1%} full coverage)")
    return out


def _build_ref_career(ref_map: pd.DataFrame, game_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Long-form per-ref expanding career stats (shift-1 for leakage prevention).
    Returns DataFrame[REF_NAME, GAME_ID, ref_hw_pct, ref_fouls, ref_poss, ref_bias].
    """
    # Expand wide ref columns to long form
    long_rows = []
    for _, row in ref_map.iterrows():
        for col in ['REF_1', 'REF_2', 'REF_3']:
            name = str(row[col]).strip()
            if name:
                long_rows.append({'GAME_ID': row['GAME_ID'], 'REF_NAME': name})
    long = pd.DataFrame(long_rows)
    if long.empty:
        return pd.DataFrame(columns=['REF_NAME','GAME_ID','ref_hw_pct',
                                     'ref_fouls','ref_poss','ref_bias'])

    stat_cols = ['GAME_ID', 'GAME_DATE', 'HOME_WIN', 'TOTAL_FOULS', 'FOUL_BIAS', 'AVG_POSS']
    long = long.merge(game_stats[stat_cols], on='GAME_ID', how='left')

    # Sort chronologically within each ref for correct expanding window
    long = long.sort_values(['REF_NAME', 'GAME_DATE']).reset_index(drop=True)

    grp = long.groupby('REF_NAME', sort=False)
    long['ref_hw_pct'] = grp['HOME_WIN'].transform(_expanding_mean_shift)
    long['ref_fouls']  = grp['TOTAL_FOULS'].transform(_expanding_mean_shift)
    long['ref_poss']   = grp['AVG_POSS'].transform(_expanding_mean_shift)
    long['ref_bias']   = grp['FOUL_BIAS'].transform(_expanding_mean_shift)

    return long[['REF_NAME', 'GAME_ID', 'ref_hw_pct', 'ref_fouls', 'ref_poss', 'ref_bias']]


def _crew_stats(ref_map: pd.DataFrame, ref_career: pd.DataFrame) -> pd.DataFrame:
    """Average per-ref career stats across the 2-3 crew members for each game."""
    lookup = ref_career.set_index(['GAME_ID', 'REF_NAME'])
    rows = []
    for _, row in ref_map.iterrows():
        gid = row['GAME_ID']
        hw, fl, ps, bs = [], [], [], []
        for col in ['REF_1', 'REF_2', 'REF_3']:
            name = str(row[col]).strip()
            if not name:
                continue
            key = (gid, name)
            if key in lookup.index:
                r = lookup.loc[key]
                hw.append(r['ref_hw_pct'])
                fl.append(r['ref_fouls'])
                ps.append(r['ref_poss'])
                bs.append(r['ref_bias'])
        def _safe_mean(lst):
            vals = [v for v in lst if v is not None and not (isinstance(v, float) and np.isnan(v))]
            return float(np.mean(vals)) if vals else np.nan

        rows.append({
            'GAME_ID':            gid,
            'ref_home_win_pct':   _safe_mean(hw),
            'ref_foul_rate':      _safe_mean(fl),
            'ref_pace_effect':    _safe_mean(ps),
            'ref_home_foul_bias': _safe_mean(bs),
        })
    return pd.DataFrame(rows)


# ── Coach features ────────────────────────────────────────────────────────────

def build_coach_features(matchup_df: pd.DataFrame, team_df=None) -> pd.DataFrame:
    """
    Add head-coach career split stats to each row of matchup_df.

    Endpoints used (via fetch_ref_coach):
      - CommonTeamRoster (index 1, Coaches): head coach name per team/season
      - Core game logs (team_df): game outcomes, IS_HOME, IS_PLAYOFFS, IS_BACK_TO_BACK

    New columns (suffixed _home / _away for each team's coach):
      coach_win_pct_home_{home,away}     : career home-game win %
      coach_win_pct_away_{home,away}     : career away-game win %
      coach_win_pct_playoffs_{home,away} : career playoff win %
      coach_ats_home_{home,away}         : career win % in back-to-back games
      coach_experience_{home,away}       : career games coached before this game

    Gap features (home coach minus away coach):
      coach_winpct_gap, coach_playoff_gap, coach_exp_gap
    """
    tdf = _prep_team_df(_load_team_df(team_df))

    # Collect all (TEAM_ID, SEASON) pairs present in matchup_df
    pairs = (
        pd.concat([
            matchup_df[['TEAM_ID_home', 'SEASON']].rename(columns={'TEAM_ID_home': 'TEAM_ID'}),
            matchup_df[['TEAM_ID_away', 'SEASON']].rename(columns={'TEAM_ID_away': 'TEAM_ID'}),
        ])
        .drop_duplicates()
    )
    team_season_pairs = list(zip(pairs['TEAM_ID'].astype(int), pairs['SEASON']))

    coach_map = frc.fetch_coach_season_map(team_season_pairs)
    coach_map['TEAM_ID'] = coach_map['TEAM_ID'].astype(int)

    # Attach coach name to game log rows
    tdf = tdf.merge(coach_map[['TEAM_ID', 'SEASON', 'COACH_NAME']],
                    on=['TEAM_ID', 'SEASON'], how='left')
    tdf['COACH_NAME'] = tdf['COACH_NAME'].fillna('__UNKNOWN__')

    career = _build_coach_career(tdf)
    joblib.dump(career, CACHE_COACH_STATS)

    # Per-(TEAM_ID, GAME_ID) lookup: join coach name -> career stats
    team_game_coach = (
        tdf[['TEAM_ID', 'GAME_ID', 'COACH_NAME']]
        .drop_duplicates(['TEAM_ID', 'GAME_ID'])
        .merge(career, on=['COACH_NAME', 'GAME_ID'], how='left')
    )

    out = matchup_df.copy()
    coach_stat_cols = [
        'coach_win_pct_home', 'coach_win_pct_away',
        'coach_win_pct_playoffs', 'coach_ats_home', 'coach_experience',
    ]
    for side in ('home', 'away'):
        team_col = f'TEAM_ID_{side}'
        sub = (
            team_game_coach[['TEAM_ID', 'GAME_ID'] + coach_stat_cols]
            .rename(columns={
                'TEAM_ID': team_col,
                **{c: f'{c}_{side}' for c in coach_stat_cols},
            })
        )
        out = out.merge(sub, on=['GAME_ID', team_col], how='left')

    out['coach_winpct_gap']  = out['coach_win_pct_home_home']     - out['coach_win_pct_home_away']
    out['coach_playoff_gap'] = out['coach_win_pct_playoffs_home'] - out['coach_win_pct_playoffs_away']
    out['coach_exp_gap']     = out['coach_experience_home']        - out['coach_experience_away']

    all_coach_cols = (
        [f'{c}_{side}' for c in coach_stat_cols for side in ('home', 'away')]
        + ['coach_winpct_gap', 'coach_playoff_gap', 'coach_exp_gap']
    )
    for col in all_coach_cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = out[col].fillna(out[col].median())

    coverage = out['coach_experience_home'].notna().mean()
    console.print(f"  [cyan]Coach features added ({coverage:.1%} coverage)")
    return out


def _build_coach_career(tdf: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pre-game expanding career stats per coach (shift-1 for leakage prevention).
    Career stats span teams — a coach who moves carries their full history.

    Returns DataFrame[COACH_NAME, GAME_ID, coach_win_pct_home, coach_win_pct_away,
                      coach_win_pct_playoffs, coach_ats_home, coach_experience].
    """
    tdf = (
        tdf[tdf['COACH_NAME'] != '__UNKNOWN__']
        .drop_duplicates(['COACH_NAME', 'GAME_ID'])
        .sort_values(['COACH_NAME', 'GAME_DATE'])
        .reset_index(drop=True)
        .copy()
    )

    # Binary indicator columns for accumulating split numerators/denominators
    tdf['_is_home']    = (tdf['IS_HOME'] == 1).astype(float)
    tdf['_is_away']    = (tdf['IS_HOME'] == 0).astype(float)
    tdf['_is_playoff'] = tdf['IS_PLAYOFFS'].astype(float)
    tdf['_is_b2b']     = tdf['IS_BACK_TO_BACK'].astype(float)
    tdf['_win_home']   = tdf['WIN'] * tdf['_is_home']
    tdf['_win_away']   = tdf['WIN'] * tdf['_is_away']
    tdf['_win_po']     = tdf['WIN'] * tdf['_is_playoff']
    tdf['_win_b2b']    = tdf['WIN'] * tdf['_is_b2b']

    num_cols = ['_win_home', '_win_away', '_win_po', '_win_b2b',
                '_is_home',  '_is_away',  '_is_playoff', '_is_b2b']
    grp = tdf.groupby('COACH_NAME', sort=False)
    for col in num_cols:
        tdf[f'{col}_cum'] = grp[col].transform(
            lambda x: x.expanding().sum().shift(1)
        )

    tdf['coach_experience'] = grp.cumcount().astype(float)

    def safe_div(num_col, den_col):
        return tdf[num_col] / tdf[den_col].replace(0, np.nan)

    tdf['coach_win_pct_home']     = safe_div('_win_home_cum', '_is_home_cum')
    tdf['coach_win_pct_away']     = safe_div('_win_away_cum', '_is_away_cum')
    tdf['coach_win_pct_playoffs'] = safe_div('_win_po_cum',   '_is_playoff_cum')
    tdf['coach_ats_home']         = safe_div('_win_b2b_cum',  '_is_b2b_cum')

    return tdf[['COACH_NAME', 'GAME_ID',
                'coach_win_pct_home', 'coach_win_pct_away',
                'coach_win_pct_playoffs', 'coach_ats_home',
                'coach_experience']]

