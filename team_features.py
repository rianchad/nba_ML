"""
team_features.py — team-level feature engineering.

Includes everything from the previous version plus:
  - Scoring splits (paint, fast break, second chance, off turnovers) [prior season]
  - Opponent/defensive stats [prior season]
  - Team clutch + hustle stats [prior season]
  - Shot-zone efficiency [prior season]
  - Travel distance + time-zone change (computed from arena coords)
  - Play-by-play rolling features (scoring runs, quarter splits, comebacks)
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, asin, sqrt
import config as C


def _roll(g, w, mp):
    return g.transform(lambda x: x.shift(1).rolling(w, min_periods=mp).mean())


def initial_prep(df):
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['IS_HOME'] = df['MATCHUP'].str.contains(r'vs\.').astype(int)
    df['WIN'] = (df['WL'] == 'W').astype(int)
    df['IS_ALTITUDE'] = (df['IS_HOME'].eq(1) &
                         df['TEAM_ABBREVIATION'].isin(C.ALTITUDE_TEAMS)).astype(int)
    # Opponent abbreviation from matchup string ("BOS vs. MIA" / "BOS @ MIA")
    df['OPP_ABBR'] = df['MATCHUP'].str.extract(r'(?:vs\.|@)\s+([A-Z]{3})')
    return df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)


def add_prior_season_column(df):
    order = sorted(df['SEASON'].unique())
    prev = {s: order[i-1] if i > 0 else None for i, s in enumerate(order)}
    df['PRIOR_SEASON'] = df['SEASON'].map(prev)
    return df


def add_rest_days(df):
    df['REST_DAYS'] = (df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(3).clip(upper=10))
    df['IS_BACK_TO_BACK'] = (df['REST_DAYS'] == 1).astype(int)
    return df


def add_schedule_stress(df):
    """Games played in last 7/14 days; consecutive road/home streaks entering each game."""
    df = df.sort_values(['TEAM_ID', 'GAME_DATE']).copy()

    new_7, new_14, road_st, home_st = {}, {}, {}, {}

    for _, g in df.groupby('TEAM_ID'):
        g = g.sort_values('GAME_DATE')
        dates   = pd.to_datetime(g['GAME_DATE']).values
        is_home = g['IS_HOME'].values
        idx     = g.index.values
        w7      = np.timedelta64(7, 'D')
        w14     = np.timedelta64(14, 'D')

        for i in range(len(g)):
            dt   = dates[i]
            past = dates[:i]
            new_7[idx[i]]  = int(((past > dt - w7)  & (past < dt)).sum())
            new_14[idx[i]] = int(((past > dt - w14) & (past < dt)).sum())

            rs = hs = 0
            for j in range(i - 1, -1, -1):
                if is_home[j] == 0: rs += 1
                else: break
            for j in range(i - 1, -1, -1):
                if is_home[j] == 1: hs += 1
                else: break
            road_st[idx[i]] = rs
            home_st[idx[i]] = hs

    df['GAMES_LAST_7']  = pd.Series(new_7)
    df['GAMES_LAST_14'] = pd.Series(new_14)
    df['ROAD_STREAK']   = pd.Series(road_st)
    df['HOME_STREAK']   = pd.Series(home_st)
    return df


def add_travel(df):
    """Distance (km) and timezone change from previous game's arena."""
    df = df.sort_values(['TEAM_ID', 'GAME_DATE']).copy()

    def venue(row):
        # Home games are played in the team's own arena; away in opponent's
        ab = row['TEAM_ABBREVIATION'] if row['IS_HOME'] == 1 else row['OPP_ABBR']
        return ab
    df['VENUE'] = df.apply(venue, axis=1)

    def haversine(a, b):
        if a not in C.ARENA_COORDS or b not in C.ARENA_COORDS:
            return np.nan
        (lat1, lon1), (lat2, lon2) = C.ARENA_COORDS[a], C.ARENA_COORDS[b]
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2-lat1, lon2-lon1
        h = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return 2*6371*asin(sqrt(h))

    df['PREV_VENUE'] = df.groupby('TEAM_ID')['VENUE'].shift(1)
    df['TRAVEL_KM'] = df.apply(
        lambda r: haversine(r['PREV_VENUE'], r['VENUE'])
        if pd.notna(r['PREV_VENUE']) else 0.0, axis=1).fillna(0.0)

    def tz_change(r):
        a, b = r['PREV_VENUE'], r['VENUE']
        if pd.isna(a) or a not in C.ARENA_TZ or b not in C.ARENA_TZ:
            return 0
        return C.ARENA_TZ[b] - C.ARENA_TZ[a]
    df['TZ_CHANGE'] = df.apply(tz_change, axis=1)
    return df


def add_time_features(df):
    df['MONTH'] = df['GAME_DATE'].dt.month
    df['DAY_OF_WEEK'] = df['GAME_DATE'].dt.dayofweek
    df['IS_WEEKEND'] = (df['GAME_DATE'].dt.dayofweek >= 5).astype(int)
    df['GAMES_INTO_SEASON'] = df.groupby(['TEAM_ID', 'SEASON']).cumcount()
    totals = df.groupby(['TEAM_ID', 'SEASON'])['GAME_DATE'].transform('count')
    df['IS_LATE_SEASON'] = (df['GAMES_INTO_SEASON'] > totals - 20).astype(int)
    return df


def add_team_rolling(df):
    cols = ['PTS','FGA','FGM','FG3A','FG3M','FTA','FTM','FG_PCT','FG3_PCT','FT_PCT',
            'REB','OREB','DREB','AST','TOV','STL','BLK','PF','PLUS_MINUS']
    for col in cols:
        if col not in df.columns: continue
        g = df.groupby('TEAM_ID')[col]
        df[f'{col}_roll10'] = _roll(g, C.ROLLING_LONG, 3)
        df[f'{col}_roll5']  = _roll(g, C.ROLLING_SHORT, 2)
        df[f'{col}_roll20'] = _roll(g, C.ROLLING_XLONG, 5)
    for w, mp in [(3,1),(5,2),(10,3),(20,5)]:
        df[f'WINRATE_roll{w}'] = df.groupby('TEAM_ID')['WIN'].transform(
            lambda x, w=w, mp=mp: x.shift(1).rolling(w, min_periods=mp).mean())
    return df


def add_pbp_rolling(df):
    """Roll the per-game play-by-play features (so they reflect current form)."""
    pbp_cols = [c for c in df.columns if c.startswith('pbp_')]
    for col in pbp_cols:
        df[f'{col}_roll10'] = df.groupby('TEAM_ID')[col].transform(
            lambda x: x.shift(1).rolling(C.ROLLING_LONG, min_periods=3).mean())
    return df


def add_win_streak(df):
    def streak_fn(wins):
        out, cur = [], 0
        for w in wins.shift(1).fillna(0.5):
            if w == 1: cur = cur+1 if cur > 0 else 1
            elif w == 0: cur = cur-1 if cur < 0 else -1
            else: cur = 0
            out.append(cur)
        return pd.Series(out, index=wins.index)
    df['WIN_STREAK'] = df.groupby('TEAM_ID')['WIN'].transform(streak_fn)
    df['LAST_GAME_PM'] = df.groupby('TEAM_ID')['PLUS_MINUS'].transform(lambda x: x.shift(1))
    df['LAST_GAME_WIN'] = df.groupby('TEAM_ID')['WIN'].transform(lambda x: x.shift(1))
    return df


def add_four_factors(df):
    eps = 1e-6
    if {'FGM_roll10','FG3M_roll10','FGA_roll10'}.issubset(df.columns):
        df['EFG_PCT_roll10'] = (df['FGM_roll10']+0.5*df['FG3M_roll10'])/(df['FGA_roll10']+eps)
    if {'TOV_roll10','FGA_roll10','FTA_roll10'}.issubset(df.columns):
        poss = df['FGA_roll10']+0.44*df['FTA_roll10']+df['TOV_roll10']
        df['TOV_PCT_roll10'] = df['TOV_roll10']/(poss+eps)
    if {'OREB_roll10','DREB_roll10'}.issubset(df.columns):
        tot = df['OREB_roll10']+df['DREB_roll10']
        df['OREB_RATE_roll10'] = df['OREB_roll10']/(tot+eps)
        df['DREB_RATE_roll10'] = df['DREB_roll10']/(tot+eps)
    if {'FTA_roll10','FGA_roll10'}.issubset(df.columns):
        df['FT_RATE_roll10'] = df['FTA_roll10']/(df['FGA_roll10']+eps)
    if {'FG3A_roll10','FGA_roll10'}.issubset(df.columns):
        df['FG3_RATE_roll10'] = df['FG3A_roll10']/(df['FGA_roll10']+eps)
    return df


def add_trends(df):
    if {'PLUS_MINUS_roll5','PLUS_MINUS_roll20'}.issubset(df.columns):
        df['PM_TREND'] = df['PLUS_MINUS_roll5']-df['PLUS_MINUS_roll20']
    if {'PTS_roll5','PTS_roll20'}.issubset(df.columns):
        df['PTS_TREND'] = df['PTS_roll5']-df['PTS_roll20']
    if {'WINRATE_roll5','WINRATE_roll20'}.issubset(df.columns):
        df['WINRATE_TREND'] = df['WINRATE_roll5']-df['WINRATE_roll20']
    return df


def add_pace(df):
    eps = 1e-6
    if {'FGA_roll10','OREB_roll10','TOV_roll10','FTA_roll10'}.issubset(df.columns):
        df['POSS_roll10'] = df['FGA_roll10']-df['OREB_roll10']+df['TOV_roll10']+0.44*df['FTA_roll10']
        df['OFF_RTG_roll10'] = df['PTS_roll10']/(df['POSS_roll10']+eps)*100
        df['NET_RTG_roll10'] = df['PLUS_MINUS_roll10']/(df['POSS_roll10']/100+eps)
    return df


def add_derived(df):
    eps = 1e-6
    df['AST_TOV_roll10'] = df['AST_roll10']/(df['TOV_roll10']+eps)
    if {'PTS_roll10','FGA_roll10','FTA_roll10'}.issubset(df.columns):
        df['TS_PCT_roll10'] = df['PTS_roll10']/(2*(df['FGA_roll10']+0.44*df['FTA_roll10'])+eps)
    return df


def add_home_away_splits(df):
    for val, label in [(1,'home'),(0,'away')]:
        mask = df['IS_HOME'] == val; col = f'WINRATE_{label}_roll10'
        df.loc[mask, col] = (df[mask].groupby('TEAM_ID')['WIN']
            .transform(lambda x: x.shift(1).rolling(C.ROLLING_LONG, min_periods=3).mean()))
        df[col] = df.groupby('TEAM_ID')[col].transform(lambda x: x.ffill())
    return df


def add_opp_adjusted(df):
    """Opponent-adjusted offensive/defensive ratings via rolling opponent quality faced."""
    avail = [c for c in ['NET_RTG_roll10', 'OFF_RTG_roll10', 'TS_PCT_roll10'] if c in df.columns]
    if not avail:
        return df

    opp = df[['GAME_ID', 'TEAM_ID'] + avail].copy()
    opp.columns = ['GAME_ID', 'OPP_ID'] + [f'OPP_{c}' for c in avail]

    df = df.merge(opp, on='GAME_ID', how='left')
    df = df[df['TEAM_ID'] != df['OPP_ID']].copy()
    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])

    if 'OPP_NET_RTG_roll10' in df.columns:
        opp_net_faced = df.groupby('TEAM_ID')['OPP_NET_RTG_roll10'].transform(
            lambda x: x.shift(1).rolling(C.ROLLING_LONG, min_periods=3).mean())
        df['OPP_NET_RTG_faced_roll10'] = opp_net_faced
        if 'OFF_RTG_roll10' in df.columns:
            df['OPP_ADJ_OFF_RTG_roll10'] = df['OFF_RTG_roll10'] - opp_net_faced
        if 'NET_RTG_roll10' in df.columns:
            df['OPP_ADJ_NET_RTG_roll10'] = df['NET_RTG_roll10'] - opp_net_faced

    if 'OPP_TS_PCT_roll10' in df.columns:
        opp_ts_faced = df.groupby('TEAM_ID')['OPP_TS_PCT_roll10'].transform(
            lambda x: x.shift(1).rolling(C.ROLLING_LONG, min_periods=3).mean())
        if 'TS_PCT_roll10' in df.columns:
            df['OPP_ADJ_TS_PCT_roll10'] = df['TS_PCT_roll10'] - opp_ts_faced

    drop = ['OPP_ID'] + [f'OPP_{c}' for c in avail]
    return df.drop(columns=drop, errors='ignore')


def add_sos(df):
    opp = df[['GAME_ID','TEAM_ID','WINRATE_roll10']].copy()
    opp.columns = ['GAME_ID','OPP_ID','OPP_WR']
    df = df.merge(opp, on='GAME_ID', how='left')
    df = df[df['TEAM_ID'] != df['OPP_ID']].copy()
    df = df.sort_values(['TEAM_ID','GAME_DATE'])
    df['SOS_roll10'] = df.groupby('TEAM_ID')['OPP_WR'].transform(
        lambda x: x.shift(1).rolling(C.ROLLING_LONG, min_periods=3).mean())
    return df.drop(columns=['OPP_ID','OPP_WR'])


def _merge_prior_team(df, src, rename_map):
    """Merge a per-team season-aggregate using PRIOR season (leak-free)."""
    if src is None or src.empty: return df
    keep = ['TEAM_ID','SEASON'] + [c for c in rename_map if c in src.columns]
    if len(keep) <= 2: return df
    s = src[keep].rename(columns={**rename_map, 'SEASON':'PRIOR_SEASON'})
    return df.merge(s, on=['TEAM_ID','PRIOR_SEASON'], how='left')


def merge_season_sources(df, advanced=None, scoring=None, misc=None,
                         opponent=None, clutch_t=None, hustle_t=None, shotzone=None):
    """Merge all season-level team aggregates as PRIOR-season features."""
    if advanced is not None:
        df = _merge_prior_team(df, advanced, {
            'OFF_RATING':'PRIOR_OFF_RATING','DEF_RATING':'PRIOR_DEF_RATING',
            'NET_RATING':'PRIOR_NET_RATING','PACE':'PRIOR_PACE'})
    if scoring is not None:
        df = _merge_prior_team(df, scoring, {
            'PCT_PTS_PAINT':'PRIOR_PCT_PTS_PAINT','PCT_PTS_FB':'PRIOR_PCT_PTS_FB',
            'PCT_PTS_2PT_MR':'PRIOR_PCT_PTS_MIDRANGE','PCT_PTS_3PT':'PRIOR_PCT_PTS_3PT',
            'PCT_AST_FGM':'PRIOR_PCT_AST_FGM'})
    if misc is not None:
        df = _merge_prior_team(df, misc, {
            'PTS_OFF_TOV':'PRIOR_PTS_OFF_TOV','PTS_2ND_CHANCE':'PRIOR_PTS_2ND_CHANCE',
            'PTS_FB':'PRIOR_PTS_FB','PTS_PAINT':'PRIOR_PTS_PAINT',
            'OPP_PTS_OFF_TOV':'PRIOR_OPP_PTS_OFF_TOV','OPP_PTS_PAINT':'PRIOR_OPP_PTS_PAINT'})
    if opponent is not None:
        df = _merge_prior_team(df, opponent, {
            'OPP_FG_PCT':'PRIOR_OPP_FG_PCT','OPP_FG3_PCT':'PRIOR_OPP_FG3_PCT',
            'OPP_PTS':'PRIOR_OPP_PTS','OPP_REB':'PRIOR_OPP_REB','OPP_AST':'PRIOR_OPP_AST'})
    if clutch_t is not None:
        df = _merge_prior_team(df, clutch_t, {
            'W_PCT':'PRIOR_CLUTCH_WPCT','PLUS_MINUS':'PRIOR_CLUTCH_PM',
            'FG_PCT':'PRIOR_CLUTCH_FG_PCT','PTS':'PRIOR_CLUTCH_PTS'})
    if hustle_t is not None:
        df = _merge_prior_team(df, hustle_t, {
            'DEFLECTIONS':'PRIOR_DEFLECTIONS','CHARGES_DRAWN':'PRIOR_CHARGES',
            'SCREEN_ASSISTS':'PRIOR_SCREEN_AST','CONTESTED_SHOTS':'PRIOR_CONTESTED',
            'LOOSE_BALLS_RECOVERED':'PRIOR_LOOSE_BALLS'})
    if shotzone is not None:
        # Shot zone columns are flattened like 'Restricted Area_FG_PCT'
        rename = {}
        for c in shotzone.columns:
            if 'Restricted' in c and 'FG_PCT' in c: rename[c] = 'PRIOR_RA_FG_PCT'
            elif 'Mid-Range' in c and 'FG_PCT' in c: rename[c] = 'PRIOR_MR_FG_PCT'
            elif 'Corner 3' in c and 'FG_PCT' in c: rename[c] = 'PRIOR_C3_FG_PCT'
            elif 'Above the Break 3' in c and 'FG_PCT' in c: rename[c] = 'PRIOR_AB3_FG_PCT'
        if rename:
            df = _merge_prior_team(df, shotzone, rename)
    return df


def engineer_team_features(team_df, sources):
    """sources: dict possibly containing advanced/scoring/misc/opponent/clutch_t/hustle_t/shotzone."""
    from rich.console import Console
    console = Console()
    pipeline = [
        ('Initial prep', initial_prep),
        ('Prior season column', add_prior_season_column),
        ('Rest days', add_rest_days),
        ('Schedule stress', add_schedule_stress),
        ('Travel & timezone', add_travel),
        ('Time features', add_time_features),
        ('Team rolling', add_team_rolling),
        ('Play-by-play rolling', add_pbp_rolling),
        ('Win streak', add_win_streak),
        ('Four Factors', add_four_factors),
        ('Trends', add_trends),
        ('Pace', add_pace),
        ('Derived', add_derived),
        ('Home/away splits', add_home_away_splits),
        ('Strength of schedule', add_sos),
        ('Opponent-adjusted stats', add_opp_adjusted),
    ]
    df = team_df
    defrag_after = {'Play-by-play rolling', 'Win streak', 'Opponent-adjusted stats'}
    for name, fn in pipeline:
        console.print(f"  [cyan]{name}...")
        df = fn(df)
        if name in defrag_after:
            df = df.copy()
    console.print("  [cyan]Merging season-level sources (prior season)...")
    df = merge_season_sources(
        df,
        advanced=sources.get('advanced'), scoring=sources.get('scoring'),
        misc=sources.get('misc'), opponent=sources.get('opponent'),
        clutch_t=sources.get('clutch_t'), hustle_t=sources.get('hustle_t'),
        shotzone=sources.get('shotzone'))
    return df