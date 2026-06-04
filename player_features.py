"""
player_features.py — per-player rolling stats, role classification,
defensive rating proxy, and lineup aggregation.

Player ROLE (Guard / Wing / Big) is derived from playing style so it works
for all 10 seasons with zero extra API calls:
  - high assists, low rebounds        -> Guard
  - high rebounds + blocks, low assist -> Big
  - in between                          -> Wing

A defensive rating proxy (DEF_SCORE) is built from steals, blocks,
defensive rebounds, and on-court plus/minus. Combined with bio/hustle/
tracking season aggregates when available.
"""

import numpy as np
import pandas as pd
import config as C
import player_advanced_features as paf


def parse_min(val):
    if pd.isna(val): return 0.0
    s = str(val)
    if ':' in s:
        p = s.split(':'); return float(p[0]) + float(p[1]) / 60
    try: return float(s)
    except: return 0.0


def _roll(g, w, mp):
    return g.transform(lambda x: x.shift(1).rolling(w, min_periods=mp).mean())


def compute_player_rolling(player_df, bio_df=None, hustle_p=None,
                           clutch_p=None, tracking=None, player_clutch_df=None,
                           player_shotzone_df=None):
    df = player_df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['MIN_FLOAT'] = df['MIN'].apply(parse_min)
    df['WIN'] = (df['WL'] == 'W').astype(int)
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

    roll_cols = ['PTS','AST','REB','OREB','DREB','STL','BLK','TOV','PF',
                 'FGA','FGM','FG3A','FG3M','FTA','FTM',
                 'FG_PCT','FG3_PCT','FT_PCT','PLUS_MINUS','MIN_FLOAT']
    for col in roll_cols:
        if col not in df.columns: continue
        g = df.groupby('PLAYER_ID')[col]
        df[f'{col}_roll10'] = _roll(g, C.ROLLING_LONG, 3)
        df[f'{col}_roll5']  = _roll(g, C.ROLLING_SHORT, 2)

    for w, mp in [(10, 3), (5, 2)]:
        df[f'WINRATE_roll{w}'] = df.groupby('PLAYER_ID')['WIN'].transform(
            lambda x, w=w, mp=mp: x.shift(1).rolling(w, min_periods=mp).mean())

    eps = 1e-6
    if {'PTS_roll10','FGA_roll10','FTA_roll10'}.issubset(df.columns):
        df['TS_PCT_roll10'] = df['PTS_roll10'] / (2*(df['FGA_roll10']+0.44*df['FTA_roll10'])+eps)
        df['PPS_roll10']    = df['PTS_roll10'] / (df['FGA_roll10']+0.44*df['FTA_roll10']+eps)

    impact_need = {'PTS_roll10','AST_roll10','REB_roll10','STL_roll10','BLK_roll10','TOV_roll10'}
    if impact_need.issubset(df.columns):
        df['IMPACT_roll10'] = (df['PTS_roll10']+1.5*df['AST_roll10']+1.2*df['REB_roll10']
                               +2.0*df['STL_roll10']+2.0*df['BLK_roll10']-df['TOV_roll10'])

    # Defensive proxy: steals + blocks + defensive boards + on-court impact
    def_need = {'STL_roll10','BLK_roll10','DREB_roll10','PLUS_MINUS_roll10'}
    if def_need.issubset(df.columns):
        df['DEF_SCORE_roll10'] = (2.0*df['STL_roll10'] + 2.0*df['BLK_roll10']
                                  + 0.4*df['DREB_roll10'] + 0.3*df['PLUS_MINUS_roll10'])

    # Hollinger game score
    gs_need = {'PTS','FGM','FGA','FTM','FTA','OREB','DREB','STL','AST','BLK','TOV','PF'}
    if gs_need.issubset(df.columns):
        df['GAME_SCORE'] = (df['PTS']+0.4*df['FGM']-0.7*df['FGA']
                            -0.4*(df['FTA']-df['FTM'])+0.7*df['OREB']+0.3*df['DREB']
                            +df['STL']+0.7*df['AST']+0.7*df['BLK']-0.4*df['PF']-df['TOV'])
        df['GAME_SCORE_roll10'] = df.groupby('PLAYER_ID')['GAME_SCORE'].transform(
            lambda x: x.shift(1).rolling(C.ROLLING_LONG, min_periods=3).mean())

    # Consistency
    df['PTS_STD_roll10'] = df.groupby('PLAYER_ID')['PTS'].transform(
        lambda x: x.shift(1).rolling(C.ROLLING_LONG, min_periods=5).std())
    df['PTS_CV_roll10'] = df['PTS_STD_roll10'] / (df['PTS_roll10'] + eps)

    if {'FGA_roll10','FTA_roll10','TOV_roll10'}.issubset(df.columns):
        df['USAGE_PROXY_roll10'] = df['FGA_roll10']+0.44*df['FTA_roll10']+df['TOV_roll10']

    # ── Player ROLE from rolling style ──
    df['ROLE'] = _classify_role(df)

    # ── Merge prior-season aggregate sources (leak-free) ──
    df = _merge_prior_season_player(df, bio_df, 'bio',
                                    ['AGE','PLAYER_HEIGHT_INCHES','PLAYER_WEIGHT'])
    if hustle_p is not None and not hustle_p.empty:
        hustle_cols = [c for c in ['DEFLECTIONS','CHARGES_DRAWN','SCREEN_ASSISTS',
                                   'LOOSE_BALLS_RECOVERED','CONTESTED_SHOTS']
                       if c in hustle_p.columns]
        df = _merge_prior_season_player(df, hustle_p, 'hustle', hustle_cols)
    if clutch_p is not None and not clutch_p.empty:
        clutch_cols = [c for c in ['PTS','PLUS_MINUS','FG_PCT'] if c in clutch_p.columns]
        df = _merge_prior_season_player(df, clutch_p, 'clutch', clutch_cols)

    # ── Advanced player features ──
    df = paf.add_shot_zone_quality(df, player_shotzone_df)
    df = paf.add_foul_trouble_rate(df, player_df)
    df = paf.add_clutch_rolling(df, player_clutch_df)

    return df


def _classify_role(df):
    """Guard / Wing / Big from rolling per-game profile."""
    ast = df.get('AST_roll10', pd.Series(0, index=df.index)).fillna(0)
    reb = df.get('REB_roll10', pd.Series(0, index=df.index)).fillna(0)
    blk = df.get('BLK_roll10', pd.Series(0, index=df.index)).fillna(0)
    big_score   = reb + 3*blk
    guard_score = ast * 2
    role = np.where((big_score > 7) & (guard_score < 8), 'BIG',
            np.where(guard_score > 8, 'GUARD', 'WING'))
    return role


def _add_prior_season(df_src):
    """Map each season to the prior season for leak-free merges."""
    order = sorted(df_src['SEASON'].unique())
    prev = {s: order[i-1] if i > 0 else None for i, s in enumerate(order)}
    return prev


def _merge_prior_season_player(df, src, tag, cols):
    """Merge a per-player season-aggregate source using PRIOR season values."""
    if src is None or src.empty or not cols:
        return df
    id_col = 'PLAYER_ID' if 'PLAYER_ID' in src.columns else None
    if id_col is None:
        return df
    keep = [id_col, 'SEASON'] + [c for c in cols if c in src.columns]
    s = src[keep].copy()
    s = s.rename(columns={c: f'{tag}_{c}' for c in cols if c in src.columns})
    s = s.rename(columns={'SEASON': 'PRIOR_SEASON'})
    # Build PRIOR_SEASON on df if not present
    if 'PLAYER_PRIOR_SEASON' not in df.columns:
        prev = _add_prior_season(df)
        df['PLAYER_PRIOR_SEASON'] = df['SEASON'].map(prev)
    df = df.merge(s, left_on=['PLAYER_ID', 'PLAYER_PRIOR_SEASON'],
                  right_on=[id_col, 'PRIOR_SEASON'], how='left')
    df = df.drop(columns=[c for c in ['PRIOR_SEASON', id_col+'_y'] if c in df.columns], errors='ignore')
    return df


def build_lineup_features(player_rolling):
    df = player_rolling.sort_values(
        ['GAME_ID','TEAM_ID','MIN_FLOAT'], ascending=[True, True, False])
    top_n = df.groupby(['GAME_ID','TEAM_ID']).head(C.TOP_N_PLAYERS)

    agg_spec = dict(
        lineup_pts_mean=('PTS_roll10','mean'), lineup_pts_max=('PTS_roll10','max'),
        lineup_pts_min=('PTS_roll10','min'),   lineup_pts_std=('PTS_roll10','std'),
        lineup_pts5_mean=('PTS_roll5','mean'),
        lineup_ast_mean=('AST_roll10','mean'), lineup_reb_mean=('REB_roll10','mean'),
        lineup_stl_mean=('STL_roll10','mean'), lineup_blk_mean=('BLK_roll10','mean'),
        lineup_tov_mean=('TOV_roll10','mean'),
        lineup_fg_pct_mean=('FG_PCT_roll10','mean'),
        lineup_fg3_pct_mean=('FG3_PCT_roll10','mean'),
        lineup_ft_pct_mean=('FT_PCT_roll10','mean'),
        lineup_pm_mean=('PLUS_MINUS_roll10','mean'), lineup_pm_max=('PLUS_MINUS_roll10','max'),
        lineup_pm5_mean=('PLUS_MINUS_roll5','mean'),
        lineup_winrate_mean=('WINRATE_roll10','mean'),
        lineup_ts_pct_mean=('TS_PCT_roll10','mean'),
        lineup_pps_mean=('PPS_roll10','mean'),
        lineup_impact_mean=('IMPACT_roll10','mean'), lineup_impact_max=('IMPACT_roll10','max'),
        lineup_def_mean=('DEF_SCORE_roll10','mean'), lineup_def_max=('DEF_SCORE_roll10','max'),
        lineup_gs_mean=('GAME_SCORE_roll10','mean'), lineup_gs_max=('GAME_SCORE_roll10','max'),
        lineup_consistency=('PTS_CV_roll10','mean'),
        lineup_usage_mean=('USAGE_PROXY_roll10','mean'),
        lineup_min_mean=('MIN_FLOAT_roll10','mean'),
    )
    # Only keep agg specs whose source column exists
    agg_spec = {k: v for k, v in agg_spec.items() if v[0] in top_n.columns}
    agg = top_n.groupby(['GAME_ID','TEAM_ID']).agg(**agg_spec).reset_index()

    # Total actual game minutes for top-5 (proxy for roster health / availability)
    min_total = top_n.groupby(['GAME_ID','TEAM_ID'])['MIN_FLOAT'].sum().reset_index()
    min_total = min_total.rename(columns={'MIN_FLOAT': 'lineup_min_total'})
    agg = agg.merge(min_total, on=['GAME_ID','TEAM_ID'], how='left')

    # Best available player's rolling PPG — wider than top-5 to detect missing stars
    active = df[df['MIN_FLOAT'] > 1]
    if 'PTS_roll10' in df.columns and not active.empty:
        star = (active.groupby(['GAME_ID','TEAM_ID'])['PTS_roll10']
                .max().reset_index()
                .rename(columns={'PTS_roll10': 'lineup_full_star_max'}))
        agg = agg.merge(star, on=['GAME_ID','TEAM_ID'], how='left')

    return agg
