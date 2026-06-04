"""
matchups.py — build game matchups and compute player-vs-player
positional matchup features.

Positional matchups answer your "strong PG vs strong forward" and
"good scorer vs DPOY-type defender" ideas:

  - Each team's top players are bucketed into GUARD / WING / BIG by play style.
  - Within each bucket, players are matched by offensive rank.
  - For each pairing we compute an offensive mismatch:
        home_player_offense  -  away_player_defense
    A big positive number = home player should dominate their man.
  - We also compute the "star vs best-defender" matchup: the home team's
    best scorer against the away team's best defender, and vice-versa.

These per-game matchup features are aggregated to the team level and
become gap features in the final matchup row.
"""

import numpy as np
import pandas as pd
import config as C


def _lineup_players(player_rolling):
    """Top-N players per (GAME_ID, TEAM_ID) with role/offense/defense."""
    df = player_rolling.sort_values(
        ['GAME_ID','TEAM_ID','MIN_FLOAT'], ascending=[True, True, False])
    top = df.groupby(['GAME_ID','TEAM_ID']).head(C.TOP_N_PLAYERS).copy()
    top['OFF'] = top.get('IMPACT_roll10', top.get('PTS_roll10', 0)).fillna(0)
    top['DEF'] = top.get('DEF_SCORE_roll10', 0)
    if 'DEF' not in top.columns or top['DEF'].isna().all():
        top['DEF'] = (2*top.get('STL_roll10',0).fillna(0)
                      + 2*top.get('BLK_roll10',0).fillna(0)
                      + 0.4*top.get('DREB_roll10',0).fillna(0))
    top['DEF'] = top['DEF'].fillna(0)
    top['ROLE'] = top.get('ROLE', 'WING')
    return top[['GAME_ID','TEAM_ID','PLAYER_ID','ROLE','OFF','DEF','PTS_roll10']]


def compute_matchup_features(player_rolling, team_meta):
    """
    team_meta: DataFrame with [GAME_ID, TEAM_ID, IS_HOME] to know which team is home.
    Returns a DataFrame keyed by GAME_ID with home-perspective matchup features.
    """
    lp = _lineup_players(player_rolling)
    meta = team_meta[['GAME_ID','TEAM_ID','IS_HOME']].drop_duplicates()
    lp = lp.merge(meta, on=['GAME_ID','TEAM_ID'], how='left')

    rows = []
    for gid, g in lp.groupby('GAME_ID'):
        home = g[g['IS_HOME'] == 1]
        away = g[g['IS_HOME'] == 0]
        if len(home) < 3 or len(away) < 3:
            continue
        rows.append(_one_game_matchup(gid, home, away))
    return pd.DataFrame([r for r in rows if r is not None])


def _match_by_role(home, away):
    """Pair home & away players within role buckets by offensive rank."""
    pairs = []
    for role in ['GUARD', 'WING', 'BIG']:
        h = home[home['ROLE'] == role].sort_values('OFF', ascending=False)
        a = away[away['ROLE'] == role].sort_values('OFF', ascending=False)
        n = min(len(h), len(a))
        for i in range(n):
            hp, ap = h.iloc[i], a.iloc[i]
            pairs.append((hp, ap))
    # Any leftover unmatched players -> match across remaining by offense rank
    matched_h = sum(min(len(home[home['ROLE']==r]), len(away[away['ROLE']==r]))
                    for r in ['GUARD','WING','BIG'])
    if matched_h < min(len(home), len(away)):
        h_left = home.sort_values('OFF', ascending=False).iloc[matched_h:]
        a_left = away.sort_values('OFF', ascending=False).iloc[matched_h:]
        for i in range(min(len(h_left), len(a_left))):
            pairs.append((h_left.iloc[i], a_left.iloc[i]))
    return pairs


def _one_game_matchup(gid, home, away):
    pairs = _match_by_role(home, away)
    if not pairs:
        return None

    # Offensive mismatches (home perspective): home offense vs away defense
    home_off_mismatch = [hp['OFF'] - ap['DEF'] for hp, ap in pairs]
    away_off_mismatch = [ap['OFF'] - hp['DEF'] for hp, ap in pairs]

    # Star vs best defender
    home_star = home.loc[home['OFF'].idxmax()]
    away_star = away.loc[away['OFF'].idxmax()]
    home_best_def = home.loc[home['DEF'].idxmax()]
    away_best_def = away.loc[away['DEF'].idxmax()]

    return {
        'GAME_ID': gid,
        # average mismatch advantage across all positional pairings
        'mu_off_mismatch_home': float(np.mean(home_off_mismatch)),
        'mu_off_mismatch_away': float(np.mean(away_off_mismatch)),
        # biggest single advantage either team has
        'mu_best_mismatch_home': float(np.max(home_off_mismatch)),
        'mu_best_mismatch_away': float(np.max(away_off_mismatch)),
        # biggest disadvantage (most negative)
        'mu_worst_mismatch_home': float(np.min(home_off_mismatch)),
        # star scorer vs opponent's best defender
        'mu_home_star_vs_def': float(home_star['OFF'] - away_best_def['DEF']),
        'mu_away_star_vs_def': float(away_star['OFF'] - home_best_def['DEF']),
        # raw best-defender quality (DPOY-type presence)
        'mu_home_best_def': float(home_best_def['DEF']),
        'mu_away_best_def': float(away_best_def['DEF']),
        # role-strength edges: who has the stronger guards / wings / bigs
        'mu_guard_edge_home': _role_edge(home, away, 'GUARD'),
        'mu_wing_edge_home':  _role_edge(home, away, 'WING'),
        'mu_big_edge_home':   _role_edge(home, away, 'BIG'),
    }


def _role_edge(home, away, role):
    h = home[home['ROLE'] == role]['OFF'].mean()
    a = away[away['ROLE'] == role]['OFF'].mean()
    h = 0 if pd.isna(h) else h
    a = 0 if pd.isna(a) else a
    return float(h - a)


def merge_lineup_onto_team(team_df, lineup_features):
    return team_df.merge(lineup_features, on=['GAME_ID', 'TEAM_ID'], how='left')


def build_matchups(df, matchup_feats=None):
    home = df[df['IS_HOME'] == 1].copy()
    away = df[df['IS_HOME'] == 0].copy()
    m = home.merge(away, on=['GAME_ID','SEASON','IS_PLAYOFFS'], suffixes=('_home','_away'))
    m['HOME_WIN'] = m['WIN_home']

    # Gap features between the two teams
    gap_pairs = [
        ('lineup_pts_max','star_pts_gap'),('lineup_pm_mean','lineup_pm_gap'),
        ('lineup_winrate_mean','lineup_wr_gap'),('lineup_impact_mean','impact_gap'),
        ('lineup_def_mean','def_gap'),('lineup_gs_mean','gs_gap'),
        ('EFG_PCT_roll10','efg_gap'),('FT_RATE_roll10','ft_rate_gap'),
        ('FG3_RATE_roll10','fg3_rate_gap'),('POSS_roll10','pace_gap'),
        ('PM_TREND','trend_gap'),('WINRATE_TREND','wr_trend_gap'),
        ('WIN_STREAK','streak_gap'),('TRAVEL_KM','travel_gap'),
        ('PRIOR_OPP_FG_PCT','def_quality_gap'),
        ('pbp_max_run_roll10','run_gap'),
        # Explicit differentials
        ('NET_RTG_roll10','net_rtg_gap'),
        ('REST_DAYS','rest_days_gap'),
        ('TS_PCT_roll10','ts_pct_gap'),
        ('WINRATE_roll10','winrate_roll10_gap'),
        # Opponent-adjusted differentials
        ('OPP_ADJ_OFF_RTG_roll10','opp_adj_off_rtg_gap'),
        ('OPP_ADJ_NET_RTG_roll10','opp_adj_net_rtg_gap'),
        # Star availability differential
        ('lineup_full_star_max','star_avail_gap'),
        # Schedule stress differentials
        ('GAMES_LAST_7','games_last_7_gap'),
        ('ROAD_STREAK','road_streak_gap'),
    ]
    for base, gap in gap_pairs:
        h, a = f'{base}_home', f'{base}_away'
        m[gap] = (m[h] - m[a]) if (h in m.columns and a in m.columns) else 0

    # Attach play-by-play matchup features (home perspective, game-level)
    if matchup_feats is not None and not matchup_feats.empty:
        m = m.merge(matchup_feats, on='GAME_ID', how='left')

    return m


def add_h2h(matchups):
    m = matchups.sort_values('GAME_DATE_home').copy()
    m['H2H_HOME_WIN'] = (
        m.groupby(['TEAM_ABBREVIATION_home','TEAM_ABBREVIATION_away'])['HOME_WIN']
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean()).fillna(0.5))
    m['H2H_MEETINGS_THIS_SEASON'] = (
        m.groupby(['SEASON','TEAM_ABBREVIATION_home','TEAM_ABBREVIATION_away']).cumcount())
    return m
