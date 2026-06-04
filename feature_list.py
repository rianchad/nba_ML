"""
feature_list.py — the master list of model features.

Organized by group. The model uses whichever of these columns actually
exist in the matchup DataFrame (missing ones are skipped and median-filled),
so it's safe to list features whose source data may not be present for
every season.
"""

TEAM_FEATURES = [
    # Win rates
    'WINRATE_roll20_home','WINRATE_roll20_away',
    'WINRATE_roll10_home','WINRATE_roll10_away',
    'WINRATE_roll5_home','WINRATE_roll5_away',
    'WINRATE_roll3_home','WINRATE_roll3_away',
    'WINRATE_home_roll10_home','WINRATE_away_roll10_away',
    # Point differential
    'PLUS_MINUS_roll20_home','PLUS_MINUS_roll20_away',
    'PLUS_MINUS_roll10_home','PLUS_MINUS_roll10_away',
    'PLUS_MINUS_roll5_home','PLUS_MINUS_roll5_away',
    # Momentum
    'WIN_STREAK_home','WIN_STREAK_away',
    'LAST_GAME_PM_home','LAST_GAME_PM_away',
    'LAST_GAME_WIN_home','LAST_GAME_WIN_away',
    'PM_TREND_home','PM_TREND_away',
    'PTS_TREND_home','PTS_TREND_away',
    'WINRATE_TREND_home','WINRATE_TREND_away',
    # Shooting efficiency
    'TS_PCT_roll10_home','TS_PCT_roll10_away',
    'EFG_PCT_roll10_home','EFG_PCT_roll10_away',
    'AST_TOV_roll10_home','AST_TOV_roll10_away',
    # Four Factors
    'TOV_PCT_roll10_home','TOV_PCT_roll10_away',
    'OREB_RATE_roll10_home','OREB_RATE_roll10_away',
    'DREB_RATE_roll10_home','DREB_RATE_roll10_away',
    'FT_RATE_roll10_home','FT_RATE_roll10_away',
    'FG3_RATE_roll10_home','FG3_RATE_roll10_away',
    # Pace & ratings
    'OFF_RTG_roll10_home','OFF_RTG_roll10_away',
    'NET_RTG_roll10_home','NET_RTG_roll10_away',
    'POSS_roll10_home','POSS_roll10_away',
    # Opponent-adjusted ratings
    'OPP_ADJ_OFF_RTG_roll10_home','OPP_ADJ_OFF_RTG_roll10_away',
    'OPP_ADJ_NET_RTG_roll10_home','OPP_ADJ_NET_RTG_roll10_away',
    'OPP_ADJ_TS_PCT_roll10_home','OPP_ADJ_TS_PCT_roll10_away',
    'OPP_NET_RTG_faced_roll10_home','OPP_NET_RTG_faced_roll10_away',
    # Schedule / travel
    'SOS_roll10_home','SOS_roll10_away',
    'REST_DAYS_home','REST_DAYS_away',
    'IS_BACK_TO_BACK_home','IS_BACK_TO_BACK_away',
    'TRAVEL_KM_home','TRAVEL_KM_away',
    'TZ_CHANGE_home','TZ_CHANGE_away',
    # Schedule stress
    'GAMES_LAST_7_home','GAMES_LAST_7_away',
    'GAMES_LAST_14_home','GAMES_LAST_14_away',
    'ROAD_STREAK_home','ROAD_STREAK_away',
    'HOME_STREAK_home','HOME_STREAK_away',
    # Time
    'MONTH','DAY_OF_WEEK','IS_WEEKEND',
    'GAMES_INTO_SEASON_home','IS_LATE_SEASON_home',
    # Context
    'IS_PLAYOFFS','IS_ALTITUDE_home',
    'H2H_HOME_WIN','H2H_MEETINGS_THIS_SEASON',
    'PLAYOFF_ROUND','SERIES_GAME_NUM','IS_MUST_WIN',
    # Prior-season advanced
    'PRIOR_OFF_RATING_home','PRIOR_OFF_RATING_away',
    'PRIOR_DEF_RATING_home','PRIOR_DEF_RATING_away',
    'PRIOR_NET_RATING_home','PRIOR_NET_RATING_away',
    'PRIOR_PACE_home','PRIOR_PACE_away',
    # Prior-season scoring splits
    'PRIOR_PCT_PTS_PAINT_home','PRIOR_PCT_PTS_PAINT_away',
    'PRIOR_PCT_PTS_FB_home','PRIOR_PCT_PTS_FB_away',
    'PRIOR_PCT_PTS_3PT_home','PRIOR_PCT_PTS_3PT_away',
    'PRIOR_PCT_AST_FGM_home','PRIOR_PCT_AST_FGM_away',
    # Prior-season misc (points-generation splits)
    'PRIOR_PTS_OFF_TOV_home','PRIOR_PTS_OFF_TOV_away',
    'PRIOR_PTS_2ND_CHANCE_home','PRIOR_PTS_2ND_CHANCE_away',
    'PRIOR_PTS_FB_home','PRIOR_PTS_FB_away',
    'PRIOR_PTS_PAINT_home','PRIOR_PTS_PAINT_away',
    'PRIOR_OPP_PTS_OFF_TOV_home','PRIOR_OPP_PTS_OFF_TOV_away',
    'PRIOR_OPP_PTS_PAINT_home','PRIOR_OPP_PTS_PAINT_away',
    # Prior-season opponent / defense
    'PRIOR_OPP_FG_PCT_home','PRIOR_OPP_FG_PCT_away',
    'PRIOR_OPP_FG3_PCT_home','PRIOR_OPP_FG3_PCT_away',
    'PRIOR_OPP_PTS_home','PRIOR_OPP_PTS_away',
    'PRIOR_OPP_REB_home','PRIOR_OPP_REB_away',
    'PRIOR_OPP_AST_home','PRIOR_OPP_AST_away',
    # Prior-season clutch
    'PRIOR_CLUTCH_WPCT_home','PRIOR_CLUTCH_WPCT_away',
    'PRIOR_CLUTCH_PM_home','PRIOR_CLUTCH_PM_away',
    'PRIOR_CLUTCH_FG_PCT_home','PRIOR_CLUTCH_FG_PCT_away',
    # Prior-season hustle
    'PRIOR_DEFLECTIONS_home','PRIOR_DEFLECTIONS_away',
    'PRIOR_CHARGES_home','PRIOR_CHARGES_away',
    'PRIOR_SCREEN_AST_home','PRIOR_SCREEN_AST_away',
    'PRIOR_CONTESTED_home','PRIOR_CONTESTED_away',
    # Prior-season shot zones
    'PRIOR_RA_FG_PCT_home','PRIOR_RA_FG_PCT_away',
    'PRIOR_MR_FG_PCT_home','PRIOR_MR_FG_PCT_away',
    'PRIOR_C3_FG_PCT_home','PRIOR_C3_FG_PCT_away',
    'PRIOR_AB3_FG_PCT_home','PRIOR_AB3_FG_PCT_away',
]

LINEUP_FEATURES = [
    'lineup_pts_mean_home','lineup_pts_mean_away',
    'lineup_pts_max_home','lineup_pts_max_away',
    'lineup_pts_min_home','lineup_pts_min_away',
    'lineup_pts_std_home','lineup_pts_std_away',
    'lineup_pts5_mean_home','lineup_pts5_mean_away',
    'lineup_ast_mean_home','lineup_ast_mean_away',
    'lineup_tov_mean_home','lineup_tov_mean_away',
    'lineup_reb_mean_home','lineup_reb_mean_away',
    'lineup_stl_mean_home','lineup_stl_mean_away',
    'lineup_blk_mean_home','lineup_blk_mean_away',
    'lineup_fg_pct_mean_home','lineup_fg_pct_mean_away',
    'lineup_fg3_pct_mean_home','lineup_fg3_pct_mean_away',
    'lineup_ft_pct_mean_home','lineup_ft_pct_mean_away',
    'lineup_ts_pct_mean_home','lineup_ts_pct_mean_away',
    'lineup_pps_mean_home','lineup_pps_mean_away',
    'lineup_pm_mean_home','lineup_pm_mean_away',
    'lineup_pm_max_home','lineup_pm_max_away',
    'lineup_pm5_mean_home','lineup_pm5_mean_away',
    'lineup_winrate_mean_home','lineup_winrate_mean_away',
    'lineup_impact_mean_home','lineup_impact_mean_away',
    'lineup_impact_max_home','lineup_impact_max_away',
    'lineup_def_mean_home','lineup_def_mean_away',
    'lineup_def_max_home','lineup_def_max_away',
    'lineup_gs_mean_home','lineup_gs_mean_away',
    'lineup_gs_max_home','lineup_gs_max_away',
    'lineup_consistency_home','lineup_consistency_away',
    'lineup_usage_mean_home','lineup_usage_mean_away',
    'lineup_min_mean_home','lineup_min_mean_away',
    # Star availability (wider than top-5 minutes)
    'lineup_full_star_max_home','lineup_full_star_max_away',
    'lineup_min_total_home','lineup_min_total_away',
    # Gaps
    'star_pts_gap','lineup_pm_gap','lineup_wr_gap','impact_gap',
    'def_gap','gs_gap','efg_gap','ft_rate_gap','fg3_rate_gap',
    'pace_gap','trend_gap','wr_trend_gap','streak_gap','travel_gap',
    'def_quality_gap','run_gap',
    # Explicit differentials (new)
    'net_rtg_gap','rest_days_gap','ts_pct_gap','winrate_roll10_gap',
    # Opponent-adjusted differentials
    'opp_adj_off_rtg_gap','opp_adj_net_rtg_gap',
    # Star availability & schedule stress differentials
    'star_avail_gap','games_last_7_gap','road_streak_gap',
]

# Player-vs-player positional matchup features (game-level, home perspective)
MATCHUP_FEATURES = [
    'mu_off_mismatch_home','mu_off_mismatch_away',
    'mu_best_mismatch_home','mu_best_mismatch_away',
    'mu_worst_mismatch_home',
    'mu_home_star_vs_def','mu_away_star_vs_def',
    'mu_home_best_def','mu_away_best_def',
    'mu_guard_edge_home','mu_wing_edge_home','mu_big_edge_home',
]

# Play-by-play rolling features (home/away perspective)
PBP_FEATURES = [
    'pbp_max_run_roll10_home','pbp_max_run_roll10_away',
    'pbp_largest_lead_roll10_home','pbp_largest_lead_roll10_away',
    'pbp_comeback_roll10_home','pbp_comeback_roll10_away',
    'pbp_clutch_net_roll10_home','pbp_clutch_net_roll10_away',
    'pbp_q1_diff_roll10_home','pbp_q1_diff_roll10_away',
    'pbp_q2_diff_roll10_home','pbp_q2_diff_roll10_away',
    'pbp_q3_diff_roll10_home','pbp_q3_diff_roll10_away',
    'pbp_q4_diff_roll10_home','pbp_q4_diff_roll10_away',
    'pbp_lead_changes_roll10_home',
    'pbp_times_tied_roll10_home',
]

REF_FEATURES = [
    # Referee crew tendencies — single value per game (refs apply to both teams)
    'ref_home_win_pct',    # crew career home-team win %
    'ref_foul_rate',       # crew career avg total fouls per game
    'ref_pace_effect',     # crew career avg estimated possessions per game
    'ref_home_foul_bias',  # crew career avg (home fouls - away fouls)
]

COACH_FEATURES = [
    # Home team's head coach career splits
    'coach_win_pct_home_home',     # career home win %
    'coach_win_pct_away_home',     # career away win %
    'coach_win_pct_playoffs_home', # career playoff win %
    'coach_ats_home_home',         # career back-to-back win %
    'coach_experience_home',       # total games coached before this game
    # Away team's head coach career splits
    'coach_win_pct_home_away',
    'coach_win_pct_away_away',
    'coach_win_pct_playoffs_away',
    'coach_ats_home_away',
    'coach_experience_away',
    # Coach differentials (home minus away)
    'coach_winpct_gap',    # home coach home-win% minus away coach home-win%
    'coach_playoff_gap',   # playoff win % differential
    'coach_exp_gap',       # experience (games) differential
]

ADVANCED_LINEUP_FEATURES = [
    # On/off splits: net rating when starting 5 on court
    'lineup_onoff_nrtg_home', 'lineup_onoff_nrtg_away',
    'lineup_onoff_games_home', 'lineup_onoff_games_away',
    # Quarter-specific runs (max scoring run per quarter)
    'q1_max_run_home', 'q1_max_run_away',
    'q2_max_run_home', 'q2_max_run_away',
    'q3_max_run_home', 'q3_max_run_away',
    'q4_max_run_home', 'q4_max_run_away',
    # Q4-weighted run (higher weight on 4th quarter)
    'q4_run_weighted_home', 'q4_run_weighted_away',
    # Aggregated player advanced stats at lineup level
    'lineup_zone_efg_pct_roll10_home', 'lineup_zone_efg_pct_roll10_away',  # mean zone efficiency
    'lineup_foul_trouble_rate_home', 'lineup_foul_trouble_rate_away',       # foul trouble risk
    'lineup_clutch_fgm_roll10_home', 'lineup_clutch_fgm_roll10_away',       # clutch FGM
    'lineup_clutch_fta_roll10_home', 'lineup_clutch_fta_roll10_away',       # clutch FTA
]

ADVANCED_PLAYER_FEATURES = [
    # Shot-zone efficiency (per player, rolled 10 games)
    # (aggregated to lineup via ADVANCED_LINEUP_FEATURES)
    # Foul-trouble rate (per player, rolled 10 games)
    # (aggregated to lineup via ADVANCED_LINEUP_FEATURES)
    # Clutch contributions (per player, final 2 min close games, rolled 10 games)
    # (aggregated to lineup via ADVANCED_LINEUP_FEATURES)
]

FEATURE_COLS = (TEAM_FEATURES + LINEUP_FEATURES + MATCHUP_FEATURES
                + PBP_FEATURES + REF_FEATURES + COACH_FEATURES + ADVANCED_LINEUP_FEATURES)
