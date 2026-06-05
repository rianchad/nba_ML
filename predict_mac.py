"""
predict_mac.py — standalone predictor for macOS.

Loads nba_mac_export.pkl (produced by export_for_mac.py on Windows) and
provides the same interactive team-mode / lineup-mode prediction UI as main.py.
No NBA API data fetching required — all data is embedded in the bundle.

Usage:
    pip install scikit-learn pandas numpy rich nba_api joblib
    python predict_mac.py
    python predict_mac.py --bundle path/to/nba_mac_export.pkl
"""

import argparse
import sys
import joblib
import numpy as np
import pandas as pd
import difflib

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box

console = Console()

MATCHUP_FEATURES = [
    'mu_off_mismatch_home', 'mu_off_mismatch_away',
    'mu_best_mismatch_home', 'mu_best_mismatch_away',
    'mu_worst_mismatch_home',
    'mu_home_star_vs_def', 'mu_away_star_vs_def',
    'mu_home_best_def', 'mu_away_best_def',
    'mu_guard_edge_home', 'mu_wing_edge_home', 'mu_big_edge_home',
]

LINEUP_FEATURES = [
    'lineup_pts_mean_home', 'lineup_pts_max_home', 'lineup_pts_min_home',
    'lineup_pts_std_home', 'lineup_pts5_mean_home',
    'lineup_ast_mean_home', 'lineup_reb_mean_home',
    'lineup_stl_mean_home', 'lineup_blk_mean_home', 'lineup_tov_mean_home',
    'lineup_fg_pct_mean_home', 'lineup_fg3_pct_mean_home', 'lineup_ft_pct_mean_home',
    'lineup_ts_pct_mean_home', 'lineup_pps_mean_home',
    'lineup_pm_mean_home', 'lineup_pm_max_home', 'lineup_pm5_mean_home',
    'lineup_winrate_mean_home', 'lineup_impact_mean_home', 'lineup_impact_max_home',
    'lineup_def_mean_home', 'lineup_def_max_home',
    'lineup_gs_mean_home', 'lineup_gs_max_home',
    'lineup_consistency_home', 'lineup_usage_mean_home', 'lineup_min_mean_home',
    'lineup_pts_mean_away', 'lineup_pts_max_away', 'lineup_pts_min_away',
    'lineup_pts_std_away', 'lineup_pts5_mean_away',
    'lineup_ast_mean_away', 'lineup_reb_mean_away',
    'lineup_stl_mean_away', 'lineup_blk_mean_away', 'lineup_tov_mean_away',
    'lineup_fg_pct_mean_away', 'lineup_fg3_pct_mean_away', 'lineup_ft_pct_mean_away',
    'lineup_ts_pct_mean_away', 'lineup_pps_mean_away',
    'lineup_pm_mean_away', 'lineup_pm_max_away', 'lineup_pm5_mean_away',
    'lineup_winrate_mean_away', 'lineup_impact_mean_away', 'lineup_impact_max_away',
    'lineup_def_mean_away', 'lineup_def_max_away',
    'lineup_gs_mean_away', 'lineup_gs_max_away',
    'lineup_consistency_away', 'lineup_usage_mean_away', 'lineup_min_mean_away',
    'star_pts_gap', 'lineup_pm_gap', 'lineup_wr_gap', 'impact_gap', 'def_gap', 'gs_gap',
]


# ── Player lookup ──────────────────────────────────────────────────────────────

def find_player(name, all_players):
    names = [p['full_name'] for p in all_players]
    attempt = name.strip().title()
    while True:
        if attempt in names:
            p = next(x for x in all_players if x['full_name'] == attempt)
            console.print(f"    [green]✓ {attempt}")
            return p['id'], attempt
        matches = difflib.get_close_matches(attempt, names, n=3, cutoff=0.45)
        if not matches:
            console.print(f"    [red]✗ No match for '{attempt}'")
            if Prompt.ask("    Try again? (y/n)", choices=['y', 'n']) == 'y':
                attempt = Prompt.ask("    Name").strip().title()
                continue
            return None, None
        t = Table(box=box.SIMPLE, show_header=False)
        t.add_column("#", style="cyan", width=4)
        t.add_column("Name")
        for i, m in enumerate(matches, 1):
            t.add_row(str(i), m)
        t.add_row("0", "[red]None of these")
        console.print(f"    Matches for '[yellow]{attempt}[/]':")
        console.print(t)
        sel = Prompt.ask("    Select").strip()
        if sel == '0':
            attempt = Prompt.ask("    Name").strip().title()
            continue
        if sel.isdigit() and 1 <= int(sel) <= len(matches):
            chosen = matches[int(sel) - 1]
            p = next(x for x in all_players if x['full_name'] == chosen)
            console.print(f"    [green]✓ {chosen}")
            return p['id'], chosen
        console.print("    [yellow]Invalid.")


def get_player_stats(player_id, player_rolling):
    rows = player_rolling[
        (player_rolling['PLAYER_ID'] == player_id) &
        player_rolling['PTS_roll10'].notna()
    ]
    return rows.iloc[-1] if len(rows) else None


def resolve_lineup(names, label, all_players, player_rolling):
    console.print(f"\n  [cyan]Resolving {label}:")
    rows = []
    for name in names:
        while True:
            pid, _ = find_player(name, all_players)
            if pid is None:
                break
            stats = get_player_stats(pid, player_rolling)
            if stats is None:
                console.print("    [yellow]No stats found.")
                if Prompt.ask("    Try another? (y/n)", choices=['y', 'n']) == 'y':
                    name = Prompt.ask("    Name").strip()
                    continue
                break
            rows.append(stats)
            break
    return rows


# ── Lineup helpers ─────────────────────────────────────────────────────────────

def aggregate_lineup(rows):
    df = pd.DataFrame(rows)
    agg = df[[c for c in df.columns if '_roll' in c]].mean()

    def mx(c):
        return df[c].max() if c in df.columns else np.nan

    return {
        'lineup_pts_mean': agg.get('PTS_roll10', np.nan),
        'lineup_pts_max': mx('PTS_roll10'),
        'lineup_pts_min': df['PTS_roll10'].min() if 'PTS_roll10' in df else np.nan,
        'lineup_pts_std': df['PTS_roll10'].std() if 'PTS_roll10' in df else np.nan,
        'lineup_pts5_mean': agg.get('PTS_roll5', np.nan),
        'lineup_ast_mean': agg.get('AST_roll10', np.nan),
        'lineup_reb_mean': agg.get('REB_roll10', np.nan),
        'lineup_stl_mean': agg.get('STL_roll10', np.nan),
        'lineup_blk_mean': agg.get('BLK_roll10', np.nan),
        'lineup_tov_mean': agg.get('TOV_roll10', np.nan),
        'lineup_fg_pct_mean': agg.get('FG_PCT_roll10', np.nan),
        'lineup_fg3_pct_mean': agg.get('FG3_PCT_roll10', np.nan),
        'lineup_ft_pct_mean': agg.get('FT_PCT_roll10', np.nan),
        'lineup_ts_pct_mean': agg.get('TS_PCT_roll10', np.nan),
        'lineup_pps_mean': agg.get('PPS_roll10', np.nan),
        'lineup_pm_mean': agg.get('PLUS_MINUS_roll10', np.nan),
        'lineup_pm_max': mx('PLUS_MINUS_roll10'),
        'lineup_pm5_mean': agg.get('PLUS_MINUS_roll5', np.nan),
        'lineup_winrate_mean': agg.get('WINRATE_roll10', np.nan),
        'lineup_impact_mean': agg.get('IMPACT_roll10', np.nan),
        'lineup_impact_max': mx('IMPACT_roll10'),
        'lineup_def_mean': agg.get('DEF_SCORE_roll10', np.nan),
        'lineup_def_max': mx('DEF_SCORE_roll10'),
        'lineup_gs_mean': agg.get('GAME_SCORE_roll10', np.nan),
        'lineup_gs_max': mx('GAME_SCORE_roll10'),
        'lineup_consistency': agg.get('PTS_CV_roll10', np.nan),
        'lineup_usage_mean': agg.get('USAGE_PROXY_roll10', np.nan),
        'lineup_min_mean': agg.get('MIN_FLOAT_roll10', np.nan),
    }


def compute_custom_matchup(home_rows, away_rows):
    def prep(rows):
        out = []
        for r in rows:
            off = r.get('IMPACT_roll10', r.get('PTS_roll10', 0)) or 0
            dfn = r.get('DEF_SCORE_roll10', 0) or 0
            role = r.get('ROLE', 'WING')
            out.append({'OFF': off, 'DEF': dfn, 'ROLE': role})
        return pd.DataFrame(out)

    h, a = prep(home_rows), prep(away_rows)
    if len(h) < 3 or len(a) < 3:
        return {k: 0 for k in MATCHUP_FEATURES}

    pairs = []
    for role in ['GUARD', 'WING', 'BIG']:
        hr = h[h['ROLE'] == role].sort_values('OFF', ascending=False)
        ar = a[a['ROLE'] == role].sort_values('OFF', ascending=False)
        for i in range(min(len(hr), len(ar))):
            pairs.append((hr.iloc[i], ar.iloc[i]))
    if not pairs:
        hs = h.sort_values('OFF', ascending=False)
        as_ = a.sort_values('OFF', ascending=False)
        for i in range(min(len(hs), len(as_))):
            pairs.append((hs.iloc[i], as_.iloc[i]))

    h_mm = [hp['OFF'] - ap['DEF'] for hp, ap in pairs]
    a_mm = [ap['OFF'] - hp['DEF'] for hp, ap in pairs]
    h_star = h.loc[h['OFF'].idxmax()]
    a_star = a.loc[a['OFF'].idxmax()]
    h_def = h.loc[h['DEF'].idxmax()]
    a_def = a.loc[a['DEF'].idxmax()]

    def edge(role):
        hh = h[h['ROLE'] == role]['OFF'].mean()
        aa = a[a['ROLE'] == role]['OFF'].mean()
        return float((0 if pd.isna(hh) else hh) - (0 if pd.isna(aa) else aa))

    return {
        'mu_off_mismatch_home': float(np.mean(h_mm)),
        'mu_off_mismatch_away': float(np.mean(a_mm)),
        'mu_best_mismatch_home': float(np.max(h_mm)),
        'mu_best_mismatch_away': float(np.max(a_mm)),
        'mu_worst_mismatch_home': float(np.min(h_mm)),
        'mu_home_star_vs_def': float(h_star['OFF'] - a_def['DEF']),
        'mu_away_star_vs_def': float(a_star['OFF'] - h_def['DEF']),
        'mu_home_best_def': float(h_def['DEF']),
        'mu_away_best_def': float(a_def['DEF']),
        'mu_guard_edge_home': edge('GUARD'),
        'mu_wing_edge_home': edge('WING'),
        'mu_big_edge_home': edge('BIG'),
    }


def lineup_to_team_context(lu, is_home, medians):
    sfx = '_home' if is_home else '_away'
    pts = lu.get('lineup_pts_mean', 15) or 15
    pm = lu.get('lineup_pm_mean', 0) or 0
    wr = lu.get('lineup_winrate_mean', 0.5) or 0.5
    ts = lu.get('lineup_ts_pct_mean', 0.55) or 0.55
    ctx = {
        f'WINRATE_roll20{sfx}': wr, f'WINRATE_roll10{sfx}': wr,
        f'WINRATE_roll5{sfx}': wr, f'WINRATE_roll3{sfx}': wr,
        f'PLUS_MINUS_roll20{sfx}': pm, f'PLUS_MINUS_roll10{sfx}': pm,
        f'PLUS_MINUS_roll5{sfx}': pm,
        f'WIN_STREAK{sfx}': 1 if wr > 0.6 else -1 if wr < 0.4 else 0,
        f'LAST_GAME_PM{sfx}': pm, f'LAST_GAME_WIN{sfx}': 1 if wr > 0.5 else 0,
        f'PM_TREND{sfx}': 0, f'PTS_TREND{sfx}': 0, f'WINRATE_TREND{sfx}': 0,
        f'TS_PCT_roll10{sfx}': ts, f'OFF_RTG_roll10{sfx}': pts / 0.95,
        f'NET_RTG_roll10{sfx}': pm,
        f'SOS_roll10{sfx}': 0.5, f'REST_DAYS{sfx}': 3,
        f'IS_BACK_TO_BACK{sfx}': 0, f'TRAVEL_KM{sfx}': 500, f'TZ_CHANGE{sfx}': 0,
    }
    if is_home:
        ctx['WINRATE_home_roll10_home'] = wr
        ctx['IS_ALTITUDE_home'] = 0
        ctx['GAMES_INTO_SEASON_home'] = 40
        ctx['IS_LATE_SEASON_home'] = 0
    else:
        ctx['WINRATE_away_roll10_away'] = wr
    return ctx


# ── Output ─────────────────────────────────────────────────────────────────────

def _print_result(home, away, hp, ap, playoffs):
    label = 'PLAYOFFS' if playoffs else 'Regular Season'
    margin = abs(hp - ap)
    conf = 'High' if margin > 0.15 else 'Medium' if margin > 0.07 else 'Low'
    cc = 'green' if conf == 'High' else 'yellow' if conf == 'Medium' else 'red'
    L = 20
    hb = '█' * int(L * hp) + '░' * (L - int(L * hp))
    ab = '█' * int(L * ap) + '░' * (L - int(L * ap))
    t = Table(title=f"{home} (home) vs {away} (away) [{label}]", box=box.ROUNDED)
    t.add_column("Team", style="cyan", width=6)
    t.add_column("Win Probability", width=22)
    t.add_column("Visual", width=24)
    t.add_row(f"[blue]{home}", f"[blue]{hp:.1%}{'  ◀' if hp > ap else ''}", f"[blue]{hb}")
    t.add_row(f"[red]{away}",  f"[red]{ap:.1%}{'  ◀' if ap > hp else ''}",  f"[red]{ab}")
    console.print()
    console.print(t)
    console.print(f"[{cc}]Confidence: {conf} ({margin:.1%} margin)\n")


# ── Prediction functions ───────────────────────────────────────────────────────

def predict_game(home, away, featured_df, model, features, medians, playoffs=False, adv_params=None):
    home, away = home.upper(), away.upper()
    anchor = ['PTS_roll10', 'WINRATE_roll10']

    def latest(ab):
        r = featured_df[featured_df['TEAM_ABBREVIATION'] == ab].dropna(subset=anchor)
        if len(r) == 0:
            raise ValueError(f"Team '{ab}' not found.")
        return r.iloc[-1]

    try:
        h, a = latest(home), latest(away)
    except ValueError as e:
        console.print(f"[red]  {e}")
        return None, None

    rec = {
        'IS_PLAYOFFS': int(playoffs), 'PLAYOFF_ROUND': 1 if playoffs else 0,
        'SERIES_GAME_NUM': 0, 'IS_MUST_WIN': 0, 'H2H_HOME_WIN': 0.5,
        'H2H_MEETINGS_THIS_SEASON': 0, 'MONTH': 3, 'DAY_OF_WEEK': 2, 'IS_WEEKEND': 0,
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
        # Rest & Travel
        if 'rest_days_home' in adv_params and 'REST_DAYS_home' in rec:
            rec['REST_DAYS_home'] = adv_params['rest_days_home']
        if 'rest_days_away' in adv_params and 'REST_DAYS_away' in rec:
            rec['REST_DAYS_away'] = adv_params['rest_days_away']
        if 'travel_km_away' in adv_params and 'TRAVEL_KM_away' in rec:
            rec['TRAVEL_KM_away'] = adv_params['travel_km_away']
        if 'tz_change_away' in adv_params and 'TZ_CHANGE_away' in rec:
            rec['TZ_CHANGE_away'] = adv_params['tz_change_away']

        # Back-to-Back & Streaks
        if 'is_back_to_back_home' in adv_params and 'IS_BACK_TO_BACK_home' in rec:
            rec['IS_BACK_TO_BACK_home'] = adv_params['is_back_to_back_home']
        if 'is_back_to_back_away' in adv_params and 'IS_BACK_TO_BACK_away' in rec:
            rec['IS_BACK_TO_BACK_away'] = adv_params['is_back_to_back_away']
        if 'win_streak_home' in adv_params and 'WIN_STREAK_home' in rec:
            rec['WIN_STREAK_home'] = adv_params['win_streak_home']
        if 'win_streak_away' in adv_params and 'WIN_STREAK_away' in rec:
            rec['WIN_STREAK_away'] = adv_params['win_streak_away']

        # Venue & Environmental
        if 'is_altitude_home' in adv_params and 'IS_ALTITUDE_home' in rec:
            rec['IS_ALTITUDE_home'] = adv_params['is_altitude_home']

        # Head-to-Head & Season Context
        if 'h2h_home_win' in adv_params and 'H2H_HOME_WIN' in rec:
            rec['H2H_HOME_WIN'] = adv_params['h2h_home_win']
        if 'h2h_meetings' in adv_params and 'H2H_MEETINGS_THIS_SEASON' in rec:
            rec['H2H_MEETINGS_THIS_SEASON'] = adv_params['h2h_meetings']
        if 'month' in adv_params and 'MONTH' in rec:
            rec['MONTH'] = adv_params['month']
        if 'day_of_week' in adv_params and 'DAY_OF_WEEK' in rec:
            rec['DAY_OF_WEEK'] = adv_params['day_of_week']
        if 'is_weekend' in adv_params and 'IS_WEEKEND' in rec:
            rec['IS_WEEKEND'] = adv_params['is_weekend']

        # Playoff-Specific
        if 'playoff_round' in adv_params and 'PLAYOFF_ROUND' in rec:
            rec['PLAYOFF_ROUND'] = adv_params['playoff_round']
        if 'series_game_num' in adv_params and 'SERIES_GAME_NUM' in rec:
            rec['SERIES_GAME_NUM'] = adv_params['series_game_num']
        if 'is_must_win' in adv_params and 'IS_MUST_WIN' in rec:
            rec['IS_MUST_WIN'] = adv_params['is_must_win']

        # Performance Trends
        if 'pts_trend_home' in adv_params and 'PTS_TREND_home' in rec:
            rec['PTS_TREND_home'] = adv_params['pts_trend_home']
        if 'pts_trend_away' in adv_params and 'PTS_TREND_away' in rec:
            rec['PTS_TREND_away'] = adv_params['pts_trend_away']

    X = pd.DataFrame([rec])[features].apply(pd.to_numeric, errors='coerce').fillna(medians)
    p = model.predict_proba(X)[0]
    _print_result(home, away, p[1], p[0], playoffs)
    return p[1], p[0]


def predict_with_lineups(home, home_names, away, away_names,
                         player_rolling, model, features, medians,
                         all_players, playoffs=False, adv_params=None):
    home, away = home.upper(), away.upper()

    h_rows = resolve_lineup(home_names, f'{home} (home)', all_players, player_rolling)
    a_rows = resolve_lineup(away_names, f'{away} (away)', all_players, player_rolling)
    if len(h_rows) < 3 or len(a_rows) < 3:
        console.print("[red]  Need at least 3 players per team.")
        return None, None

    h_lu, a_lu = aggregate_lineup(h_rows), aggregate_lineup(a_rows)
    mu = compute_custom_matchup(h_rows, a_rows)

    rec = {
        'IS_PLAYOFFS': int(playoffs), 'PLAYOFF_ROUND': 1 if playoffs else 0,
        'SERIES_GAME_NUM': 0, 'IS_MUST_WIN': 0, 'H2H_HOME_WIN': 0.5,
        'H2H_MEETINGS_THIS_SEASON': 0, 'MONTH': 3, 'DAY_OF_WEEK': 2, 'IS_WEEKEND': 0,
    }
    rec.update(lineup_to_team_context(h_lu, True, medians))
    rec.update(lineup_to_team_context(a_lu, False, medians))
    rec.update(mu)

    gap_map = {
        'star_pts_gap': 'lineup_pts_max', 'lineup_pm_gap': 'lineup_pm_mean',
        'lineup_wr_gap': 'lineup_winrate_mean', 'impact_gap': 'lineup_impact_mean',
        'def_gap': 'lineup_def_mean', 'gs_gap': 'lineup_gs_mean',
    }
    for col in LINEUP_FEATURES:
        if col in gap_map:
            k = gap_map[col]
            rec[col] = (h_lu.get(k, 0) or 0) - (a_lu.get(k, 0) or 0)
        elif col.endswith('_gap'):
            rec[col] = 0
        elif '_home' in col:
            rec[col] = h_lu.get(col.replace('_home', ''), np.nan)
        elif '_away' in col:
            rec[col] = a_lu.get(col.replace('_away', ''), np.nan)

    if adv_params:
        # Rest & Travel
        if 'rest_days_home' in adv_params and 'REST_DAYS_home' in rec:
            rec['REST_DAYS_home'] = adv_params['rest_days_home']
        if 'rest_days_away' in adv_params and 'REST_DAYS_away' in rec:
            rec['REST_DAYS_away'] = adv_params['rest_days_away']
        if 'travel_km_away' in adv_params and 'TRAVEL_KM_away' in rec:
            rec['TRAVEL_KM_away'] = adv_params['travel_km_away']
        if 'tz_change_away' in adv_params and 'TZ_CHANGE_away' in rec:
            rec['TZ_CHANGE_away'] = adv_params['tz_change_away']

        # Back-to-Back & Streaks
        if 'is_back_to_back_home' in adv_params and 'IS_BACK_TO_BACK_home' in rec:
            rec['IS_BACK_TO_BACK_home'] = adv_params['is_back_to_back_home']
        if 'is_back_to_back_away' in adv_params and 'IS_BACK_TO_BACK_away' in rec:
            rec['IS_BACK_TO_BACK_away'] = adv_params['is_back_to_back_away']
        if 'win_streak_home' in adv_params and 'WIN_STREAK_home' in rec:
            rec['WIN_STREAK_home'] = adv_params['win_streak_home']
        if 'win_streak_away' in adv_params and 'WIN_STREAK_away' in rec:
            rec['WIN_STREAK_away'] = adv_params['win_streak_away']

        # Venue & Environmental
        if 'is_altitude_home' in adv_params and 'IS_ALTITUDE_home' in rec:
            rec['IS_ALTITUDE_home'] = adv_params['is_altitude_home']

        # Head-to-Head & Season Context
        if 'h2h_home_win' in adv_params and 'H2H_HOME_WIN' in rec:
            rec['H2H_HOME_WIN'] = adv_params['h2h_home_win']
        if 'h2h_meetings' in adv_params and 'H2H_MEETINGS_THIS_SEASON' in rec:
            rec['H2H_MEETINGS_THIS_SEASON'] = adv_params['h2h_meetings']
        if 'month' in adv_params and 'MONTH' in rec:
            rec['MONTH'] = adv_params['month']
        if 'day_of_week' in adv_params and 'DAY_OF_WEEK' in rec:
            rec['DAY_OF_WEEK'] = adv_params['day_of_week']
        if 'is_weekend' in adv_params and 'IS_WEEKEND' in rec:
            rec['IS_WEEKEND'] = adv_params['is_weekend']

        # Playoff-Specific
        if 'playoff_round' in adv_params and 'PLAYOFF_ROUND' in rec:
            rec['PLAYOFF_ROUND'] = adv_params['playoff_round']
        if 'series_game_num' in adv_params and 'SERIES_GAME_NUM' in rec:
            rec['SERIES_GAME_NUM'] = adv_params['series_game_num']
        if 'is_must_win' in adv_params and 'IS_MUST_WIN' in rec:
            rec['IS_MUST_WIN'] = adv_params['is_must_win']

        # Performance Trends
        if 'pts_trend_home' in adv_params and 'PTS_TREND_home' in rec:
            rec['PTS_TREND_home'] = adv_params['pts_trend_home']
        if 'pts_trend_away' in adv_params and 'PTS_TREND_away' in rec:
            rec['PTS_TREND_away'] = adv_params['pts_trend_away']

    X = (pd.DataFrame([rec])
         .reindex(columns=features)
         .apply(pd.to_numeric, errors='coerce')
         .fillna(medians))
    p = model.predict_proba(X)[0]
    _print_result(home, away, p[1], p[0], playoffs)

    for rows, title, color in [
        (h_rows, f'Home — {home}', 'blue'),
        (a_rows, f'Away — {away}', 'red'),
    ]:
        t = Table(title=f'{title} (last 10g)', box=box.ROUNDED)
        for c, w in [('Player', 24), ('Role', 6), ('PTS', 6),
                     ('OFF', 6), ('DEF', 6), ('TS%', 7), ('+/-', 7)]:
            t.add_column(c,
                         style=color if c == 'Player' else None,
                         justify='left' if c in ('Player', 'Role') else 'right',
                         width=w)
        for r in rows:
            off = r.get('IMPACT_roll10', r.get('PTS_roll10', 0)) or 0
            t.add_row(
                str(r.get('PLAYER_NAME', '?')), str(r.get('ROLE', '?')),
                f"{r.get('PTS_roll10', 0):.1f}", f"{off:.1f}",
                f"{r.get('DEF_SCORE_roll10', 0):.1f}",
                f"{r.get('TS_PCT_roll10', 0):.3f}",
                f"{r.get('PLUS_MINUS_roll10', 0):+.1f}",
            )
        console.print(t)

    console.print(
        f"[dim]Positional edge (home): avg mismatch "
        f"{mu['mu_off_mismatch_home'] - mu['mu_off_mismatch_away']:+.1f}, "
        f"star vs best defender {mu['mu_home_star_vs_def']:+.1f}[/dim]\n"
    )
    return p[1], p[0]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NBA Win Probability Predictor (Mac)")
    parser.add_argument('--bundle', default='nba_mac_export.pkl',
                        help='Path to the export bundle (default: nba_mac_export.pkl)')
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]NBA Win Probability Predictor — Mac Edition",
        border_style="cyan", padding=(1, 2),
    ))

    # ── Load bundle ───────────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Loading bundle: [white]{args.bundle}")
    try:
        bundle = joblib.load(args.bundle)
    except FileNotFoundError:
        console.print(f"[red]Bundle not found: {args.bundle}")
        console.print("[yellow]Run export_for_mac.py on Windows first, then transfer the file.")
        sys.exit(1)

    model         = bundle['model']
    features      = bundle['features']
    medians       = bundle['medians']
    featured_df   = bundle['featured_df']
    player_rolling = bundle['player_rolling']

    console.print("[green]Bundle loaded.")

    # ── Load static player list (ships with nba_api, no network call needed) ──
    try:
        from nba_api.stats.static import players as nba_players_static
        all_players = nba_players_static.get_players()
    except ImportError:
        console.print("[red]nba_api not installed. Run: pip install nba_api")
        sys.exit(1)

    # ── Quick examples ────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Quick examples:")
    predict_game('BOS', 'MIA', featured_df, model, features, medians)
    predict_game('OKC', 'DEN', featured_df, model, features, medians)

    # ── Advanced parameters helper ───────────────────────────────────────────
    def get_advanced_params():
        """Get optional advanced parameters from user."""
        params = {}
        console.print("\n[cyan]Optional Advanced Parameters (press Enter to skip):")

        # ── Rest & Travel ──
        console.print("\n  [yellow]Rest & Travel:")
        rest = Prompt.ask("    Rest days home team (default 3)", default="").strip()
        if rest:
            try:
                params['rest_days_home'] = int(rest)
            except ValueError:
                console.print("    [yellow]Invalid, skipping")

        rest_away = Prompt.ask("    Rest days away team (default 3)", default="").strip()
        if rest_away:
            try:
                params['rest_days_away'] = int(rest_away)
            except ValueError:
                console.print("    [yellow]Invalid, skipping")

        travel = Prompt.ask("    Travel km away team (default 500)", default="").strip()
        if travel:
            try:
                params['travel_km_away'] = int(travel)
            except ValueError:
                console.print("    [yellow]Invalid, skipping")

        tz = Prompt.ask("    Timezone change away team in hours (default 0)", default="").strip()
        if tz:
            try:
                params['tz_change_away'] = int(tz)
            except ValueError:
                console.print("    [yellow]Invalid, skipping")

        # ── Back-to-Back & Momentum ──
        console.print("\n  [yellow]Back-to-Back & Momentum:")
        btb_home = Prompt.ask("    Home team back-to-back? (y/n)", default="").strip()
        if btb_home.lower() in ['y', 'n']:
            params['is_back_to_back_home'] = 1 if btb_home.lower() == 'y' else 0

        btb_away = Prompt.ask("    Away team back-to-back? (y/n)", default="").strip()
        if btb_away.lower() in ['y', 'n']:
            params['is_back_to_back_away'] = 1 if btb_away.lower() == 'y' else 0

        win_streak_home = Prompt.ask("    Home team win streak (e.g., -2 for loss streak)", default="").strip()
        if win_streak_home:
            try:
                params['win_streak_home'] = int(win_streak_home)
            except ValueError:
                console.print("    [yellow]Invalid, skipping")

        win_streak_away = Prompt.ask("    Away team win streak (e.g., 3 for 3-game win streak)", default="").strip()
        if win_streak_away:
            try:
                params['win_streak_away'] = int(win_streak_away)
            except ValueError:
                console.print("    [yellow]Invalid, skipping")

        # ── Venue & Environmental ──
        console.print("\n  [yellow]Venue & Environmental:")
        altitude = Prompt.ask("    High altitude? (y/n) [Denver, etc.]", default="").strip()
        if altitude.lower() in ['y', 'n']:
            params['is_altitude_home'] = 1 if altitude.lower() == 'y' else 0

        # ── Head-to-Head & Series ──
        console.print("\n  [yellow]Head-to-Head & Season Context:")
        h2h = Prompt.ask("    Home team H2H win rate (0.0-1.0, e.g., 0.6 for 60%)", default="").strip()
        if h2h:
            try:
                h2h_val = float(h2h)
                if 0 <= h2h_val <= 1:
                    params['h2h_home_win'] = h2h_val
                else:
                    console.print("    [yellow]Must be 0-1, skipping")
            except ValueError:
                console.print("    [yellow]Invalid, skipping")

        h2h_meetings = Prompt.ask("    H2H meetings this season (default 0)", default="").strip()
        if h2h_meetings:
            try:
                params['h2h_meetings'] = int(h2h_meetings)
            except ValueError:
                console.print("    [yellow]Invalid, skipping")

        month = Prompt.ask("    Month (1-12, default 3 for March)", default="").strip()
        if month:
            try:
                m = int(month)
                if 1 <= m <= 12:
                    params['month'] = m
                else:
                    console.print("    [yellow]Must be 1-12, skipping")
            except ValueError:
                console.print("    [yellow]Invalid, skipping")

        day_of_week = Prompt.ask("    Day of week (0=Mon, 1=Tue... 6=Sun, default 2)", default="").strip()
        if day_of_week:
            try:
                d = int(day_of_week)
                if 0 <= d <= 6:
                    params['day_of_week'] = d
                else:
                    console.print("    [yellow]Must be 0-6, skipping")
            except ValueError:
                console.print("    [yellow]Invalid, skipping")

        is_weekend = Prompt.ask("    Is weekend game? (y/n)", default="").strip()
        if is_weekend.lower() in ['y', 'n']:
            params['is_weekend'] = 1 if is_weekend.lower() == 'y' else 0

        # ── Playoff-Specific ──
        console.print("\n  [yellow]Playoff-Specific (if in playoffs):")
        playoff_round = Prompt.ask("    Playoff round (1=first round, 4=Finals, default 1)", default="").strip()
        if playoff_round:
            try:
                pr = int(playoff_round)
                if 1 <= pr <= 4:
                    params['playoff_round'] = pr
                else:
                    console.print("    [yellow]Must be 1-4, skipping")
            except ValueError:
                console.print("    [yellow]Invalid, skipping")

        series_game = Prompt.ask("    Series game number (1-7, e.g., 5 for Game 5)", default="").strip()
        if series_game:
            try:
                sg = int(series_game)
                if 1 <= sg <= 7:
                    params['series_game_num'] = sg
                else:
                    console.print("    [yellow]Must be 1-7, skipping")
            except ValueError:
                console.print("    [yellow]Invalid, skipping")

        is_must_win = Prompt.ask("    Must-win game? (y/n)", default="").strip()
        if is_must_win.lower() in ['y', 'n']:
            params['is_must_win'] = 1 if is_must_win.lower() == 'y' else 0

        # ── Team Performance Trends ──
        console.print("\n  [yellow]Performance Trends (optional adjustments):")
        pts_trend_home = Prompt.ask("    Home team pts trend (-10 to 10, recent scoring change)", default="").strip()
        if pts_trend_home:
            try:
                params['pts_trend_home'] = int(pts_trend_home)
            except ValueError:
                console.print("    [yellow]Invalid, skipping")

        pts_trend_away = Prompt.ask("    Away team pts trend (-10 to 10)", default="").strip()
        if pts_trend_away:
            try:
                params['pts_trend_away'] = int(pts_trend_away)
            except ValueError:
                console.print("    [yellow]Invalid, skipping")

        return params

    # ── Interactive loop ──────────────────────────────────────────────────────
    console.print(Panel(
        "[cyan]1. Team mode (quick)\n2. Lineup mode (quick)\n3. Team mode (advanced)\n4. Lineup mode (advanced)",
        title="[bold cyan]Interactive", border_style="cyan", padding=(1, 2),
    ))
    while True:
        try:
            choice = Prompt.ask("\n[bold cyan]Choose (1/2/3/4, Ctrl+C to exit)", choices=['1', '2', '3', '4'])
            playoffs = Prompt.ask("  Playoffs? (y/n)", choices=['y', 'n']) == 'y'
            adv_params = {}

            if choice in ['1', '3']:
                h = Prompt.ask("  Home team").strip()
                a = Prompt.ask("  Away team").strip()
                if choice == '3':
                    adv_params = get_advanced_params()
                predict_game(h, a, featured_df, model, features, medians,
                           playoffs=playoffs, adv_params=adv_params)
            else:
                h = Prompt.ask("  Home team").strip().upper()
                hn = [Prompt.ask(f"    {h} player {i}").strip() for i in range(1, 6)]
                a = Prompt.ask("  Away team").strip().upper()
                an = [Prompt.ask(f"    {a} player {i}").strip() for i in range(1, 6)]
                if choice == '4':
                    adv_params = get_advanced_params()
                predict_with_lineups(h, hn, a, an, player_rolling,
                                     model, features, medians, all_players,
                                     playoffs=playoffs, adv_params=adv_params)
        except KeyboardInterrupt:
            console.print("\n[cyan]Done.")
            break


if __name__ == '__main__':
    main()
