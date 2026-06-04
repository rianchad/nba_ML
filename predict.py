"""
predict.py — team-mode and lineup-mode prediction.

Lineup mode reconstructs the full feature row from the chosen players,
including the player-vs-player positional matchup features, and derives
team context from lineup quality so hypothetical lineups behave sensibly.
"""

import difflib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich import box

import config as C
from feature_list import TEAM_FEATURES, LINEUP_FEATURES, MATCHUP_FEATURES, PBP_FEATURES
from nba_api.stats.static import players as nba_players_static

console = Console()


# ── Player lookup ──────────────────────────────────────
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
            if Prompt.ask("    Try again? (y/n)", choices=['y','n']) == 'y':
                attempt = Prompt.ask("    Name").strip().title(); continue
            return None, None
        t = Table(box=box.SIMPLE, show_header=False)
        t.add_column("#", style="cyan", width=4); t.add_column("Name")
        for i, m in enumerate(matches, 1): t.add_row(str(i), m)
        t.add_row("0", "[red]None of these")
        console.print(f"    Matches for '[yellow]{attempt}[/]':"); console.print(t)
        sel = Prompt.ask("    Select").strip()
        if sel == '0':
            attempt = Prompt.ask("    Name").strip().title(); continue
        if sel.isdigit() and 1 <= int(sel) <= len(matches):
            chosen = matches[int(sel)-1]
            p = next(x for x in all_players if x['full_name'] == chosen)
            console.print(f"    [green]✓ {chosen}")
            return p['id'], chosen
        console.print("    [yellow]Invalid.")


def get_player_stats(player_id, player_rolling):
    rows = player_rolling[(player_rolling['PLAYER_ID']==player_id) &
                          player_rolling['PTS_roll10'].notna()]
    return rows.iloc[-1] if len(rows) else None


def resolve_lineup(names, label, all_players, player_rolling):
    console.print(f"\n  [cyan]Resolving {label}:")
    rows = []
    for name in names:
        while True:
            pid, _ = find_player(name, all_players)
            if pid is None: break
            stats = get_player_stats(pid, player_rolling)
            if stats is None:
                console.print("    [yellow]No stats found.")
                if Prompt.ask("    Try another? (y/n)", choices=['y','n']) == 'y':
                    name = Prompt.ask("    Name").strip(); continue
                break
            rows.append(stats); break
    return rows


# ── Lineup aggregation ─────────────────────────────────
def aggregate_lineup(rows):
    df = pd.DataFrame(rows)
    agg = df[[c for c in df.columns if '_roll' in c]].mean()
    def mx(c): return df[c].max() if c in df.columns else np.nan
    return {
        'lineup_pts_mean': agg.get('PTS_roll10', np.nan),
        'lineup_pts_max': mx('PTS_roll10'), 'lineup_pts_min': df['PTS_roll10'].min() if 'PTS_roll10' in df else np.nan,
        'lineup_pts_std': df['PTS_roll10'].std() if 'PTS_roll10' in df else np.nan,
        'lineup_pts5_mean': agg.get('PTS_roll5', np.nan),
        'lineup_ast_mean': agg.get('AST_roll10', np.nan), 'lineup_reb_mean': agg.get('REB_roll10', np.nan),
        'lineup_stl_mean': agg.get('STL_roll10', np.nan), 'lineup_blk_mean': agg.get('BLK_roll10', np.nan),
        'lineup_tov_mean': agg.get('TOV_roll10', np.nan),
        'lineup_fg_pct_mean': agg.get('FG_PCT_roll10', np.nan),
        'lineup_fg3_pct_mean': agg.get('FG3_PCT_roll10', np.nan),
        'lineup_ft_pct_mean': agg.get('FT_PCT_roll10', np.nan),
        'lineup_ts_pct_mean': agg.get('TS_PCT_roll10', np.nan),
        'lineup_pps_mean': agg.get('PPS_roll10', np.nan),
        'lineup_pm_mean': agg.get('PLUS_MINUS_roll10', np.nan), 'lineup_pm_max': mx('PLUS_MINUS_roll10'),
        'lineup_pm5_mean': agg.get('PLUS_MINUS_roll5', np.nan),
        'lineup_winrate_mean': agg.get('WINRATE_roll10', np.nan),
        'lineup_impact_mean': agg.get('IMPACT_roll10', np.nan), 'lineup_impact_max': mx('IMPACT_roll10'),
        'lineup_def_mean': agg.get('DEF_SCORE_roll10', np.nan), 'lineup_def_max': mx('DEF_SCORE_roll10'),
        'lineup_gs_mean': agg.get('GAME_SCORE_roll10', np.nan), 'lineup_gs_max': mx('GAME_SCORE_roll10'),
        'lineup_consistency': agg.get('PTS_CV_roll10', np.nan),
        'lineup_usage_mean': agg.get('USAGE_PROXY_roll10', np.nan),
        'lineup_min_mean': agg.get('MIN_FLOAT_roll10', np.nan),
    }


def compute_custom_matchup(home_rows, away_rows):
    """Replicate positional matchup logic for a custom lineup (home perspective)."""
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
    for role in ['GUARD','WING','BIG']:
        hr = h[h['ROLE']==role].sort_values('OFF', ascending=False)
        ar = a[a['ROLE']==role].sort_values('OFF', ascending=False)
        for i in range(min(len(hr), len(ar))):
            pairs.append((hr.iloc[i], ar.iloc[i]))
    if not pairs:  # fallback: match by overall offense
        hs = h.sort_values('OFF', ascending=False); as_ = a.sort_values('OFF', ascending=False)
        for i in range(min(len(hs), len(as_))): pairs.append((hs.iloc[i], as_.iloc[i]))

    h_mm = [hp['OFF']-ap['DEF'] for hp, ap in pairs]
    a_mm = [ap['OFF']-hp['DEF'] for hp, ap in pairs]
    h_star, a_star = h.loc[h['OFF'].idxmax()], a.loc[a['OFF'].idxmax()]
    h_def, a_def = h.loc[h['DEF'].idxmax()], a.loc[a['DEF'].idxmax()]
    def edge(role):
        hh = h[h['ROLE']==role]['OFF'].mean(); aa = a[a['ROLE']==role]['OFF'].mean()
        return float((0 if pd.isna(hh) else hh) - (0 if pd.isna(aa) else aa))
    return {
        'mu_off_mismatch_home': float(np.mean(h_mm)), 'mu_off_mismatch_away': float(np.mean(a_mm)),
        'mu_best_mismatch_home': float(np.max(h_mm)), 'mu_best_mismatch_away': float(np.max(a_mm)),
        'mu_worst_mismatch_home': float(np.min(h_mm)),
        'mu_home_star_vs_def': float(h_star['OFF']-a_def['DEF']),
        'mu_away_star_vs_def': float(a_star['OFF']-h_def['DEF']),
        'mu_home_best_def': float(h_def['DEF']), 'mu_away_best_def': float(a_def['DEF']),
        'mu_guard_edge_home': edge('GUARD'), 'mu_wing_edge_home': edge('WING'),
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
        f'PLUS_MINUS_roll20{sfx}': pm, f'PLUS_MINUS_roll10{sfx}': pm, f'PLUS_MINUS_roll5{sfx}': pm,
        f'WIN_STREAK{sfx}': 1 if wr>0.6 else -1 if wr<0.4 else 0,
        f'LAST_GAME_PM{sfx}': pm, f'LAST_GAME_WIN{sfx}': 1 if wr>0.5 else 0,
        f'PM_TREND{sfx}': 0, f'PTS_TREND{sfx}': 0, f'WINRATE_TREND{sfx}': 0,
        f'TS_PCT_roll10{sfx}': ts, f'OFF_RTG_roll10{sfx}': pts/0.95, f'NET_RTG_roll10{sfx}': pm,
        f'SOS_roll10{sfx}': 0.5, f'REST_DAYS{sfx}': 3, f'IS_BACK_TO_BACK{sfx}': 0,
        f'TRAVEL_KM{sfx}': 500, f'TZ_CHANGE{sfx}': 0,
    }
    if is_home:
        ctx['WINRATE_home_roll10_home'] = wr; ctx['IS_ALTITUDE_home'] = 0
        ctx['GAMES_INTO_SEASON_home'] = 40; ctx['IS_LATE_SEASON_home'] = 0
    else:
        ctx['WINRATE_away_roll10_away'] = wr
    return ctx


def _print_result(home, away, hp, ap, playoffs):
    label = 'PLAYOFFS' if playoffs else 'Regular Season'
    margin = abs(hp-ap)
    conf = 'High' if margin>0.15 else 'Medium' if margin>0.07 else 'Low'
    cc = 'green' if conf=='High' else 'yellow' if conf=='Medium' else 'red'
    L = 20
    hb = '█'*int(L*hp)+'░'*(L-int(L*hp)); ab = '█'*int(L*ap)+'░'*(L-int(L*ap))
    t = Table(title=f"{home} (home) vs {away} (away) [{label}]", box=box.ROUNDED)
    t.add_column("Team", style="cyan", width=6)
    t.add_column("Win Probability", width=22); t.add_column("Visual", width=24)
    t.add_row(f"[blue]{home}", f"[blue]{hp:.1%}{'  ◀' if hp>ap else ''}", f"[blue]{hb}")
    t.add_row(f"[red]{away}",  f"[red]{ap:.1%}{'  ◀' if ap>hp else ''}",  f"[red]{ab}")
    console.print(); console.print(t)
    console.print(f"[{cc}]Confidence: {conf} ({margin:.1%} margin)\n")


def predict_game(home, away, featured_df, art, playoffs=False):
    home, away = home.upper(), away.upper()
    feats, medians = art['features'], art['medians']
    model = art['model']
    anchor = ['PTS_roll10','WINRATE_roll10']

    def latest(ab):
        r = featured_df[featured_df['TEAM_ABBREVIATION']==ab].dropna(subset=anchor)
        if len(r)==0: raise ValueError(f"Team '{ab}' not found.")
        return r.iloc[-1]
    try: h, a = latest(home), latest(away)
    except ValueError as e: console.print(f"[red]  {e}"); return None, None

    rec = {'IS_PLAYOFFS': int(playoffs), 'PLAYOFF_ROUND': 1 if playoffs else 0,
           'SERIES_GAME_NUM': 0, 'IS_MUST_WIN': 0, 'H2H_HOME_WIN': 0.5,
           'H2H_MEETINGS_THIS_SEASON': 0, 'MONTH': 3, 'DAY_OF_WEEK': 2, 'IS_WEEKEND': 0}
    for col in feats:
        if col in rec: continue
        if col in MATCHUP_FEATURES or col.endswith('_gap'):
            rec[col] = 0; continue
        base = col.replace('_home','').replace('_away','')
        src = h if '_home' in col else a
        rec[col] = src.get(base, np.nan)
    X = pd.DataFrame([rec])[feats].apply(pd.to_numeric, errors='coerce').fillna(medians)
    p = model.predict_proba(X)[0]
    _print_result(home, away, p[1], p[0], playoffs)
    return p[1], p[0]


def predict_with_lineups(home, home_names, away, away_names,
                         featured_df, player_rolling, art, playoffs=False):
    home, away = home.upper(), away.upper()
    feats, medians, model = art['features'], art['medians'], art['model']
    all_players = nba_players_static.get_players()

    h_rows = resolve_lineup(home_names, f'{home} (home)', all_players, player_rolling)
    a_rows = resolve_lineup(away_names, f'{away} (away)', all_players, player_rolling)
    if len(h_rows) < 3 or len(a_rows) < 3:
        console.print("[red]  Need at least 3 players per team."); return None, None

    h_lu, a_lu = aggregate_lineup(h_rows), aggregate_lineup(a_rows)
    mu = compute_custom_matchup(h_rows, a_rows)

    rec = {'IS_PLAYOFFS': int(playoffs), 'PLAYOFF_ROUND': 1 if playoffs else 0,
           'SERIES_GAME_NUM': 0, 'IS_MUST_WIN': 0, 'H2H_HOME_WIN': 0.5,
           'H2H_MEETINGS_THIS_SEASON': 0, 'MONTH': 3, 'DAY_OF_WEEK': 2, 'IS_WEEKEND': 0}
    rec.update(lineup_to_team_context(h_lu, True, medians))
    rec.update(lineup_to_team_context(a_lu, False, medians))
    rec.update(mu)

    # Lineup features + gaps
    gap_map = {'star_pts_gap':'lineup_pts_max','lineup_pm_gap':'lineup_pm_mean',
               'lineup_wr_gap':'lineup_winrate_mean','impact_gap':'lineup_impact_mean',
               'def_gap':'lineup_def_mean','gs_gap':'lineup_gs_mean'}
    for col in LINEUP_FEATURES:
        if col in gap_map:
            k = gap_map[col]; rec[col] = (h_lu.get(k,0) or 0) - (a_lu.get(k,0) or 0)
        elif col.endswith('_gap'): rec[col] = 0
        elif '_home' in col: rec[col] = h_lu.get(col.replace('_home',''), np.nan)
        elif '_away' in col: rec[col] = a_lu.get(col.replace('_away',''), np.nan)

    X = pd.DataFrame([rec]).reindex(columns=feats).apply(pd.to_numeric, errors='coerce').fillna(medians)
    p = model.predict_proba(X)[0]
    _print_result(home, away, p[1], p[0], playoffs)

    # Player tables
    for rows, title, color in [(h_rows, f'Home — {home}', 'blue'), (a_rows, f'Away — {away}', 'red')]:
        t = Table(title=f'{title} (last 10g)', box=box.ROUNDED)
        for c, w in [('Player',24),('Role',6),('PTS',6),('OFF',6),('DEF',6),('TS%',7),('+/-',7)]:
            t.add_column(c, style=color if c=='Player' else None,
                         justify='left' if c in ('Player','Role') else 'right', width=w)
        for r in rows:
            off = r.get('IMPACT_roll10', r.get('PTS_roll10',0)) or 0
            t.add_row(str(r.get('PLAYER_NAME','?')), str(r.get('ROLE','?')),
                      f"{r.get('PTS_roll10',0):.1f}", f"{off:.1f}",
                      f"{r.get('DEF_SCORE_roll10',0):.1f}",
                      f"{r.get('TS_PCT_roll10',0):.3f}", f"{r.get('PLUS_MINUS_roll10',0):+.1f}")
        console.print(t)

    # Show the key matchup read-out
    console.print(f"[dim]Positional edge (home): avg mismatch {mu['mu_off_mismatch_home']-mu['mu_off_mismatch_away']:+.1f}, "
                  f"star vs best defender {mu['mu_home_star_vs_def']:+.1f}[/dim]\n")
    return p[1], p[0]