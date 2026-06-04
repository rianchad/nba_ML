"""
fetch_pbp.py — resumable play-by-play fetch + per-game feature extraction.

This is the overnight job (~13,000 calls). It is fully resumable: progress
is saved after every batch, so if it crashes or you stop it, rerunning
picks up exactly where it left off.

For each game it derives team-level features that are otherwise impossible
to get from box scores:
  - scoring runs (largest run, number of 6-0+ runs for and against)
  - quarter-by-quarter point differential (Q1..Q4)
  - lead changes, times tied
  - largest lead, comeback magnitude
  - clutch (last 5 min, within 5) net points

These are stored per (GAME_ID, TEAM_ID) and later rolled like any other
per-game stat (so they reflect current-season form without leakage).
"""

import os, time, random, joblib
import numpy as np
import pandas as pd
from rich.console import Console

import config as C

console = Console()

from nba_api.stats.endpoints import playbyplayv3


def _pbp_sleep():
    time.sleep(random.uniform(C.PBP_SLEEP_MIN, C.PBP_SLEEP_MAX))


def _process_one_game(pbp_df, game_id):
    """
    Turn a single game's play-by-play (V3 schema) into team-level features.
    V3 has scoreHome / scoreAway columns (int, 0 when no score yet).
    Returns a dict keyed by GAME_ID with home/away perspective features.
    """
    df = pbp_df.copy()
    # Keep only rows where a score is recorded (both columns non-zero after tip-off)
    df['scoreHome'] = pd.to_numeric(df['scoreHome'], errors='coerce').fillna(0)
    df['scoreAway'] = pd.to_numeric(df['scoreAway'], errors='coerce').fillna(0)
    df['period'] = pd.to_numeric(df['period'], errors='coerce').fillna(1).astype(int)

    # Build margin series (home - away), forward-fill within the sequence
    scored = df[(df['scoreHome'] > 0) | (df['scoreAway'] > 0)].copy()
    if len(scored) == 0:
        return None
    scored['MARGIN'] = (scored['scoreHome'] - scored['scoreAway']).astype(float)
    df = scored  # work with scored rows only

    margins = df['MARGIN'].values
    # Lead changes = sign flips of margin
    signs = np.sign(margins)
    lead_changes = int(np.sum(np.abs(np.diff(signs[signs != 0])) == 2)) if len(signs[signs != 0]) > 1 else 0
    times_tied = int(np.sum(margins == 0))
    largest_lead_home = float(np.max(margins)) if len(margins) else 0.0
    largest_lead_away = float(-np.min(margins)) if len(margins) else 0.0

    # Quarter point differentials (home perspective) from end-of-quarter margins
    q_diffs = {}
    prev_margin = 0
    for q in [1, 2, 3, 4]:
        qd = df[df['period'] == q]
        if len(qd):
            end_margin = qd['MARGIN'].iloc[-1]
            q_diffs[f'Q{q}_DIFF'] = float(end_margin - prev_margin)
            prev_margin = end_margin
        else:
            q_diffs[f'Q{q}_DIFF'] = 0.0

    # Clutch: period >= 4 and within 5 points, final stretch
    clutch = df[(df['period'] >= 4) & (df['MARGIN'].abs() <= 5)]
    clutch_net_home = float(clutch['MARGIN'].iloc[-1] - clutch['MARGIN'].iloc[0]) if len(clutch) > 1 else 0.0

    # Scoring runs (home perspective): consecutive margin increases
    run_for_home = 0; max_run_home = 0
    run_for_away = 0; max_run_away = 0
    diffs = np.diff(margins)
    for d in diffs:
        if d > 0:
            run_for_home += d; run_for_away = 0
            max_run_home = max(max_run_home, run_for_home)
        elif d < 0:
            run_for_away += -d; run_for_home = 0
            max_run_away = max(max_run_away, run_for_away)
    # Comeback: did the trailing team end up winning? magnitude = largest deficit overcome by winner
    final_margin = margins[-1]
    if final_margin > 0:   # home won
        comeback_home = float(max(0, -np.min(margins)))   # biggest deficit home overcame
        comeback_away = 0.0
    elif final_margin < 0: # away won
        comeback_away = float(max(0, np.max(margins)))
        comeback_home = 0.0
    else:
        comeback_home = comeback_away = 0.0

    return {
        'GAME_ID': game_id,
        'pbp_lead_changes': lead_changes,
        'pbp_times_tied': times_tied,
        'pbp_largest_lead_home': largest_lead_home,
        'pbp_largest_lead_away': largest_lead_away,
        'pbp_max_run_home': float(max_run_home),
        'pbp_max_run_away': float(max_run_away),
        'pbp_clutch_net_home': clutch_net_home,
        'pbp_comeback_home': comeback_home,
        'pbp_comeback_away': comeback_away,
        **q_diffs,
    }


def fetch_and_process_pbp(team_df, batch_save=50):
    """
    Fetch play-by-play for every game in team_df and process to features.
    Resumable — saves progress to CACHE_PBP_RAW after each batch.
    Final processed features cached to CACHE_PBP.
    """
    os.makedirs(C.CACHE_DIR, exist_ok=True)

    if os.path.exists(C.CACHE_PBP):
        console.print(f"  [green]cache hit  -> {C.CACHE_PBP}")
        return joblib.load(C.CACHE_PBP)

    # Unique games (one row per game, not per team)
    games = team_df[['GAME_ID']].drop_duplicates()['GAME_ID'].tolist()

    # Resume
    processed = {}
    if os.path.exists(C.CACHE_PBP_RAW):
        processed = joblib.load(C.CACHE_PBP_RAW)
        console.print(f"  [green]resuming — {len(processed):,}/{len(games):,} games already done")

    remaining = [g for g in games if g not in processed]
    console.print(f"  [yellow]play-by-play: {len(remaining):,} games to fetch "
                  f"(~{len(remaining)*0.5/60:.0f} min)")

    for i, gid in enumerate(remaining, 1):
        _pbp_sleep()
        try:
            pbp = playbyplayv3.PlayByPlayV3(game_id=gid).get_data_frames()[0]
            feats = _process_one_game(pbp, gid)
            if feats:
                processed[gid] = feats
        except Exception as e:
            console.print(f"[yellow]    {gid}: {e}")
            processed[gid] = {'GAME_ID': gid}   # mark done so we don't retry forever
        if i % batch_save == 0:
            joblib.dump(processed, C.CACHE_PBP_RAW)
            console.print(f"      saved progress: {len(processed):,}/{len(games):,}")

    joblib.dump(processed, C.CACHE_PBP_RAW)
    pbp_df = pd.DataFrame([v for v in processed.values() if len(v) > 1])
    joblib.dump(pbp_df, C.CACHE_PBP)
    console.print(f"  [green]play-by-play processed -> {len(pbp_df):,} games")
    return pbp_df


def attach_pbp_to_team(team_df, pbp_df):
    """
    pbp_df is keyed by GAME_ID with home/away features.
    Map them onto each team row based on whether the team was home or away.
    """
    if pbp_df is None or pbp_df.empty:
        return team_df

    df = team_df.merge(pbp_df, on='GAME_ID', how='left')
    is_home = df['IS_HOME'] == 1

    # Convert home/away-keyed pbp features into team-perspective columns
    pairs = [
        ('pbp_largest_lead', 'pbp_largest_lead_home', 'pbp_largest_lead_away'),
        ('pbp_max_run',      'pbp_max_run_home',      'pbp_max_run_away'),
        ('pbp_comeback',     'pbp_comeback_home',     'pbp_comeback_away'),
    ]
    for out, hcol, acol in pairs:
        if hcol in df.columns and acol in df.columns:
            df[out] = np.where(is_home, df[hcol], df[acol])
    if 'pbp_clutch_net_home' in df.columns:
        df['pbp_clutch_net'] = np.where(is_home, df['pbp_clutch_net_home'], -df['pbp_clutch_net_home'])
    for q in [1, 2, 3, 4]:
        col = f'Q{q}_DIFF'
        if col in df.columns:
            df[f'pbp_q{q}_diff'] = np.where(is_home, df[col], -df[col])
    for shared in ['pbp_lead_changes', 'pbp_times_tied']:
        if shared in df.columns:
            df[shared] = df[shared]   # game-level, same for both teams
    return df
