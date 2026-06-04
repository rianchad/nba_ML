"""
fetch.py — all NBA API data fetching with per-source caching.

Each data source caches to its own file so a crash or partial run never
forces a full re-fetch. Season-level aggregate stats (clutch, hustle,
tracking, opponent, scoring) are fetched once per season; on/off is
per-team-per-season.

All of these season-aggregate sources are merged later using the PRIOR
season's value (handled in team_features) to avoid data leakage.
"""

import os, time, random, joblib
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

import config as C

console = Console()

from nba_api.stats.endpoints import (
    leaguegamelog, leaguedashteamstats, leaguedashteamclutch,
    leaguedashplayerclutch, leaguehustlestatsteam, leaguehustlestatsplayer,
    leaguedashteamshotlocations, leaguedashplayershotlocations,
    leaguedashptstats, leaguedashplayerbiostats,
    teamplayeronoffdetails,
)
from nba_api.stats.static import teams as nba_teams_static


def _sleep():
    time.sleep(random.uniform(C.SLEEP_MIN, C.SLEEP_MAX))


def _progress(desc, total):
    return Progress(SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"))


def _cached(cache_file, build_fn, label):
    """Generic cache wrapper: load if exists else build + save."""
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    if os.path.exists(cache_file):
        console.print(f"  [green]cache hit  -> {cache_file}")
        return joblib.load(cache_file)
    console.print(f"  [yellow]fetching   -> {label}")
    data = build_fn()
    joblib.dump(data, cache_file)
    console.print(f"  [green]cached     -> {cache_file}")
    return data


# ── Core game logs (team + player) ─────────────────────
def _fetch_game_logs():
    def grab(level, stype):
        frames = []
        for season in C.SEASONS:
            _sleep()
            try:
                logs = leaguegamelog.LeagueGameLog(
                    season=season, player_or_team_abbreviation=level,
                    season_type_all_star=stype)
                df = logs.get_data_frames()[0]
                df['SEASON'] = season
                df['IS_PLAYOFFS'] = 1 if stype == 'Playoffs' else 0
                frames.append(df)
            except Exception as e:
                console.print(f"[yellow]    {season} {level} {stype}: {e}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    with _progress("", 1):
        console.print("    team logs...")
        team = pd.concat([grab('T', 'Regular Season'), grab('T', 'Playoffs')], ignore_index=True)
        console.print("    player logs...")
        player = pd.concat([grab('P', 'Regular Season'), grab('P', 'Playoffs')], ignore_index=True)
    return {'team_df': team, 'player_df': player}


def fetch_core():
    return _cached(C.CACHE_CORE, _fetch_game_logs, "core game logs")


# ── Generic season-level team-stat fetcher ─────────────
def _fetch_team_measure(measure_type, per_mode='PerGame'):
    def build():
        frames = []
        for season in C.SEASONS:
            _sleep()
            try:
                s = leaguedashteamstats.LeagueDashTeamStats(
                    season=season,
                    measure_type_detailed_defense=measure_type,
                    per_mode_detailed=per_mode)
                df = s.get_data_frames()[0]; df['SEASON'] = season
                frames.append(df)
            except Exception as e:
                console.print(f"[yellow]    {season} {measure_type}: {e}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return build


def fetch_advanced(): return _cached(C.CACHE_ADVANCED, _fetch_team_measure('Advanced'), "advanced team stats")
def fetch_scoring():  return _cached(C.CACHE_SCORING,  _fetch_team_measure('Scoring'),  "scoring splits")
def fetch_misc():     return _cached(C.CACHE_MISC,     _fetch_team_measure('Misc'),     "misc splits")
def fetch_opponent(): return _cached(C.CACHE_OPPONENT, _fetch_team_measure('Opponent'), "opponent (defense) stats")


# ── Clutch (team + player) ─────────────────────────────
def _fetch_clutch_team():
    frames = []
    for season in C.SEASONS:
        _sleep()
        try:
            s = leaguedashteamclutch.LeagueDashTeamClutch(
                season=season, per_mode_detailed='PerGame',
                measure_type_detailed_defense='Base')
            df = s.get_data_frames()[0]; df['SEASON'] = season; frames.append(df)
        except Exception as e:
            console.print(f"[yellow]    {season} clutch-team: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fetch_clutch_player():
    frames = []
    for season in C.SEASONS:
        _sleep()
        try:
            s = leaguedashplayerclutch.LeagueDashPlayerClutch(
                season=season, per_mode_detailed='PerGame',
                measure_type_detailed_defense='Base')
            df = s.get_data_frames()[0]; df['SEASON'] = season; frames.append(df)
        except Exception as e:
            console.print(f"[yellow]    {season} clutch-player: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_clutch_team():   return _cached(C.CACHE_CLUTCH_T, _fetch_clutch_team,   "team clutch stats")
def fetch_clutch_player(): return _cached(C.CACHE_CLUTCH_P, _fetch_clutch_player, "player clutch stats")


# ── Hustle (team + player) — 2015-16 onward ────────────
def _fetch_hustle_team():
    frames = []
    for season in C.SEASONS:
        _sleep()
        try:
            s = leaguehustlestatsteam.LeagueHustleStatsTeam(
                season=season, per_mode_time='PerGame')
            df = s.get_data_frames()[0]; df['SEASON'] = season; frames.append(df)
        except Exception as e:
            console.print(f"[yellow]    {season} hustle-team: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fetch_hustle_player():
    frames = []
    for season in C.SEASONS:
        _sleep()
        try:
            s = leaguehustlestatsplayer.LeagueHustleStatsPlayer(
                season=season, per_mode_time='PerGame')
            df = s.get_data_frames()[0]; df['SEASON'] = season; frames.append(df)
        except Exception as e:
            console.print(f"[yellow]    {season} hustle-player: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_hustle_team():   return _cached(C.CACHE_HUSTLE_T, _fetch_hustle_team,   "team hustle stats")
def fetch_hustle_player(): return _cached(C.CACHE_HUSTLE_P, _fetch_hustle_player, "player hustle stats")


# ── Shot zones ─────────────────────────────────────────
def _fetch_shotzone():
    frames = []
    for season in C.SEASONS:
        _sleep()
        try:
            s = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
                season=season, per_mode_detailed='PerGame',
                distance_range='By Zone')
            df = s.get_data_frames()[0]
            # Columns are a MultiIndex (zone, stat) — flatten
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join([str(x) for x in c if x]).strip('_') for c in df.columns]
            df['SEASON'] = season
            frames.append(df)
        except Exception as e:
            console.print(f"[yellow]    {season} shotzone: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_shotzone(): return _cached(C.CACHE_SHOTZONE, _fetch_shotzone, "shot-zone stats")


# ── Player shot locations (zone efficiency per player) ──
def _fetch_player_shotzone():
    frames = []
    for season in C.SEASONS:
        _sleep()
        try:
            s = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
                season=season, per_mode_detailed='PerGame',
                distance_range='By Zone')
            df = s.get_data_frames()[0]
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join([str(x) for x in c if x]).strip('_') for c in df.columns]
            df['SEASON'] = season
            frames.append(df)
        except Exception as e:
            console.print(f"[yellow]    {season} player shotzone: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_player_shotzone(): return _cached(C.CACHE_PLAYER_SHOTZONE, _fetch_player_shotzone, "player shot-zone stats")


# ── Player tracking (multiple measure types) ───────────
def _fetch_tracking():
    measure_types = ['Drives', 'CatchShoot', 'PullUpShot', 'Possessions',
                     'Passing', 'Defense']
    out = {}
    for mt in measure_types:
        frames = []
        for season in C.SEASONS:
            _sleep()
            try:
                s = leaguedashptstats.LeagueDashPtStats(
                    season=season, per_mode_simple='PerGame',
                    player_or_team='Player', pt_measure_type=mt)
                df = s.get_data_frames()[0]; df['SEASON'] = season; frames.append(df)
            except Exception as e:
                console.print(f"[yellow]    {season} tracking-{mt}: {e}")
        out[mt] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out


def fetch_tracking(): return _cached(C.CACHE_TRACKING, _fetch_tracking, "player tracking stats")


# ── Player bio (positions, height, age) ────────────────
def _fetch_bio():
    frames = []
    for season in C.SEASONS:
        _sleep()
        try:
            s = leaguedashplayerbiostats.LeagueDashPlayerBioStats(season=season)
            df = s.get_data_frames()[0]; df['SEASON'] = season; frames.append(df)
        except Exception as e:
            console.print(f"[yellow]    {season} bio: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_bio(): return _cached(C.CACHE_BIO, _fetch_bio, "player bio stats")


# ── On/off splits (per team per season) ────────────────
def _fetch_onoff():
    teams = nba_teams_static.get_teams()
    frames = []
    total = len(teams) * len(C.SEASONS)
    done = 0
    for season in C.SEASONS:
        for t in teams:
            _sleep()
            done += 1
            try:
                s = teamplayeronoffdetails.TeamPlayerOnOffDetails(
                    team_id=t['id'], season=season, per_mode_detailed='PerGame')
                dfs = s.get_data_frames()
                # data set [1] = on-court, [2] = off-court (varies by version)
                for idx, tag in [(1, 'ON'), (2, 'OFF')]:
                    if len(dfs) > idx:
                        d = dfs[idx].copy()
                        d['SEASON'] = season; d['SPLIT'] = tag
                        frames.append(d)
            except Exception as e:
                console.print(f"[yellow]    {season} {t['abbreviation']} on/off: {e}")
            if done % 30 == 0:
                console.print(f"      on/off progress: {done}/{total}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_onoff(): return _cached(C.CACHE_ONOFF, _fetch_onoff, "on/off splits (slow ~300 calls)")