"""
fetch_ref_coach.py — fetch and cache referee assignments and head-coach mappings.

Endpoints used:
  - BoxScoreSummaryV3 (Officials table, index 3): referee names per game
  - CommonTeamRoster  (Coaches table,   index 1): head coach name per team/season

Both tables are cached as parquet files under cache/ so only new games / new
team-seasons need network calls on subsequent runs.
"""

import os, time, random
import pandas as pd
from rich.console import Console

import config as C

console = Console()

CACHE_REF_MAP   = f'{C.CACHE_DIR}/ref_game_map.parquet'
CACHE_COACH_MAP = f'{C.CACHE_DIR}/coach_season_map.parquet'

_MAX_RETRIES = 4
_SLEEP_MIN   = 0.3
_SLEEP_MAX   = 0.7


def _sleep():
    time.sleep(random.uniform(_SLEEP_MIN, _SLEEP_MAX))


def _call_with_backoff(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on HTTP/Timeout errors."""
    from requests.exceptions import HTTPError, Timeout, ConnectionError as CError
    delay = 2.0
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            _sleep()
            return fn(*args, **kwargs)
        except (HTTPError, Timeout, CError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                console.print(f"[yellow]    backoff {attempt + 1}/{_MAX_RETRIES - 1}: {exc}")
                time.sleep(min(delay, 30.0))
                delay *= 2
        except Exception:
            raise
    raise last_exc


# ── Referee game map ──────────────────────────────────────────────────────────

def _extract_ref_names(bs) -> list[str]:
    """
    Extract official names from a BoxScoreSummaryV3 response.
    V3: officials at index 3, columns: firstName, familyName.
    Falls back to V2 column names (FIRST_NAME, LAST_NAME) if needed.
    """
    try:
        officials = bs.get_data_frames()[3]
        if 'firstName' in officials.columns and 'familyName' in officials.columns:
            return (officials['firstName'].str.strip() + ' ' +
                    officials['familyName'].str.strip()).tolist()
        if 'FIRST_NAME' in officials.columns and 'LAST_NAME' in officials.columns:
            return (officials['FIRST_NAME'].str.strip() + ' ' +
                    officials['LAST_NAME'].str.strip()).tolist()
    except Exception:
        pass
    return []


def fetch_ref_game_map(game_ids):
    """
    Return a DataFrame[GAME_ID, REF_1, REF_2, REF_3] mapping each game to its
    assigned officials, fetched via BoxScoreSummaryV3 (Officials table, index 3).

    Results are cached to cache/ref_game_map.parquet and resumed on re-runs so
    only unseen GAME_IDs trigger network calls.
    """
    from nba_api.stats.endpoints import boxscoresummaryv3

    os.makedirs(C.CACHE_DIR, exist_ok=True)
    existing: dict[str, list[str]] = {}

    if os.path.exists(CACHE_REF_MAP):
        cached = pd.read_parquet(CACHE_REF_MAP)
        existing = {
            row['GAME_ID']: [row['REF_1'], row['REF_2'], row['REF_3']]
            for _, row in cached.iterrows()
        }
        console.print(f"  [green]ref map cache hit -> {len(existing):,} games")

    remaining = [g for g in game_ids if g not in existing]
    if remaining:
        console.print(f"  [yellow]fetching ref assignments for {len(remaining):,} games...")
        for i, gid in enumerate(remaining):
            try:
                bs    = _call_with_backoff(boxscoresummaryv3.BoxScoreSummaryV3, game_id=gid)
                names = _extract_ref_names(bs)
            except Exception as exc:
                console.print(f"[yellow]    {gid}: {exc}")
                names = []

            while len(names) < 3:
                names.append('')
            existing[gid] = names[:3]

            if (i + 1) % 100 == 0:
                _save_ref_map(existing)
                console.print(f"  [cyan]    checkpoint {i + 1}/{len(remaining)}")

        _save_ref_map(existing)
        console.print(f"  [green]ref map saved -> {CACHE_REF_MAP}")

    rows = [
        {'GAME_ID': gid, 'REF_1': refs[0], 'REF_2': refs[1], 'REF_3': refs[2]}
        for gid, refs in existing.items()
    ]
    return pd.DataFrame(rows)


def _save_ref_map(data: dict):
    rows = [
        {'GAME_ID': gid, 'REF_1': refs[0], 'REF_2': refs[1], 'REF_3': refs[2]}
        for gid, refs in data.items()
    ]
    pd.DataFrame(rows).to_parquet(CACHE_REF_MAP, index=False)


# ── Coach season map ──────────────────────────────────────────────────────────

def fetch_coach_season_map(team_season_pairs):
    """
    Return a DataFrame[TEAM_ID, SEASON, COACH_NAME] mapping each team/season to
    its head coach, fetched via CommonTeamRoster (Coaches table, index 1).

    Results are cached to cache/coach_season_map.parquet.
    """
    from nba_api.stats.endpoints import commonteamroster

    os.makedirs(C.CACHE_DIR, exist_ok=True)
    existing: dict[tuple, str] = {}

    if os.path.exists(CACHE_COACH_MAP):
        cached = pd.read_parquet(CACHE_COACH_MAP)
        existing = {
            (int(row['TEAM_ID']), row['SEASON']): row['COACH_NAME']
            for _, row in cached.iterrows()
        }
        console.print(f"  [green]coach map cache hit -> {len(existing):,} team-seasons")

    remaining = [(int(t), s) for t, s in team_season_pairs if (int(t), s) not in existing]
    if remaining:
        console.print(f"  [yellow]fetching coach assignments for {len(remaining):,} team-seasons...")
        for team_id, season in remaining:
            try:
                roster    = _call_with_backoff(
                    commonteamroster.CommonTeamRoster,
                    team_id=str(team_id), season=season,
                )
                coaches   = roster.get_data_frames()[1]
                head      = _extract_head_coach(coaches)
                existing[(team_id, season)] = head
            except Exception as exc:
                console.print(f"[yellow]    ({team_id}, {season}): {exc}")
                existing[(team_id, season)] = ''

        _save_coach_map(existing)
        console.print(f"  [green]coach map saved -> {CACHE_COACH_MAP}")

    rows = [
        {'TEAM_ID': t, 'SEASON': s, 'COACH_NAME': name}
        for (t, s), name in existing.items()
    ]
    return pd.DataFrame(rows)


def _extract_head_coach(coaches_df: pd.DataFrame) -> str:
    """Return 'FIRST LAST' for the head coach row, or '' if not found."""
    if coaches_df.empty:
        return ''
    # Primary: IS_ASSISTANT == 0 flags the head coach
    if 'IS_ASSISTANT' in coaches_df.columns:
        head = coaches_df[coaches_df['IS_ASSISTANT'] == 0]
        if not head.empty:
            return _name(head.iloc[0])
    # Fallback: COACH_TYPE contains 'Head'
    if 'COACH_TYPE' in coaches_df.columns:
        head = coaches_df[coaches_df['COACH_TYPE'].str.contains('Head', case=False, na=False)]
        if not head.empty:
            return _name(head.iloc[0])
    return ''


def _name(row) -> str:
    fn = str(row.get('FIRST_NAME', '')).strip()
    ln = str(row.get('LAST_NAME', '')).strip()
    return f"{fn} {ln}".strip()


def _save_coach_map(data: dict):
    rows = [
        {'TEAM_ID': t, 'SEASON': s, 'COACH_NAME': name}
        for (t, s), name in data.items()
    ]
    pd.DataFrame(rows).to_parquet(CACHE_COACH_MAP, index=False)
