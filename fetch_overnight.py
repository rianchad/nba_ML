"""
fetch_overnight.py — run this ONCE, overnight, to fetch and process all
play-by-play data (~13,000 API calls, roughly 2-3 hours).

It is fully resumable: if it crashes or you stop it (Ctrl+C), just run it
again and it continues from where it left off. When finished it writes
cache/pbp_features.pkl, which main.py automatically picks up on its next run.

Usage:
    python fetch_overnight.py
"""

from rich.console import Console
import fetch
import fetch_pbp

console = Console()

if __name__ == '__main__':
    console.print("[bold cyan]Overnight play-by-play fetch\n")
    console.print("[yellow]This will take ~2-3 hours. Safe to stop and resume.\n")

    core = fetch.fetch_core()      # uses cache if main.py already fetched it
    team_df = core['team_df']
    fetch_pbp.fetch_and_process_pbp(team_df, batch_save=50)

    console.print("\n[green]Done. Play-by-play features cached.")
    console.print("[green]Next time you run main.py they'll be included automatically.")