"""
export_for_mac.py — bundle the trained model + runtime data into a single
file that can be transferred to a Mac and used for predictions without
re-fetching any data from the NBA API.

Usage (run after main.py has already built and cached everything):
    python export_for_mac.py

Output:
    nba_mac_export.pkl  (~20–80 MB depending on cache state)

Transfer to Mac:
    scp nba_mac_export.pkl user@macbook:~/nba/
    # or AirDrop / USB / cloud drive
"""

import os
import joblib
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    console.print(Panel(
        "[bold cyan]NBA Model — Mac Export",
        border_style="cyan", padding=(1, 2),
    ))

    # ── Build / reload from cache ─────────────────────────────────────────────
    console.print("\n[bold cyan]Loading pipeline (uses cached data — no API calls)...")
    from main import build_everything
    featured, player_rolling, _, art = build_everything()

    # ── Bundle ────────────────────────────────────────────────────────────────
    bundle = {
        'model':          art['model'],
        'features':       art['features'],
        'medians':        art['medians'],
        'featured_df':    featured,
        'player_rolling': player_rolling,
    }

    out = 'nba_mac_export.pkl'
    console.print(f"\n[bold cyan]Saving export bundle -> [green]{out}")
    joblib.dump(bundle, out, compress=3)

    size_mb = os.path.getsize(out) / 1_048_576
    console.print(f"[green]Done! File size: {size_mb:.1f} MB")
    console.print("\n[cyan]Transfer to Mac:")
    console.print(f"  [white]scp {out} user@macbook:~/nba/")
    console.print("  [dim]— or use AirDrop / USB / cloud drive")
    console.print("\nOn the Mac, run:")
    console.print("  [white]python predict_mac.py")


if __name__ == '__main__':
    main()
