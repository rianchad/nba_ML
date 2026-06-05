"""
main.py — orchestrates the full pipeline.

Run order:
  1.  python fetch_overnight.py     (optional, run once overnight for play-by-play)
  2.  python main.py                (fetches fast data, trains, predicts)

The first run of main.py fetches all the fast season-level data (~10 min)
and caches it. Play-by-play is loaded from cache if fetch_overnight.py has
been run; otherwise the model simply trains without play-by-play features.
"""

import os, joblib
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

import config as C
import fetch
import fetch_pbp
import fetch_player_pbp
import player_features as pf
import player_advanced_features as paf
import team_features as tf
import matchups as mu
import ref_coach_features as rcf
import lineup_advanced_features as laf
import model as M
import graphs as G
import predict as P

console = Console()


def build_everything():
    # ── 1. Fetch ──────────────────────────────────────
    console.print("\n[bold cyan][1/5] Fetching data (cached per-source)...")
    core = fetch.fetch_core()
    team_df, player_df = core['team_df'], core['player_df']

    sources = {}
    sources['advanced'] = fetch.fetch_advanced()
    sources['scoring']  = fetch.fetch_scoring()
    sources['misc']     = fetch.fetch_misc()
    sources['opponent'] = fetch.fetch_opponent()
    sources['clutch_t'] = fetch.fetch_clutch_team()
    sources['hustle_t'] = fetch.fetch_hustle_team()
    sources['shotzone'] = fetch.fetch_shotzone()
    clutch_p = fetch.fetch_clutch_player()
    hustle_p = fetch.fetch_hustle_player()
    bio_df   = fetch.fetch_bio()
    player_shotzone = fetch.fetch_player_shotzone()
    onoff_df = fetch.fetch_onoff()

    # Play-by-play (only if overnight fetch already produced the cache)
    pbp_df = None
    player_clutch_df = None
    quarter_runs_df = None
    if os.path.exists(C.CACHE_PBP):
        pbp_df = joblib.load(C.CACHE_PBP)
        console.print(f"  [green]play-by-play cache found ({len(pbp_df):,} games)")
        # Extract per-player clutch stats and quarter runs from PBP
        player_clutch_df = fetch_player_pbp.extract_player_clutch_stats(pbp_df)
        quarter_runs_df = fetch_player_pbp.extract_quarter_runs(pbp_df)
        console.print(f"  [cyan]extracted player clutch stats ({len(player_clutch_df):,} player-games)")
        console.print(f"  [cyan]extracted quarter runs ({len(quarter_runs_df):,} team-games)")
    else:
        console.print("  [yellow]no play-by-play cache — run fetch_overnight.py to add it")

    # ── 2. Player features ────────────────────────────
    console.print("\n[bold cyan][2/5] Player features...")
    player_rolling = pf.compute_player_rolling(player_df, bio_df=bio_df,
                                               hustle_p=hustle_p, clutch_p=clutch_p,
                                               player_clutch_df=player_clutch_df,
                                               player_shotzone_df=player_shotzone)
    lineup_features = pf.build_lineup_features(player_rolling)

    # ── 3. Team features ──────────────────────────────
    console.print("\n[bold cyan][3/5] Team features...")
    featured = tf.initial_prep(team_df)
    if pbp_df is not None:
        featured = fetch_pbp.attach_pbp_to_team(featured, pbp_df)
    # Re-run full team pipeline (initial_prep is idempotent enough; we re-derive)
    featured = tf.engineer_team_features(
        featured if pbp_df is not None else team_df, sources)
    if pbp_df is not None:
        # attach pbp again post-pipeline isn't needed; add_pbp_rolling ran inside pipeline
        pass
    featured = mu.merge_lineup_onto_team(featured, lineup_features)

    # ── 4. Matchups + train ───────────────────────────
    console.print("\n[bold cyan][4/5] Matchups + positional matchups + train...")
    console.print("  [cyan]Computing player-vs-player positional matchups...")
    matchup_feats = mu.compute_matchup_features(
        player_rolling, featured[['GAME_ID','TEAM_ID','IS_HOME']])
    matchups = mu.build_matchups(featured, matchup_feats)
    matchups = mu.add_h2h(matchups)

    console.print("  [cyan]Adding advanced lineup features...")
    matchups = laf.build_lineup_onoff_splits(matchups, onoff_df)
    matchups = laf.build_quarter_run_features(matchups, quarter_runs_df)
    matchups = laf.aggregate_player_advanced_to_lineup(matchups, player_rolling)

    console.print("  [cyan]Adding referee features...")
    matchups = rcf.build_ref_features(matchups, team_df)
    console.print("  [cyan]Adding coach features...")
    matchups = rcf.build_coach_features(matchups, team_df)
    console.print(f"  [cyan]Total matchups: {len(matchups):,}")
    art = M.train_model(matchups)

    # Save model bundle for export/website
    joblib.dump({'model': art['model'], 'features': art['features'],
                 'medians': art['medians']}, C.MODEL_FILE)
    console.print(f"  [green]Model saved -> {C.MODEL_FILE}")

    return featured, player_rolling, matchups, art


def main():
    console.print(Panel("[bold cyan]NBA Win Probability Predictor — Full Build",
                        border_style="cyan", padding=(1,2)))
    featured, player_rolling, matchups, art = build_everything()

    console.print("\n[bold cyan][5/5] Graphs...")
    G.generate_all(art, matchups)

    console.print("[bold cyan]=== Examples ===")
    P.predict_game('BOS','MIA', featured, art)
    P.predict_game('OKC','DEN', featured, art)
    P.predict_with_lineups(
        'BOS', ['Jayson Tatum','Jaylen Brown','Jrue Holiday','Kristaps Porzingis','Al Horford'],
        'MIA', ['Jimmy Butler','Tyler Herro','Bam Adebayo','Duncan Robinson','Kyle Lowry'],
        featured, player_rolling, art)

    # Interactive
    console.print(Panel("[cyan]1. Team mode\n2. Lineup mode",
                        title="[bold cyan]Interactive", border_style="cyan", padding=(1,2)))
    while True:
        try:
            choice = Prompt.ask("\n[bold cyan]Choose (1/2, Ctrl+C to exit)", choices=['1','2'])
            playoffs = Prompt.ask("  Playoffs? (y/n)", choices=['y','n']) == 'y'
            if choice == '1':
                h = Prompt.ask("  Home team").strip()
                a = Prompt.ask("  Away team").strip()
                P.predict_game(h, a, featured, art, playoffs=playoffs)
            else:
                h = Prompt.ask("  Home team").strip().upper()
                hn = [Prompt.ask(f"    {h} player {i}").strip() for i in range(1,6)]
                a = Prompt.ask("  Away team").strip().upper()
                an = [Prompt.ask(f"    {a} player {i}").strip() for i in range(1,6)]
                P.predict_with_lineups(h, hn, a, an, featured, player_rolling, art, playoffs=playoffs)
        except KeyboardInterrupt:
            console.print("\n[cyan]Done."); break
    

if __name__ == '__main__':
    main()
