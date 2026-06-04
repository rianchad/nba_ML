"""
graphs.py — analysis and diagnostic plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console

from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import config as C

console = Console()
plt.rcParams.update(C.PLOT_RC)


def _save(fig, name):
    fig.savefig(f'{name}.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    console.print(f"  [green]Saved: {name}.png")
    plt.close(fig)


def calibration(res):
    fig, ax = plt.subplots(figsize=(7,5))
    CalibrationDisplay.from_predictions(res['y_test'], res['y_proba'],
                                        n_bins=10, ax=ax, name='Model', color=C.BLUE)
    ax.plot([0,1],[0,1],'w--',alpha=0.4); ax.set_title('Calibration Curve', pad=12)
    _save(fig, 'g01_calibration')


def feature_importance(res):
    top = res['imp_df'].head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11,9))
    def color(f):
        if f.startswith('mu_'): return C.TEAL
        if f.startswith('pbp_'): return C.ORANGE
        if 'lineup' in f: return C.GREEN
        if 'PRIOR' in f: return C.PURPLE
        return C.BLUE if '_home' in f else C.RED
    ax.barh(top['feature'], top['importance'], color=[color(f) for f in top['feature']], edgecolor='none')
    ax.set_title('Top 25 Feature Importances\nTeal=matchup Orange=play-by-play Green=lineup Purple=prior-season', pad=12)
    ax.grid(axis='x'); _save(fig, 'g02_feature_importance')


def probability_dist(res):
    y_t, y_p = np.array(res['y_test']), res['y_proba']
    fig, ax = plt.subplots(figsize=(8,5)); bins = np.linspace(0,1,30)
    ax.hist(y_p[y_t==0], bins=bins, alpha=0.7, color=C.RED, label='Home lost', density=True)
    ax.hist(y_p[y_t==1], bins=bins, alpha=0.7, color=C.GREEN, label='Home won', density=True)
    ax.axvline(0.5, color='white', ls='--', alpha=0.5); ax.legend()
    ax.set_title('Predicted Probability Distribution', pad=12)
    _save(fig, 'g03_probability_dist')


def confusion(res):
    fig, ax = plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay(confusion_matrix(res['y_test'], res['y_pred']),
                           display_labels=['Away','Home']).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix [{C.TEST_SEASON}]', pad=12)
    for t in ax.texts: t.set_color('white'); t.set_fontsize(14)
    _save(fig, 'g04_confusion_matrix')


def home_court(matchups):
    hca = matchups.groupby('SEASON')['HOME_WIN'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(hca['SEASON'], hca['HOME_WIN'], color=C.BLUE, marker='o', lw=2, ms=8)
    ax.axhline(0.5, color='white', ls='--', alpha=0.3)
    ax.fill_between(hca['SEASON'], 0.5, hca['HOME_WIN'], alpha=0.15, color=C.BLUE)
    ax.set_ylim(0.4,0.7); ax.tick_params(axis='x', rotation=45)
    ax.set_title('Home Court Advantage Over Time', pad=12)
    _save(fig, 'g05_home_court')


def _binned_winrate(matchups, col, title, fname, color=C.BLUE, bins=6):
    if col not in matchups.columns: return
    b = pd.cut(matchups[col], bins=bins)
    grp = matchups.groupby(b, observed=True)['HOME_WIN'].agg(['mean','count']).reset_index()
    grp = grp[grp['count'] >= 30]
    if len(grp) == 0: return
    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(range(len(grp)), grp['mean'], color=color, alpha=0.8, edgecolor='none')
    ax.axhline(0.5, color='white', ls='--', alpha=0.4); ax.set_ylim(0,0.9)
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels([str(x) for x in grp[col]], rotation=25, fontsize=8)
    ax.set_title(title, pad=12); _save(fig, fname)


def four_factors(matchups):
    factors = {'EFG_PCT_roll10':'eFG%','TOV_PCT_roll10':'TOV rate',
               'OREB_RATE_roll10':'OREB rate','FT_RATE_roll10':'FT rate'}
    corr = {}
    for col, lab in factors.items():
        h, a = f'{col}_home', f'{col}_away'
        if h in matchups.columns and a in matchups.columns:
            corr[lab] = (matchups[h]-matchups[a]).corr(matchups['HOME_WIN'])
    if not corr: return
    fig, ax = plt.subplots(figsize=(8,5))
    labels, vals = zip(*sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True))
    ax.barh(labels, vals, color=[C.GREEN if v>0 else C.RED for v in vals], edgecolor='none')
    ax.axvline(0, color='white', ls='--', alpha=0.4)
    ax.set_title("Four Factors — Correlation with Home Win", pad=12)
    _save(fig, 'g06_four_factors')


def matchup_mismatch(matchups):
    """Does the positional offensive-mismatch edge predict winning?"""
    if 'mu_off_mismatch_home' not in matchups.columns: return
    matchups = matchups.copy()
    matchups['net_mismatch'] = matchups['mu_off_mismatch_home'] - matchups.get('mu_off_mismatch_away', 0)
    _binned_winrate(matchups, 'net_mismatch',
                    'Positional Matchup Edge vs Win Rate\n(home offensive mismatch minus away)',
                    'g07_matchup_edge', color=C.TEAL, bins=6)


def star_vs_defender(matchups):
    if 'mu_home_star_vs_def' not in matchups.columns: return
    _binned_winrate(matchups, 'mu_home_star_vs_def',
                    'Home Star vs Away Best Defender\n(star offense minus defender quality)',
                    'g08_star_vs_def', color=C.TEAL, bins=6)


def clutch_analysis(matchups):
    if 'PRIOR_CLUTCH_WPCT_home' not in matchups.columns: return
    matchups = matchups.copy()
    matchups['clutch_gap'] = matchups['PRIOR_CLUTCH_WPCT_home'] - matchups.get('PRIOR_CLUTCH_WPCT_away', 0)
    _binned_winrate(matchups, 'clutch_gap',
                    'Prior-Season Clutch Win% Edge vs Win Rate',
                    'g09_clutch', color=C.YELLOW, bins=6)


def travel_analysis(matchups):
    if 'TRAVEL_KM_away' not in matchups.columns: return
    _binned_winrate(matchups, 'TRAVEL_KM_away',
                    'Away Team Travel Distance vs Home Win Rate\n(more travel = home edge?)',
                    'g10_travel', color=C.ORANGE, bins=6)


def quarter_analysis(matchups):
    """Play-by-play: does Q4 performance predict wins?"""
    if 'pbp_q4_diff_roll10_home' not in matchups.columns: return
    matchups = matchups.copy()
    matchups['q4_gap'] = matchups['pbp_q4_diff_roll10_home'] - matchups.get('pbp_q4_diff_roll10_away', 0)
    _binned_winrate(matchups, 'q4_gap',
                    'Rolling Q4 Point-Diff Edge vs Win Rate (from play-by-play)',
                    'g11_q4', color=C.ORANGE, bins=6)


def rolling_accuracy(res):
    r = pd.DataFrame({'date': pd.to_datetime(res['dates']),
                      'correct': (np.array(res['y_test'])==np.array(res['y_pred'])).astype(int)}
                     ).sort_values('date')
    r['roll'] = r['correct'].rolling(30, min_periods=10).mean()
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(r['date'], r['roll'], color=C.BLUE, lw=2)
    ax.axhline(r['correct'].mean(), color=C.YELLOW, ls='--', label=f"Avg: {r['correct'].mean():.1%}")
    ax.axhline(0.5, color='white', ls=':', alpha=0.3)
    ax.set_ylim(0.3,0.9); ax.legend(); ax.tick_params(axis='x', rotation=30)
    ax.set_title(f'Rolling Accuracy [{C.TEST_SEASON}]', pad=12)
    _save(fig, 'g12_rolling_accuracy')


def feature_correlation(matchups, res):
    feats = [f for f in res['features'][:30] if f in matchups.columns]
    sub = matchups[feats + ['HOME_WIN']].dropna()
    corr = sub.corr()[['HOME_WIN']].drop('HOME_WIN').sort_values('HOME_WIN', ascending=False)
    fig, ax = plt.subplots(figsize=(6,13))
    ax.barh(corr.index, corr['HOME_WIN'],
            color=[C.GREEN if v>0 else C.RED for v in corr['HOME_WIN']], edgecolor='none')
    ax.axvline(0, color='white', ls='--', alpha=0.4)
    ax.set_title('Feature Correlation with Home Win', pad=12)
    _save(fig, 'g13_feature_correlation')


def generate_all(res, matchups):
    console.print("\n[bold cyan]Generating graphs...")
    calibration(res)
    feature_importance(res)
    probability_dist(res)
    confusion(res)
    home_court(matchups)
    four_factors(matchups)
    matchup_mismatch(matchups)
    star_vs_defender(matchups)
    clutch_analysis(matchups)
    travel_analysis(matchups)
    quarter_analysis(matchups)
    rolling_accuracy(res)
    feature_correlation(matchups, res)
    console.print("[green]  Graphs saved.\n")