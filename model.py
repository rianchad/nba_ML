"""
model.py — train the calibrated gradient-boosting classifier.
"""

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

import config as C
from feature_list import FEATURE_COLS

console = Console()


def train_model(matchups):
    matchups = matchups.loc[:, ~matchups.columns.duplicated()]
    available = list(dict.fromkeys(c for c in FEATURE_COLS if c in matchups.columns))
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        console.print(f"  [yellow]{len(missing)} listed features not present "
                      f"(missing data) — skipping them")

    keep = available + ['HOME_WIN', 'SEASON', 'GAME_DATE_home']
    data = matchups[keep].dropna(subset=['HOME_WIN'])
    train = data[data['SEASON'] != C.TEST_SEASON]
    test  = data[data['SEASON'] == C.TEST_SEASON]

    train_feats = train[available].apply(pd.to_numeric, errors='coerce')
    test_feats  = test[available].apply(pd.to_numeric, errors='coerce')
    medians = train_feats.median()
    X_train = train_feats.fillna(medians)
    y_train = train['HOME_WIN'].astype(int)
    X_test  = test_feats.fillna(medians)
    y_test  = test['HOME_WIN'].astype(int)
    dates   = test['GAME_DATE_home'].values

    console.print(f"  Training: [cyan]{len(X_train):,}[/] | Test: [cyan]{len(X_test):,}[/] | "
                  f"Features: [cyan]{len(available)}")

    base = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_child_weight=25, random_state=42,
        device='cuda', tree_method='hist', eval_metric='logloss',
        verbosity=0)
    model = CalibratedClassifierCV(base, cv=5, method='isotonic')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)

    console.print(Panel(f"[bold cyan]Test Accuracy ({C.TEST_SEASON}): {acc:.1%}",
                        border_style="cyan", padding=(1, 2)))
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))

    base_fit = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_child_weight=25, random_state=42,
        device='cuda', tree_method='hist', eval_metric='logloss',
        verbosity=0).fit(X_train, y_train)
    imp_df = pd.DataFrame({'feature': X_train.columns.tolist(),
                           'importance': base_fit.feature_importances_}
                          ).sort_values('importance', ascending=False)
    console.print("  [cyan]Top 20 features:")
    for _, r in imp_df.head(20).iterrows():
        console.print(f"    {r['feature']:<42} [cyan]{'█'*int(r['importance']*500)}[/]  {r['importance']:.4f}")

    train_cols = X_train.columns.tolist()
    return {
        'model': model, 'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred,
        'y_proba': y_proba, 'features': train_cols, 'medians': medians,
        'imp_df': imp_df, 'dates': dates,
    }