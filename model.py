"""
model.py — train the calibrated gradient-boosting classifier.
"""

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

    naive_acc = max(y_test.mean(), 1 - y_test.mean())
    console.print(Panel(
        f"[bold cyan]Test Accuracy ({C.TEST_SEASON}): {acc:.1%}\n"
        f"[white]Naive baseline (always pick home): {naive_acc:.1%}\n"
        f"[{'green' if acc-naive_acc > 0.08 else 'yellow'}]Lift over naive: {acc-naive_acc:+.1%}",
        border_style="cyan", padding=(1, 2)))
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))

    # Accuracy by confidence band
    eval_df = pd.DataFrame({'prob': y_proba, 'actual': y_test.values})
    eval_df['margin'] = (eval_df['prob'] - 0.5).abs()
    eval_df['correct'] = (eval_df['prob'] > 0.5) == eval_df['actual'].astype(bool)
    bands = [
        ('Close   (<10% from 50/50)', eval_df['margin'] < 0.10),
        ('Medium  (10–20%)',          eval_df['margin'].between(0.10, 0.20)),
        ('Confident (>20%)',          eval_df['margin'] > 0.20),
    ]
    console.print("  [bold cyan]Accuracy by model confidence:")
    for label, mask in bands:
        sub = eval_df[mask]
        if len(sub) == 0:
            continue
        pct = sub['correct'].mean()
        bar = '█' * int(pct * 30) + '░' * (30 - int(pct * 30))
        color = 'green' if pct > 0.72 else 'yellow' if pct > 0.60 else 'red'
        console.print(f"    {label}: [{color}]{bar} {pct:.1%}[/] ({len(sub)} games)")

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