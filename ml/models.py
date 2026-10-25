"""The two classifiers, built identically everywhere (so scripts cannot drift apart)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build_rf(seed: int = 42, n_estimators: int = 200, balance: bool = False, n_jobs: int = -1):
    """Random Forest. balance=True re-weights the rare class (use when classes are very uneven)."""
    return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs,
                                  class_weight="balanced" if balance else None)


def build_xgb(seed: int = 42, y_train=None, n_estimators: int = 300, balance: bool = False, n_jobs: int = -1):
    """XGBoost. balance=True sets scale_pos_weight = negatives / positives."""
    from xgboost import XGBClassifier  # imported here so RF-only scripts work without xgboost

    spw = 1.0
    if balance and y_train is not None:
        pos = max(int(np.sum(y_train)), 1)
        spw = float(len(y_train) - pos) / pos
    return XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=n_jobs,
                         tree_method="hist", scale_pos_weight=spw)
