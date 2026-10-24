"""Train + evaluate the Random Forest classifier (benign vs malicious DoH).

Usage (from the repo root):
    python ml/train_rf.py --data ml/data --seed 42
    python ml/train_rf.py --data ml/data --seed 42 --balance      # re-weight the rare class
Outputs:
    ml/artifacts/rf_seed42.joblib             model + feature list (used by cross_resolver_test.py)
    results/ml_rf_seed42.json                 metrics
    results/ml_rf_importance_seed42.csv       which features mattered
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd

from evaluate import evaluate_model, print_report, save_metrics
from models import build_rf
from preprocess import make_split, save_split

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO_ROOT / "ml" / "data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--artifacts", default=str(REPO_ROOT / "ml" / "artifacts"))
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    a = ap.parse_args()

    sp = make_split(a.data, a.seed, a.test_size)
    save_split(sp, a.artifacts, a.seed, a.test_size)
    model = build_rf(a.seed, a.n_estimators, a.balance)
    t0 = time.time()
    model.fit(sp.X_train, sp.y_train)
    fit_s = time.time() - t0
    m = evaluate_model(model, sp.X_test, sp.y_test, "random_forest")
    m.update({"seed": a.seed, "balanced": a.balance, "fit_seconds": round(fit_s, 2), "n_train": len(sp.y_train)})
    print_report(m)

    Path(a.artifacts).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": sp.feature_cols, "seed": a.seed, "name": "rf"},
                Path(a.artifacts) / f"rf_seed{a.seed}.joblib")
    save_metrics(m, Path(a.results) / f"ml_rf_seed{a.seed}.json")
    imp = pd.DataFrame({"feature": sp.feature_cols, "importance": model.feature_importances_})
    imp.sort_values("importance", ascending=False).to_csv(Path(a.results) / f"ml_rf_importance_seed{a.seed}.csv", index=False)
    print("top features:", ", ".join(imp.sort_values("importance", ascending=False)["feature"].head(5)))


if __name__ == "__main__":
    main()
