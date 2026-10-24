"""Train + evaluate the XGBoost classifier. Reuses evaluate.py, so its numbers are
directly comparable with train_rf.py.

Usage (from the repo root):
    python ml/train_xgb.py --data ml/data --seed 42
Outputs:
    ml/artifacts/xgb_seed42.joblib, results/ml_xgb_seed42.json, results/ml_xgb_importance_seed42.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd

from evaluate import evaluate_model, print_report, save_metrics
from models import build_xgb
from preprocess import make_split, save_split

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO_ROOT / "ml" / "data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--artifacts", default=str(REPO_ROOT / "ml" / "artifacts"))
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    a = ap.parse_args()

    sp = make_split(a.data, a.seed, a.test_size)
    save_split(sp, a.artifacts, a.seed, a.test_size)
    model = build_xgb(a.seed, sp.y_train, a.n_estimators, a.balance)
    t0 = time.time()
    model.fit(sp.X_train, sp.y_train)
    fit_s = time.time() - t0
    m = evaluate_model(model, sp.X_test, sp.y_test, "xgboost")
    m.update({"seed": a.seed, "balanced": a.balance, "fit_seconds": round(fit_s, 2), "n_train": len(sp.y_train)})
    print_report(m)

    Path(a.artifacts).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": sp.feature_cols, "seed": a.seed, "name": "xgb"},
                Path(a.artifacts) / f"xgb_seed{a.seed}.joblib")
    save_metrics(m, Path(a.results) / f"ml_xgb_seed{a.seed}.json")
    imp = pd.DataFrame({"feature": sp.feature_cols, "importance": model.feature_importances_})
    imp.sort_values("importance", ascending=False).to_csv(Path(a.results) / f"ml_xgb_importance_seed{a.seed}.csv", index=False)
    print("top features:", ", ".join(imp.sort_values("importance", ascending=False)["feature"].head(5)))


if __name__ == "__main__":
    main()
