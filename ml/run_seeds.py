"""Repeat training/evaluation of BOTH models over several random seeds (Day 19) and
report mean +/- std - so no reviewer can say "is that one lucky split?".

Each seed changes the train/test split AND the model's randomness.

Usage (from the repo root):
    python ml/run_seeds.py --data ml/data --seeds 42 43 44 45 46
Outputs:
    results/ml_multiseed.csv           one row per (model, seed)
    results/ml_multiseed_summary.csv   mean and std per model and metric
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate import evaluate_model
from load_dataset import load_prepared
from models import build_rf, build_xgb
from preprocess import split_prepared

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate",
           "balanced_accuracy", "mcc", "pr_auc"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO_ROOT / "ml" / "data"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    a = ap.parse_args()

    X, y, info = load_prepared(a.data)   # load + clean ONCE, split per seed
    rows = []
    for seed in a.seeds:
        sp = split_prepared(X, y, info, seed, a.test_size)
        for name, model in (("random_forest", build_rf(seed, balance=a.balance)),
                            ("xgboost", build_xgb(seed, sp.y_train, balance=a.balance))):
            model.fit(sp.X_train, sp.y_train)
            m = evaluate_model(model, sp.X_test, sp.y_test, name)
            m["seed"] = seed
            rows.append(m)
            print(f"seed {seed} {name:<14} acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
                  f"rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")
    df = pd.DataFrame(rows)
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ml_multiseed.csv", index=False)
    g = df.groupby("model")[METRICS].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    g.insert(0, "n_seeds", df.groupby("model")["seed"].nunique())
    g.reset_index().to_csv(out / "ml_multiseed_summary.csv", index=False)
    print("\nmean +/- std over", len(a.seeds), "seeds")
    for model_name, r in g.iterrows():
        print(f"  {model_name:<14}" + "  ".join(f"{m}={r[m + '_mean']:.4f}+/-{r[m + '_std']:.4f}" for m in METRICS[:5]))


if __name__ == "__main__":
    main()
