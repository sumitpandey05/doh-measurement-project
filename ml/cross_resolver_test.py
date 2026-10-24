"""Cross-resolver generalisation test (Day 18).

Take a model trained on the public dataset and run it on the benign DoH flows YOU captured
(capture/pcap_to_csv.py flows ...). Because your flows are benign by construction, every
"malicious" prediction is a FALSE POSITIVE. A high false-positive rate on your own traffic
means the model does not generalise to traffic it was not trained on - a reportable finding.

BEFORE you trust a surprising result, read the scale check it prints: if your features live
on a very different scale than the training data (different feature definition, wrong units)
the "finding" is a bug, not science.

Usage (from the repo root):
    python ml/cross_resolver_test.py --model ml/artifacts/rf_seed42.joblib \
        --own results/own_flows_A.csv results/own_flows_B.csv --train-data ml/data
Outputs:
    results/cross_resolver_<model>.csv             per-resolver / per-capture summary
    results/cross_resolver_<model>_scalecheck.csv  feature scale comparison (needs --train-data)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from load_dataset import align_columns, load_prepared

REPO_ROOT = Path(__file__).resolve().parents[1]


def scale_check(train_X: pd.DataFrame, own_X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in train_X.columns:
        lo, med, hi = train_X[c].quantile([0.01, 0.5, 0.99])
        own_med = own_X[c].median()
        rows.append({"feature": c, "train_p01": lo, "train_median": med, "train_p99": hi,
                     "own_median": own_med, "outside_train_p01_p99": bool(own_med < lo or own_med > hi)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a .joblib saved by train_rf.py / train_xgb.py")
    ap.add_argument("--own", nargs="+", required=True, help="flow CSV(s) from pcap_to_csv.py flows")
    ap.add_argument("--train-data", help="dataset folder, only for the scale check")
    ap.add_argument("--results", default=str(REPO_ROOT / "results"))
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    model, features, name = bundle["model"], bundle["features"], bundle.get("name", "model")
    frames = []
    for p in a.own:
        df = pd.read_csv(p)
        df["_file"] = Path(p).name
        frames.append(df)
    own = pd.concat(frames, ignore_index=True)
    X = align_columns(own, features)
    keep = X.notna().all(axis=1)
    if (~keep).any():
        print(f"[note] dropped {int((~keep).sum())} own rows with NaN/inf features")
    own, X = own[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
    if X.empty:
        raise SystemExit("no usable rows in --own files")

    own["pred_malicious"] = model.predict(X)
    own["p_malicious"] = model.predict_proba(X)[:, 1]
    group_cols = [c for c in ("Resolver", "_file") if c in own.columns]
    rows = []
    for key, g in [("ALL", own)] + [(f"{c}={v}", gg) for c in group_cols for v, gg in own.groupby(c)]:
        rows.append({"group": key, "n_flows": len(g), "predicted_malicious": int(g["pred_malicious"].sum()),
                     "false_positive_rate": float(g["pred_malicious"].mean()),
                     "mean_p_malicious": float(g["p_malicious"].mean())})
    summary = pd.DataFrame(rows)
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / f"cross_resolver_{name}.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print("(all own flows are benign, so predicted_malicious = false positives)")

    if a.train_data:
        train_X, _, _ = load_prepared(a.train_data)
        sc = scale_check(train_X[features], X)
        sc.to_csv(out / f"cross_resolver_{name}_scalecheck.csv", index=False)
        bad = sc[sc["outside_train_p01_p99"]]
        if len(bad):
            print(f"\n[SCALE WARNING] {len(bad)}/{len(sc)} features have an own-data median outside the training "
                  "1st-99th percentile range:")
            print("  " + ", ".join(bad["feature"].head(12)))
            print("  -> check feature definitions/units in capture/pcap_to_csv.py BEFORE concluding anything about "
                  "generalisation")
        else:
            print("\nscale check: all feature medians are inside the training 1st-99th percentile range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
