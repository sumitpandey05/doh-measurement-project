"""Load and clean the CIRA-CIC-DoHBrw-2020 flow CSVs (or any CSV with the same schema).

Use the "Layer 2" files: the ones whose Label column says Benign / Malicious
(benign DoH vs malicious/tunnelling DoH). The Layer-1 files (DoH vs NonDoH) are
a different task and are rejected with a clear message.

What cleaning does, and why:
  * drops identifier columns (SourceIP, DestinationIP, SourcePort, DestinationPort,
    TimeStamp): they let a model memorise WHO sent a flow instead of HOW it behaves
    (data leakage -> unrealistically high accuracy, and it would not generalise)
  * turns Label into 0 = Benign, 1 = Malicious
  * replaces +/-inf with NaN and drops rows with missing features (counted and reported)

Usage (from the repo root):
    python ml/load_dataset.py --data ml/data          # summary of every CSV under ml/data
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMNS_NORM = {"sourceip", "destinationip", "sourceport", "destinationport", "timestamp"}
LABEL_MAP = {"benign": 0, "malicious": 1}
REPO_ROOT = Path(__file__).resolve().parents[1]


def norm(name) -> str:
    """'Packet Length-Mean' / 'PacketLengthMean' / 'packet_length_mean' -> 'packetlengthmean'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_csvs(path) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    files = sorted(p.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files under {p} - download the dataset into that folder (see README)")
    return files


def load_flows(paths) -> pd.DataFrame:
    """Concatenate CSVs in a fixed (sorted) order so every teammate gets identical row order."""
    frames = []
    for p in sorted(Path(x) for x in paths):
        df = pd.read_csv(p, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        df["_source_file"] = p.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def label_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if norm(c) == "label":
            return c
    raise ValueError("no 'Label' column found")


def prepare(df: pd.DataFrame):
    """Return (X, y, info). X = numeric feature DataFrame, y = 0/1 numpy array."""
    lab = label_column(df)
    raw = df[lab].astype(str).str.strip().str.lower()
    unknown = sorted(set(raw.unique()) - set(LABEL_MAP))
    if unknown:
        raise ValueError(f"unexpected label values {unknown}. Expected Benign/Malicious. "
                         "Layer-1 files (DoH/NonDoH) are a different task - use the Layer-2 files.")
    y_all = raw.map(LABEL_MAP).to_numpy(dtype=int)
    feats = [c for c in df.columns
             if c != lab and norm(c) not in ID_COLUMNS_NORM and not c.startswith("_")
             and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].replace([np.inf, -np.inf], np.nan)
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        X = X.drop(columns=all_nan)
    keep = X.notna().all(axis=1).to_numpy()
    info = {"rows_loaded": int(len(df)), "rows_dropped_nan_inf": int((~keep).sum()),
            "dropped_all_nan_columns": all_nan, "feature_columns": list(X.columns)}
    return X[keep].reset_index(drop=True), y_all[keep], info


def load_prepared(data):
    df = load_flows(find_csvs(data))
    return prepare(df)


def align_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Reorder/rename df's columns to exactly feature_cols (matching ignoring case/punctuation)."""
    by_norm = {norm(c): c for c in df.columns}
    missing = [c for c in feature_cols if norm(c) not in by_norm]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    out = df[[by_norm[norm(c)] for c in feature_cols]].copy()
    out.columns = feature_cols
    return out.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO_ROOT / "ml" / "data"))
    a = ap.parse_args()
    files = find_csvs(a.data)
    print("files:", ", ".join(f.name for f in files))
    X, y, info = prepare(load_flows(files))
    n1, n0 = int(y.sum()), int((y == 0).sum())
    print(f"rows loaded {info['rows_loaded']}, dropped (NaN/inf) {info['rows_dropped_nan_inf']}, kept {len(y)}")
    print(f"benign (0): {n0}   malicious (1): {n1}   ratio malicious:benign = {n1 / max(n0, 1):.2f}")
    print(f"{X.shape[1]} features:", ", ".join(X.columns))
    if info["dropped_all_nan_columns"]:
        print("columns that were entirely NaN and removed:", info["dropped_all_nan_columns"])
    if max(n0, n1) / max(min(n0, n1), 1) > 3:
        print("[note] strong class imbalance: report precision/recall/F1/AUC, not accuracy alone")


if __name__ == "__main__":
    main()

