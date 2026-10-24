"""Deterministic train/test split + a manifest both teammates can compare.

Why: for "run independently on both machines" to mean anything, you must be able to
PROVE you trained on the same split. The manifest holds hashes of the data and of the
split indices; if your two manifests are byte-identical, you have identical splits.

Usage (from the repo root):
    python ml/preprocess.py --data ml/data --seed 42
    # then compare:  sha256sum ml/artifacts/split_seed42_manifest.json   (same on both machines)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from load_dataset import load_prepared

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_cols: list
    train_idx: np.ndarray
    test_idx: np.ndarray
    data_fingerprint: str
    info: dict


def fingerprint(X: pd.DataFrame, y: np.ndarray) -> str:
    """sha256 over the raw float64 feature values + labels. Depends only on the numbers, not on
    pandas internals, so it is the same on any machine that loaded the same CSV files."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X.to_numpy(dtype="float64")).tobytes())
    h.update(np.asarray(y, dtype=np.int64).tobytes())
    return h.hexdigest()


def split_prepared(X: pd.DataFrame, y: np.ndarray, info: dict, seed: int = 42, test_size: float = 0.2) -> Split:
    """Stratified split: both sets keep the benign/malicious ratio. random_state is passed
    explicitly (a global numpy seed alone is NOT enough for sklearn)."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    return Split(X.iloc[tr], X.iloc[te], y[tr], y[te], list(X.columns), tr, te, fingerprint(X, y), info)


def make_split(data, seed: int = 42, test_size: float = 0.2) -> Split:
    X, y, info = load_prepared(data)
    return split_prepared(X, y, info, seed, test_size)


def save_split(sp: Split, out_dir, seed: int, test_size: float = 0.2) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"split_seed{seed}_train.npy", sp.train_idx)
    np.save(out / f"split_seed{seed}_test.npy", sp.test_idx)
    manifest = {
        "seed": seed, "test_size": test_size,
        "n_train": int(len(sp.train_idx)), "n_test": int(len(sp.test_idx)),
        "train_malicious": int(sp.y_train.sum()), "test_malicious": int(sp.y_test.sum()),
        "n_features": len(sp.feature_cols), "feature_columns": sp.feature_cols,
        "rows_dropped_nan_inf": sp.info["rows_dropped_nan_inf"],
        "data_sha256": sp.data_fingerprint,
        "train_idx_sha256": hashlib.sha256(sp.train_idx.astype(np.int64).tobytes()).hexdigest(),
        "test_idx_sha256": hashlib.sha256(sp.test_idx.astype(np.int64).tobytes()).hexdigest(),
    }
    path = out / f"split_seed{seed}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    # Library versions go in a SEPARATE file: they differ between laptops, and they must not
    # change the manifest hash that you compare. If the hashes differ, compare these files.
    (out / f"split_seed{seed}_versions.json").write_text(json.dumps(
        {"python": platform.python_version(), "sklearn": sklearn.__version__,
         "pandas": pd.__version__, "numpy": np.__version__}, indent=2, sort_keys=True))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO_ROOT / "ml" / "data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "ml" / "artifacts"))
    a = ap.parse_args()
    sp = make_split(a.data, a.seed, a.test_size)
    path = save_split(sp, a.out_dir, a.seed, a.test_size)
    print(f"train {len(sp.y_train)} (malicious {int(sp.y_train.sum())}), "
          f"test {len(sp.y_test)} (malicious {int(sp.y_test.sum())}), {len(sp.feature_cols)} features")
    print(f"data sha256  : {sp.data_fingerprint}")
    print(f"manifest     : {path}")
    print("Teammate check: both run `sha256sum` on that manifest - the hashes must match.")


if __name__ == "__main__":
    main()
