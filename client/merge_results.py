"""Merge both teammates' raw CSVs into one file (results/merged_raw.csv).

Run this after you both `git pull`. It checks every file has the expected
columns (a different column order/name = someone edited the client locally),
drops exact duplicate rows, and prints a per person/profile overview.

Usage:
    python client/merge_results.py
    python client/merge_results.py --out results/merged_raw.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import RAW_COLUMNS, REPO_ROOT


def merge(results_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    frames, problems = [], []
    for p in sorted(results_dir.glob("raw_*.csv")):
        df = pd.read_csv(p)
        if set(df.columns) != set(RAW_COLUMNS):
            problems.append(f"{p.name}: columns differ (missing {sorted(set(RAW_COLUMNS) - set(df.columns))}, "
                            f"extra {sorted(set(df.columns) - set(RAW_COLUMNS))})")
            continue
        frames.append(df[RAW_COLUMNS])
    if not frames:
        raise SystemExit("no valid raw_*.csv files found")
    merged = pd.concat(frames, ignore_index=True).drop_duplicates()
    return merged, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "merged_raw.csv"))
    a = ap.parse_args()
    merged, problems = merge(Path(a.results_dir))
    merged.to_csv(a.out, index=False)
    print(f"merged {len(merged)} rows -> {a.out}")
    print(merged.groupby(["person", "profile"]).agg(runs=("run_id", "nunique"), rows=("seq", "size")).to_string())
    for msg in problems:
        print("[skipped]", msg)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
