"""Turn raw measurement CSVs into the numbers that go into the paper.

Views:
  summarize()          - statistics over ALL queries in a group
  summarize_by_run()   - mean of each repetition first, then mean +/- std ACROSS
                         repetitions. Queries inside one run are correlated, so
                         this is the honest "mean +/- std over repeated trials".
  jitter()             - mean absolute difference between successive RTTs of the
                         same method (how erratic the latency is), averaged over runs
  consistency()        - do classic DNS and DoH return the same answer?
  handshake_overhead() - what the TLS/TCP handshake costs: doh_cold - doh_warm

Usage:
    python client/summarize.py                       # prints both tables
    python client/summarize.py --out results/summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import REPO_ROOT

DEFAULT_BY = ["profile", "resolver", "method"]


def load_raw(results_dir=REPO_ROOT / "results", pattern: str = "raw_*.csv") -> pd.DataFrame:
    files = sorted(Path(results_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern} in {results_dir}")
    frames = []
    for p in files:
        df = pd.read_csv(p)
        df["source_file"] = p.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame, by=DEFAULT_BY) -> pd.DataFrame:
    by = list(by)
    ok = df[df["status"] == "ok"]
    stats = ok.groupby(by)["rtt_ms"].agg(
        n_ok="count", mean_ms="mean", median_ms="median", std_ms="std",
        p95_ms=lambda s: s.quantile(0.95), min_ms="min", max_ms="max")
    totals = df.groupby(by).agg(n_total=("status", "size"),
                                n_fail=("status", lambda s: int((s != "ok").sum())))
    out = totals.join(stats, how="left")
    out["success_rate"] = (out["n_total"] - out["n_fail"]) / out["n_total"]
    return out.reset_index()


def summarize_by_run(df: pd.DataFrame, by=DEFAULT_BY) -> pd.DataFrame:
    by = list(by)
    ok = df[df["status"] == "ok"]
    per_run = ok.groupby(by + ["run_id"])["rtt_ms"].mean().rename("run_mean_ms").reset_index()
    out = per_run.groupby(by)["run_mean_ms"].agg(n_runs="count", mean_of_runs_ms="mean",
                                                 std_of_runs_ms="std").reset_index()
    return out


def jitter(df: pd.DataFrame, by=DEFAULT_BY) -> pd.DataFrame:
    """Jitter = mean absolute successive difference of RTT within one run and one method
    (queries taken in the order they were sent), then mean +/- std across runs."""
    by = list(by)
    ok = df[df["status"] == "ok"].sort_values("seq")

    def masd(series: pd.Series) -> float:
        d = series.diff().abs().dropna()
        return float(d.mean()) if len(d) else np.nan

    per_run = ok.groupby(by + ["run_id"])["rtt_ms"].apply(masd).rename("jitter_ms").reset_index()
    return per_run.groupby(by)["jitter_ms"].agg(n_runs="count", jitter_ms_mean="mean",
                                                jitter_ms_std="std").reset_index()


def consistency(df: pd.DataFrame, by=("profile", "resolver")) -> pd.DataFrame:
    """For every (run, domain, resolver) compare what the three methods returned.

    identical_rate : all methods returned exactly the same set of IPs
    overlap_rate   : all methods share at least one IP
    rcode_agree    : all methods returned the same response code
    NOTE: big sites rotate IPs between queries (DNS load balancing), so identical_rate can be
    below 1 even when nothing is wrong - judge DoH against the classic_udp baseline, not
    against 100 %.
    """
    by = list(by)
    ok = df[df["status"] == "ok"]
    rows = []
    for (run, dom, *keys), g in ok.groupby(["run_id", "domain"] + by):
        if g["method"].nunique() < 2:
            continue
        sets = [frozenset(str(x).split(";")) if isinstance(x, str) and x else frozenset() for x in g["resolved_ips"]]
        overlap = all(sets) and bool(frozenset.intersection(*sets))
        rows.append(dict(zip(by, keys)) | {"identical": len(set(sets)) == 1, "overlap": bool(overlap),
                                           "rcode_agree": g["rcode"].nunique() == 1})
    if not rows:
        return pd.DataFrame(columns=by + ["n_cases", "identical_rate", "overlap_rate", "rcode_agree_rate"])
    per = pd.DataFrame(rows)
    return per.groupby(by).agg(n_cases=("identical", "size"), identical_rate=("identical", "mean"),
                               overlap_rate=("overlap", "mean"), rcode_agree_rate=("rcode_agree", "mean")).reset_index()


def handshake_overhead(df: pd.DataFrame, by=("profile", "resolver")) -> pd.DataFrame:
    """Mean RTT of doh_cold minus doh_warm = the cost of setting up TCP+TLS for one lookup."""
    by = list(by)
    ok = df[df["status"] == "ok"]
    piv = ok.pivot_table(index=by, columns="method", values="rtt_ms", aggfunc="mean")
    out = pd.DataFrame({"doh_cold_ms": piv.get("doh_cold"), "doh_warm_ms": piv.get("doh_warm"),
                        "classic_udp_ms": piv.get("classic_udp")})
    out["handshake_overhead_ms"] = out["doh_cold_ms"] - out["doh_warm_ms"]
    out["doh_warm_vs_udp_ms"] = out["doh_warm_ms"] - out["classic_udp_ms"]
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--by", nargs="+", default=DEFAULT_BY)
    ap.add_argument("--out", help="write the all-queries summary CSV here")
    a = ap.parse_args()
    df = load_raw(a.results_dir)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    s = summarize(df, a.by)
    print(f"{len(df)} rows from {df['source_file'].nunique()} files\n")
    print(s.round(2).to_string(index=False))
    print("\nmean +/- std across repeated runs:")
    print(summarize_by_run(df, a.by).round(2).to_string(index=False))
    print("\njitter (mean absolute successive RTT difference, ms):")
    print(jitter(df, a.by).round(2).to_string(index=False))
    print("\nanswer consistency between methods:")
    print(consistency(df).round(3).to_string(index=False))
    print("\ncost of encryption (ms):")
    print(handshake_overhead(df).round(2).to_string(index=False))
    if a.out:
        s.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
