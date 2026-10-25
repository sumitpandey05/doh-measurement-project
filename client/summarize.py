"""Turn raw measurement CSVs into the numbers that go into the paper.

Latency is heavy-tailed: one UDP timeout + retry adds whole seconds to a single row and
drags the MEAN far away from what a typical lookup costs. So the paper uses the MEDIAN
(typical lookup) and the 95th percentile (tail), with confidence intervals from a
run-level (cluster) bootstrap. Means are still printed, but never compared on their own.

Views:
  summarize()          - statistics over ALL queries in a group (median, p95, mean, ...)
  summarize_by_run()   - median of each repetition, then mean +/- std ACROSS repetitions
  median_ci()          - median with a 95 % bootstrap CI that resamples whole runs
  paired_overhead()    - per (run, domain, resolver): doh_cold - classic, doh_warm - classic,
                         doh_cold - doh_warm, then the median of those paired differences
  jitter()             - mean absolute difference between successive RTTs of one method
  consistency()        - do classic DNS and DoH return the same answer?
  handshake_overhead() - median doh_cold - median doh_warm per group (kept for old notebooks)

Profiles named 'smoke' (quick tests) are skipped unless --keep-smoke is given.

Usage:
    python client/summarize.py
    python client/summarize.py --out results/summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import REPO_ROOT

DEFAULT_BY = ["profile", "resolver", "method"]
PROFILE_ORDER = ["clean", "moderate", "high_jitter", "lossy"]


def load_raw(results_dir=REPO_ROOT / "results", pattern: str = "raw_*.csv",
             exclude_profiles=("smoke",)) -> pd.DataFrame:
    files = sorted(Path(results_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern} in {results_dir}")
    frames = []
    for p in files:
        df = pd.read_csv(p)
        df["source_file"] = p.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if exclude_profiles:
        df = df[~df["profile"].isin(exclude_profiles)].reset_index(drop=True)
    return df


def _q(p):
    def f(s):
        return s.quantile(p)
    f.__name__ = f"p{int(p * 100):02d}"
    return f


def summarize(df: pd.DataFrame, by=DEFAULT_BY) -> pd.DataFrame:
    by = list(by)
    ok = df[df["status"] == "ok"]
    stats = ok.groupby(by)["rtt_ms"].agg(
        n_ok="count", median_ms="median", p25_ms=_q(.25), p75_ms=_q(.75), p95_ms=_q(.95),
        mean_ms="mean", std_ms="std", min_ms="min", max_ms="max")
    totals = df.groupby(by).agg(n_total=("status", "size"),
                                n_fail=("status", lambda s: int((s != "ok").sum())),
                                n_timeout=("status", lambda s: int((s == "timeout").sum())))
    out = totals.join(stats, how="left")
    out["success_rate"] = (out["n_total"] - out["n_fail"]) / out["n_total"]
    if "attempts" in ok.columns:
        out = out.join(ok.assign(retried=ok["attempts"].fillna(1) > 1)
                         .groupby(by)["retried"].mean().rename("retry_rate"), how="left")
    return out.reset_index()


def summarize_by_run(df: pd.DataFrame, by=DEFAULT_BY) -> pd.DataFrame:
    by = list(by)
    ok = df[df["status"] == "ok"]
    per_run = ok.groupby(by + ["run_id"])["rtt_ms"].agg(run_median_ms="median", run_mean_ms="mean").reset_index()
    out = per_run.groupby(by).agg(n_runs=("run_id", "count"),
                                  median_of_runs_ms=("run_median_ms", "mean"),
                                  std_of_run_medians_ms=("run_median_ms", "std"),
                                  mean_of_runs_ms=("run_mean_ms", "mean"),
                                  std_of_runs_ms=("run_mean_ms", "std")).reset_index()
    return out


def _cluster_bootstrap(values_by_run: list[np.ndarray], stat, n_boot: int, rng) -> tuple[float, float]:
    k = len(values_by_run)
    if k == 0:
        return np.nan, np.nan
    sizes = np.array([len(v) for v in values_by_run])
    flat = np.concatenate(values_by_run)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, k)
        boots[b] = stat(np.concatenate([flat[starts[i]:starts[i] + sizes[i]] for i in pick]))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def median_ci(df: pd.DataFrame, by=DEFAULT_BY, n_boot: int = 2000, seed: int = 0,
              value: str = "rtt_ms") -> pd.DataFrame:
    """Median and p95 of `value` with 95 % CIs. Whole runs are resampled (queries inside a
    run share the same network moment, so they are not independent)."""
    by = list(by)
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    rng = np.random.default_rng(seed)
    rows = []
    for keys, g in ok.groupby(by):
        runs = [x[value].to_numpy(float) for _, x in g.groupby("run_id")]
        lo, hi = _cluster_bootstrap(runs, np.median, n_boot, rng)
        lo95, hi95 = _cluster_bootstrap(runs, lambda v: np.percentile(v, 95), n_boot, rng)
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(dict(zip(by, keys)) | {"n": len(g), "n_runs": len(runs),
                                           "median_ms": float(g[value].median()), "ci_lo": lo, "ci_hi": hi,
                                           "p95_ms": float(g[value].quantile(.95)),
                                           "p95_ci_lo": lo95, "p95_ci_hi": hi95})
    return pd.DataFrame(rows)


def paired_differences(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, domain, resolver) with the three methods side by side, and their
    differences. Pairing removes domain and moment-to-moment network effects."""
    ok = df[df["status"] == "ok"]
    keys = ["person", "profile", "run_id", "domain", "resolver"]
    keys = [k for k in keys if k in ok.columns]
    w = ok.pivot_table(index=keys, columns="method", values="rtt_ms", aggfunc="first").reset_index()
    for m in ("classic_udp", "doh_cold", "doh_warm"):
        if m not in w.columns:
            w[m] = np.nan
    w["cold_minus_udp"] = w["doh_cold"] - w["classic_udp"]
    w["warm_minus_udp"] = w["doh_warm"] - w["classic_udp"]
    w["cold_minus_warm"] = w["doh_cold"] - w["doh_warm"]
    return w


def paired_overhead(df: pd.DataFrame, by=("profile", "resolver"), n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    by = list(by)
    w = paired_differences(df)
    rng = np.random.default_rng(seed)
    rows = []
    for keys, g in w.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        for col in ("cold_minus_udp", "warm_minus_udp", "cold_minus_warm"):
            gg = g.dropna(subset=[col])
            runs = [x[col].to_numpy(float) for _, x in gg.groupby("run_id")]
            lo, hi = _cluster_bootstrap(runs, np.median, n_boot, rng) if runs else (np.nan, np.nan)
            row[f"{col}_n"] = len(gg)
            row[f"{col}_median_ms"] = float(gg[col].median()) if len(gg) else np.nan
            row[f"{col}_ci_lo"], row[f"{col}_ci_hi"] = lo, hi
            row[f"{col}_share_positive"] = float((gg[col] > 0).mean()) if len(gg) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def jitter(df: pd.DataFrame, by=DEFAULT_BY) -> pd.DataFrame:
    """Jitter = mean absolute successive difference of RTT within one run and one method
    (queries taken in the order they were sent), then mean +/- std across runs."""
    by = list(by)
    ok = df[df["status"] == "ok"].sort_values("seq")

    def masd(series: pd.Series) -> float:
        d = series.diff().abs().dropna()
        return float(d.mean()) if len(d) else np.nan

    def mesd(series: pd.Series) -> float:
        d = series.diff().abs().dropna()
        return float(d.median()) if len(d) else np.nan

    per_run = ok.groupby(by + ["run_id"])["rtt_ms"].agg(jitter_ms=masd, jitter_median_ms=mesd).reset_index()
    return per_run.groupby(by).agg(n_runs=("run_id", "count"), jitter_ms_mean=("jitter_ms", "mean"),
                                   jitter_ms_std=("jitter_ms", "std"),
                                   jitter_median_ms=("jitter_median_ms", "mean")).reset_index()


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


def handshake_overhead(df: pd.DataFrame, by=("profile", "resolver"), stat: str = "median") -> pd.DataFrame:
    """doh_cold minus doh_warm = the cost of setting up TCP+TLS for one lookup.
    stat='median' (default, robust) or 'mean'."""
    by = list(by)
    ok = df[df["status"] == "ok"]
    piv = ok.pivot_table(index=by, columns="method", values="rtt_ms", aggfunc=stat)
    out = pd.DataFrame({"doh_cold_ms": piv.get("doh_cold"), "doh_warm_ms": piv.get("doh_warm"),
                        "classic_udp_ms": piv.get("classic_udp")})
    out["handshake_overhead_ms"] = out["doh_cold_ms"] - out["doh_warm_ms"]
    out["doh_warm_vs_udp_ms"] = out["doh_warm_ms"] - out["classic_udp_ms"]
    return out.reset_index()


def order_profiles(df: pd.DataFrame, col: str = "profile") -> pd.DataFrame:
    present = list(df[col].unique())
    cats = [p for p in PROFILE_ORDER if p in present] + sorted(set(present) - set(PROFILE_ORDER))
    df = df.copy()
    df[col] = pd.Categorical(df[col], cats, ordered=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--by", nargs="+", default=DEFAULT_BY)
    ap.add_argument("--keep-smoke", action="store_true", help="also include profile 'smoke'")
    ap.add_argument("--out", help="write the all-queries summary CSV here")
    a = ap.parse_args()
    df = load_raw(a.results_dir, exclude_profiles=() if a.keep_smoke else ("smoke",))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    s = summarize(df, a.by)
    print(f"{len(df)} rows from {df['source_file'].nunique()} files\n")
    cols = [c for c in a.by] + ["n_total", "success_rate", "median_ms", "p95_ms", "mean_ms", "retry_rate"]
    print(s[[c for c in cols if c in s.columns]].round(2).to_string(index=False))
    print("\nmedian with 95% CI (bootstrap over runs):")
    print(median_ci(df, a.by).round(1).to_string(index=False))
    print("\npaired overhead per (run, domain): median difference, ms:")
    po = paired_overhead(df)
    print(po[[c for c in po.columns if c.endswith(("_median_ms",)) or c in ("profile", "resolver")]]
          .round(1).to_string(index=False))
    print("\njitter (mean absolute successive RTT difference, ms):")
    print(jitter(df, a.by).round(2).to_string(index=False))
    print("\nanswer consistency between methods:")
    print(consistency(df).round(3).to_string(index=False))
    if a.out:
        s.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
