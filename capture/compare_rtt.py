"""Cross-check: does the RTT our client measured match the RTT visible on the wire?

Takes classic-UDP queries only (their wire RTT is unambiguous: query packet ->
answer packet). A small constant gap (client RTT a bit larger than wire RTT, a
few tenths of a ms) is normal: Python + OS overhead. Hundreds of ms of gap means
the client's timer wraps too much code.

Usage:
    python capture/pcap_to_csv.py udp-rtt capture/pcaps/x.pcap --out results/x_udp_rtt.csv
    python capture/compare_rtt.py --client results/raw_A_clean_r1_*.csv --wire results/x_udp_rtt.csv
"""
from __future__ import annotations

import argparse
import glob

import pandas as pd


def compare(client_csvs: list[str], wire_csv: str, tolerance_s: float = 0.25) -> pd.DataFrame:
    client = pd.concat([pd.read_csv(p) for p in client_csvs], ignore_index=True)
    client = client[(client["method"] == "classic_udp") & (client["status"] == "ok")]
    client = client.sort_values("send_ts")
    wire = pd.read_csv(wire_csv)
    wire["domain"] = wire["qname"].astype(str).str.rstrip(".").str.lower()  # 'google.com.' -> 'google.com'
    wire = wire.sort_values("ts_query")
    client["domain"] = client["domain"].astype(str).str.lower()
    merged = pd.merge_asof(client, wire, left_on="send_ts", right_on="ts_query",
                           left_by=["server", "domain"], right_by=["server_ip", "domain"],
                           tolerance=tolerance_s, direction="forward")
    merged = merged.dropna(subset=["rtt_ms_y"]).copy()
    merged["gap_ms"] = merged["rtt_ms_x"] - merged["rtt_ms_y"]
    return merged.rename(columns={"rtt_ms_x": "client_rtt_ms", "rtt_ms_y": "wire_rtt_ms"})[
        ["run_id", "seq", "domain", "resolver", "server", "client_rtt_ms", "wire_rtt_ms", "gap_ms"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--client", nargs="+", required=True, help="raw_*.csv file(s) (globs allowed)")
    ap.add_argument("--wire", required=True, help="CSV from `pcap_to_csv.py udp-rtt`")
    ap.add_argument("--tolerance", type=float, default=0.25, help="max seconds between client send and wire query")
    ap.add_argument("--out", help="write the matched rows here")
    a = ap.parse_args()
    files = [f for pat in a.client for f in sorted(glob.glob(pat))]
    if not files:
        raise SystemExit("no client CSVs matched")
    m = compare(files, a.wire, a.tolerance)
    if m.empty:
        print("no matches - are client and capture from the same run / same clock / same server IPs?")
        return 1
    g = m["gap_ms"]
    print(f"matched {len(m)} queries\n"
          f"gap (client - wire), ms: mean={g.mean():.3f} median={g.median():.3f} "
          f"p95={g.quantile(.95):.3f} max={g.max():.3f} min={g.min():.3f}")
    if g.median() > 5:
        print("[warn] median gap > 5 ms: narrow the client's timing window to just send/recv")
    if a.out:
        m.to_csv(a.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
