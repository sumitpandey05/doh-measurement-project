"""Privacy audit: which of the domains you looked up can a PASSIVE observer read in the pcap?

This is the "privacy" half of the project's latency-vs-privacy trade-off. For every domain
the client queried we search the captured packets for it, separately for
  * classic DNS  (UDP/TCP port 53)   - expected: visible (plaintext)
  * DoH          (port 443)          - expected: NOT visible (encrypted inside TLS)
We also list the TLS server names (SNI) an observer sees on DoH connections: they reveal
WHICH RESOLVER you use, never WHAT you asked.

How a domain is searched for: in DNS wire format (b"\\x06google\\x03com\\x00") and as plain
dotted text. Only the first 700 payload bytes of each packet are kept, which is enough for a
DNS query.

Usage (from the repo root):
    python capture/privacy_audit.py capture/pcaps/x.pcap --client "results/raw_A_clean_r1_*.csv" \
        --out results/privacy_audit_A.csv
    # non-standard ports (local mock servers): --classic-ports 15353 --doh-ports 18443

Remember for the paper: a plain-HTTP DoH endpoint (only used by the local mock) is NOT private;
with real DoH (HTTPS) the exposure rate should be 0 %.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from pcap_to_csv import parse_sni, read_packets

REPO_ROOT = Path(__file__).resolve().parents[1]


def wire_name(domain: str) -> bytes:
    labels = domain.rstrip(".").lower().split(".")
    return b"".join(bytes([len(label)]) + label.encode() for label in labels) + b"\x00"


def audit(pcap, domains, classic_ports=(53,), doh_ports=(443,)) -> tuple[pd.DataFrame, list[str]]:
    classic_ports, doh_ports = set(classic_ports), set(doh_ports)
    needles = {d: (wire_name(d), d.lower().encode()) for d in domains}
    seen = {"classic_dns": set(), "doh": set()}
    packets = {"classic_dns": 0, "doh": 0}
    sni_names: set[str] = set()
    for p in read_packets(pcap):
        ports = {p.sport, p.dport}
        if ports & classic_ports:
            channel = "classic_dns"
        elif p.proto == "TCP" and ports & doh_ports:
            channel = "doh"
            if p.first_payload:
                name = parse_sni(p.first_payload)
                if name:
                    sni_names.add(name)
        else:
            continue
        packets[channel] += 1
        blob = p.first_payload.lower()
        if not blob:
            continue
        for d, (w, t) in needles.items():
            if d not in seen[channel] and (w in blob or t in blob):
                seen[channel].add(d)
    rows = []
    for channel in ("classic_dns", "doh"):
        rows.append({"channel": channel, "packets_inspected": packets[channel],
                     "domains_queried": len(domains), "domains_visible_to_observer": len(seen[channel]),
                     "exposure_rate": len(seen[channel]) / len(domains) if domains else float("nan")})
    return pd.DataFrame(rows), sorted(sni_names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcap")
    ap.add_argument("--client", nargs="+", help="raw_*.csv file(s) of the same run (globs allowed): the domains queried")
    ap.add_argument("--domains", default=str(REPO_ROOT / "results" / "domains.csv"),
                    help="used when --client is not given")
    ap.add_argument("--classic-ports", type=int, nargs="+", default=[53])
    ap.add_argument("--doh-ports", type=int, nargs="+", default=[443])
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.client:
        files = [f for pat in a.client for f in sorted(glob.glob(pat))]
        domains = sorted(set(pd.concat([pd.read_csv(f) for f in files])["domain"].astype(str)))
    else:
        domains = sorted(set(pd.read_csv(a.domains)["domain"].astype(str)))
    table, sni = audit(a.pcap, domains, a.classic_ports, a.doh_ports)
    pd.set_option("display.width", 160)
    print(table.round(3).to_string(index=False))
    print("TLS server names an observer sees on DoH connections (= which resolver you use):",
          ", ".join(sni) if sni else "none seen")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out = table.copy()
        out["sni_seen"] = ";".join(sni)
        out["pcap"] = Path(a.pcap).name
        out.to_csv(a.out, index=False)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
