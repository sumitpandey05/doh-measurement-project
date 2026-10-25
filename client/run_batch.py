"""Run the full measurement: domains x resolvers x methods, repeated N times.

Every repetition writes its own CSV (results/raw_<run_id>.csv) plus a small
.meta.json (tool versions, arguments, counts) for the Methodology section.
Scripts take everything as arguments - nothing is hard-coded - so both team
members run the identical command on their own machine.

Usage (from the repo root, venv active):
    python client/run_batch.py --person A --profile clean --repeats 3
    python client/run_batch.py --person B --profile lossy --repeats 3 --udp-retries 2
    python client/run_batch.py --person A --limit 5 --only cloudflare     # quick smoke test

DoH uses HTTP/2 by default (--http1 to force the old behaviour). The .meta.json also
stores the `tc qdisc` rule that was active, so every run carries proof of the network
condition it was measured under.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import dns.version
import httpx

from common import METHODS, RAW_COLUMNS, REPO_ROOT
from dual_query import SessionPool, load_resolvers, measure


def load_domains(path) -> list[str]:
    domains = []
    if not Path(path).exists():
        raise SystemExit(f"{path} not found - create it first with:  python client/domain_list.py")
    with open(path, newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        if "domain" in sample.splitlines()[0].lower():
            domains = [row["domain"].strip() for row in csv.DictReader(f) if row.get("domain")]
        else:
            domains = [ln.strip() for ln in f if ln.strip()]
    if not domains:
        raise SystemExit(f"no domains found in {path} - run: python client/domain_list.py")
    return domains


def netem_state() -> dict:
    """Which tc/netem rule is active on the default-route interface right now."""
    try:
        route = subprocess.run(["ip", "route", "get", "1.1.1.1"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"\bdev\s+(\S+)", route)
        iface = m.group(1) if m else ""
        qdisc = subprocess.run(["tc", "qdisc", "show", "dev", iface], capture_output=True, text=True,
                               timeout=5).stdout.strip() if iface else ""
        return {"interface": iface, "qdisc": qdisc, "netem_active": "netem" in qdisc}
    except Exception as exc:
        return {"interface": "", "qdisc": f"unavailable: {exc}", "netem_active": None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--person", required=True, help="A or B - who is running this")
    ap.add_argument("--profile", default="clean", help="network condition label (clean, moderate, high_jitter, lossy)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42, help="seed for shuffling order (reproducible)")
    ap.add_argument("--domains", default=str(REPO_ROOT / "results" / "domains.csv"))
    ap.add_argument("--resolvers", default=str(REPO_ROOT / "client" / "resolvers.yaml"))
    ap.add_argument("--only", nargs="*", help="only these resolver names")
    ap.add_argument("--limit", type=int, help="only the first N domains (smoke test)")
    ap.add_argument("--qtype", default="A")
    ap.add_argument("--timeout", type=float, default=5.0, help="DoH timeout (s)")
    ap.add_argument("--udp-timeout", type=float, default=2.0,
                    help="wait per UDP attempt before a retry (s); stub resolvers use 1-5 s")
    ap.add_argument("--udp-retries", type=int, default=0, help="extra UDP attempts after a timeout")
    ap.add_argument("--http1", action="store_true", help="force HTTP/1.1 for DoH (default HTTP/2)")
    ap.add_argument("--pause", type=float, default=0.02, help="seconds between queries")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    resolvers = load_resolvers(args.resolvers, args.only)
    domains = load_domains(args.domains)
    if args.limit:
        domains = domains[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(domains) * len(resolvers) * len(METHODS)
    print(f"{len(domains)} domains x {len(resolvers)} resolvers x {len(METHODS)} methods "
          f"= {total} queries per repetition, {args.repeats} repetition(s)")
    net = netem_state()
    print(f"network: {net['interface'] or '?'}  netem={'ON' if net['netem_active'] else 'off'}  "
          f"DoH over {'HTTP/1.1' if args.http1 else 'HTTP/2'}")
    if net["netem_active"] is False and args.profile not in ("clean", "smoke"):
        print(f"[warn] profile '{args.profile}' but no netem rule is active - "
              "did you forget netem/apply_condition.sh?", file=sys.stderr)
    if net["netem_active"] and args.profile == "clean":
        print("[warn] profile 'clean' but a netem rule IS active - run netem/clear_condition.sh",
              file=sys.stderr)

    for rep in range(1, args.repeats + 1):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{args.person}_{args.profile}_r{rep}_{stamp}"
        out_csv = out_dir / f"raw_{run_id}.csv"
        rng = random.Random(args.seed * 1000 + rep)
        order = list(domains)
        rng.shuffle(order)

        pool = SessionPool(resolvers, args.timeout, not args.http1)
        counts: Counter = Counter()
        started = time.time()
        seq = 0
        interrupted = False
        probes = {}
        try:
            probes = pool.prime()
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
                writer.writeheader()
                for domain in order:
                    for resolver in resolvers:
                        for method, server, res in measure(domain, resolver, pool, args.qtype,
                                                           args.timeout, args.udp_retries, rng,
                                                           udp_timeout=args.udp_timeout):
                            seq += 1
                            row = {"person": args.person, "profile": args.profile, "run_id": run_id,
                                   "seq": seq, "domain": domain, "resolver": resolver.name,
                                   "method": method, "qtype": args.qtype, "server": server, **res}
                            writer.writerow(row)
                            counts[res["status"]] += 1
                            if args.pause:
                                time.sleep(args.pause)
                        f.flush()
                    if seq % 60 < len(resolvers) * len(METHODS):
                        print(f"  [r{rep}] {seq}/{total}  ok={counts['ok']}  other={seq - counts['ok']}", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print("\ninterrupted - partial results were kept", file=sys.stderr)
        finally:
            pool.close()
            meta = {
                "run_id": run_id, "args": vars(args), "queries_written": seq,
                "status_counts": dict(counts), "interrupted": interrupted,
                "started_epoch": started, "finished_epoch": time.time(),
                "python": platform.python_version(), "platform": platform.platform(),
                "dnspython": dns.version.version, "httpx": httpx.__version__,
                "doh_http_version": "HTTP/1.1" if args.http1 else "HTTP/2",
                "doh_probe": {k: {"status": v["status"], "http_version": v.get("http_version", "")}
                              for k, v in probes.items()},
                "doh_pinned_ip": {r.name: pool.warm[r.name].ip for r in resolvers},
                "network": net,
            }
            with open(out_dir / f"raw_{run_id}.meta.json", "w") as mf:
                json.dump(meta, mf, indent=2)
        print(f"run {run_id}: {seq} rows -> {out_csv.name}  status={dict(counts)}")
        if interrupted:
            return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
