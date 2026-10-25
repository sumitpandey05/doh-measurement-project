"""Send the SAME question three ways and time each:

    classic_udp : plaintext DNS over UDP/53
    doh_cold    : DoH, new TCP+TLS connection per query (handshake included)
    doh_warm    : DoH over one reused connection (handshake excluded)

Usage:
    python client/dual_query.py google.com
    python client/dual_query.py google.com --only cloudflare --udp-retries 2
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml

from classic_dns import query_classic
from common import METHODS, REPO_ROOT
from doh_client import DoHSession


@dataclass(frozen=True)
class Resolver:
    name: str
    classic_ip: str
    doh_url: str
    classic_port: int = 53
    tls_verify: bool = True

    @property
    def doh_host(self) -> str:
        return urlparse(self.doh_url).hostname or ""


def load_resolvers(path, only=None) -> list[Resolver]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    out = []
    for item in cfg["resolvers"]:
        out.append(Resolver(
            name=str(item["name"]),
            classic_ip=str(item["classic_ip"]),
            doh_url=str(item["doh_url"]),
            classic_port=int(item.get("classic_port", 53)),
            tls_verify=bool(item.get("tls_verify", True)),
        ))
    if only:
        wanted = set(only)
        out = [r for r in out if r.name in wanted]
        missing = wanted - {r.name for r in out}
        if missing:
            raise SystemExit(f"unknown resolver(s) {sorted(missing)}; check {path}")
    if not out:
        raise SystemExit("no resolvers configured")
    return out


class SessionPool:
    """One warm (reused) and one cold (fresh-connection) DoH session per resolver."""

    def __init__(self, resolvers, timeout: float = 5.0, http2: bool = True):
        self.warm = {r.name: DoHSession(r.doh_url, reuse=True, timeout=timeout, http2=http2,
                                        verify=r.tls_verify) for r in resolvers}
        self.cold = {r.name: DoHSession(r.doh_url, reuse=False, timeout=timeout, http2=http2,
                                        verify=r.tls_verify) for r in resolvers}

    def prime(self, probe_domain: str = "example.com") -> dict:
        """Open each warm connection with one UNLOGGED query so the handshake is not
        counted in the first doh_warm row. Returns name -> probe result and warns loudly
        if a resolver's DoH endpoint does not work at all (e.g. HTTP 505 = wrong HTTP
        version), so a broken resolver is noticed before 450 rows are written."""
        probes = {}
        for name, session in self.warm.items():
            r = session.query(probe_domain)
            probes[name] = r
            if r["status"] != "ok":
                print(f"[warn] DoH probe to {name} failed: {r['status']} {r['error']} "
                      f"(http_version={r.get('http_version') or '?'})", file=sys.stderr)
        return probes

    def close(self) -> None:
        for s in list(self.warm.values()) + list(self.cold.values()):
            s.close()


def run_method(method: str, domain: str, resolver: Resolver, pool: SessionPool,
               qtype: str = "A", timeout: float = 5.0, udp_retries: int = 0,
               udp_timeout: float | None = None):
    """Run ONE method; returns (server_label, result_dict).
    udp_timeout = wait per UDP attempt before a retry (defaults to `timeout`)."""
    if method == "classic_udp":
        return resolver.classic_ip, query_classic(domain, resolver.classic_ip, qtype,
                                                  port=resolver.classic_port,
                                                  timeout=udp_timeout or timeout, retries=udp_retries)
    if method == "doh_cold":
        return resolver.doh_url, pool.cold[resolver.name].query(domain, qtype)
    if method == "doh_warm":
        return resolver.doh_url, pool.warm[resolver.name].query(domain, qtype)
    raise ValueError(f"unknown method {method!r}")


def measure(domain, resolver, pool, qtype="A", timeout=5.0, udp_retries=0,
            rng: random.Random | None = None, methods=METHODS, udp_timeout: float | None = None):
    """Run every method once for (domain, resolver). The order is shuffled when an
    rng is given, so no method is systematically first (and so hits a cold cache)."""
    order = list(methods)
    if rng is not None:
        rng.shuffle(order)
    rows = []
    for m in order:
        server, res = run_method(m, domain, resolver, pool, qtype, timeout, udp_retries, udp_timeout)
        rows.append((m, server, res))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Query one domain via classic DNS and DoH")
    ap.add_argument("domain")
    ap.add_argument("--resolvers", default=str(REPO_ROOT / "client" / "resolvers.yaml"))
    ap.add_argument("--only", nargs="*", help="resolver names to use, e.g. cloudflare")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--udp-retries", type=int, default=0)
    ap.add_argument("--http1", action="store_true", help="force HTTP/1.1 for DoH (default HTTP/2)")
    a = ap.parse_args()

    resolvers = load_resolvers(a.resolvers, a.only)
    pool = SessionPool(resolvers, a.timeout, not a.http1)
    try:
        pool.prime()
        print(f"{'resolver':<11}{'method':<13}{'status':<10}{'rtt_ms':>9}  {'http':<9}ips")
        for r in resolvers:
            for m, _server, res in measure(a.domain, r, pool, timeout=a.timeout,
                                           udp_retries=a.udp_retries):
                print(f"{r.name:<11}{m:<13}{res['status']:<10}{res['rtt_ms']:>9.1f}  "
                      f"{res.get('http_version') or '-':<9}{res['resolved_ips'][:40]}")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
