"""Classic DNS over UDP (port 53) - the baseline in the comparison.

Usage:
    python client/classic_dns.py google.com 1.1.1.1
    python client/classic_dns.py google.com 1.1.1.1 --retries 2
"""
from __future__ import annotations

import argparse
import json
import time

import dns.exception
import dns.flags
import dns.query

from common import build_query, make_result, short_error


def query_classic(domain: str, server: str, qtype: str = "A", port: int = 53,
                  timeout: float = 5.0, retries: int = 0) -> dict:
    """Send one plaintext UDP query and time it.

    * Only the network send/receive sits inside the timing window; parsing the
      answer happens after the stopwatch stops.
    * If the answer is truncated (TC bit) we repeat over TCP, like real stubs.
    * retries > 0 re-sends after a timeout (rtt then includes the wait), which
      mimics a real stub resolver and matters on lossy links.
    """
    q = build_query(domain, qtype)
    send_ts = time.time()
    t0 = time.perf_counter()
    attempts = 0
    last_exc: BaseException | None = None
    while attempts <= retries:
        attempts += 1
        try:
            resp = dns.query.udp(q, server, timeout=timeout, port=port)
            if resp.flags & dns.flags.TC:
                resp = dns.query.tcp(q, server, timeout=timeout, port=port)
            t1 = time.perf_counter()
            return make_result("ok", send_ts, t0, t1, resp, attempts=attempts)
        except dns.exception.Timeout as exc:
            last_exc = exc
            continue
        except Exception as exc:  # unreachable network, bad response, ...
            t1 = time.perf_counter()
            return make_result("error", send_ts, t0, t1, error=short_error(exc), attempts=attempts)
    t1 = time.perf_counter()
    return make_result("timeout", send_ts, t0, t1, error=short_error(last_exc), attempts=attempts)


def main() -> None:
    ap = argparse.ArgumentParser(description="One classic UDP DNS query")
    ap.add_argument("domain")
    ap.add_argument("server", help="resolver IP, e.g. 1.1.1.1")
    ap.add_argument("--qtype", default="A")
    ap.add_argument("--port", type=int, default=53)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--retries", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(query_classic(a.domain, a.server, a.qtype, a.port, a.timeout, a.retries), indent=2))


if __name__ == "__main__":
    main()
