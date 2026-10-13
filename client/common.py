"""Shared helpers for the DNS / DoH measurement client.

Everything that has to be *identical* for classic DNS and DoH lives here, so
that the two methods are compared like-for-like:
  * the same DNS question (same EDNS0 settings) is built by build_query()
  * the same result fields come back from every query function
  * the same CSV columns (RAW_COLUMNS) are written for every method
"""
from __future__ import annotations

from pathlib import Path

import dns.message
import dns.rcode
import dns.rdatatype

REPO_ROOT = Path(__file__).resolve().parents[1]

# One row of the raw measurement CSV. Order matters (it is the CSV header).
RAW_COLUMNS = [
    "person",        # who ran it: A / B
    "profile",       # network condition: clean, moderate, high_jitter, lossy ...
    "run_id",        # unique id of one repetition
    "seq",           # running number of the query inside the run
    "domain",        # the name that was looked up
    "resolver",      # cloudflare / google / quad9
    "method",        # classic_udp | doh_cold | doh_warm
    "qtype",         # A / AAAA ...
    "server",        # classic: resolver IP, DoH: URL
    "send_ts",       # epoch seconds just before sending (lets you align with a pcap)
    "recv_ts",       # epoch seconds when the answer was in hand
    "rtt_ms",        # latency in milliseconds (perf_counter based, not wall clock)
    "status",        # ok | timeout | error | http_<code> | bad_response
    "rcode",         # NOERROR / NXDOMAIN / ...
    "answer_count",  # number of records in the answer section
    "resolved_ips",  # A/AAAA addresses returned, ';' separated
    "attempts",      # UDP attempts used (1 unless --udp-retries is set)
    "error",         # short error text if status != ok
]

METHODS = ("classic_udp", "doh_cold", "doh_warm")


def build_query(domain: str, qtype: str = "A", qid: int | None = None) -> dns.message.Message:
    """Build the DNS question. Same EDNS0 options for classic DNS and DoH."""
    return dns.message.make_query(domain, qtype, use_edns=0, payload=1232, id=qid)


def short_error(exc: BaseException | None) -> str:
    if exc is None:
        return ""
    return f"{type(exc).__name__}: {exc}".replace("\n", " ")[:200]


def make_result(status, send_ts, t0, t1, msg=None, error="", attempts=1) -> dict:
    """Common result dict returned by every query function.

    t0/t1 are time.perf_counter() values; recv_ts is derived from them so that
    rtt_ms and the two timestamps are always consistent with each other.
    """
    rtt = (t1 - t0) * 1000.0
    result = {
        "send_ts": round(send_ts, 6),
        "recv_ts": round(send_ts + (t1 - t0), 6),
        "rtt_ms": round(rtt, 3),
        "status": status,
        "rcode": "",
        "answer_count": 0,
        "resolved_ips": "",
        "attempts": attempts,
        "error": error,
    }
    if msg is not None:
        ips = []
        for rrset in msg.answer:
            if rrset.rdtype in (dns.rdatatype.A, dns.rdatatype.AAAA):
                ips.extend(rr.address for rr in rrset)
        result["rcode"] = dns.rcode.to_text(msg.rcode())
        result["answer_count"] = sum(len(r) for r in msg.answer)
        result["resolved_ips"] = ";".join(ips)
    return result
