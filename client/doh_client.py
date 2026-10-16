"""DNS-over-HTTPS (RFC 8484) client.

Two modes, because they measure different things:
  * cold (reuse=False): a brand-new TCP + TLS connection for EVERY query.
    rtt therefore includes the TCP and TLS handshakes = the "price of
    encryption" a first lookup pays.
  * warm (reuse=True): one long-lived HTTPS connection reused for many
    queries - what a browser does. Handshake cost is paid once, not timed.

Usage:
    python client/doh_client.py google.com https://cloudflare-dns.com/dns-query
    python client/doh_client.py google.com https://dns.google/dns-query --cold
"""
from __future__ import annotations

import argparse
import base64
import json
import time

import dns.message
import httpx

from common import build_query, make_result, short_error

DOH_MIME = "application/dns-message"


class DoHSession:
    def __init__(self, url: str, reuse: bool = True, timeout: float = 5.0,
                 http2: bool = False, verify: bool = True, use_get: bool = False):
        if http2:
            try:
                import h2  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("HTTP/2 needs the 'h2' package: pip install 'httpx[http2]'") from exc
        self.url = url
        self.reuse = reuse
        self.timeout = timeout
        self.http2 = http2
        self.use_get = use_get
        # Build the (slow) SSL context ONCE so that creating a client for a cold
        # query does not add certificate-loading time to the measured latency.
        self._verify = httpx.create_ssl_context() if verify else False
        self._client = self._new_client() if reuse else None

    def _new_client(self) -> httpx.Client:
        return httpx.Client(http2=self.http2, timeout=self.timeout, verify=self._verify,
                            headers={"accept": DOH_MIME})

    def query(self, domain: str, qtype: str = "A") -> dict:
        # RFC 8484 section 4.1: use DNS ID 0 so identical questions are cache friendly.
        wire = build_query(domain, qtype, qid=0).to_wire()
        client = self._client if self.reuse else self._new_client()  # created OUTSIDE the timer
        send_ts = time.time()
        t0 = time.perf_counter()
        try:
            if self.use_get:
                token = base64.urlsafe_b64encode(wire).rstrip(b"=").decode()
                resp = client.get(self.url, params={"dns": token})
            else:
                resp = client.post(self.url, content=wire, headers={"content-type": DOH_MIME})
            t1 = time.perf_counter()
        except httpx.TimeoutException as exc:
            return make_result("timeout", send_ts, t0, time.perf_counter(), error=short_error(exc))
        except (httpx.HTTPError, OSError) as exc:
            return make_result("error", send_ts, t0, time.perf_counter(), error=short_error(exc))
        finally:
            if not self.reuse:
                client.close()  # TLS shutdown happens after the timer stopped

        if resp.status_code != 200:
            return make_result(f"http_{resp.status_code}", send_ts, t0, t1, error=resp.reason_phrase)
        try:
            msg = dns.message.from_wire(resp.content)  # parsing is outside the timing window
        except Exception as exc:
            return make_result("bad_response", send_ts, t0, t1, error=short_error(exc))
        return make_result("ok", send_ts, t0, t1, msg)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="One DoH query")
    ap.add_argument("domain")
    ap.add_argument("url", help="e.g. https://cloudflare-dns.com/dns-query")
    ap.add_argument("--qtype", default="A")
    ap.add_argument("--cold", action="store_true", help="fresh connection (includes TLS handshake)")
    ap.add_argument("--get", action="store_true", help="use GET instead of POST")
    ap.add_argument("--http2", action="store_true")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification (local tests only)")
    ap.add_argument("--timeout", type=float, default=5.0)
    a = ap.parse_args()
    with DoHSession(a.url, reuse=not a.cold, timeout=a.timeout, http2=a.http2,
                    verify=not a.insecure, use_get=a.get) as s:
        print(json.dumps(s.query(a.domain, a.qtype), indent=2))


if __name__ == "__main__":
    main()
