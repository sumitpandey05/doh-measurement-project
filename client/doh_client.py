"""DNS-over-HTTPS (RFC 8484) client.

Two modes, because they measure different things:
  * cold (reuse=False): a brand-new TCP + TLS connection for EVERY query.
    rtt therefore includes the TCP and TLS handshakes = the "price of
    encryption" a first lookup pays.
  * warm (reuse=True): one long-lived HTTPS connection reused for many
    queries - what a browser does. Handshake cost is paid once, not timed.

HTTP/2 is the default. RFC 8484 recommends it, browsers use it, and some
resolvers no longer accept DoH over HTTP/1.1 at all (Quad9 answers 505 since
December 2025). Use --http1 only to reproduce old runs.

The DoH hostname (e.g. cloudflare-dns.com) is resolved ONCE when the session is
created and the connection is then made to that IP, with the real name sent as
TLS SNI and HTTP Host. Without this, every cold query would also time a system
DNS lookup of the resolver's own name, which is not part of the DoH cost.

Usage:
    python client/doh_client.py google.com https://cloudflare-dns.com/dns-query
    python client/doh_client.py google.com https://dns.google/dns-query --cold
"""
from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import socket
import time
from urllib.parse import urlparse, urlunparse

import dns.message
import httpx

from common import build_query, make_result, short_error

DOH_MIME = "application/dns-message"


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def pin_url(url: str, family: int = socket.AF_INET) -> tuple[str, str | None, str | None]:
    """Resolve the DoH hostname once. Returns (url_with_ip, hostname, ip).
    If the URL already uses an IP (local tests) nothing changes."""
    parts = urlparse(url)
    host = parts.hostname or ""
    if not host or _is_ip(host) or host == "localhost":
        return url, None, None
    port = parts.port or (443 if parts.scheme == "https" else 80)
    ip = socket.getaddrinfo(host, port, family, socket.SOCK_STREAM)[0][4][0]
    netloc = (f"[{ip}]" if ":" in ip else ip) + (f":{parts.port}" if parts.port else "")
    return urlunparse(parts._replace(netloc=netloc)), host, ip


class DoHSession:
    def __init__(self, url: str, reuse: bool = True, timeout: float = 5.0,
                 http2: bool = True, verify: bool = True, use_get: bool = False,
                 pin_ip: bool = True):
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
        self.host, self.ip = None, None
        self._target = url
        if pin_ip:
            try:
                self._target, self.host, self.ip = pin_url(url)
            except OSError:
                self._target, self.host, self.ip = url, None, None
        self._headers = {"accept": DOH_MIME}
        if self.host:
            self._headers["host"] = self.host
        self._extensions = {"sni_hostname": self.host} if self.host else {}
        # Build the (slow) SSL context ONCE so that creating a client for a cold
        # query does not add certificate-loading time to the measured latency.
        self._verify = httpx.create_ssl_context() if verify else False
        self._client = self._new_client() if reuse else None

    def _new_client(self) -> httpx.Client:
        return httpx.Client(http2=self.http2, timeout=self.timeout, verify=self._verify,
                            headers=self._headers)

    def query(self, domain: str, qtype: str = "A") -> dict:
        # RFC 8484 section 4.1: use DNS ID 0 so identical questions are cache friendly.
        wire = build_query(domain, qtype, qid=0).to_wire()
        client = self._client if self.reuse else self._new_client()  # created OUTSIDE the timer
        send_ts = time.time()
        t0 = time.perf_counter()
        try:
            if self.use_get:
                token = base64.urlsafe_b64encode(wire).rstrip(b"=").decode()
                resp = client.get(self._target, params={"dns": token}, extensions=self._extensions)
            else:
                resp = client.post(self._target, content=wire, headers={"content-type": DOH_MIME},
                                   extensions=self._extensions)
            t1 = time.perf_counter()
        except httpx.TimeoutException as exc:
            return make_result("timeout", send_ts, t0, time.perf_counter(), error=short_error(exc))
        except (httpx.HTTPError, OSError) as exc:
            return make_result("error", send_ts, t0, time.perf_counter(), error=short_error(exc))
        finally:
            if not self.reuse:
                client.close()  # TLS shutdown happens after the timer stopped

        ver = resp.http_version
        if resp.status_code != 200:
            return make_result(f"http_{resp.status_code}", send_ts, t0, t1, error=resp.reason_phrase,
                               http_version=ver)
        try:
            msg = dns.message.from_wire(resp.content)  # parsing is outside the timing window
        except Exception as exc:
            return make_result("bad_response", send_ts, t0, t1, error=short_error(exc), http_version=ver)
        return make_result("ok", send_ts, t0, t1, msg, http_version=ver)

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
    ap.add_argument("--http1", action="store_true", help="force HTTP/1.1 (default is HTTP/2)")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification (local tests only)")
    ap.add_argument("--timeout", type=float, default=5.0)
    a = ap.parse_args()
    with DoHSession(a.url, reuse=not a.cold, timeout=a.timeout, http2=not a.http1,
                    verify=not a.insecure, use_get=a.get) as s:
        print(json.dumps(s.query(a.domain, a.qtype), indent=2))


if __name__ == "__main__":
    main()
