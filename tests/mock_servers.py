"""Local fake DNS + DoH servers so the whole pipeline can be tested offline.

Starts, on 127.0.0.1:
  * a UDP DNS server  (answers every A query with a deterministic 10.x.y.z)
  * a DoH server over plain HTTP
  * a DoH server over HTTPS (self-signed cert, needs `openssl`)
All three can add an artificial delay. Used by tests/test_pipeline.py, and
runnable by hand:

    python tests/mock_servers.py --udp-port 15353 --http-port 18053 --tls-port 18443 --delay-ms 20
and then point a resolvers file at them (see tests/resolvers_local.yaml).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import dns.message
import dns.rrset


def _answer(wire: bytes) -> bytes:
    q = dns.message.from_wire(wire)
    resp = dns.message.make_response(q)
    name = q.question[0].name
    h = hashlib.sha256(name.to_text().encode()).digest()
    ip = f"10.{h[0]}.{h[1]}.{h[2]}"
    if q.question[0].rdtype == dns.rdatatype.A:
        resp.answer.append(dns.rrset.from_text(name, 300, "IN", "A", ip))
    return resp.to_wire()


class UdpDnsServer:
    def __init__(self, port: int, delay_ms: float = 0.0, host: str = "127.0.0.1"):
        self.delay = delay_ms / 1000.0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.2)
        self.port = self.sock.getsockname()[1]
        self.queries = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            self.queries += 1
            try:
                out = _answer(data)
            except Exception:
                continue
            if self.delay:
                time.sleep(self.delay)
            self.sock.sendto(out, addr)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.sock.close()


class _DohHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, so a "warm" client really reuses the connection
    # Send headers+body in ONE write with Nagle off. Without this, headers and body go out as
    # two small segments and the Nagle/delayed-ACK interaction adds a random ~40 ms stall to
    # reused connections - a test-server artifact that would pollute the latency numbers.
    wbufsize = 65536
    disable_nagle_algorithm = True

    def log_message(self, *a):
        pass

    def _reply(self, wire: bytes):
        if self.server.delay:
            time.sleep(self.server.delay)
        body = _answer(wire)
        self.send_response(200)
        self.send_header("Content-Type", "application/dns-message")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.server.requests += 1
        self._reply(self.rfile.read(n))

    def do_GET(self):
        qs = parse_qs(urlparse(self.path).query)
        token = qs["dns"][0]
        self.server.requests += 1
        self._reply(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))


class DohServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, delay_ms: float = 0.0, certfile=None, keyfile=None,
                 host: str = "127.0.0.1"):
        super().__init__((host, port), _DohHandler)
        self.delay = delay_ms / 1000.0
        self.connections = 0   # TCP connections accepted
        self.requests = 0      # HTTP requests served
        self.port = self.server_address[1]
        self.tls = bool(certfile)
        if certfile:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile, keyfile)
            self.socket = ctx.wrap_socket(self.socket, server_side=True)
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    def get_request(self):
        sock, addr = super().get_request()
        self.connections += 1
        return sock, addr

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self.shutdown()
        self.server_close()


def make_self_signed(tmpdir: Path):
    cert, key = tmpdir / "cert.pem", tmpdir / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key),
                    "-out", str(cert), "-days", "2", "-subj", "/CN=127.0.0.1",
                    "-addext", "subjectAltName=IP:127.0.0.1"],
                   check=True, capture_output=True)
    return str(cert), str(key)


class MockWorld:
    """Context manager that starts all three servers on free ports."""

    def __init__(self, delay_ms: float = 0.0, udp_port: int = 0, http_port: int = 0, tls_port: int = 0):
        self.delay_ms = delay_ms
        self._ports = (udp_port, http_port, tls_port)

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        cert, key = make_self_signed(Path(self._tmp.name))
        up, hp, tp = self._ports
        self.udp = UdpDnsServer(up, self.delay_ms).start()
        self.http = DohServer(hp, self.delay_ms).start()
        self.tls = DohServer(tp, self.delay_ms, cert, key).start()
        return self

    def __exit__(self, *exc):
        self.udp.stop()
        self.http.stop()
        self.tls.stop()
        self._tmp.cleanup()

    def resolvers_yaml(self) -> str:
        return (
            "resolvers:\n"
            "  - name: local_http\n    classic_ip: 127.0.0.1\n"
            f"    classic_port: {self.udp.port}\n    doh_url: http://127.0.0.1:{self.http.port}/dns-query\n"
            "  - name: local_tls\n    classic_ip: 127.0.0.1\n"
            f"    classic_port: {self.udp.port}\n    doh_url: https://127.0.0.1:{self.tls.port}/dns-query\n"
            "    tls_verify: false\n"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--udp-port", type=int, default=15353)
    ap.add_argument("--http-port", type=int, default=18053)
    ap.add_argument("--tls-port", type=int, default=18443)
    ap.add_argument("--delay-ms", type=float, default=0.0)
    a = ap.parse_args()
    with MockWorld(a.delay_ms, a.udp_port, a.http_port, a.tls_port) as w:
        print(f"UDP DNS  127.0.0.1:{w.udp.port}\nDoH http 127.0.0.1:{w.http.port}\nDoH tls  127.0.0.1:{w.tls.port}")
        print("resolvers yaml for these servers:\n" + w.resolvers_yaml())
        print("Ctrl-C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
