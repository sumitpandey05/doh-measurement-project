"""Run any command while tshark records the DNS / DoH packets it produces.

Why: the ML half of the project needs the real on-the-wire packets (sizes and
timing of encrypted flows), not just the client-side timestamps.

Usage (from the repo root, venv active):
    python capture/capture_runner.py --label clean_A -- \
        python client/run_batch.py --person A --profile clean --repeats 1

Output (both git-ignored / local only):
    capture/pcaps/<label>_<timestamp>.pcap
    capture/pcaps/<label>_<timestamp>.meta.json   (interface, filter, resolver IPs)

Needs tshark with permission to capture without sudo:
    sudo dpkg-reconfigure wireshark-common   # answer Yes
    sudo usermod -aG wireshark $USER         # then log out / back in (WSL: wsl --shutdown)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def detect_interface() -> str:
    """Interface used for the default route (e.g. eth0 in WSL2); falls back to 'any'."""
    try:
        out = subprocess.run(["ip", "route", "get", "1.1.1.1"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"\bdev\s+(\S+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "any"


def resolve_resolver_ips(resolvers_yaml: Path) -> tuple[dict, dict, dict]:
    """name -> [ips] (classic IP + whatever the DoH hostname resolves to right now),
    name -> DoH hostname (used later to label flows by TLS SNI), name -> DoH port."""
    with open(resolvers_yaml) as f:
        cfg = yaml.safe_load(f)
    ips, hosts, ports = {}, {}, {}
    for r in cfg["resolvers"]:
        name = str(r["name"])
        found = {str(r["classic_ip"])}
        url = urlparse(str(r["doh_url"]))
        host = url.hostname or ""
        hosts[name] = host
        ports[name] = url.port or (443 if url.scheme == "https" else 80)
        try:
            for info in socket.getaddrinfo(host, None):
                found.add(info[4][0])
        except OSError:
            print(f"[warn] could not resolve {host}", file=sys.stderr)
        ips[name] = sorted(found)
    return ips, hosts, ports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="capture", help="prefix of the pcap name, e.g. clean_A")
    ap.add_argument("--interface", help="default: auto-detect (route to 1.1.1.1)")
    ap.add_argument("--filter", default="udp port 53 or tcp port 443",
                    help="tshark/BPF capture filter. NOTE: DoH is on 443, NOT 53")
    ap.add_argument("--resolvers", default=str(REPO_ROOT / "client" / "resolvers.yaml"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "capture" / "pcaps"))
    ap.add_argument("--startup-wait", type=float, default=30.0, help="max seconds to wait for tshark to start")
    ap.add_argument("command", nargs=argparse.REMAINDER, help="-- <command to run while capturing>")
    a = ap.parse_args()

    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd:
        ap.error("give a command after --, e.g. -- python client/run_batch.py --person A")
    if shutil.which("tshark") is None:
        print("tshark not found: sudo apt install tshark", file=sys.stderr)
        return 2

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pcap = out_dir / f"{a.label}_{stamp}.pcap"
    iface = a.interface or detect_interface()
    ips, hosts, ports = resolve_resolver_ips(Path(a.resolvers))

    meta = {"pcap": pcap.name, "interface": iface, "filter": a.filter, "label": a.label,
            "resolver_ips": ips, "resolver_hosts": hosts, "resolver_ports": ports, "command": cmd,
            "started_epoch": time.time()}

    tshark_cmd = ["tshark", "-i", iface, "-f", a.filter, "-n", "-q", "-F", "pcap", "-w", str(pcap)]
    print("capture:", " ".join(tshark_cmd))
    proc = subprocess.Popen(tshark_cmd, stderr=subprocess.PIPE, text=True)
    # tshark can take several seconds to start. Do NOT guess with a fixed sleep (the first
    # queries would be missing from the pcap): wait until it says "Capturing on ...".
    ready, log = threading.Event(), []

    def _watch_stderr():
        for line in proc.stderr:
            log.append(line.rstrip())
            if "Capturing on" in line:
                ready.set()
        ready.set()  # stderr closed -> tshark ended; the poll() check below reports it

    threading.Thread(target=_watch_stderr, daemon=True).start()
    ready.wait(timeout=a.startup_wait)
    time.sleep(0.5)
    if proc.poll() is not None or not any("Capturing on" in l for l in log):
        print("tshark did not start capturing:\n" + "\n".join(log[-10:]), file=sys.stderr)
        print("(permission problem? see the header of this file)", file=sys.stderr)
        proc.kill()
        return 3

    try:
        rc = subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        rc = 130
    finally:
        time.sleep(1.5)  # let the last packets reach the capture
        proc.send_signal(signal.SIGINT)  # tshark finalises the file on SIGINT
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    meta["finished_epoch"] = time.time()
    meta["command_exit_code"] = rc
    with open(pcap.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    size = pcap.stat().st_size if pcap.exists() else 0
    print(f"saved {pcap} ({size/1024:.1f} KiB) + meta; command exit code {rc}")
    if size < 100:
        print("[warn] pcap is (almost) empty - wrong interface or filter?", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
