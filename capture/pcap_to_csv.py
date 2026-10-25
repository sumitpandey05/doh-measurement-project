"""Turn a pcap into CSVs for analysis and ML.  Three sub-commands:

  flows    TCP flows to the DoH servers -> one row per flow with the SAME feature
           columns as the CIRA-CIC-DoHBrw-2020 dataset (so a model trained on the
           dataset can be tested on your own capture).
  packets  one row per packet (time, size, direction)  - handy for plots/debugging.
  udp-rtt  match classic-DNS queries with their answers inside the pcap -> RTT
           measured ON THE WIRE (used to cross-check the client's own timing).

Usage (from the repo root):
    python capture/pcap_to_csv.py flows   capture/pcaps/x.pcap --meta capture/pcaps/x.meta.json \
                                          --out results/own_flows_A.csv
    python capture/pcap_to_csv.py packets capture/pcaps/x.pcap --out results/x_packets.csv
    python capture/pcap_to_csv.py udp-rtt capture/pcaps/x.pcap --out results/x_udp_rtt.csv

Feature definitions follow DoHMeter / DoHlyzer, the tool that produced the
CIRA-CIC-DoHBrw-2020 CSVs (github.com/ahlashkari/DoHlyzer, meter/features/*.py):
  * packet length  = frame length = IP packet size + 14 bytes Ethernet header
  * PacketTime     = time of each packet RELATIVE TO THE FIRST packet of the flow
                     (not the gap between packets)
  * ResponseTime   = time between a client->server packet and the server->client
                     packet that immediately follows it (any packet, data or not)
  * variance       = population variance; std = sqrt(variance); mode = smallest of
                     the most frequent values (scipy.stats.mode)
  * SkewFromMedian = 3 * (mean - median) / std,  -10 when std == 0
  * SkewFromMode   = (mean - mode) / std,        -10 when std == 0
  * CoefficientofVariation = std / mean,         -1 when mean == 0
  * Flow*Rate      = bytes / duration,           -1 when duration == 0
  * a flow ends after 40 s of silence or when it is longer than 120 s
A flow without any client->server->client turn has no ResponseTime median; DoHMeter
writes NaN there and such rows are dropped, exactly like the dataset loader does.
Remaining differences (capture point, flow cutting during garbage collection) are
small but real: state them in the paper.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMNS = ["SourceIP", "DestinationIP", "SourcePort", "DestinationPort", "TimeStamp"]
BASE_FEATURES = ["Duration", "FlowBytesSent", "FlowSentRate", "FlowBytesReceived", "FlowReceivedRate"]
STAT_SUFFIXES = ["Variance", "StandardDeviation", "Mean", "Median", "Mode",
                 "SkewFromMedian", "SkewFromMode", "CoefficientofVariation"]
STAT_GROUPS = ["PacketLength", "PacketTime", "ResponseTimeTime"]  # (sic) matches the dataset's spelling
FEATURE_COLUMNS = BASE_FEATURES + [g + s for g in STAT_GROUPS for s in STAT_SUFFIXES]
FLOW_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + ["Label", "Resolver", "Capture"]
ETHERNET_HEADER = 14
IDLE_TIMEOUT = 40.0
MAX_FLOW_DURATION = 120.0


# ----------------------------------------------------------------------------- packets
@dataclass
class Pkt:
    ts: float
    src: str
    dst: str
    sport: int
    dport: int
    proto: str          # TCP / UDP
    length: int         # IP packet length in bytes
    payload: int        # transport payload length in bytes
    first_payload: bytes = b""


def read_packets(pcap: str | Path):
    """Yield Pkt for every TCP/UDP packet (IPv4 or IPv6)."""
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
    from scapy.utils import PcapReader

    with PcapReader(str(pcap)) as reader:
        for pkt in reader:
            if IP in pkt:
                ip = pkt[IP]
                total = int(ip.len)
                l4_len = total - int(ip.ihl) * 4
                src, dst = ip.src, ip.dst
            elif IPv6 in pkt:
                ip = pkt[IPv6]
                l4_len = int(ip.plen)
                total = l4_len + 40
                src, dst = ip.src, ip.dst
            else:
                continue
            if TCP in pkt:
                t = pkt[TCP]
                payload = l4_len - int(t.dataofs) * 4  # NOT len(payload): Ethernet padding would inflate it
                raw = bytes(t.payload)[:payload] if payload > 0 else b""
                yield Pkt(float(pkt.time), src, dst, int(t.sport), int(t.dport), "TCP", total,
                          max(payload, 0), raw[:700])
            elif UDP in pkt:
                u = pkt[UDP]
                payload = int(u.len) - 8
                raw = bytes(u.payload)[:payload] if payload > 0 else b""
                yield Pkt(float(pkt.time), src, dst, int(u.sport), int(u.dport), "UDP", total,
                          max(payload, 0), raw)


# ----------------------------------------------------------------------------- TLS SNI
def parse_sni(payload: bytes) -> str | None:
    """Server name from a TLS ClientHello (plaintext in TLS1.2/1.3 unless ECH). None if absent."""
    try:
        if len(payload) < 44 or payload[0] != 0x16 or payload[5] != 0x01:
            return None
        pos = 5 + 4 + 2 + 32                      # record hdr, handshake hdr, version, random
        pos += 1 + payload[pos]                   # session id
        pos += 2 + struct.unpack("!H", payload[pos:pos + 2])[0]   # cipher suites
        pos += 1 + payload[pos]                   # compression
        end = pos + 2 + struct.unpack("!H", payload[pos:pos + 2])[0]
        pos += 2
        while pos + 4 <= min(end, len(payload)):
            etype, elen = struct.unpack("!HH", payload[pos:pos + 4])
            pos += 4
            if etype == 0 and elen >= 5:
                nlen = struct.unpack("!H", payload[pos + 3:pos + 5])[0]
                return payload[pos + 5:pos + 5 + nlen].decode("ascii", "ignore") or None
            pos += elen
    except Exception:
        return None
    return None


# ----------------------------------------------------------------------------- flows
@dataclass
class Flow:
    client_ip: str
    client_port: int
    server_ip: str
    server_port: int
    ts: list = field(default_factory=list)
    length: list = field(default_factory=list)
    c2s: list = field(default_factory=list)
    has_data: list = field(default_factory=list)
    hello: bytes = b""


def build_flows(packets, server_ports, proto="TCP", flow_timeout=IDLE_TIMEOUT,
                max_duration=MAX_FLOW_DURATION) -> list[Flow]:
    """Group packets into bidirectional flows. A flow ends after `flow_timeout` s of silence
    or once it has lasted `max_duration` s (DoHMeter: 40 s idle, flows cut after ~90-120 s)."""
    server_ports = set(server_ports)
    open_flows: dict = {}
    flows: list[Flow] = []
    for p in sorted((p for p in packets if p.proto == proto), key=lambda p: p.ts):
        if p.dport in server_ports and p.sport not in server_ports:
            c2s, cip, cport, sip, sport = True, p.src, p.sport, p.dst, p.dport
        elif p.sport in server_ports and p.dport not in server_ports:
            c2s, cip, cport, sip, sport = False, p.dst, p.dport, p.src, p.sport
        else:
            continue
        key = (cip, cport, sip, sport)
        fl = open_flows.get(key)
        if fl is None or p.ts - fl.ts[-1] > flow_timeout or (max_duration and p.ts - fl.ts[0] > max_duration):
            fl = Flow(cip, cport, sip, sport)
            open_flows[key] = fl
            flows.append(fl)
        fl.ts.append(p.ts)
        fl.length.append(p.length + ETHERNET_HEADER)
        fl.c2s.append(c2s)
        fl.has_data.append(p.payload > 0)
        if c2s and p.payload > 0 and not fl.hello:
            fl.hello = p.first_payload
    return flows


def _mode(v: np.ndarray) -> float:
    vals, counts = np.unique(v, return_counts=True)
    return float(vals[counts.argmax()])


def _stats(values) -> list[float]:
    """DoHMeter statistics, in STAT_SUFFIXES order:
    [variance, std, mean, median, mode, skew_from_median, skew_from_mode, coeff_of_variation]"""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return [-1.0, -1.0, -1.0, float("nan"), -1.0, float("nan"), 0.0, 1.0]
    mean, median = float(v.mean()), float(np.median(v))
    var = float(v.var())
    std = float(np.sqrt(var))
    mode = _mode(v)
    skew_med = 3 * (mean - median) / std if std != 0 else -10.0
    skew_mode = (mean - mode) / std if std != 0 else -10.0
    cv = std / mean if mean != 0 else -1.0
    return [var, std, mean, median, mode, skew_med, skew_mode, cv]


def flow_features(fl: Flow) -> dict:
    ts = np.asarray(fl.ts, dtype=float)
    ln = np.asarray(fl.length, dtype=float)
    c2s = np.asarray(fl.c2s, dtype=bool)
    forward = c2s if c2s[0] else ~c2s          # DoHMeter: direction of the FIRST packet = forward
    duration = float(ts.max() - ts.min())
    sent, recv = float(ln[forward].sum()), float(ln[~forward].sum())
    rel_times = ts - ts[0]
    resp = [ts[i] - ts[i - 1] for i in range(1, len(ts)) if forward[i - 1] and not forward[i]]
    feats = {
        "Duration": duration,
        "FlowBytesSent": sent,
        "FlowSentRate": sent / duration if duration > 0 else -1.0,
        "FlowBytesReceived": recv,
        "FlowReceivedRate": recv / duration if duration > 0 else -1.0,
    }
    for group, values in (("PacketLength", ln), ("PacketTime", rel_times), ("ResponseTimeTime", resp)):
        for suffix, x in zip(STAT_SUFFIXES, _stats(values)):
            feats[group + suffix] = x
    return feats


def label_resolver(fl: Flow, meta: dict | None) -> str:
    if not meta:
        return "unknown"
    sni = parse_sni(fl.hello) if fl.hello else None
    if sni:
        for name, host in meta.get("resolver_hosts", {}).items():
            if host and host.lower() == sni.lower():
                return name
    ports = meta.get("resolver_ports", {})
    for name, ips in meta.get("resolver_ips", {}).items():
        if fl.server_ip in ips and ports.get(name, fl.server_port) == fl.server_port:
            return name
    return "unknown"


def flows_to_frame(pcap, meta=None, server_ports=(443,), min_packets=4, flow_timeout=IDLE_TIMEOUT,
                   label="Benign", only_known=True, max_duration=MAX_FLOW_DURATION) -> pd.DataFrame:
    flows = build_flows(read_packets(pcap), server_ports, "TCP", flow_timeout, max_duration)
    rows = []
    for fl in flows:
        if len(fl.ts) < min_packets:
            continue
        resolver = label_resolver(fl, meta)
        if only_known and meta and resolver == "unknown":
            continue
        row = {"SourceIP": fl.client_ip, "DestinationIP": fl.server_ip, "SourcePort": fl.client_port,
               "DestinationPort": fl.server_port, "TimeStamp": fl.ts[0]}
        row.update(flow_features(fl))
        row.update({"Label": label, "Resolver": resolver, "Capture": Path(pcap).name})
        rows.append(row)
    return pd.DataFrame(rows, columns=FLOW_COLUMNS)


# ----------------------------------------------------------------------------- packets csv / udp rtt
def packets_to_frame(pcap, server_ports=(53, 443)) -> pd.DataFrame:
    sp = set(server_ports)
    rows = []
    for p in read_packets(pcap):
        if p.dport in sp:
            d = "c2s"
        elif p.sport in sp:
            d = "s2c"
        else:
            continue
        rows.append({"ts": p.ts, "direction": d, "src": p.src, "dst": p.dst, "sport": p.sport,
                     "dport": p.dport, "proto": p.proto, "length": p.length, "payload_len": p.payload})
    return pd.DataFrame(rows, columns=["ts", "direction", "src", "dst", "sport", "dport", "proto",
                                       "length", "payload_len"])


def udp_dns_rtt(pcap, server_ports=(53,)) -> pd.DataFrame:
    """Pair each UDP DNS query with its answer (same client ip/port, DNS id, server)."""
    import dns.flags
    import dns.message

    sp = set(server_ports)
    pending, rows = {}, []
    for p in read_packets(pcap):
        if p.proto != "UDP" or not p.first_payload:
            continue
        try:
            msg = dns.message.from_wire(p.first_payload)
        except Exception:
            continue
        is_response = bool(msg.flags & dns.flags.QR)
        if not is_response and p.dport in sp:
            pending[(p.src, p.sport, msg.id, p.dst)] = (p.ts, msg.question[0].name.to_text() if msg.question else "")
        elif is_response and p.sport in sp:
            hit = pending.pop((p.dst, p.dport, msg.id, p.src), None)
            if hit:
                rows.append({"ts_query": hit[0], "ts_answer": p.ts, "rtt_ms": (p.ts - hit[0]) * 1000.0,
                             "server_ip": p.src, "dns_id": msg.id, "qname": hit[1]})
    return pd.DataFrame(rows, columns=["ts_query", "ts_answer", "rtt_ms", "server_ip", "dns_id", "qname"])


# ----------------------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("flows", help="per-flow features (dataset-compatible columns)")
    f.add_argument("pcap")
    f.add_argument("--meta", help="the .meta.json written by capture_runner.py (labels the resolver)")
    f.add_argument("--out", required=True)
    f.add_argument("--server-ports", type=int, nargs="+", default=[443])
    f.add_argument("--min-packets", type=int, default=4)
    f.add_argument("--flow-timeout", type=float, default=IDLE_TIMEOUT, help="idle seconds that end a flow")
    f.add_argument("--max-duration", type=float, default=MAX_FLOW_DURATION, help="cut flows longer than this")
    f.add_argument("--label", default="Benign", help="label to write (own traffic is benign by construction)")
    f.add_argument("--keep-unknown", action="store_true", help="keep flows not matched to a resolver")

    k = sub.add_parser("packets", help="one row per packet")
    k.add_argument("pcap")
    k.add_argument("--out", required=True)
    k.add_argument("--server-ports", type=int, nargs="+", default=[53, 443])

    u = sub.add_parser("udp-rtt", help="on-the-wire RTT of classic DNS")
    u.add_argument("pcap")
    u.add_argument("--out", required=True)
    u.add_argument("--server-ports", type=int, nargs="+", default=[53])

    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.cmd == "flows":
        meta = json.load(open(a.meta)) if a.meta else None
        df = flows_to_frame(a.pcap, meta, a.server_ports, a.min_packets, a.flow_timeout, a.label,
                            only_known=not a.keep_unknown, max_duration=a.max_duration)
        if df.empty:
            print("[warn] no flows extracted - check --server-ports, the filter used for capture, and --meta",
                  file=sys.stderr)
        else:
            print(df["Resolver"].value_counts().to_string())
    elif a.cmd == "packets":
        df = packets_to_frame(a.pcap, a.server_ports)
    else:
        df = udp_dns_rtt(a.pcap, a.server_ports)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
