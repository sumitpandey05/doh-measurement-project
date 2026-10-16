"""Automated checks for the whole project. Run from the repo root:   pytest -q

Everything runs offline against the local mock servers / synthetic data, so a green run
means the CODE works. It does not prove anything about real resolvers or the real dataset.
"""
from __future__ import annotations

import json
import socket
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mock_servers import MockWorld, _answer

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


@pytest.fixture(scope="module")
def world():
    with MockWorld(delay_ms=10) as w:
        yield w


# ----------------------------------------------------------------------------- client
def test_domain_list_fallback(tmp_path):
    out = tmp_path / "domains.csv"
    subprocess.run([PY, str(ROOT / "client/domain_list.py"), "--out", str(out)], check=True, capture_output=True)
    df = pd.read_csv(out)
    assert len(df) == 50 and df["domain"].is_unique


def test_all_three_methods_agree_and_are_timed(world):
    from classic_dns import query_classic
    from doh_client import DoHSession

    classic = query_classic("example.org", "127.0.0.1", port=world.udp.port, timeout=2)
    with DoHSession(f"http://127.0.0.1:{world.http.port}/dns-query", reuse=True) as s:
        warm = s.query("example.org")
    with DoHSession(f"https://127.0.0.1:{world.tls.port}/dns-query", reuse=False, verify=False) as s:
        cold = s.query("example.org")
    for r in (classic, warm, cold):
        assert r["status"] == "ok" and r["rcode"] == "NOERROR" and r["answer_count"] == 1
        assert r["rtt_ms"] >= 9, "mock adds 10 ms; measured RTT must include it"
    assert classic["resolved_ips"] == warm["resolved_ips"] == cold["resolved_ips"]


def test_warm_reuses_connection_cold_does_not(world):
    from doh_client import DoHSession

    url = f"https://127.0.0.1:{world.tls.port}/dns-query"
    before = world.tls.connections
    with DoHSession(url, reuse=True, verify=False) as s:
        for _ in range(5):
            assert s.query("example.org")["status"] == "ok"
    assert world.tls.connections - before == 1, "warm session must use ONE connection"
    before = world.tls.connections
    with DoHSession(url, reuse=False, verify=False) as s:
        for _ in range(5):
            assert s.query("example.org")["status"] == "ok"
    assert world.tls.connections - before == 5, "cold session must open a new connection per query"


def test_doh_get_method(world):
    from doh_client import DoHSession

    with DoHSession(f"http://127.0.0.1:{world.http.port}/dns-query", use_get=True) as s:
        assert s.query("example.org")["status"] == "ok"


def test_failures_are_recorded_not_raised(world):
    from classic_dns import query_classic
    from doh_client import DoHSession

    assert query_classic("example.org", "127.0.0.1", port=9, timeout=0.4)["status"] in ("timeout", "error")
    assert DoHSession("http://127.0.0.1:9/dns-query", timeout=0.4).query("example.org")["status"] == "error"


def test_udp_retry_recovers_from_one_lost_packet():
    from classic_dns import query_classic

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(3)
    port = sock.getsockname()[1]

    def flaky():
        data, addr = sock.recvfrom(4096)      # first query: silently dropped
        data, addr = sock.recvfrom(4096)      # retry: answered
        sock.sendto(_answer(data), addr)

    t = threading.Thread(target=flaky, daemon=True)
    t.start()
    r = query_classic("example.org", "127.0.0.1", port=port, timeout=0.5, retries=1)
    t.join(2)
    sock.close()
    assert r["status"] == "ok" and r["attempts"] == 2 and r["rtt_ms"] >= 500


def test_run_batch_end_to_end_and_summaries(world, tmp_path):
    from common import RAW_COLUMNS

    (tmp_path / "res.yaml").write_text(world.resolvers_yaml())
    dom = tmp_path / "domains.csv"
    subprocess.run([PY, str(ROOT / "client/domain_list.py"), "--out", str(dom)], check=True, capture_output=True)
    for person in ("A", "B"):
        r = subprocess.run([PY, str(ROOT / "client/run_batch.py"), "--person", person, "--profile", "clean",
                            "--repeats", "2", "--limit", "4", "--domains", str(dom), "--resolvers",
                            str(tmp_path / "res.yaml"), "--out-dir", str(tmp_path), "--pause", "0"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    files = sorted(tmp_path.glob("raw_*.csv"))
    assert len(files) == 4 and len(list(tmp_path.glob("raw_*.meta.json"))) == 4
    df = pd.read_csv(files[0])
    assert list(df.columns) == RAW_COLUMNS
    assert len(df) == 4 * 2 * 3 and set(df.method) == {"classic_udp", "doh_cold", "doh_warm"}
    assert (df.status == "ok").all()

    from merge_results import merge
    from summarize import load_raw, summarize, summarize_by_run

    merged, problems = merge(tmp_path)
    assert not problems and len(merged) == 4 * 4 * 2 * 3
    s = summarize(load_raw(tmp_path))
    assert (s.success_rate == 1).all() and len(s) == 2 * 3
    assert summarize_by_run(load_raw(tmp_path)).n_runs.eq(4).all()


def test_merge_flags_files_with_wrong_columns(tmp_path):
    from merge_results import merge

    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "raw_bad.csv", index=False)
    from common import RAW_COLUMNS
    pd.DataFrame([{c: 1 for c in RAW_COLUMNS}]).to_csv(tmp_path / "raw_good.csv", index=False)
    merged, problems = merge(tmp_path)
    assert len(merged) == 1 and len(problems) == 1 and "raw_bad.csv" in problems[0]


# ----------------------------------------------------------------------------- capture / features
def _client_hello(sni: str) -> bytes:
    name = sni.encode()
    sni_list = struct.pack("!BH", 0, len(name)) + name
    ext = struct.pack("!HH", 0, len(sni_list) + 2) + struct.pack("!H", len(sni_list)) + sni_list
    body = (b"\x03\x03" + b"\x11" * 32 + b"\x00" + struct.pack("!H", 2) + b"\x13\x02" + b"\x01\x00"
            + struct.pack("!H", len(ext)) + ext)
    hs = b"\x01" + struct.pack("!I", len(body))[1:] + body
    return b"\x16\x03\x01" + struct.pack("!H", len(hs)) + hs


def test_sni_parser():
    from pcap_to_csv import parse_sni

    assert parse_sni(_client_hello("cloudflare-dns.com")) == "cloudflare-dns.com"
    assert parse_sni(b"GET / HTTP/1.1") is None
    assert parse_sni(b"") is None


def test_flow_features_match_hand_calculation(tmp_path):
    """A 6-packet flow whose features we can compute on paper (incl. Ethernet padding)."""
    from scapy.all import Ether, IP, Padding, Raw, TCP, wrpcap

    from pcap_to_csv import flows_to_frame

    t0 = 1_700_000_000.0
    spec = [  # (time offset, client->server?, payload bytes)
        (0.00, True, 0), (0.01, False, 0), (0.02, True, 0), (0.03, True, 100), (0.08, False, 300), (0.09, True, 0)]
    pkts = []
    for off, up, n in spec:
        src, dst = ("10.0.0.2", "1.1.1.1") if up else ("1.1.1.1", "10.0.0.2")
        sp, dp = (40000, 443) if up else (443, 40000)
        p = Ether() / IP(src=src, dst=dst) / TCP(sport=sp, dport=dp, flags="PA" if n else "A")
        if n:
            p = p / Raw(b"x" * n)
        else:
            p = p / Padding(b"\x00" * 6)  # real NICs pad tiny frames; must NOT count as payload
        p.time = t0 + off
        pkts.append(p)
    pcap = tmp_path / "f.pcap"
    wrpcap(str(pcap), pkts)

    df = flows_to_frame(pcap, meta=None, server_ports=(443,), min_packets=4)
    assert len(df) == 1
    r = df.iloc[0]
    assert r.SourceIP == "10.0.0.2" and r.DestinationIP == "1.1.1.1" and r.DestinationPort == 443
    assert r.Duration == pytest.approx(0.09, abs=1e-4)
    assert r.FlowBytesSent == 40 + 40 + 140 + 40       # SYN-less: ack, data(100+40), ack, first pkt
    assert r.FlowBytesReceived == 40 + 340
    assert r.FlowSentRate == pytest.approx(260 / 0.09, rel=1e-3)
    assert r.PacketLengthMean == pytest.approx((40 * 4 + 140 + 340) / 6)
    assert r.PacketLengthMedian == 40 and r.PacketLengthMode == 40
    assert r.ResponseTimeTimeMean == pytest.approx(0.05, abs=1e-4)   # request@0.03 -> answer@0.08
    assert r.PacketTimeMean == pytest.approx(0.018, abs=1e-4)


def test_flows_are_labelled_by_resolver_via_port_and_ip():
    from pcap_to_csv import Flow, label_resolver

    meta = {"resolver_hosts": {"a": "a.example", "b": "b.example"},
            "resolver_ips": {"a": ["127.0.0.1"], "b": ["127.0.0.1"]}, "resolver_ports": {"a": 8053, "b": 8443}}
    assert label_resolver(Flow("c", 1, "127.0.0.1", 8053), meta) == "a"
    assert label_resolver(Flow("c", 1, "127.0.0.1", 8443), meta) == "b"
    assert label_resolver(Flow("c", 1, "9.9.9.9", 8443), meta) == "unknown"


def test_udp_rtt_pairs_queries_with_answers(tmp_path):
    import dns.message
    from scapy.all import Ether, IP, UDP, wrpcap

    from pcap_to_csv import udp_dns_rtt

    q = dns.message.make_query("example.com", "A")
    q.id = 4242
    resp = dns.message.make_response(q)
    pk = [Ether() / IP(src="10.0.0.2", dst="8.8.8.8") / UDP(sport=5555, dport=53) / q.to_wire(),
          Ether() / IP(src="8.8.8.8", dst="10.0.0.2") / UDP(sport=53, dport=5555) / resp.to_wire()]
    pk[0].time, pk[1].time = 1_700_000_000.000, 1_700_000_000.025
    wrpcap(str(tmp_path / "u.pcap"), pk)
    df = udp_dns_rtt(tmp_path / "u.pcap")
    assert len(df) == 1 and df.rtt_ms[0] == pytest.approx(25, abs=0.05) and df.server_ip[0] == "8.8.8.8"


# ----------------------------------------------------------------------------- ML
@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    from make_synthetic_dataset import make

    d = tmp_path_factory.mktemp("syn")
    b, m = make(300, 900, seed=1)
    b.to_csv(d / "l2-benign.csv", index=False)
    m.to_csv(d / "l2-malicious.csv", index=False)
    return d


def test_loader_cleans_and_drops_identifier_columns(synthetic):
    from load_dataset import load_prepared

    X, y, info = load_prepared(synthetic)
    assert info["rows_dropped_nan_inf"] == 8 and len(X) == len(y) == 1200 - 8
    assert not {"SourceIP", "DestinationIP", "SourcePort", "DestinationPort", "TimeStamp", "Label"} & set(X.columns)
    assert np.isfinite(X.to_numpy()).all() and set(y) == {0, 1}


def test_loader_rejects_layer1_labels(tmp_path):
    from load_dataset import load_prepared

    pd.DataFrame({"Duration": [1, 2], "Label": ["DoH", "NonDoH"]}).to_csv(tmp_path / "l1.csv", index=False)
    with pytest.raises(ValueError, match="Layer-2"):
        load_prepared(tmp_path)


def test_split_is_deterministic_and_stratified(synthetic, tmp_path):
    from preprocess import make_split, save_split

    s1, s2 = make_split(synthetic, 42), make_split(synthetic, 42)
    assert (s1.train_idx == s2.train_idx).all() and s1.data_fingerprint == s2.data_fingerprint
    p1 = save_split(s1, tmp_path / "a", 42)
    p2 = save_split(s2, tmp_path / "b", 42)
    assert p1.read_bytes() == p2.read_bytes(), "manifests of two machines must be byte-identical"
    assert abs(s1.y_train.mean() - s1.y_test.mean()) < 0.01       # stratified
    assert make_split(synthetic, 7).train_idx.tolist() != s1.train_idx.tolist()


def test_models_train_and_evaluate(synthetic):
    from evaluate import evaluate_model
    from models import build_rf, build_xgb
    from preprocess import make_split

    sp = make_split(synthetic, 42)
    for name, model in (("rf", build_rf(42, n_estimators=60)), ("xgb", build_xgb(42, sp.y_train, 60, balance=True))):
        model.fit(sp.X_train, sp.y_train)
        m = evaluate_model(model, sp.X_test, sp.y_test, name)
        assert m["accuracy"] > 0.8 and m["roc_auc"] > 0.9
        assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == m["n_test"]


def test_cross_resolver_script(synthetic, tmp_path):
    import joblib

    from models import build_xgb
    from preprocess import make_split

    sp = make_split(synthetic, 42)
    model = build_xgb(42, sp.y_train, 60)
    model.fit(sp.X_train, sp.y_train)
    # fake "own traffic" CSV built from benign rows, labelled with two resolvers
    joblib.dump({"model": model, "features": sp.feature_cols, "name": "xgb"}, tmp_path / "m.joblib")
    own = sp.X_test[sp.y_test == 0].head(30).copy()
    own["Resolver"] = ["cloudflare"] * 15 + ["google"] * 15
    own.to_csv(tmp_path / "own.csv", index=False)
    r = subprocess.run([PY, str(ROOT / "ml/cross_resolver_test.py"), "--model", str(tmp_path / "m.joblib"),
                        "--own", str(tmp_path / "own.csv"), "--train-data", str(synthetic),
                        "--results", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    summ = pd.read_csv(tmp_path / "cross_resolver_xgb.csv")
    assert {"ALL", "Resolver=cloudflare", "Resolver=google"} <= set(summ.group)
    assert (tmp_path / "cross_resolver_xgb_scalecheck.csv").exists()


def test_run_seeds_and_comparison_table(synthetic, tmp_path):
    r = subprocess.run([PY, str(ROOT / "ml/run_seeds.py"), "--data", str(synthetic), "--seeds", "1", "2",
                        "--results", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    s = pd.read_csv(tmp_path / "ml_multiseed_summary.csv")
    assert set(s.model) == {"random_forest", "xgboost"} and (s.n_seeds == 2).all()
    prior = tmp_path / "prior_work.csv"
    subprocess.run([PY, str(ROOT / "ml/make_comparison_table.py"), "--init", "--prior", str(prior)], check=True,
                   capture_output=True)
    r = subprocess.run([PY, str(ROOT / "ml/make_comparison_table.py"), "--prior", str(prior), "--summary",
                        str(tmp_path / "ml_multiseed_summary.csv"), "--out-dir", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    tex = (tmp_path / "comparison_table.tex").read_text()
    assert "\\toprule" in tex and "This work" in tex and "--" in tex


# ----------------------------------------------------------------------------- extra analysis + privacy
def _fake_raw():
    rows = []
    for run in ("r1", "r2"):
        seq = 0
        for dom in ("a.com", "b.com", "c.com"):
            for method, rtt, ips in (("classic_udp", 10, "1.1.1.1;2.2.2.2"), ("doh_cold", 30, "1.1.1.1"),
                                     ("doh_warm", 12, "9.9.9.9")):
                seq += 1
                rows.append(dict(person="A", profile="clean", run_id=run, seq=seq, domain=dom, resolver="x",
                                 method=method, rtt_ms=rtt + (seq % 3), status="ok", rcode="NOERROR",
                                 resolved_ips=ips))
    return pd.DataFrame(rows)


def test_jitter_consistency_and_handshake_overhead():
    from summarize import consistency, handshake_overhead, jitter

    df = _fake_raw()
    j = jitter(df)
    assert set(j.method) == {"classic_udp", "doh_cold", "doh_warm"} and (j.n_runs == 2).all()
    assert (j.jitter_ms_mean >= 0).all()
    c = consistency(df).iloc[0]
    # classic{1,2} vs cold{1} share 1.1.1.1, but warm{9.9.9.9} shares nothing -> never identical / never overlapping
    assert c.n_cases == 6 and c.identical_rate == 0 and c.overlap_rate == 0 and c.rcode_agree_rate == 1
    h = handshake_overhead(df).iloc[0]
    assert h.handshake_overhead_ms == pytest.approx(h.doh_cold_ms - h.doh_warm_ms) and h.handshake_overhead_ms > 15


def test_privacy_audit_sees_plaintext_but_not_tls(tmp_path):
    import dns.message
    from scapy.all import Ether, IP, Raw, TCP, UDP, wrpcap

    from privacy_audit import audit

    q = dns.message.make_query("secret-site.org", "A").to_wire()
    hello = _client_hello("cloudflare-dns.com")
    pk = [Ether() / IP(src="10.0.0.2", dst="8.8.8.8") / UDP(sport=5555, dport=53) / q,                    # plaintext query
          Ether() / IP(src="10.0.0.2", dst="1.1.1.1") / TCP(sport=40000, dport=443, flags="PA") / Raw(hello),  # TLS hello
          Ether() / IP(src="10.0.0.2", dst="1.1.1.1") / TCP(sport=40000, dport=443, flags="PA") / Raw(bytes(range(200)))]  # "encrypted"
    for i, p in enumerate(pk):
        p.time = 1_700_000_000 + i
    wrpcap(str(tmp_path / "p.pcap"), pk)
    table, sni = audit(tmp_path / "p.pcap", ["secret-site.org", "other.net"])
    t = table.set_index("channel")
    assert t.loc["classic_dns", "domains_visible_to_observer"] == 1 and t.loc["classic_dns", "exposure_rate"] == 0.5
    assert t.loc["doh", "domains_visible_to_observer"] == 0
    assert sni == ["cloudflare-dns.com"]
