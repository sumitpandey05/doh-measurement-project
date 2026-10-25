# DoH Measurement & Classification

**An empirical study of DNS-over-HTTPS (DoH): how much does encryption cost in latency, how does it behave on degraded networks, what does it hide from a passive observer, and can machine learning still tell benign DoH traffic from malicious DoH traffic using only encrypted flow metadata?**

Course project — *BCSE308L/P Computer Networks* · conference-paper track · team of two, working remotely.

---

## Table of contents

1. [Overview](#1-overview)
2. [Research objectives](#2-research-objectives)
3. [How it works](#3-how-it-works)
4. [Repository structure](#4-repository-structure)
5. [Requirements](#5-requirements)
6. [Installation](#6-installation)
7. [Quick start](#7-quick-start)
8. [Usage guide](#8-usage-guide)
9. [Outputs](#9-outputs)
10. [Working as a two-person team](#10-working-as-a-two-person-team)
11. [Testing](#11-testing)
12. [Methodology notes and known limitations](#12-methodology-notes-and-known-limitations)
13. [Project status and verification](#13-project-status-and-verification)
14. [Troubleshooting](#14-troubleshooting)
15. [References](#15-references)
16. [Authors and license](#16-authors-and-license)

---

## 1. Overview

DNS translates names such as `example.org` into IP addresses, traditionally in **plaintext** over UDP port 53, so anyone on the path can see which sites you look up. **DNS-over-HTTPS** (RFC 8484) wraps the same question in an encrypted HTTPS connection. That fixes the privacy problem but raises two questions this project answers with data:

* **Performance:** what does encryption cost, on a clean link *and* on a bad one (high latency, jitter, packet loss)?
* **Security visibility:** because DoH looks like ordinary HTTPS, network defenders can no longer read DNS. Can they still detect malicious DoH (e.g. tunnelling) from packet sizes and timing alone, and does such a classifier transfer to traffic it was never trained on?

The repository contains one reproducible pipeline for both halves:

```
MEASURE  ── plain DNS vs DoH, several resolvers, repeated, optionally on emulated bad networks
WATCH    ── record the packets, extract flow features, audit what an observer can read
CLASSIFY ── Random Forest / XGBoost on a public benchmark dataset, then test on your own captures
```

## 2. Research objectives

1. Build a controlled **dual-path measurement client** that sends identical queries over classic UDP DNS and over DoH to multiple public resolvers (Cloudflare, Google, Quad9).
2. Quantify **latency, jitter, answer consistency, handshake overhead and reliability** on clean and artificially degraded networks (`tc netem`).
3. Extract **payload-free flow features** from encrypted DoH traffic and train **Random Forest and XGBoost** classifiers (benign vs malicious) on the CIRA-CIC-DoHBrw-2020 benchmark.
4. Evaluate **generalisation** of the trained models on independently captured benign DoH traffic, per resolver.
5. Measure the **privacy gain** — what fraction of queried domains a passive observer can read in classic DNS vs DoH.
6. Benchmark the results against prior work and keep everything **reproducible across two machines**.

## 3. How it works

```
 domain_list.py ─► results/domains.csv ─┐
 resolvers.yaml ────────────────────────┤
                                        ▼
        ┌──────────── netem/run_with_condition.sh  (optional: degrade the network)
        │             capture/capture_runner.py    (optional: record packets)
        ▼
   client/run_batch.py ── dual_query ── classic_udp │ doh_cold │ doh_warm
        │                                   │
        ▼                                   ▼
 results/raw_*.csv                  capture/pcaps/*.pcap (+ .meta.json)
        │                                   │
        │                  ┌────────────────┼──────────────────────┐
        ▼                  ▼                ▼                      ▼
 summarize.py        pcap_to_csv.py    compare_rtt.py       privacy_audit.py
 merge_results.py    flows → own_flows (is my timer honest?)  (what can an observer read?)
 notebooks/                │
                           ▼
 ml/data (CIRA-CIC) ─► load_dataset ─► preprocess ─► train_rf / train_xgb / run_seeds
                                                          │
                              ml/cross_resolver_test.py ◄─┘  (model vs your own flows)
                                                          ▼
                                          make_comparison_table.py ─► paper tables/figures
```

**The three measured methods**

| Method | What it is | What its latency includes |
|---|---|---|
| `classic_udp` | plaintext DNS, UDP/53 | one network round trip |
| `doh_cold` | DoH over a **new** TCP+TLS connection per query | handshake + request |
| `doh_warm` | DoH over **one reused** connection (browser-like) | request only |

`doh_cold − doh_warm` is the **handshake overhead**; `doh_warm − classic_udp` is the steady-state price of encryption.

## 4. Repository structure

```
doh-measurement-project/
├── client/                    measurement instrument
│   ├── common.py              shared query builder, result format, CSV schema
│   ├── domain_list.py         create results/domains.csv (pinned Tranco list or built-in 50)
│   ├── classic_dns.py         one plaintext UDP query, timed
│   ├── doh_client.py          one DoH query (RFC 8484), cold or warm
│   ├── dual_query.py          the three methods for one (domain, resolver)
│   ├── resolvers.yaml         Cloudflare, Google, Quad9
│   ├── run_batch.py           the experiment runner
│   ├── summarize.py           latency / jitter / consistency / handshake statistics
│   └── merge_results.py       merge both teammates' CSVs
├── capture/                   packets and features
│   ├── capture_runner.py      run any command under a tshark capture
│   ├── pcap_to_csv.py         flows (dataset-compatible features) | packets | UDP RTT
│   ├── compare_rtt.py         client timer vs wire timing
│   └── privacy_audit.py       what a passive observer can read
├── netem/                     network-condition emulation (Linux / WSL2)
│   ├── apply_condition.sh · clear_condition.sh · run_with_condition.sh
│   └── apply_condition_notes.md   equivalents for macOS / Windows
├── ml/                        classification
│   ├── load_dataset.py · preprocess.py · models.py · evaluate.py
│   ├── train_rf.py · train_xgb.py · run_seeds.py
│   └── cross_resolver_test.py · make_comparison_table.py
├── notebooks/                 week1_analysis · rtt_crosscheck · netem_analysis
│                              dataset_explore · model_comparison
├── results/                   small outputs (CSV / JSON / PNG) — committed
├── tests/                     offline mock servers, synthetic data, 20 automated tests
└── requirements.txt · .gitignore · README.md
```

Created at run time (not in git): `capture/pcaps/` (raw captures), `ml/data/` (dataset), `ml/artifacts/*.joblib` (models).

## 5. Requirements

| | |
|---|---|
| **OS** | Ubuntu 22.04+ (native) or **Windows 11 / 10 with WSL2 + Ubuntu** (recommended on Windows). macOS works except for netem. |
| **Python** | 3.9 or newer (developed and tested on 3.12) |
| **System packages** | `tshark`, `iproute2` (provides `tc`), `git` |
| **Python packages** | see `requirements.txt` — `dnspython`, `httpx[http2]`, `PyYAML`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`, `scapy`, `matplotlib`, `seaborn`, `jupyter`, `pytest` |
| **Hardware** | any laptop; ~1–2 GB free disk for captures; no GPU |
| **Network** | outbound UDP/53 and TCP/443 to public resolvers (campus networks sometimes block these) |
| **Data** | CIRA-CIC-DoHBrw-2020 from the Canadian Institute for Cybersecurity dataset page (manual download) |

> `tc netem` needs a Linux kernel with the netem module. Inside WSL2 run `wsl --update` if you see *"Specified qdisc kind is unknown"*. On native Windows use Clumsy; on macOS use Network Link Conditioner — see `netem/apply_condition_notes.md`, and **disclose any non-netem tool in your paper**.

## 6. Installation

```bash
# 1. system packages (Ubuntu / WSL2)
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git tshark iproute2
#    answer YES to "Should non-superusers be able to capture packets?"
sudo usermod -aG wireshark "$USER"        # then close and reopen the terminal
                                          # (WSL: run `wsl --shutdown` in PowerShell first)

# 2. project environment
cd ~/cn-project/doh-measurement-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. verify
pytest -q                                 # expected: 22 passed
tshark --version && tc -V
```

**Windows (WSL2):** `wsl --install -d Ubuntu`, restart, open the *Ubuntu* app and run the commands above *inside* it. Do not mix native Windows Python with WSL tools.

## 7. Quick start

Everything is run **from the repository root** with the virtual environment active.

```bash
git pull                                                     # always first
python client/domain_list.py                                 # → results/domains.csv
python client/dual_query.py google.com                       # sanity check: 3 methods, 1 domain
python client/run_batch.py --person A --profile clean --repeats 3 --udp-retries 2
python client/summarize.py                                   # latency, jitter, consistency, handshake cost
```

No internet, or want to dry-run first? Start the built-in fake servers:

```bash
python tests/mock_servers.py --delay-ms 10                   # terminal 1
python client/run_batch.py --person A --limit 5 --resolvers tests/resolvers_local.yaml   # terminal 2
```

## 8. Usage guide

### 8.1 Measure latency — `client/`

```bash
python client/domain_list.py --list-id <TRANCO_ID> --count 50    # pinned list (cite it); omit for built-in 50
python client/run_batch.py --person A --profile clean --repeats 3 --udp-retries 2
python client/run_batch.py --person A --limit 5 --only cloudflare   # quick smoke test
```

| Option | Meaning |
|---|---|
| `--person A\|B` | who is running it (required) |
| `--profile` | label of the network condition (`clean`, `moderate`, `high_jitter`, `lossy`) — **must match what you applied** |
| `--repeats N` | independent repetitions (one CSV each) |
| `--udp-retries N` | extra UDP attempts after a timeout. **Use the same value in every run** (recommended: 2) |
| `--only NAME…` / `--limit N` | restrict resolvers / domains |
| `--http1` | force DoH over HTTP/1.1 (default is HTTP/2; Quad9 rejects HTTP/1.1 with 505 since Dec 2025) |
| `--udp-timeout` | seconds to wait per UDP attempt before a retry (default 2) |
| `--seed` | seed for shuffling domain and method order (reproducible) |

One repetition = 50 domains × 3 resolvers × 3 methods = **450 queries** (about 2–4 min on a clean link). Each run writes `results/raw_<person>_<profile>_r<k>_<time>.csv` plus a `.meta.json` (arguments, library versions, platform).

### 8.2 Degrade the network — `netem/`

| Profile | Added to every outgoing packet |
|---|---|
| `clean` | nothing |
| `moderate` | 50 ± 10 ms delay, 0.5 % loss |
| `high_jitter` | 80 ± 40 ms delay (normal distribution) |
| `lossy` | 30 ± 5 ms delay, 3 % loss |

```bash
bash netem/apply_condition.sh moderate      # as a normal user — the script calls sudo itself
ping -c 5 1.1.1.1                           # should now be ~50 ms slower
bash netem/clear_condition.sh

# the safe way for real experiments (condition is removed even on crash / Ctrl-C):
bash netem/run_with_condition.sh lossy -- \
    python client/run_batch.py --person A --profile lossy --repeats 3 --udp-retries 2
```

netem shapes **outgoing packets only**, so "50 ms" adds about 50 ms to the round trip.

### 8.3 Watch packets — `capture/`

```bash
# record the packets produced by any command
python capture/capture_runner.py --label own_A -- \
    python client/run_batch.py --person A --profile clean --repeats 2 --udp-retries 2

# per-flow features, same column names as the CIRA-CIC dataset
python capture/pcap_to_csv.py flows capture/pcaps/<file>.pcap \
    --meta capture/pcaps/<file>.meta.json --out results/own_flows_A.csv

# is the client's stopwatch honest? (wire RTT vs client RTT for classic DNS)
python capture/pcap_to_csv.py udp-rtt capture/pcaps/<file>.pcap --out results/own_udp_rtt.csv
python capture/compare_rtt.py --client "results/raw_A_clean_r*.csv" --wire results/own_udp_rtt.csv

# what can a passive observer read?
python capture/privacy_audit.py capture/pcaps/<file>.pcap \
    --client "results/raw_A_clean_r*.csv" --out results/privacy_audit_A.csv
```

The capture filter is `udp port 53 or tcp port 443` — **DoH travels on 443, not 53**.

### 8.4 Classify — `ml/`

1. Download **CIRA-CIC-DoHBrw-2020** and place the **Layer-2** CSVs (`Label` = `Benign` / `Malicious`) in `ml/data/`.
2. Verify and train:

```bash
python ml/load_dataset.py --data ml/data            # class balance, features, dropped rows
python ml/preprocess.py   --data ml/data --seed 42  # deterministic split + manifest
python ml/train_rf.py     --data ml/data --seed 42
python ml/train_xgb.py    --data ml/data --seed 42
python ml/run_seeds.py    --data ml/data --seeds 42 43 44 45 46     # mean ± std over seeds

python ml/cross_resolver_test.py --model ml/artifacts/rf_seed42.joblib \
    --own results/own_flows_A.csv results/own_flows_B.csv --train-data ml/data

python ml/generalization.py --data ml/data --seed 42    # dedup / time-ordered / unseen-tool tests

python ml/make_comparison_table.py --init           # then fill results/prior_work.csv by hand
python ml/make_comparison_table.py                  # → CSV + LaTeX table
```

No dataset yet? `python tests/make_synthetic_dataset.py --out ml/data_synthetic` creates **fake** data to dry-run the pipeline. **Never report results from it.**

### 8.5 Analyse — `notebooks/` and `analysis/`

`python analysis/paper_figures.py` rebuilds every measurement figure and table used in the paper (medians, bootstrap CIs, paired differences) into `results/paper/`.

Open in Jupyter or VS Code; each notebook locates the repository root itself.

| Notebook | Purpose |
|---|---|
| `week1_analysis.ipynb` | first look: success rates, RTT boxplots, teammate comparison, privacy table |
| `rtt_crosscheck.ipynb` | client timing vs wire timing |
| `netem_analysis.ipynb` | latency vs network condition, handshake cost, jitter, consistency, reliability |
| `dataset_explore.ipynb` | class balance, discriminative features, correlations |
| `model_comparison.ipynb` | RF vs XGBoost metrics, confusion matrices, feature importance |

## 9. Outputs

**Raw measurement schema** (`results/raw_*.csv`)

| Column | Meaning |
|---|---|
| `person`, `profile`, `run_id`, `seq` | who, network condition, repetition id, query number |
| `domain`, `resolver`, `method`, `qtype`, `server` | what was asked, of whom, how |
| `send_ts`, `recv_ts`, `rtt_ms` | epoch timestamps and latency (ms) |
| `status`, `rcode`, `answer_count`, `resolved_ips` | outcome and answer |
| `attempts`, `error` | UDP tries used; short error text |

**Everything that goes into the paper**

| Paper element | File |
|---|---|
| Latency vs network condition (figure) | `results/fig_netem_rtt.png` |
| Latency boxplot | `results/fig_week1_rtt_boxplot.png` |
| Latency / jitter / consistency / handshake tables | `results/netem_summary.csv`, notebook tables |
| Privacy table | `results/privacy_audit_*.csv` |
| Classifier metrics | `results/ml_rf_seed42.json`, `ml_xgb_seed42.json`, `ml_multiseed_summary.csv` |
| Confusion matrices | `results/fig_confusion_matrices.png` |
| Generalisation (false-positive rate per resolver) | `results/cross_resolver_*.csv` |
| Comparison with prior work | `results/comparison_table.csv` / `.tex` (add `\usepackage{booktabs}`) |
| Reproducibility evidence | `results/raw_*.meta.json`, `ml/artifacts/split_seed42_manifest.json` |

## 10. Working as a two-person team

* **Identical commands, different `--person`.** Nothing is hard-coded; each of you runs the same command on your own network.
* **Git carries small files only:** code, `results/*.csv|json|png`, notebooks, the split manifest. Raw `.pcap` files, the dataset and trained models stay local (see `.gitignore`).
* **Daily rhythm:** `git pull` → work → `git add . && git commit -m "…" && git push`.
* **Merge results:** `python client/merge_results.py` → `results/merged_raw.csv` (rejects files with different columns and tells you which).
* **Prove identical ML splits:** both run `python ml/preprocess.py --seed 42`, then `sha256sum ml/artifacts/split_seed42_manifest.json` — the hashes must match. (Library versions are stored separately in `split_seed42_versions.json`; compare those if the hashes differ.)
* **Branches:** short-lived feature branches (`personA/…`, `personB/…`) merged through pull requests; pair-program the shared core.
* **Prose** (the paper) lives in Overleaf, not git; daily notes in a shared sheet.

## 11. Testing

```bash
pytest -q          # 22 tests, about 15 s, no internet needed
```

The tests run against local mock DNS/DoH servers (plain HTTP and real TLS) and synthetic data. They check, among others: warm sessions reuse one connection while cold sessions open a new one each time; failures are recorded as rows, not crashes; flow features match a hand calculation (including the Ethernet-padding trap); the privacy audit sees plaintext but not TLS; the train/test split and manifests are byte-identical across runs.

A green run proves the **code** works. It says nothing about real resolvers or the real dataset.

## 12. Methodology notes and known limitations

State these in the paper.

1. **Cached answers.** The same popular domain is queried by all three methods, so public resolvers answer from cache. This isolates *transport* overhead (UDP vs TLS), not recursive-resolution time. Method order is shuffled per query.
2. **Emulated degradation.** netem is egress-only and an approximation of real lossy links. Any teammate using Clumsy / Network Link Conditioner must say so.
3. **UDP has no retransmission,** TCP does. Report the `--udp-retries` value used; keep it identical across profiles.
4. **Re-implemented features.** `pcap_to_csv.py` approximates the dataset's CICFlowMeter features (packet length = IP bytes; PacketTime = gaps between packets; ResponseTime = client data → server data gap; population variance; skew = (mean − median/mode)/std). `cross_resolver_test.py` prints a scale check to catch mismatches.
5. **Flow composition differs.** A cold DoH query is one short flow; warm queries share one long flow; the dataset contains browser traffic.
6. **"Cross-resolver" means transfer, not retraining.** Models are trained on the public dataset and tested on your own **benign** captures per resolver, so the metric is the false-positive rate (no recall).
7. **Malicious traffic comes only from the public dataset;** this project does not generate attacks.
8. **Offline pipeline,** not a real-time detector. **Two vantage points** (one network per teammate) — do not generalise to "the internet".

## 13. Project status and verification

| Verified (automated, in a Linux sandbox) | To be confirmed on your machines |
|---|---|
| All code paths and 22 tests (mock servers, real TLS, real tshark capture on loopback, flow features, privacy audit, deterministic split, both models, multi-seed, cross-resolver, comparison table) | Real Cloudflare / Google / Quad9 measurements |
| Client-vs-wire RTT gap of about 0.6 ms on a real capture | `tc netem` actually shaping traffic (`ping` check, §8.2) |
| All five notebooks execute end-to-end | The real dataset's exact file names and labels (`load_dataset.py` reports mismatches) |
| netem scripts always clean up (stub `tc`; success, failure and Ctrl-C) | Tranco download (`--list-id`); WSL2 specifics |

## 14. Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `tshark did not start capturing` | no capture permission → `sudo dpkg-reconfigure wireshark-common` (Yes), `sudo usermod -aG wireshark $USER`, reopen terminal |
| Empty pcap | wrong interface or filter → `--interface eth0`; remember DoH is port **443** |
| Every UDP query times out | network blocks UDP/53 → try a phone hotspot; note it as a finding |
| `http_400` on DoH | wrong endpoint → use `/dns-query` (not Google's JSON `/resolve`) |
| `http_505` on Quad9 DoH | HTTP/1.1 is no longer accepted by Quad9 → use the default HTTP/2 (do not pass `--http1`); `pip install 'httpx[http2]'` |
| `git pull`: *SSL certificate verification failed: certificate signer not trusted* | WSL does not trust the certificate shown by your network (antivirus / campus proxy that inspects HTTPS) → `sudo apt install --reinstall ca-certificates && sudo update-ca-certificates`; if it still fails, export that proxy's root certificate from Windows, copy it to `/usr/local/share/ca-certificates/<name>.crt` and run `sudo update-ca-certificates` again. Never set `http.sslVerify false`. |
| `Specified qdisc kind is unknown` | kernel lacks netem → WSL: `wsl --update` then `wsl --shutdown` |
| Everything is slow after an experiment | a netem rule was left behind → `bash netem/clear_condition.sh` |
| `domains.csv not found` | run `python client/domain_list.py` |
| Merge says "columns differ" | a teammate edited the client without `git pull` |
| `unexpected label values ['DoH','NonDoH']` | you loaded Layer-1 files → use the Layer-2 files (Benign / Malicious) |
| Accuracy exactly 1.0 | data leakage → check the feature list from `load_dataset.py` contains no identifier or label-derived column |
| Many features flagged by the scale check | suspect feature extraction before claiming poor generalisation |
| Different split hashes between teammates | different data files or library versions → compare `split_seed42_versions.json` |

## 15. References

* P. McManus, P. Hoffman — **RFC 8484: DNS Queries over HTTPS (DoH)**, IETF, 2018.
* V. Le Pochat, T. Van Goethem, S. Tajalizadehkhoob, M. Korczyński, W. Joosen — *Tranco: A Research-Oriented Top Sites Ranking Hardened Against Manipulation*, NDSS 2019. <https://tranco-list.eu>
* **CIRA-CIC-DoHBrw-2020**, Canadian Institute for Cybersecurity (University of New Brunswick) with CIRA — cite it exactly as requested on the dataset page.
* Related work for the paper's background section is listed in the project's idea document (`CN_Project_Ideas.md`).

## 16. Authors and license

* **Authors:** `Mahi Bagla` · `Sumit Pandey` — `CSE specialisation (information security)`
* **Course:** BCSE308L / BCSE308P Computer Networks
* **License:** `MIT`
* **Ethics:** this project only observes your own traffic and uses a published dataset for malicious samples. Do not run DoH-tunnelling tools or any attack on networks you do not own.
