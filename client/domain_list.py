"""Create results/domains.csv - the fixed list of domains every run will query.

Why: a defensible, reproducible domain list is what lets another researcher
repeat your measurement. Use a *pinned* Tranco list (cite it in the paper).

Usage (from the repo root):
    # recommended: pin a Tranco list. Get the list ID from https://tranco-list.eu
    # ("permanent link" of the list you want), then:
    python client/domain_list.py --list-id ABCDE --count 50

    # offline / fallback: built-in list of 50 popular domains
    python client/domain_list.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]

FALLBACK_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "microsoft.com", "apple.com",
    "amazon.com", "wikipedia.org", "instagram.com", "linkedin.com", "netflix.com",
    "cloudflare.com", "github.com", "whatsapp.com", "yahoo.com", "bing.com",
    "reddit.com", "adobe.com", "zoom.us", "office.com", "live.com",
    "baidu.com", "qq.com", "taobao.com", "tiktok.com", "pinterest.com",
    "twitch.tv", "spotify.com", "dropbox.com", "wordpress.com", "stackoverflow.com",
    "mozilla.org", "nytimes.com", "bbc.com", "cnn.com", "paypal.com",
    "ebay.com", "salesforce.com", "oracle.com", "ibm.com", "intel.com",
    "nvidia.com", "samsung.com", "yandex.ru", "flipkart.com", "hotstar.com",
    "irctc.co.in", "naukri.com", "amazon.in", "vit.ac.in", "x.com",
]
assert len(FALLBACK_DOMAINS) == 50 and len(set(FALLBACK_DOMAINS)) == 50


def fetch_tranco(list_id: str, count: int) -> list[str]:
    """Download the top-`count` domains of a pinned Tranco list (CSV: rank,domain)."""
    url = f"https://tranco-list.eu/download/{list_id}/{count}"
    r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30, follow_redirects=True)
    r.raise_for_status()
    domains = []
    for line in r.text.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2 and parts[1]:
            domains.append(parts[1].strip())
    if len(domains) < count:
        raise RuntimeError(f"Tranco returned only {len(domains)} domains (wanted {count})")
    return domains[:count]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-id", help="pinned Tranco list ID (recommended for the paper)")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "domains.csv"))
    args = ap.parse_args()

    if args.list_id:
        try:
            domains = fetch_tranco(args.list_id, args.count)
            source = f"tranco:{args.list_id}"
        except Exception as exc:  # network blocked, bad id, ...
            print(f"[warn] Tranco download failed ({exc}); using built-in fallback list", file=sys.stderr)
            domains, source = FALLBACK_DOMAINS[: args.count], "builtin-fallback"
    else:
        domains, source = FALLBACK_DOMAINS[: args.count], "builtin-fallback"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "domain", "source"])
        for i, d in enumerate(domains, 1):
            w.writerow([i, d, source])
    print(f"wrote {len(domains)} domains ({source}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
