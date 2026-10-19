"""Create a FAKE dataset with the same columns as CIRA-CIC-DoHBrw-2020 (Layer-2 style).

ONLY for dry-running the ML pipeline before you have downloaded the real dataset, and for
the automated tests. The numbers are random: NEVER report results from this data.

    python tests/make_synthetic_dataset.py --out ml/data_synthetic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "capture"))
from pcap_to_csv import FEATURE_COLUMNS, ID_COLUMNS  # noqa: E402


def make(n_benign=1500, n_malicious=4500, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    def block(n, shift, label):
        base = rng.lognormal(mean=shift, sigma=1.0, size=(n, len(FEATURE_COLUMNS)))
        df = pd.DataFrame(base, columns=FEATURE_COLUMNS)
        df.insert(0, "SourceIP", [f"192.168.{rng.integers(0, 255)}.{rng.integers(1, 254)}" for _ in range(n)])
        df.insert(1, "DestinationIP", "104.16.248.249")
        df.insert(2, "SourcePort", rng.integers(20000, 60000, n))
        df.insert(3, "DestinationPort", 443)
        df.insert(4, "TimeStamp", pd.date_range("2020-01-01", periods=n, freq="s").astype(str))
        df["Label"] = label
        return df

    benign = block(n_benign, 0.0, "Benign")
    mal = block(n_malicious, 0.6, "Malicious")
    # dirt the real files are known to need cleaning for
    mal.loc[mal.index[:5], "FlowSentRate"] = np.inf
    benign.loc[benign.index[:3], "PacketTimeMean"] = np.nan
    return benign, mal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="ml/data_synthetic")
    ap.add_argument("--benign", type=int, default=1500)
    ap.add_argument("--malicious", type=int, default=4500)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    b, m = make(a.benign, a.malicious)
    b.to_csv(out / "l2-benign.csv", index=False)
    m.to_csv(out / "l2-malicious.csv", index=False)
    print(f"wrote FAKE data to {out} ({len(b)} benign, {len(m)} malicious rows) - do not report results from it")


if __name__ == "__main__":
    main()
