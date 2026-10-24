"""Build the "ours vs prior work" table (Day 20) as CSV and as a LaTeX table for Overleaf.

You type the prior papers' numbers into results/prior_work.csv BY HAND, copying them from the
papers. This script never invents numbers: blank cells appear as "--" (= not reported).

Usage (from the repo root):
    python ml/make_comparison_table.py --init        # creates results/prior_work.csv template
    # ... edit results/prior_work.csv ...
    python ml/make_comparison_table.py               # needs results/ml_multiseed_summary.csv
Outputs: results/comparison_table.csv and results/comparison_table.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
COLS = ["Paper", "Method", "Dataset", "Accuracy", "Precision", "Recall", "F1"]
NICE = {"random_forest": "Random Forest", "xgboost": "XGBoost"}


def tex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return "--"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", action="store_true", help="write an empty prior_work.csv template and exit")
    ap.add_argument("--prior", default=str(REPO_ROOT / "results" / "prior_work.csv"))
    ap.add_argument("--summary", default=str(REPO_ROOT / "results" / "ml_multiseed_summary.csv"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    a = ap.parse_args()

    if a.init:
        Path(a.prior).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([[f"(paper {i}: author, venue, year)", "", "CIRA-CIC-DoHBrw-2020", "", "", "", ""] for i in (1, 2, 3)],
                     columns=COLS).to_csv(a.prior, index=False)
        print(f"template written: {a.prior}  - fill it from the papers (leave unknown cells blank)")
        return 0

    prior = pd.read_csv(a.prior, dtype=str)
    missing = [c for c in COLS if c not in prior.columns]
    if missing:
        raise SystemExit(f"{a.prior} is missing columns {missing}")
    summ = pd.read_csv(a.summary)
    rows = [prior[COLS].copy()]
    ours = []
    for _, r in summ.iterrows():
        def pm(metric):
            return f"{r[metric + '_mean']:.4f} ± {r[metric + '_std']:.4f}"
        ours.append(["This work", NICE.get(r["model"], r["model"]) + f" (n={int(r['n_seeds'])} seeds)",
                     "CIRA-CIC-DoHBrw-2020", pm("accuracy"), pm("precision"), pm("recall"), pm("f1")])
    table = pd.concat([rows[0], pd.DataFrame(ours, columns=COLS)], ignore_index=True)
    table = table.fillna("")
    out = Path(a.out_dir)
    table.to_csv(out / "comparison_table.csv", index=False)

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Comparison with prior work on DoH traffic classification (-- = not reported).}",
             r"\label{tab:comparison}", r"\begin{tabular}{lllcccc}", r"\toprule",
             " & ".join(COLS) + r" \\", r"\midrule"]
    for _, r in table.iterrows():
        cells = [tex_escape(fmt(r[c])).replace("±", r"$\pm$") for c in COLS]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "comparison_table.tex").write_text("\n".join(lines) + "\n")
    print(table.to_string(index=False))
    print(f"\nwrote {out / 'comparison_table.csv'} and {out / 'comparison_table.tex'} "
          "(add \\usepackage{booktabs} to the preamble)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
