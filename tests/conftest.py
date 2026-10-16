"""Make the project's folders importable from the tests (run `pytest` from the repo root)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("client", "capture", "ml", "tests"):
    sys.path.insert(0, str(ROOT / sub))
