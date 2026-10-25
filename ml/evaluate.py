"""One evaluation function used by EVERY model, so metrics are computed identically
(a fair comparison). Reports accuracy, precision, recall, F1, ROC-AUC, the confusion
matrix and the false-positive rate (benign flows wrongly flagged as malicious), plus three
metrics that stay honest under the 12.6 : 1 class imbalance: balanced accuracy, Matthews
correlation coefficient (MCC) and PR-AUC (average precision).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)


def evaluate_model(model, X, y, name: str = "model") -> dict:
    y = np.asarray(y)
    pred = model.predict(X)
    try:
        score = model.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, score))
        pr_auc = float(average_precision_score(y, score))
    except Exception:  # only one class present, or no predict_proba
        auc = pr_auc = float("nan")
    both = len(np.unique(y)) > 1
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "model": name, "n_test": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred)) if both else float("nan"),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else float("nan"),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def print_report(m: dict) -> None:
    print(f"\n=== {m['model']} (n_test={m['n_test']}) ===")
    print(f"accuracy {m['accuracy']:.4f} | precision {m['precision']:.4f} | recall {m['recall']:.4f} | "
          f"F1 {m['f1']:.4f} | ROC-AUC {m['roc_auc']:.4f} | FPR {m['false_positive_rate']:.4f}")
    print(f"balanced acc {m['balanced_accuracy']:.4f} | MCC {m['mcc']:.4f} | PR-AUC {m['pr_auc']:.4f}")
    print("confusion matrix  (rows = truth, cols = predicted)")
    print(f"              pred benign  pred malicious")
    print(f"true benign   {m['tn']:>11}  {m['fp']:>14}")
    print(f"true malicious{m['fn']:>11}  {m['tp']:>14}")


def save_metrics(m: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(m, indent=2))
