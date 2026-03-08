"""
=============================================================
  Chest X-Ray Competition — Evaluation Script
  Metric: Macro F1-Score
=============================================================
Usage:
  python score.py path/to/predictions.txt

predictions.txt must have one integer (0, 1, or 2) per line,
one per test image, in order (0.jpeg, 1.jpeg, ...).
=============================================================
"""

import sys
import json
import numpy as np
from pathlib import Path

LABELS_PATH = "competition_data/test_labels.npy"  # organizer keeps this

def macro_f1(y_true, y_pred, num_classes=3):
    f1_scores = []
    for cls in range(num_classes):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
        print(f"  Class {cls}: Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
    return np.mean(f1_scores)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python score.py predictions.txt")
        sys.exit(1)

    pred_path   = sys.argv[1]
    labels_path = sys.argv[2] if len(sys.argv) > 2 else LABELS_PATH

    preds  = np.array([int(line.strip()) for line in open(pred_path)])
    labels = np.load(labels_path)

    if len(preds) != len(labels):
        print(f"ERROR: Got {len(preds)} predictions but expected {len(labels)}")
        sys.exit(1)

    print(f"\nEvaluating {pred_path}...")
    score = macro_f1(labels, preds)
    print(f"\n  ✅ Macro F1-Score: {score:.4f}")

    result = {"macro_f1": round(float(score), 4)}
    with open("score.json", "w") as f:
        json.dump(result, f)
    print(f"  Saved to score.json")
