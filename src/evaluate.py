"""Evaluation helpers for the departure-delay classifier.

Delays are the minority class (about one flight in five), so accuracy is
misleading: predicting "on time" for every flight is about 80% accurate and useless.
The headline metric is PR-AUC (average precision), with precision and recall
reported at a decision threshold chosen on the validation months.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def best_f1_threshold(y_true, scores):
    """Threshold that maximizes F1 (the balance of precision and recall)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = np.nanargmax(f1[:-1])   # the last point has no threshold
    return float(thresholds[best])


def evaluate(y_true, scores, threshold):
    """Ranking metrics plus precision/recall/F1 at one threshold."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    return {
        "PR-AUC": average_precision_score(y_true, scores),
        "ROC-AUC": roc_auc_score(y_true, scores) if np.unique(scores).size > 1 else 0.5,
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "share flagged": pred.mean(),
    }


def comparison_table(rows):
    """rows: dict of model name -> evaluate(...) output."""
    return pd.DataFrame(rows).T
