# -*- coding: utf-8 -*-
"""Utility helpers required by traintest.py.
Minimal re-implementation providing:
 - AverageMeter
 - d_prime
 - calculate_stats
The metrics implementation is simplified to keep training running even if the
original (full) AST repository utilities are absent.
"""
from __future__ import annotations
import math
import numpy as np
from typing import List, Dict

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
except Exception:  # sklearn not installed: provide fallbacks
    roc_auc_score = None
    average_precision_score = None
    confusion_matrix = None
    precision_recall_curve = None


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def d_prime(auc: float) -> float:
    """Convert AUC to d-prime (signal detection theory)."""
    # clamp to avoid inf
    auc = min(max(auc, 1e-6), 1 - 1e-6)
    return math.sqrt(2) * math.erfcinv(2 * (1 - auc))  # erfcinv formulation


def _binary_confusion(y_true: np.ndarray, y_pred: np.ndarray):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return np.array([[tn, fp], [fn, tp]])


def calculate_stats(predictions, targets) -> List[Dict]:
    """Return a list of dicts with required keys.

    Parameters
    ----------
    predictions : torch.Tensor | np.ndarray, shape (N, C)
        After sigmoid (or soft probabilities).
    targets : torch.Tensor | np.ndarray, shape (N, C) one-hot or multi-hot.

    Returns
    -------
    list(dict): minimal fields used downstream.
    """
    if hasattr(predictions, 'detach'):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(targets, 'detach'):
        targets = targets.detach().cpu().numpy()

    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if targets.ndim == 1:
        targets = targets[:, None]

    N, C = predictions.shape
    # derive per-sample class labels
    if C == 1:
        prob_pos = predictions[:, 0]
        y_true = targets[:, 0].astype(int)
        y_pred = (prob_pos >= 0.5).astype(int)
        conf = _binary_confusion(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, prob_pos) if roc_auc_score else 0.0
        except Exception:
            auc = 0.0
        try:
            ap = average_precision_score(y_true, prob_pos) if average_precision_score else 0.0
        except Exception:
            ap = 0.0
        acc = (y_pred == y_true).mean()
        # PR curve (simplified)
        if precision_recall_curve:
            precisions, recalls, _ = precision_recall_curve(y_true, prob_pos)
        else:
            precisions, recalls = np.array([0, 1]), np.array([0, 1])
        primary = dict(acc=acc, AP=ap, auc=auc, conf_matrix=conf,
                        precisions=precisions, recalls=recalls)
        # duplicate second dict for compatibility with code expecting stats[1]
        return [primary, primary.copy()]

    # Multi-class path
    y_true = targets.argmax(axis=1)
    y_pred = predictions.argmax(axis=1)
    acc = (y_true == y_pred).mean()
    # create confusion matrix
    if confusion_matrix:
        conf = confusion_matrix(y_true, y_pred, labels=list(range(C)))
    else:
        conf = np.zeros((C, C), dtype=int)
        for t, p in zip(y_true, y_pred):
            conf[t, p] += 1

    # macro-AUC (one-vs-rest) if sklearn available; else 0
    aucs = []
    if roc_auc_score:
        for c in range(C):
            try:
                aucs.append(roc_auc_score((y_true == c).astype(int), predictions[:, c]))
            except Exception:
                pass
    macro_auc = float(np.mean(aucs)) if aucs else 0.0

    # approximate AP macro
    aps = []
    if average_precision_score:
        for c in range(C):
            try:
                aps.append(average_precision_score((y_true == c).astype(int), predictions[:, c]))
            except Exception:
                pass
    macro_ap = float(np.mean(aps)) if aps else 0.0

    # dummy precision/recall curves (not used heavily downstream for multi-class)
    precisions = np.array([0, 1])
    recalls = np.array([0, 1])

    primary = dict(acc=acc, AP=macro_ap, auc=macro_auc, conf_matrix=conf,
                   precisions=precisions, recalls=recalls)
    return [primary, primary.copy()]
