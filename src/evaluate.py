"""Model evaluation utilities for ManSanket (मनसंकेत).

Computes comprehensive metrics:
- Overall Accuracy
- Macro & Weighted Precision, Recall, F1-score
- Per-class precision, recall, f1, support
- Confusion matrix (counts and normalized)
- Classification report dictionary
"""
import numpy as np
from typing import Dict, Any, List, Union, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from src.config import EMOTION_LABELS


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute detailed evaluation metrics for multi-class classification.
    
    Args:
        y_true: Ground truth integer labels.
        y_pred: Predicted integer labels.
        target_names: List of class names. Defaults to EMOTION_LABELS.
        
    Returns:
        Dictionary containing overall, per-class, and matrix metrics.
    """
    if target_names is None:
        target_names = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]

    acc = float(accuracy_score(y_true, y_pred))
    
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(len(target_names)))
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
    cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))), normalize="true")

    per_class_metrics = {}
    for i, name in enumerate(target_names):
        per_class_metrics[name] = {
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1_score": round(float(per_class_f1[i]), 4),
            "support": int(per_class_supp[i])
        }

    labels_list = list(range(len(target_names)))
    report_dict = classification_report(
        y_true, y_pred, labels=labels_list, target_names=target_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
        "classification_report": report_dict,
        "labels": target_names
    }
