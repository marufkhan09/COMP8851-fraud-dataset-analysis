"""Shared evaluator for the COMP8851 fraud-GNN benchmark.

Every model must score predictions through this module. A model that computes
its own metrics is not comparable, because differences in averaging, positive
class, zero-division handling or threshold choice silently change the numbers.

Protocol rules implemented here (master plan v4.4, sections 7.3 and 10.1):

* Checkpoint selection uses validation **AUPRC**.
* Thresholds are swept on **validation** data only, 0.01..0.99 in 0.01 steps,
  maximising Macro-F1, tie-broken by fraud recall and then by closeness to 0.5.
* The chosen threshold is applied unchanged to the test set.
* The positive class is always label ``1`` (fraud / anomaly).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .protocol import THRESHOLD_MAX, THRESHOLD_MIN, THRESHOLD_STEP

#: Metric keys reported for every run, in reporting order.
METRIC_KEYS = (
    "auroc",
    "auprc",
    "macro_f1",
    "fraud_f1",
    "fraud_precision",
    "fraud_recall",
    "gmean",
    "accuracy",
)

POSITIVE_LABEL = 1


def _as_arrays(labels: Sequence[int], probabilities: Sequence[float]):
    y_true = np.asarray(labels).reshape(-1).astype(np.int64)
    y_score = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Label/probability shape mismatch: {y_true.shape} vs {y_score.shape}"
        )
    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty prediction set.")
    unseen = set(np.unique(y_true).tolist()) - {0, 1}
    if unseen:
        raise ValueError(f"Expected binary labels 0/1; found extra values {sorted(unseen)}")
    if not np.isfinite(y_score).all():
        raise ValueError("Probabilities contain NaN or infinity.")
    return y_true, y_score


def threshold_grid() -> np.ndarray:
    """Return the frozen validation threshold grid (0.01..0.99 step 0.01)."""
    count = int(round((THRESHOLD_MAX - THRESHOLD_MIN) / THRESHOLD_STEP)) + 1
    return np.round(np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, count), 10)


def select_threshold(labels: Sequence[int],
                     probabilities: Sequence[float]) -> Tuple[float, float]:
    """Choose a decision threshold on validation data under protocol v4.4.

    Maximises Macro-F1; ties are broken first by higher fraud recall, then by
    closeness to 0.5. Returns ``(threshold, macro_f1_at_threshold)``.

    This function must only ever be given validation labels. Passing test
    labels here would select the threshold from test data, which the protocol
    forbids.
    """
    y_true, y_score = _as_arrays(labels, probabilities)

    best = None  # (macro_f1, fraud_recall, -|t - 0.5|, threshold)
    for threshold in threshold_grid():
        predictions = (y_score >= threshold).astype(np.int64)
        macro = f1_score(y_true, predictions, average="macro", zero_division=0)
        rec = recall_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        key = (float(macro), float(rec), -abs(float(threshold) - 0.5), float(threshold))
        if best is None or key > best:
            best = key
    assert best is not None
    return best[3], best[0]


def metric_bundle(labels: Sequence[int], probabilities: Sequence[float],
                  threshold: float) -> Dict[str, float]:
    """Compute the full required metric set at a fixed threshold.

    ``threshold`` must have been selected on validation data, never here.
    """
    y_true, y_score = _as_arrays(labels, probabilities)
    predictions = (y_score >= float(threshold)).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    # AUROC and AUPRC are undefined when only one class is present. That is a
    # data problem, not a metric to silently fill with 0.
    present = set(np.unique(y_true).tolist())
    single_class = len(present) < 2

    bundle: Dict[str, float] = {
        "auroc": float("nan") if single_class else float(roc_auc_score(y_true, y_score)),
        "auprc": float("nan") if single_class else float(average_precision_score(y_true, y_score)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "fraud_f1": float(f1_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "fraud_precision": float(
            precision_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "fraud_recall": float(
            recall_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "gmean": float(np.sqrt(tpr * tnr)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "positives": int(tp + fn),
        "negatives": int(tn + fp),
        "evaluated_nodes": int(y_true.size),
    }
    if single_class:
        bundle["metric_note"] = (
            "AUROC/AUPRC undefined: evaluation set contains a single class."
        )
    return bundle


def selection_value(metrics: Dict[str, float], metric: str = "auprc") -> float:
    """Extract the checkpoint-selection scalar from a validation metric bundle."""
    if metric not in metrics:
        raise KeyError(f"Selection metric {metric!r} not present in metric bundle.")
    value = float(metrics[metric])
    if not np.isfinite(value):
        # An undefined selection metric must not silently win or lose the
        # comparison; treat it as the worst possible value and let the caller's
        # log record why.
        return float("-inf")
    return value


def evaluate_validation(labels: Sequence[int],
                        probabilities: Sequence[float]) -> Dict[str, float]:
    """Score a validation set: select the threshold, then report all metrics."""
    threshold, _ = select_threshold(labels, probabilities)
    return metric_bundle(labels, probabilities, threshold)


def summarise_seeds(runs: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metric bundles across training seeds.

    Returns ``{metric: {"mean": m, "std": s, "n": k}}`` using the sample
    standard deviation (ddof=1) when more than one seed is present, matching
    the reporting convention in master plan section 10.2.
    """
    summary: Dict[str, Dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = [float(run[key]) for run in runs if key in run and np.isfinite(run[key])]
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "n": int(array.size),
        }
    return summary
