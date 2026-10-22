"""Evaluation metrics and tracking utilities for TrustLens.

Includes:
- AUROC, AUPR (Average Precision)
- False Positive Rate at 95% True Positive Rate (FPR@95TPR)
- Multi-metric aggregation
- Formatted metrics table generation
- MetricsTracker for logging and DataFrame export across corruption experiments
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute overall classification accuracy.

    Args:
        predictions: Predicted class indices of shape (N,).
        labels: Ground truth class indices of shape (N,).

    Returns:
        Scalar accuracy value in [0.0, 1.0].
    """
    preds = np.asarray(predictions)
    lbls = np.asarray(labels)
    if len(preds) == 0:
        return 0.0
    return float(np.mean(preds == lbls))


def _parse_scores_and_labels(
    arg1: Union[np.ndarray, List],
    arg2: Union[np.ndarray, List],
) -> Tuple[np.ndarray, np.ndarray]:
    """Disambiguate continuous scores and binary labels regardless of argument order."""
    a = np.asarray(arg1)
    b = np.asarray(arg2)

    a_is_binary = len(np.unique(a)) <= 2 and np.all(np.isin(a, [0, 1, 0.0, 1.0, True, False]))
    b_is_binary = len(np.unique(b)) <= 2 and np.all(np.isin(b, [0, 1, 0.0, 1.0, True, False]))

    if a_is_binary and not b_is_binary:
        labels, scores = a, b
    elif b_is_binary and not a_is_binary:
        scores, labels = a, b
    else:
        scores, labels = a, b

    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def compute_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute the Area Under the Receiver Operating Characteristic (AUROC) curve.

    Args:
        scores: Continuous anomaly or failure scores of shape (N,), or labels if swapped.
        labels: Binary ground truth array of shape (N,) (1 for positive class, 0 for negative).

    Returns:
        AUROC score in [0.0, 1.0], or NaN if only one class is present in labels.
    """
    s, y = _parse_scores_and_labels(scores, labels)

    if len(np.unique(y)) < 2:
        logger.warning("compute_auroc encountered single-class labels; returning NaN.")
        return float("nan")

    return float(roc_auc_score(y, s))


def compute_aupr(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute the Area Under the Precision-Recall curve (AUPR / Average Precision).

    Args:
        scores: Continuous anomaly or failure scores of shape (N,), or labels if swapped.
        labels: Binary ground truth array of shape (N,) (1 for positive class, 0 for negative).

    Returns:
        AUPR score in [0.0, 1.0], or NaN if only one class is present in labels.
    """
    s, y = _parse_scores_and_labels(scores, labels)

    if len(np.unique(y)) < 2:
        logger.warning("compute_aupr encountered single-class labels; returning NaN.")
        return float("nan")

    return float(average_precision_score(y, s))


def compute_fpr_at_tpr(
    scores: np.ndarray,
    labels: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """Calculate the False Positive Rate (FPR) when True Positive Rate (TPR) is at least target_tpr.

    Standard benchmark metric for Out-of-Distribution (OOD) and failure detection.

    Args:
        scores: Continuous failure scores of shape (N,), or labels if swapped.
        labels: Binary ground truth array of shape (N,) (1=positive/failure, 0=negative/correct).
        target_tpr: Target recall / true positive rate (default 0.95).

    Returns:
        FPR value in [0.0, 1.0].
    """
    s, y = _parse_scores_and_labels(scores, labels)

    if len(np.unique(y)) < 2:
        logger.warning("compute_fpr_at_tpr encountered single-class labels; returning NaN.")
        return float("nan")

    fpr, tpr, _ = roc_curve(y, s)
    indices = np.where(tpr >= target_tpr)[0]

    if len(indices) == 0:
        return 1.0
    return float(fpr[indices[0]])


def compute_all_metrics(
    failure_scores: np.ndarray,
    is_correct: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute a comprehensive suite of evaluation metrics for failure detection.

    Treats model misclassifications (is_correct == 0) as positive failure cases.

    Args:
        failure_scores: Predicted failure probability or risk score per sample.
        is_correct: Binary correctness flag per sample (1=correct, 0=misclassification).
        predictions: Optional class predictions.
        labels: Optional ground truth class targets.

    Returns:
        Dictionary mapping metric names to scalar float values:
            - 'auroc': Area under ROC curve for failure detection
            - 'aupr': Area under Precision-Recall curve
            - 'fpr_at_95tpr': FPR at 95% TPR
            - 'failure_rate': Proportion of incorrect predictions
            - 'accuracy': (If predictions and labels provided) Top-1 accuracy
            - 'macro_f1': (If predictions and labels provided) Macro-averaged F1
    """
    scores = np.asarray(failure_scores, dtype=np.float64)
    # Binary failure target: 1 = failure, 0 = correct
    y_failure = (np.asarray(is_correct) == 0).astype(int)

    metrics: Dict[str, float] = {
        "auroc": compute_auroc(scores, y_failure),
        "aupr": compute_aupr(scores, y_failure),
        "fpr_at_95tpr": compute_fpr_at_tpr(scores, y_failure, target_tpr=0.95),
        "failure_rate": float(np.mean(y_failure)) if len(y_failure) > 0 else 0.0,
    }

    if predictions is not None and labels is not None:
        preds = np.asarray(predictions)
        lbls = np.asarray(labels)
        metrics["accuracy"] = compute_accuracy(preds, lbls)
        metrics["macro_f1"] = float(
            f1_score(lbls, preds, average="macro", zero_division=0)
        )
        metrics["error_rate"] = 1.0 - metrics["accuracy"]

    return metrics


def format_metrics_table(
    metrics_dict: Dict[str, Any],
    title: str = "Evaluation Metrics",
) -> str:
    """Format a metrics dictionary into an aligned ASCII markdown table.

    Args:
        metrics_dict: Dictionary mapping metric names to values.
        title: Header title for the table.

    Returns:
        Formatted multi-line markdown table string.
    """
    lines = [
        f"### {title}",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
    ]

    for key, val in metrics_dict.items():
        if key.isupper() or any(c in key for c in ["@", "_"]):
            name = key.replace("_", " ")
        else:
            name = key.replace("_", " ").title()
        if isinstance(val, float):
            if np.isnan(val):
                formatted_val = "N/A"
            elif 0.0 <= abs(val) < 1.0:
                formatted_val = f"{val:.4f}"
            else:
                formatted_val = f"{val:.2f}"
        else:
            formatted_val = str(val)
        lines.append(f"| {name} | {formatted_val} |")

    return "\n".join(lines)


class MetricsTracker:
    """Accumulates and exports experimental evaluation results across configurations.

    Enables aggregation over corruptions, severities, detector baselines,
    and supports exporting to Pandas DataFrames and CSV files.
    """

    def __init__(self) -> None:
        """Initialize empty MetricsTracker."""
        self._records: List[Dict[str, Any]] = []

    @property
    def records(self) -> List[Dict[str, Any]]:
        """List of recorded experiment dictionaries."""
        return self._records

    def update(self, metrics: Dict[str, Any]) -> None:
        """Update tracker with a dictionary of metrics."""
        self._records.append(dict(metrics))

    def get_averages(self) -> Dict[str, float]:
        """Compute average value for all numeric metrics recorded."""
        if not self._records:
            return {}
        df = pd.DataFrame(self._records)
        num_cols = df.select_dtypes(include=[np.number]).columns
        return {col: float(df[col].mean()) for col in num_cols}

    def add_result(
        self,
        experiment_name: str,
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record evaluation results for a completed experiment.

        Args:
            experiment_name: Identifier name for the experiment run.
            metrics: Dictionary of evaluated metrics (e.g. AUROC, AUPR).
            metadata: Optional dictionary of supplementary parameters
                (e.g., corruption_type, severity, model_name).
        """
        record: Dict[str, Any] = {"experiment": experiment_name}
        if metadata is not None:
            record.update(metadata)
        record.update(metrics)
        self._records.append(record)
        logger.debug("Recorded metrics for experiment: %s", experiment_name)

    def to_dataframe(self) -> pd.DataFrame:
        """Export all recorded experimental results as a pandas DataFrame.

        Returns:
            pandas DataFrame containing all tracked records.
        """
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame(self._records)

    def summary(self) -> str:
        """Generate a human-readable text summary of all accumulated experiments.

        Returns:
            Formatted string representation of tracked experiments.
        """
        df = self.to_dataframe()
        if df.empty:
            return "No experiment records tracked yet."
        return df.to_string(index=False)

    def save_csv(self, path: str) -> None:
        """Write all accumulated experiment records to a CSV file.

        Args:
            path: Destination file path for the CSV output.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info("Saved %d metric records to %s", len(self._records), path)
