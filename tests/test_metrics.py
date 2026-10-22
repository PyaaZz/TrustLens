"""Unit tests for the metrics module.

Tests cover:
- AUROC with perfect, random, and inverted separator scores
- AUPR (Precision-Recall AUC) calculation
- False Positive Rate at target True Positive Rate (FPR@95TPR)
- Top-1 and multi-class accuracy computation
- MetricsTracker recording, aggregation, and formatting
"""

import math
from typing import Dict, Any
import numpy as np
import pytest

from src.metrics import (
    MetricsTracker,
    compute_accuracy,
    compute_all_metrics,
    compute_aupr,
    compute_auroc,
    compute_fpr_at_tpr,
    format_metrics_table,
)


class TestAUROC:
    """Tests for AUROC calculation across known distributions."""

    def test_auroc_perfect_separator(self) -> None:
        """When positive scores are strictly greater than negative scores, AUROC = 1.0."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        auroc = compute_auroc(y_true, scores)
        assert math.isclose(auroc, 1.0, abs_tol=1e-5)

    def test_auroc_inverted_separator(self) -> None:
        """When positive scores are strictly less than negative scores, AUROC = 0.0."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
        auroc = compute_auroc(y_true, scores)
        assert math.isclose(auroc, 0.0, abs_tol=1e-5)

    def test_auroc_random_separator(self) -> None:
        """AUROC of random noise predictor on large sample should approximate 0.5."""
        np.random.seed(42)
        n = 5000
        y_true = np.random.randint(0, 2, size=n)
        scores = np.random.uniform(0, 1, size=n)
        auroc = compute_auroc(y_true, scores)
        assert 0.45 <= auroc <= 0.55


class TestAUPR:
    """Tests for Area Under the Precision-Recall curve."""

    def test_aupr_perfect(self) -> None:
        """Perfect ranking should achieve AUPR of 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        aupr = compute_aupr(y_true, scores)
        assert math.isclose(aupr, 1.0, abs_tol=1e-4)

    def test_aupr_bounds(self) -> None:
        """AUPR score must always fall within [0.0, 1.0]."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        scores = np.random.uniform(0, 1, size=100)
        aupr = compute_aupr(y_true, scores)
        assert 0.0 <= aupr <= 1.0


class TestFPRAtTPR:
    """Tests for FPR at specified TPR (e.g. 95%)."""

    def test_fpr_at_95tpr_perfect(self) -> None:
        """For perfectly separated distributions, FPR at 95% TPR should be 0.0."""
        y_true = np.array([0] * 50 + [1] * 50)
        scores = np.array([0.1] * 50 + [0.9] * 50)
        fpr = compute_fpr_at_tpr(y_true, scores, target_tpr=0.95)
        assert math.isclose(fpr, 0.0, abs_tol=1e-4)

    def test_fpr_at_95tpr_bounds(self) -> None:
        """FPR at 95% TPR must fall within [0.0, 1.0]."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=200)
        scores = np.random.randn(200)
        fpr = compute_fpr_at_tpr(y_true, scores, target_tpr=0.95)
        assert 0.0 <= fpr <= 1.0


class TestAccuracy:
    """Tests for accuracy computation."""

    def test_compute_accuracy_perfect(self) -> None:
        """All correct predictions yield accuracy 1.0."""
        preds = np.array([0, 1, 2, 3, 4])
        targets = np.array([0, 1, 2, 3, 4])
        acc = compute_accuracy(preds, targets)
        assert math.isclose(acc, 1.0, abs_tol=1e-5)

    def test_compute_accuracy_partial(self) -> None:
        """Partial matching returns correct fraction."""
        preds = np.array([0, 1, 2, 3])
        targets = np.array([0, 1, 9, 9])
        acc = compute_accuracy(preds, targets)
        assert math.isclose(acc, 0.5, abs_tol=1e-5)


class TestMetricsTracker:
    """Tests for the MetricsTracker logging class."""

    def test_metrics_tracker_logging(self) -> None:
        """Tracker stores metric values and computes summary statistics."""
        tracker = MetricsTracker()
        tracker.update({"auroc": 0.85, "aupr": 0.80})
        tracker.update({"auroc": 0.95, "aupr": 0.90})

        avg_metrics = tracker.get_averages()
        assert math.isclose(avg_metrics["auroc"], 0.90, abs_tol=1e-5)
        assert math.isclose(avg_metrics["aupr"], 0.85, abs_tol=1e-5)

    def test_format_metrics_table(self) -> None:
        """format_metrics_table should output a formatted string containing keys."""
        metrics_dict = {"AUROC": 0.9234, "AUPR": 0.8876, "FPR@95": 0.0451}
        table_str = format_metrics_table(metrics_dict)
        assert isinstance(table_str, str)
        assert "AUROC" in table_str
        assert "0.9234" in table_str or "0.92" in table_str
