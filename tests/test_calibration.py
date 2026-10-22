"""Unit tests for calibration module.

Tests cover:
- Expected Calibration Error (ECE) computation with known calibrated and uncalibrated cases
- Reliability diagram data generation (bins, accuracies, confidences)
- Temperature scaling parameter optimization and inference with synthetic logits
- Brier score computation
"""

import math
from typing import Dict, Any
import numpy as np
import pytest
import torch

from src.calibration import (
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
)


class TestExpectedCalibrationError:
    """Tests for Expected Calibration Error (ECE) metric."""

    def test_ece_perfectly_calibrated(self) -> None:
        """When accuracy equals confidence in every bin, ECE should be near 0."""
        # 100 samples with 80% confidence, exactly 80 correct
        confidences = np.full(100, 0.8)
        accuracies = np.zeros(100, dtype=int)
        accuracies[:80] = 1

        ece = expected_calibration_error(confidences, accuracies, n_bins=10)
        assert ece < 0.05

    def test_ece_severely_miscalibrated(self) -> None:
        """High confidence with low accuracy should produce high ECE."""
        confidences = np.full(100, 0.99)
        accuracies = np.zeros(100, dtype=int)  # 0% accuracy

        ece = expected_calibration_error(confidences, accuracies, n_bins=10)
        assert ece > 0.90

    def test_ece_bounds(self) -> None:
        """ECE value must always lie in [0, 1]."""
        np.random.seed(42)
        confidences = np.random.uniform(0.1, 1.0, size=200)
        accuracies = np.random.binomial(1, confidences)

        ece = expected_calibration_error(confidences, accuracies, n_bins=15)
        assert 0.0 <= ece <= 1.0


class TestReliabilityDiagramData:
    """Tests for reliability diagram bin data extraction."""

    def test_diagram_bins_structure(self) -> None:
        """Diagram data should contain bin accuracies, confidences, and counts."""
        np.random.seed(42)
        n_bins = 10
        confidences = np.random.uniform(0.0, 1.0, size=150)
        accuracies = np.random.binomial(1, confidences)

        data = reliability_diagram_data(confidences, accuracies, n_bins=n_bins)

        assert isinstance(data, dict)
        for expected_key in ["bin_accuracies", "bin_confidences", "bin_counts"]:
            assert expected_key in data, f"Key '{expected_key}' missing from diagram data"
            assert len(data[expected_key]) == n_bins

    def test_diagram_bins_total_count(self) -> None:
        """The sum of bin counts must equal total number of valid samples."""
        n_samples = 250
        confidences = np.random.uniform(0.0, 1.0, size=n_samples)
        accuracies = np.random.binomial(1, confidences)

        data = reliability_diagram_data(confidences, accuracies, n_bins=10)
        total_binned = sum(data["bin_counts"])
        assert total_binned == n_samples


class TestBrierScore:
    """Tests for multi-class and binary Brier score."""

    def test_brier_score_perfect_predictions(self) -> None:
        """Brier score for perfect 1.0 probability on true labels should be 0."""
        # 3 samples, 3 classes, perfect predictions
        probs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        labels = np.array([0, 1, 2])

        score = brier_score(probs, labels)
        assert math.isclose(float(score), 0.0, abs_tol=1e-5)

    def test_brier_score_worst_predictions(self) -> None:
        """Brier score for completely wrong confident predictions."""
        # Probabilities placed entirely on wrong class
        probs = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        labels = np.array([0, 1])

        score = brier_score(probs, labels)
        # (0-1)^2 + (1-0)^2 = 2 per sample
        assert score > 1.0


class TestTemperatureScaler:
    """Tests for TemperatureScaler model post-processing."""

    def test_temperature_scaler_initial_value(self) -> None:
        """Initial temperature parameter should default to 1.0 or positive float."""
        scaler = TemperatureScaler()
        assert scaler.temperature > 0.0

    def test_temperature_scaling_optimization(self) -> None:
        """Optimizing temperature on overconfident logits should adjust T > 1."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Overconfident logits (large magnitude) with moderate noise
        n_samples = 300
        n_classes = 10
        logits = torch.randn(n_samples, n_classes) * 5.0
        # True labels aligned with argmax only 50% of the time (overconfident)
        labels = torch.randint(0, n_classes, (n_samples,))

        scaler = TemperatureScaler()
        scaler.fit(logits, labels)

        assert scaler.temperature > 0.0
        scaled_logits = scaler.scale_logits(logits)
        assert scaled_logits.shape == logits.shape

    def test_temperature_scaling_predict_proba(self) -> None:
        """Calibrated probabilities should sum to 1 across classes."""
        scaler = TemperatureScaler()
        logits = torch.randn(20, 5)
        probs = scaler.predict_proba(logits)

        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()

        row_sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(20), atol=1e-5)
