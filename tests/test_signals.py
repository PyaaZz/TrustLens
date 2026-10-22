"""Unit tests for the reliability signals module.

Tests cover:
- Max Softmax Probability (MSP) with known distributions and edge cases
- Predictive entropy (uniform distribution, one-hot distribution, zero handling)
- Free energy score computation
- Feature distance computation (Mahalanobis / KNN distance)
- Synthetic data edge cases (zeros, ones, near-zero probabilities)
"""

import math
from typing import Dict, Any
import numpy as np
import pytest
import torch

from src.signals import (
    compute_all_signals,
    energy_score,
    feature_distance,
    max_softmax_probability,
    predictive_entropy,
)


class TestMaxSoftmaxProbability:
    """Tests for max softmax probability (MSP) signal."""

    def test_msp_known_one_hot(self) -> None:
        """One-hot probability distribution should have MSP equal to 1.0."""
        probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        msp = max_softmax_probability(probs)
        if isinstance(msp, torch.Tensor):
            msp = msp.detach().cpu().numpy()
        np.testing.assert_allclose(msp, [1.0, 1.0, 1.0], atol=1e-5)

    def test_msp_known_uniform(self) -> None:
        """Uniform probability distribution over 10 classes should have MSP = 0.1."""
        num_classes = 10
        probs = np.full((5, num_classes), 1.0 / num_classes)
        msp = max_softmax_probability(probs)
        if isinstance(msp, torch.Tensor):
            msp = msp.detach().cpu().numpy()
        np.testing.assert_allclose(msp, np.full(5, 0.1), atol=1e-5)

    def test_msp_tensor_input(self) -> None:
        """MSP should support PyTorch Tensor inputs."""
        probs = torch.tensor([[0.7, 0.2, 0.1], [0.4, 0.5, 0.1]])
        msp = max_softmax_probability(probs)
        if isinstance(msp, torch.Tensor):
            msp = msp.detach().cpu().numpy()
        np.testing.assert_allclose(msp, [0.7, 0.5], atol=1e-5)

    def test_msp_edge_cases_extremes(self) -> None:
        """MSP handles 1D arrays or extreme bounds without error."""
        single_sample = np.array([0.05, 0.90, 0.05])
        msp = max_softmax_probability(single_sample)
        if isinstance(msp, torch.Tensor):
            msp = msp.detach().cpu().numpy()
        val = float(np.squeeze(msp))
        assert math.isclose(val, 0.90, abs_tol=1e-4)


class TestPredictiveEntropy:
    """Tests for predictive entropy signal."""

    def test_entropy_one_hot(self) -> None:
        """Entropy of a deterministic (one-hot) distribution should be 0."""
        probs = np.array([[1.0, 0.0, 0.0, 0.0]])
        entropy = predictive_entropy(probs)
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.detach().cpu().numpy()
        val = float(np.squeeze(entropy))
        assert math.isclose(val, 0.0, abs_tol=1e-5)

    def test_entropy_uniform_is_maximum(self) -> None:
        """Uniform distribution should achieve theoretical maximum entropy ln(K)."""
        k = 10
        probs = np.full((1, k), 1.0 / k)
        entropy = predictive_entropy(probs)
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.detach().cpu().numpy()
        val = float(np.squeeze(entropy))
        # Support either natural log or log2 base
        expected_ln = math.log(k)
        expected_log2 = math.log2(k)
        assert math.isclose(val, expected_ln, rel_tol=1e-3) or math.isclose(
            val, expected_log2, rel_tol=1e-3
        )

    def test_entropy_monotonicity(self) -> None:
        """More uncertain distributions should have higher entropy."""
        sharp = np.array([[0.9, 0.05, 0.05]])
        flat = np.array([[0.4, 0.3, 0.3]])
        ent_sharp = predictive_entropy(sharp)
        ent_flat = predictive_entropy(flat)
        if isinstance(ent_sharp, torch.Tensor):
            ent_sharp = ent_sharp.detach().cpu().numpy()
            ent_flat = ent_flat.detach().cpu().numpy()
        assert float(np.squeeze(ent_flat)) > float(np.squeeze(ent_sharp))

    def test_entropy_zero_probabilities(self) -> None:
        """Entropy calculation should handle zeros safely without NaN or Inf."""
        probs = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        entropy = predictive_entropy(probs)
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.detach().cpu().numpy()
        assert not np.isnan(entropy).any()
        assert not np.isinf(entropy).any()


class TestEnergyScore:
    """Tests for free energy score."""

    def test_energy_score_basic(self) -> None:
        """Energy score computed from logits should match negative logsumexp."""
        logits = np.array([[2.0, 1.0, 0.0], [-1.0, -2.0, -3.0]])
        energy = energy_score(logits, temperature=1.0)
        if isinstance(energy, torch.Tensor):
            energy = energy.detach().cpu().numpy()
        # E(x; T) = -T * log(sum(exp(f_i / T)))
        expected = -1.0 * np.log(np.sum(np.exp(logits), axis=1))
        np.testing.assert_allclose(np.squeeze(energy), expected, atol=1e-4)

    def test_energy_score_temperature_scaling(self) -> None:
        """Energy score should scale properly with temperature."""
        logits = np.array([[4.0, 2.0, 0.0]])
        energy_t1 = energy_score(logits, temperature=1.0)
        energy_t2 = energy_score(logits, temperature=2.0)
        if isinstance(energy_t1, torch.Tensor):
            energy_t1 = energy_t1.detach().cpu().numpy()
            energy_t2 = energy_t2.detach().cpu().numpy()
        assert float(np.squeeze(energy_t1)) != float(np.squeeze(energy_t2))


class TestFeatureDistance:
    """Tests for feature distance calculations."""

    def test_feature_distance_synthetic_data(self) -> None:
        """Distance to centroids or reference features should be minimal at centroid."""
        np.random.seed(42)
        n_samples = 20
        dim = 8
        features = np.random.randn(n_samples, dim)
        reference_mean = np.mean(features, axis=0, keepdims=True)

        # Distance of reference_mean to itself should be approximately 0
        dist_self = feature_distance(reference_mean, reference_mean)
        if isinstance(dist_self, torch.Tensor):
            dist_self = dist_self.detach().cpu().numpy()
        np.testing.assert_allclose(float(np.squeeze(dist_self)), 0.0, atol=1e-4)

        # Random points should have strictly positive distance
        dist = feature_distance(features, reference_mean)
        if isinstance(dist, torch.Tensor):
            dist = dist.detach().cpu().numpy()
        assert np.all(dist > 0)


class TestComputeAllSignals:
    """Tests for the unified compute_all_signals function."""

    def test_compute_all_signals_returns_dict(self) -> None:
        """compute_all_signals should return a dictionary with all expected keys."""
        np.random.seed(42)
        n_samples = 15
        num_classes = 10
        logits = np.random.randn(n_samples, num_classes)
        features = np.random.randn(n_samples, 32)
        train_features = np.random.randn(50, 32)
        train_labels = np.random.randint(0, num_classes, size=50)

        signals = compute_all_signals(
            logits=logits,
            features=features,
            train_features=train_features,
            train_labels=train_labels,
        )

        assert isinstance(signals, dict)
        for key in ["msp", "entropy", "energy", "feature_distance"]:
            assert key in signals, f"Key '{key}' missing from compute_all_signals output"
            assert len(signals[key]) == n_samples
