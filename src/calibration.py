"""Probability calibration and reliability assessment for TrustLens.

Includes:
- Temperature scaling (Guo et al., 2017) using L-BFGS-B optimization
- Expected Calibration Error (ECE)
- Reliability diagram data preparation
- Multi-class Brier score
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class TemperatureScaler:
    """Post-hoc temperature scaling calibrator for deep neural networks.

    Learns a single scalar temperature T > 0 on validation logits to minimize
    Negative Log-Likelihood (cross-entropy), softening or sharpening the softmax
    distribution without altering model accuracy or rankings.
    """

    def __init__(self, initial_temperature: float = 1.0) -> None:
        """Initialize TemperatureScaler.

        Args:
            initial_temperature: Starting temperature parameter (must be > 0).
        """
        if initial_temperature <= 0:
            raise ValueError(
                f"Initial temperature must be positive, got {initial_temperature}"
            )
        self._temperature: float = float(initial_temperature)
        self._is_fitted: bool = False

    @property
    def temperature(self) -> float:
        """Learned temperature parameter T."""
        return self._temperature

    @property
    def is_fitted(self) -> bool:
        """Whether the temperature scaler has been fitted on data."""
        return self._is_fitted

    def fit(
        self,
        logits: Union[np.ndarray, Any],
        labels: Union[np.ndarray, Any],
        max_iter: int = 100,
    ) -> "TemperatureScaler":
        """Optimize temperature parameter T on validation logits and labels.

        Args:
            logits: Unnormalized network logits of shape (N, C) (NumPy or PyTorch).
            labels: Ground truth class targets of shape (N,) (NumPy or PyTorch).
            max_iter: Maximum iterations for L-BFGS-B optimizer.

        Returns:
            Fitted instance of self.
        """
        if hasattr(logits, "detach"):
            logits = logits.detach().cpu().numpy()
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().numpy()

        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)

        if z.ndim != 2:
            raise ValueError(f"Expected logits of shape (N, C), got ndim={z.ndim}")
        if len(z) != len(y):
            raise ValueError(
                f"Logits length ({len(z)}) does not match labels length ({len(y)})"
            )

        n_samples, n_classes = z.shape

        # One-hot representation of labels
        y_one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
        y_one_hot[np.arange(n_samples), y] = 1.0

        def _nll_loss_and_grad(t_arr: np.ndarray) -> Tuple[float, np.ndarray]:
            t = float(t_arr[0])
            if t <= 0:
                return 1e9, np.array([0.0])

            scaled = z / t
            max_val = np.max(scaled, axis=-1, keepdims=True)
            exp_z = np.exp(scaled - max_val)
            sum_exp = np.sum(exp_z, axis=-1, keepdims=True)
            log_sum_exp = max_val.squeeze(-1) + np.log(sum_exp.squeeze(-1) + 1e-12)

            # NLL = 1/N * sum(logsumexp(z/t) - z_y/t)
            correct_logits = z[np.arange(n_samples), y]
            loss = float(np.mean(log_sum_exp - correct_logits / t))

            # Analytical gradient w.r.t temperature T
            # grad = 1 / (N * T^2) * sum(z_y - sum_k p_k * z_k)
            probs = exp_z / (sum_exp + 1e-12)
            expected_z = np.sum(probs * z, axis=-1)
            grad = float(np.mean((expected_z - correct_logits) / (t**2)))

            return loss, np.array([grad])

        initial_t = np.array([self._temperature], dtype=np.float64)
        bounds = [(1e-3, 50.0)]

        res = minimize(
            fun=_nll_loss_and_grad,
            x0=initial_t,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": max_iter},
        )

        if res.success:
            self._temperature = float(res.x[0])
            self._is_fitted = True
            logger.info("Temperature scaling converged to T = %.4f", self._temperature)
        else:
            self._temperature = float(res.x[0])
            self._is_fitted = True
            logger.warning(
                "Temperature scaling optimization terminated with message: %s. Using T = %.4f",
                res.message,
                self._temperature,
            )

        return self

    def scale_logits(self, logits: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """Scale unnormalized logits by the learned temperature T.

        Args:
            logits: Logits array or tensor.

        Returns:
            Scaled logits with identical type and shape.
        """
        if hasattr(logits, "detach"):
            return logits / self._temperature
        return np.asarray(logits, dtype=np.float64) / self._temperature

    def predict_proba(self, logits: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """Compute calibrated softmax probabilities (alias / wrapper for transform).

        Args:
            logits: Unnormalized logits array or PyTorch tensor.

        Returns:
            Calibrated softmax probabilities with identical input type.
        """
        is_torch = hasattr(logits, "detach")
        arr = logits.detach().cpu().numpy() if is_torch else np.asarray(logits)
        probs = self.transform(arr)
        if is_torch:
            import torch
            return torch.from_numpy(probs).to(logits.device)
        return probs

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply learned temperature to compute calibrated softmax probabilities.

        Args:
            logits: Logits array of shape (N, C) or (C,).

        Returns:
            Calibrated probability array of shape (N, C) or (C,).
        """
        arr = np.asarray(logits, dtype=np.float64)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = arr[np.newaxis, :]

        scaled = arr / self._temperature
        # Numerically stable softmax
        max_val = np.max(scaled, axis=-1, keepdims=True)
        exp_z = np.exp(scaled - max_val)
        probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

        return probs.squeeze(0) if is_1d else probs

    def fit_transform(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit temperature scaler and immediately transform logits.

        Args:
            logits: Validation logits of shape (N, C).
            labels: Ground truth targets of shape (N,).

        Returns:
            Calibrated probabilities of shape (N, C).
        """
        self.fit(logits, labels)
        return self.transform(logits)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute the Expected Calibration Error (ECE) with equal-width confidence bins.

    ECE measures the difference between expected accuracy and confidence:
        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        probabilities: Softmax probability matrix of shape (N, C) or 1D confidences of shape (N,).
        labels: Ground truth target indices or binary correctness flags of shape (N,).
        n_bins: Number of equal-width bins spanning [0, 1].

    Returns:
        Float value of ECE in [0, 1].
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(labels)

    if len(probs) == 0:
        return 0.0

    if probs.ndim == 1:
        confidences = np.clip(probs, 0.0, 1.0)
        accuracies = y_true.astype(float)
    else:
        confidences = np.max(probs, axis=-1)
        predictions = np.argmax(probs, axis=-1)
        accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def reliability_diagram_data(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Compute bin-level statistics required to plot a Reliability Diagram.

    Args:
        probabilities: Softmax probability matrix of shape (N, C) or 1D confidences of shape (N,).
        labels: Ground truth target indices or binary correctness flags of shape (N,).
        n_bins: Number of equal-width calibration bins.

    Returns:
        Dictionary containing:
            - 'bin_confidences': Average confidence per bin (shape (n_bins,))
            - 'bin_accuracies': Empirical accuracy per bin (shape (n_bins,))
            - 'bin_counts': Number of samples in each bin (shape (n_bins,))
            - 'bin_edges': Array of bin boundary values (shape (n_bins + 1,))
            - 'bin_centers': Midpoint of each bin (shape (n_bins,))
            - 'ece': Scalar Expected Calibration Error
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(labels)

    if probs.ndim == 1:
        confidences = np.clip(probs, 0.0, 1.0)
        accuracies = y_true.astype(float)
    else:
        confidences = np.max(probs, axis=-1)
        predictions = np.argmax(probs, axis=-1)
        accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    bin_confidences = np.zeros(n_bins, dtype=np.float64)
    bin_accuracies = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    total_samples = len(confidences)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        count = int(np.sum(in_bin))
        bin_counts[i] = count

        if count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_accuracies[i] = bin_acc
            bin_confidences[i] = bin_conf
            ece += (count / total_samples) * abs(bin_acc - bin_conf)
        else:
            bin_accuracies[i] = 0.0
            bin_confidences[i] = bin_centers[i]
            bin_confidences[i] = bin_centers[i]

    return {
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "ece": float(ece),
    }


def brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute the multi-class Brier score.

    The Brier score is the mean squared error between predicted class probabilities
    and the one-hot encoded ground truth:
        BS = (1 / N) * sum_{i=1}^N sum_{c=1}^C (p_{ic} - y_{ic})^2

    Lower scores indicate superior calibration and probabilistic accuracy.

    Args:
        probabilities: Softmax probability matrix of shape (N, C).
        labels: Ground truth target indices of shape (N,).

    Returns:
        Float Brier score value in [0, 2].
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(labels, dtype=np.int64)

    if len(probs) == 0:
        return 0.0

    n_samples, n_classes = probs.shape
    one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    one_hot[np.arange(n_samples), y_true] = 1.0

    score = float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))
    return score
