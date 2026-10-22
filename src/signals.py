"""Reliability signal calculators for TrustLens.

Provides uncertainty, confidence, and distance-based reliability signals:
- Maximum Softmax Probability (MSP)
- Predictive Entropy (Shannon Entropy)
- Energy Score (Liu et al., 2020)
- Feature Space Distance (Mahalanobis & k-NN)

All operations are implemented in pure NumPy without PyTorch dependencies.
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def max_softmax_probability(probabilities: np.ndarray) -> np.ndarray:
    """Compute the Maximum Softmax Probability (MSP) confidence score.

    Higher values indicate greater model confidence in the primary prediction.

    Args:
        probabilities: Softmax probability array of shape (N, num_classes) or (num_classes,).

    Returns:
        NumPy array of shape (N,) containing the maximum probability for each sample.

    Raises:
        ValueError: If probabilities array is empty or has invalid shape.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.size == 0:
        return np.array([], dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]

    msp = np.max(probs, axis=-1)
    return msp


def predictive_entropy(
    probabilities: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute the Shannon predictive entropy from class probabilities.

    Higher entropy values signify higher prediction ambiguity and epistemic uncertainty.

    Args:
        probabilities: Softmax probability array of shape (N, num_classes) or (num_classes,).
        epsilon: Small smoothing constant to avoid log(0).

    Returns:
        NumPy array of shape (N,) containing entropy values H(p) = -sum(p * log(p)).
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.size == 0:
        return np.array([], dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]

    # Clamp probabilities to avoid numerical instability
    clamped_probs = np.clip(probs, epsilon, 1.0)
    # Re-normalize across classes
    clamped_probs = clamped_probs / np.sum(clamped_probs, axis=-1, keepdims=True)

    entropy = -np.sum(clamped_probs * np.log(clamped_probs), axis=-1)
    return entropy


def energy_score(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute the Helmholtz free energy score for out-of-distribution detection.

    Energy is defined as: E(x; f, T) = -T * log(sum(exp(logits / T))).
    In-distribution samples typically exhibit lower (more negative) energy values.

    Args:
        logits: Unnormalized network logits of shape (N, num_classes) or (num_classes,).
        temperature: Scaling temperature parameter T > 0.

    Returns:
        NumPy array of shape (N,) containing sample-wise energy scores.

    Raises:
        ValueError: If temperature is non-positive.
    """
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be strictly positive, got {temperature}")

    arr = np.asarray(logits, dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    scaled = arr / temperature
    # Numerically stable Log-Sum-Exp computation
    max_val = np.max(scaled, axis=-1, keepdims=True)
    lse = max_val.squeeze(-1) + np.log(
        np.sum(np.exp(scaled - max_val), axis=-1) + 1e-12
    )

    energy = -temperature * lse
    return energy


def feature_distance(
    features: np.ndarray,
    reference_features: np.ndarray,
    reference_labels: Optional[np.ndarray] = None,
    method: str = "mahalanobis",
    k: int = 5,
    reg_lambda: float = 1e-4,
) -> np.ndarray:
    """Compute distance of query features from a reference (in-distribution) set.

    Supports:
    - 'mahalanobis': Class-conditional Gaussian models with tied covariance.
                     Returns the minimum Mahalanobis distance to any class centroid.
    - 'knn': Average Euclidean distance to the k-nearest neighbors in reference features.

    Args:
        features: Query feature matrix of shape (N, D).
        reference_features: Reference (e.g. training) feature matrix of shape (M, D).
        reference_labels: Reference class labels of shape (M,). Required for
                          class-conditional Mahalanobis distance.
        method: Distance calculation method ('mahalanobis' or 'knn').
        k: Number of nearest neighbors to consider when method='knn'.
        reg_lambda: Covariance regularization parameter to ensure non-singularity.

    Returns:
        NumPy array of shape (N,) containing distance values for each query sample.

    Raises:
        ValueError: If method is unsupported or feature dimensions do not match.
    """
    feats = np.asarray(features, dtype=np.float64)
    ref_feats = np.asarray(reference_features, dtype=np.float64)

    if feats.ndim == 1:
        feats = feats[np.newaxis, :]
    if ref_feats.ndim == 1:
        ref_feats = ref_feats[np.newaxis, :]

    n_samples, dim = feats.shape
    if ref_feats.shape[1] != dim:
        raise ValueError(
            f"Dimension mismatch: query features dim={dim}, reference dim={ref_feats.shape[1]}"
        )

    method = method.lower()
    if method == "mahalanobis":
        if reference_labels is not None:
            ref_labels = np.asarray(reference_labels)
            classes = np.unique(ref_labels)
            num_classes = len(classes)

            # Compute class-conditional empirical means
            means = np.zeros((num_classes, dim), dtype=np.float64)
            for idx, c in enumerate(classes):
                c_mask = ref_labels == c
                means[idx] = np.mean(ref_feats[c_mask], axis=0)

            # Compute shared / tied empirical covariance matrix
            cov = np.zeros((dim, dim), dtype=np.float64)
            for idx, c in enumerate(classes):
                c_mask = ref_labels == c
                diff = ref_feats[c_mask] - means[idx]
                cov += np.dot(diff.T, diff)
            cov /= max(len(ref_feats), 1)
        else:
            # Single global mean and covariance
            means = np.mean(ref_feats, axis=0, keepdims=True)
            diff = ref_feats - means
            cov = np.dot(diff.T, diff) / max(len(ref_feats), 1)

        # Apply diagonal regularization to ensure positive-definiteness
        cov += reg_lambda * np.eye(dim, dtype=np.float64)
        try:
            precision = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            precision = np.linalg.inv(cov + 1e-2 * np.eye(dim))

        # Compute minimum Mahalanobis distance across all classes
        # d_M(x, mu_c) = sqrt((x - mu_c)^T * Sigma^{-1} * (x - mu_c))
        min_distances = np.full(n_samples, np.inf, dtype=np.float64)

        for mean_c in means:
            delta = feats - mean_c  # (N, D)
            # Quadratic form: row-wise dot product of delta and (delta @ precision)
            projected = np.dot(delta, precision)  # (N, D)
            dists_sq = np.sum(delta * projected, axis=1)  # (N,)
            dists = np.sqrt(np.maximum(dists_sq, 0.0))
            min_distances = np.minimum(min_distances, dists)

        return min_distances

    elif method == "knn":
        k_val = min(k, len(ref_feats))
        chunk_size = 500
        all_knn_dists = []

        ref_sq = np.sum(ref_feats**2, axis=1)  # (M,)

        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = feats[start_idx:end_idx]  # (B, D)
            chunk_sq = np.sum(chunk**2, axis=1, keepdims=True)  # (B, 1)

            # Pairwise squared Euclidean distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a b^T
            dist_sq = chunk_sq + ref_sq - 2.0 * np.dot(chunk, ref_feats.T)
            dist_sq = np.maximum(dist_sq, 0.0)

            # Partial sort to extract k nearest neighbors
            if k_val < dist_sq.shape[1]:
                k_smallest_sq = np.partition(dist_sq, kth=k_val - 1, axis=1)[:, :k_val]
            else:
                k_smallest_sq = dist_sq

            k_smallest = np.sqrt(k_smallest_sq)
            chunk_mean_dist = np.mean(k_smallest, axis=1)
            all_knn_dists.append(chunk_mean_dist)

        return np.concatenate(all_knn_dists, axis=0)

    else:
        raise ValueError(
            f"Unsupported distance method '{method}'. Supported methods: 'mahalanobis', 'knn'"
        )


def compute_all_signals(
    probabilities: Optional[np.ndarray] = None,
    logits: Optional[np.ndarray] = None,
    features: Optional[np.ndarray] = None,
    reference_features: Optional[np.ndarray] = None,
    reference_labels: Optional[np.ndarray] = None,
    train_features: Optional[np.ndarray] = None,
    train_labels: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Compute all available reliability and uncertainty signals for a set of predictions.

    Args:
        probabilities: Class probabilities of shape (N, C) (optional if logits provided).
        logits: Optional unnormalized logits of shape (N, C).
        features: Optional query feature matrix of shape (N, D).
        reference_features: Optional reference feature matrix of shape (M, D).
        reference_labels: Optional reference labels of shape (M,).
        train_features: Alias for reference_features.
        train_labels: Alias for reference_labels.

    Returns:
        Dictionary mapping signal names to score arrays of shape (N,):
        - 'msp': Maximum softmax probability (confidence)
        - 'entropy': Predictive Shannon entropy (uncertainty)
        - 'energy': Free energy score
        - 'feature_distance': Penultimate feature distance (Mahalanobis)
        - 'mahalanobis': Mahalanobis distance to reference distribution
        - 'knn_distance': k-NN feature distance
    """
    if reference_features is None:
        reference_features = train_features
    if reference_labels is None:
        reference_labels = train_labels

    if probabilities is None:
        if logits is not None:
            z = np.asarray(logits, dtype=np.float64)
            max_val = np.max(z, axis=-1, keepdims=True)
            exp_z = np.exp(z - max_val)
            probabilities = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        else:
            raise ValueError("Either probabilities or logits must be provided to compute_all_signals.")

    signals: Dict[str, np.ndarray] = {}

    signals["msp"] = max_softmax_probability(probabilities)
    signals["entropy"] = predictive_entropy(probabilities)

    if logits is not None:
        signals["energy"] = energy_score(logits)
    else:
        # Approximate energy using log probabilities
        approx_logits = np.log(np.clip(probabilities, 1e-12, 1.0))
        signals["energy"] = energy_score(approx_logits)

    if features is not None and reference_features is not None:
        logger.debug("Computing Mahalanobis and KNN feature distances...")
        dist_mah = feature_distance(
            features=features,
            reference_features=reference_features,
            reference_labels=reference_labels,
            method="mahalanobis",
        )
        signals["mahalanobis"] = dist_mah
        signals["feature_distance"] = dist_mah
        signals["knn_distance"] = feature_distance(
            features=features,
            reference_features=reference_features,
            reference_labels=reference_labels,
            method="knn",
        )

    return signals
