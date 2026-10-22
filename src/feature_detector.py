"""Feature-based out-of-distribution and failure detector for TrustLens.

Extracts penultimate representations from deep neural networks and evaluates
distance from in-distribution representations using class-conditional Mahalanobis
distance (Lee et al., 2018) or k-Nearest Neighbors distance (Sun et al., 2022).
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def extract_features(
    model: nn.Module,
    data_loader: DataLoader,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer feature vectors and targets using a forward hook.

    Hooks into the model's 'avgpool' layer, flattens spatial activations,
    and returns arrays of representations and ground truth labels.

    Args:
        model: PyTorch neural network.
        data_loader: DataLoader yielding (images, labels).
        device: Device to run inference on.

    Returns:
        Tuple of:
            - features (np.ndarray): Penultimate features of shape (N, feature_dim).
            - labels (np.ndarray): Target labels of shape (N,).

    Raises:
        AttributeError: If model lacks an 'avgpool' layer.
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    # Find avgpool layer
    target_layer = None
    if hasattr(model, "avgpool"):
        target_layer = model.avgpool
    elif hasattr(model, "module") and hasattr(model.module, "avgpool"):
        target_layer = model.module.avgpool
    else:
        raise AttributeError(
            "Model does not have an 'avgpool' layer required for feature extraction."
        )

    features_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    def _avgpool_hook(_module: nn.Module, _input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        flat_feats = torch.flatten(output, 1)
        features_list.append(flat_feats.detach().cpu().numpy())

    hook_handle = target_layer.register_forward_hook(_avgpool_hook)

    try:
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                _ = model(inputs)
                labels_list.append(targets.cpu().numpy())
    finally:
        hook_handle.remove()

    if not features_list:
        return np.array([]), np.array([])

    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    logger.debug(
        "Extracted features shape: %s, labels shape: %s",
        all_features.shape,
        all_labels.shape,
    )
    return all_features, all_labels


class FeatureDetector:
    """Feature-space anomaly and failure detector.

    Supports class-conditional Mahalanobis distance scoring and
    k-Nearest Neighbor (k-NN) distance scoring.
    """

    def __init__(
        self,
        method: str = "mahalanobis",
        k: int = 5,
        reg_lambda: float = 1e-4,
    ) -> None:
        """Initialize FeatureDetector.

        Args:
            method: Detection distance method ('mahalanobis' or 'knn').
            k: Number of nearest neighbors if method='knn'.
            reg_lambda: Diagonal regularization added to the covariance matrix.

        Raises:
            ValueError: If method is not 'mahalanobis' or 'knn'.
        """
        method_clean = method.lower().strip()
        if method_clean not in ("mahalanobis", "knn"):
            raise ValueError(
                f"Invalid method '{method}'. Supported methods: 'mahalanobis', 'knn'"
            )

        self.method: str = method_clean
        self.k: int = k
        self.reg_lambda: float = reg_lambda

        self._is_fitted: bool = False
        # Mahalanobis parameters
        self.class_means: Optional[np.ndarray] = None
        self.precision_matrix: Optional[np.ndarray] = None
        self.classes: Optional[np.ndarray] = None

        # KNN parameters
        self.knn_model: Optional[NearestNeighbors] = None

    @property
    def is_fitted(self) -> bool:
        """Whether the detector has been fitted with in-distribution reference features."""
        return self._is_fitted

    def fit(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> "FeatureDetector":
        """Fit detector on in-distribution reference features and class labels.

        Args:
            features: Reference feature matrix of shape (N, D).
            labels: In-distribution target labels of shape (N,).
                    Required for class-conditional Mahalanobis distance.

        Returns:
            Fitted instance of self.
        """
        feats = np.asarray(features, dtype=np.float64)
        n_samples, dim = feats.shape

        if self.method == "mahalanobis":
            if labels is None:
                raise ValueError(
                    "Labels must be provided for class-conditional Mahalanobis fitting."
                )

            lbls = np.asarray(labels)
            self.classes = np.unique(lbls)
            num_classes = len(self.classes)

            # Compute class-conditional empirical means
            self.class_means = np.zeros((num_classes, dim), dtype=np.float64)
            for idx, c in enumerate(self.classes):
                c_mask = lbls == c
                self.class_means[idx] = np.mean(feats[c_mask], axis=0)

            # Compute shared / tied covariance matrix
            cov = np.zeros((dim, dim), dtype=np.float64)
            for idx, c in enumerate(self.classes):
                c_mask = lbls == c
                diff = feats[c_mask] - self.class_means[idx]
                cov += np.dot(diff.T, diff)
            cov /= max(n_samples, 1)

            # Diagonal regularization
            cov += self.reg_lambda * np.eye(dim, dtype=np.float64)
            try:
                self.precision_matrix = np.linalg.pinv(cov)
            except np.linalg.LinAlgError:
                self.precision_matrix = np.linalg.inv(cov + 1e-2 * np.eye(dim))

            logger.info(
                "Fitted Mahalanobis FeatureDetector with %d classes, dim=%d",
                num_classes,
                dim,
            )

        elif self.method == "knn":
            k_val = min(self.k, n_samples)
            self.knn_model = NearestNeighbors(
                n_neighbors=k_val,
                metric="euclidean",
                algorithm="auto",
            )
            self.knn_model.fit(feats)
            logger.info(
                "Fitted KNN FeatureDetector with k=%d on %d samples, dim=%d",
                k_val,
                n_samples,
                dim,
            )

        self._is_fitted = True
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Compute anomaly/failure scores for query features.

        Higher scores indicate greater deviation from in-distribution representations
        (greater risk of failure or OOD).

        Args:
            features: Query feature matrix of shape (M, D) or (D,).

        Returns:
            NumPy array of shape (M,) containing anomaly scores.

        Raises:
            RuntimeError: If called before detector is fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureDetector must be fitted before calling score().")

        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim == 1:
            feats = feats[np.newaxis, :]

        n_samples, _ = feats.shape

        if self.method == "mahalanobis":
            assert self.class_means is not None and self.precision_matrix is not None
            min_distances = np.full(n_samples, np.inf, dtype=np.float64)

            for mean_c in self.class_means:
                delta = feats - mean_c
                projected = np.dot(delta, self.precision_matrix)
                dist_sq = np.sum(delta * projected, axis=1)
                dists = np.sqrt(np.maximum(dist_sq, 0.0))
                min_distances = np.minimum(min_distances, dists)

            return min_distances

        elif self.method == "knn":
            assert self.knn_model is not None
            # Extract distance to k nearest neighbors
            distances, _ = self.knn_model.kneighbors(feats)
            # Use mean distance across the k neighbors as anomaly score
            scores = np.mean(distances, axis=1)
            return scores

        else:
            raise RuntimeError(f"Unknown method {self.method}")
