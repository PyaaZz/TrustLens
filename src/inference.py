"""Batch inference engine for TrustLens.

Provides single-loader inference with feature extraction hooks, multi-corruption
batch inference for CIFAR-10-C, and serialization of inference results.
"""

from dataclasses import dataclass
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_loader import CIFAR10C_CORRUPTIONS, get_cifar10c_loader

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Container for batch model inference outputs and runtime statistics.

    Attributes:
        predictions: Class predictions of shape (N,).
        probabilities: Softmax probabilities of shape (N, num_classes).
        labels: Ground-truth target labels of shape (N,).
        features: Penultimate layer representations of shape (N, feature_dim), if extracted.
        logits: Unnormalized logits of shape (N, num_classes), if available.
        inference_time: Total elapsed wall-clock inference time in seconds.
    """

    predictions: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray
    features: Optional[np.ndarray] = None
    logits: Optional[np.ndarray] = None
    inference_time: float = 0.0

    @property
    def is_correct(self) -> np.ndarray:
        """Binary array indicating whether each prediction matches ground truth (1) or not (0)."""
        return (self.predictions == self.labels).astype(int)

    @property
    def accuracy(self) -> float:
        """Overall classification accuracy across all samples."""
        if len(self.labels) == 0:
            return 0.0
        return float(np.mean(self.is_correct))

    def save(self, path: str) -> None:
        """Serialize InferenceResult to a compressed numpy .npz archive.

        Args:
            path: Destination file path ending with .npz.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data_to_save: Dict[str, Any] = {
            "predictions": self.predictions,
            "probabilities": self.probabilities,
            "labels": self.labels,
            "inference_time": np.array(self.inference_time, dtype=np.float64),
        }
        if self.features is not None:
            data_to_save["features"] = self.features
        if self.logits is not None:
            data_to_save["logits"] = self.logits

        np.savez_compressed(path, **data_to_save)
        logger.debug("Saved InferenceResult to %s", path)

    @classmethod
    def load(cls, path: str) -> "InferenceResult":
        """Deserialize an InferenceResult instance from a .npz file.

        Args:
            path: File path to a previously saved .npz archive.

        Returns:
            InferenceResult object with all restored arrays.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Inference result file not found: {path}")

        with np.load(path) as data:
            predictions = data["predictions"]
            probabilities = data["probabilities"]
            labels = data["labels"]
            features = data["features"] if "features" in data else None
            logits = data["logits"] if "logits" in data else None
            inference_time = (
                float(data["inference_time"]) if "inference_time" in data else 0.0
            )

        return cls(
            predictions=predictions,
            probabilities=probabilities,
            labels=labels,
            features=features,
            logits=logits,
            inference_time=inference_time,
        )


def run_inference(
    model: nn.Module,
    data_loader: DataLoader,
    device: Union[str, torch.device] = "cpu",
    extract_features: bool = False,
) -> InferenceResult:
    """Run model inference across a complete DataLoader and collect results.

    Optionally registers a forward hook on the model's `avgpool` layer
    to capture penultimate-layer feature activations.

    Args:
        model: PyTorch classification model.
        data_loader: DataLoader yielding (inputs, targets).
        device: Device on which inference is performed.
        extract_features: If True, extract penultimate representations.

    Returns:
        InferenceResult containing predictions, probabilities, labels,
        optional features, logits, and elapsed runtime.
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    hook_handle = None
    extracted_features_list: List[np.ndarray] = []

    if extract_features:
        # Locate avgpool layer (supports standard ResNet or DataParallel)
        target_layer = None
        if hasattr(model, "avgpool"):
            target_layer = model.avgpool
        elif hasattr(model, "module") and hasattr(model.module, "avgpool"):
            target_layer = model.module.avgpool

        if target_layer is not None:
            def _avgpool_hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                # Output shape typically (B, D, 1, 1); flatten to (B, D)
                flat_out = torch.flatten(output, 1)
                extracted_features_list.append(flat_out.detach().cpu().numpy())

            hook_handle = target_layer.register_forward_hook(_avgpool_hook)
        else:
            logger.warning(
                "Requested feature extraction, but 'avgpool' layer was not found in model."
            )

    all_predictions: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []

    start_time = time.perf_counter()

    try:
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_logits.append(logits.detach().cpu().numpy())
                all_probabilities.append(probs.detach().cpu().numpy())
                all_predictions.append(preds.detach().cpu().numpy())
                all_labels.append(targets.cpu().numpy())
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    elapsed_time = time.perf_counter() - start_time

    if not all_predictions:
        return InferenceResult(
            predictions=np.array([]),
            probabilities=np.array([]),
            labels=np.array([]),
            features=None,
            logits=None,
            inference_time=elapsed_time,
        )

    predictions = np.concatenate(all_predictions, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    features = (
        np.concatenate(extracted_features_list, axis=0)
        if (extract_features and extracted_features_list)
        else None
    )

    logger.debug(
        "Inference completed for %d samples in %.3fs (acc=%.4f)",
        len(predictions),
        elapsed_time,
        float(np.mean(predictions == labels)),
    )

    return InferenceResult(
        predictions=predictions,
        probabilities=probabilities,
        labels=labels,
        features=features,
        logits=logits,
        inference_time=elapsed_time,
    )


def batch_inference_cifar10c(
    model: nn.Module,
    corruption_types: Optional[Sequence[str]] = None,
    severities: Optional[Sequence[int]] = None,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 128,
    data_dir: str = "./data",
    extract_features: bool = False,
) -> Dict[Tuple[str, int], InferenceResult]:
    """Execute batch inference across a grid of CIFAR-10-C corruptions and severities.

    Args:
        model: PyTorch model.
        corruption_types: List of corruption names. Defaults to all 19 CIFAR-10-C types.
        severities: Sequence of severity levels (1 to 5). Defaults to (1, 2, 3, 4, 5).
        device: Compute device.
        batch_size: Batch size for DataLoader.
        data_dir: Base directory containing CIFAR-10-C.
        extract_features: Whether to extract penultimate layer features.

    Returns:
        Dictionary mapping (corruption_type, severity) to InferenceResult.
    """
    if corruption_types is None:
        corruption_types = CIFAR10C_CORRUPTIONS
    if severities is None:
        severities = [1, 2, 3, 4, 5]

    results: Dict[Tuple[str, int], InferenceResult] = {}
    total_runs = len(corruption_types) * len(severities)
    current_run = 0

    logger.info(
        "Starting CIFAR-10-C batch inference: %d corruptions x %d severities (%d total runs)",
        len(corruption_types),
        len(severities),
        total_runs,
    )

    for corruption in corruption_types:
        for severity in severities:
            current_run += 1
            logger.info(
                "[%d/%d] Evaluating CIFAR-10-C: %s (severity %d)",
                current_run,
                total_runs,
                corruption,
                severity,
            )
            loader = get_cifar10c_loader(
                corruption_type=corruption,
                severity=severity,
                batch_size=batch_size,
                data_dir=data_dir,
            )
            result = run_inference(
                model=model,
                data_loader=loader,
                device=device,
                extract_features=extract_features,
            )
            results[(corruption, severity)] = result

    return results
