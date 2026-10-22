"""TrustLens source package.

Provides data loading, model inference, reliability signals,
calibration, feature-based OOD detection, failure detection, and metrics.
"""

from src.data_loader import (
    CIFAR10_CLASSES,
    CIFAR10C_CORRUPTIONS,
    CIFAR10CDataset,
    download_cifar10c,
    get_cifar10_loaders,
    get_cifar10c_loader,
)
from src.inference import InferenceResult, batch_inference_cifar10c, run_inference
from src.signals import (
    compute_all_signals,
    energy_score,
    feature_distance,
    max_softmax_probability,
    predictive_entropy,
)
from src.calibration import (
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
)
from src.feature_detector import FeatureDetector, extract_features
from src.failure_detector import (
    FailureDetector,
    evaluate_detector,
    find_optimal_threshold,
)
from src.metrics import (
    MetricsTracker,
    compute_accuracy,
    compute_all_metrics,
    compute_aupr,
    compute_auroc,
    compute_fpr_at_tpr,
    format_metrics_table,
)

__all__ = [
    # Data loaders
    "CIFAR10_CLASSES",
    "CIFAR10C_CORRUPTIONS",
    "CIFAR10CDataset",
    "download_cifar10c",
    "get_cifar10_loaders",
    "get_cifar10c_loader",
    # Inference
    "InferenceResult",
    "run_inference",
    "batch_inference_cifar10c",
    # Signals
    "max_softmax_probability",
    "predictive_entropy",
    "energy_score",
    "feature_distance",
    "compute_all_signals",
    # Calibration
    "TemperatureScaler",
    "expected_calibration_error",
    "reliability_diagram_data",
    "brier_score",
    # Feature detection
    "FeatureDetector",
    "extract_features",
    # Failure detection
    "FailureDetector",
    "evaluate_detector",
    "find_optimal_threshold",
    # Metrics
    "compute_auroc",
    "compute_aupr",
    "compute_fpr_at_tpr",
    "compute_accuracy",
    "compute_all_metrics",
    "format_metrics_table",
    "MetricsTracker",
]
