"""Combined failure detection system for deep neural networks.

Aggregates multiple reliability and uncertainty signals into a unified
failure prediction score using machine learning classifiers (Logistic Regression,
Multi-Layer Perceptron) or calibrated ensemble voting.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def evaluate_detector(
    failure_scores: np.ndarray,
    is_correct: np.ndarray,
) -> Dict[str, float]:
    """Evaluate failure detector performance against actual correctness labels.

    Treats misclassifications (is_correct == 0) as the positive failure class (1),
    and correct classifications (is_correct == 1) as the negative class (0).

    Args:
        failure_scores: Continuous failure risk scores of shape (N,).
        is_correct: Ground truth binary array of shape (N,) (1=correct, 0=error).

    Returns:
        Dictionary with:
            - 'auroc': Area Under the Receiver Operating Characteristic curve.
            - 'aupr': Area Under the Precision-Recall curve (Average Precision).
            - 'fpr_at_95tpr': False Positive Rate when True Positive Rate is 95%.
    """
    scores = np.asarray(failure_scores, dtype=np.float64)
    # Binary failure target: 1 if incorrect, 0 if correct
    y_true = (np.asarray(is_correct) == 0).astype(int)

    if len(np.unique(y_true)) < 2:
        logger.warning(
            "evaluate_detector called with only one class present in ground truth. Returning nan metrics."
        )
        return {"auroc": float("nan"), "aupr": float("nan"), "fpr_at_95tpr": float("nan")}

    auroc = float(roc_auc_score(y_true, scores))
    aupr = float(average_precision_score(y_true, scores))

    # Compute FPR at 95% TPR
    fpr, tpr, _ = roc_curve(y_true, scores)
    # Find smallest FPR where TPR >= 0.95
    valid_indices = np.where(tpr >= 0.95)[0]
    if len(valid_indices) > 0:
        fpr_at_95tpr = float(fpr[valid_indices[0]])
    else:
        fpr_at_95tpr = 1.0

    return {
        "auroc": auroc,
        "aupr": aupr,
        "fpr_at_95tpr": fpr_at_95tpr,
    }


def find_optimal_threshold(
    failure_scores: np.ndarray,
    is_correct: np.ndarray,
    target_fpr: float = 0.05,
) -> float:
    """Find decision threshold that bounds the False Positive Rate on correct samples.

    A false positive occurs when a correct prediction is incorrectly flagged as a failure.
    At target_fpr=0.05, at most 5% of correct samples are falsely flagged.

    Args:
        failure_scores: Continuous failure risk scores of shape (N,).
        is_correct: Binary correctness array (1=correct, 0=error).
        target_fpr: Maximum acceptable false alarm rate on correct predictions.

    Returns:
        Scalar threshold float. Scores >= threshold are classified as failures.
    """
    scores = np.asarray(failure_scores, dtype=np.float64)
    correct_mask = np.asarray(is_correct) == 1

    if not np.any(correct_mask):
        return float(np.min(scores))

    correct_scores = scores[correct_mask]
    # Quantile corresponding to 1 - target_fpr
    threshold = float(np.quantile(correct_scores, 1.0 - target_fpr))
    return threshold


class FailureDetector:
    """Multi-signal failure detector combining confidence, uncertainty, and distance cues.

    Supports combination via:
    - 'logistic': Logistic Regression with class re-weighting
    - 'mlp': Multi-Layer Perceptron neural network
    - 'threshold': Ensemble voting across per-signal decision thresholds
    """

    def __init__(
        self,
        signals_to_use: Optional[List[str]] = None,
        combination_method: str = "logistic",
        random_state: int = 42,
    ) -> None:
        """Initialize FailureDetector.

        Args:
            signals_to_use: List of signal dictionary keys to use as features.
                Defaults to ['msp', 'entropy', 'energy'].
            combination_method: Classifier method ('logistic', 'mlp', 'threshold').
            random_state: Random state for deterministic model training.

        Raises:
            ValueError: If combination_method is invalid.
        """
        if signals_to_use is None:
            signals_to_use = ["msp", "entropy", "energy"]

        comb_method = combination_method.lower().strip()
        if comb_method not in ("logistic", "mlp", "threshold"):
            raise ValueError(
                f"Unknown combination_method '{combination_method}'. "
                "Supported methods: 'logistic', 'mlp', 'threshold'"
            )

        self.signals_to_use: List[str] = list(signals_to_use)
        self.combination_method: str = comb_method
        self.random_state: int = random_state

        self.scaler: StandardScaler = StandardScaler()
        self.model: Any = None
        self._is_fitted: bool = False

        # Threshold method specific state: threshold and polarity (+1 if higher=failure, -1 if lower=failure)
        self._signal_thresholds: Dict[str, float] = {}
        self._signal_polarities: Dict[str, int] = {}

    @property
    def is_fitted(self) -> bool:
        """Whether the detector has been trained."""
        return self._is_fitted

    def _extract_feature_matrix(
        self, signal_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Extract and horizontally stack requested signals from input dictionary.

        Args:
            signal_dict: Dictionary mapping signal names to 1D arrays.

        Returns:
            2D NumPy array of shape (N, num_signals).

        Raises:
            KeyError: If an expected signal name is missing from signal_dict.
        """
        columns: List[np.ndarray] = []
        for signal_name in self.signals_to_use:
            if signal_name not in signal_dict:
                raise KeyError(
                    f"Required signal '{signal_name}' not found in signal_dict. "
                    f"Available signals: {list(signal_dict.keys())}"
                )
            col = np.asarray(signal_dict[signal_name], dtype=np.float64)
            if col.ndim > 1:
                col = col.ravel()
            columns.append(col)

        if not columns:
            return np.empty((0, 0), dtype=np.float64)

        return np.column_stack(columns)

    def fit(
        self,
        signal_dict: Dict[str, np.ndarray],
        is_correct: np.ndarray,
    ) -> "FailureDetector":
        """Train failure detector using calculated signals and correctness labels.

        Args:
            signal_dict: Mapping of signal names to array of scores.
            is_correct: Binary correctness array (1=correct, 0=error/failure).

        Returns:
            Fitted instance of self.
        """
        X = self._extract_feature_matrix(signal_dict)
        y = (np.asarray(is_correct) == 0).astype(int)

        if len(np.unique(y)) < 2:
            raise ValueError(
                "Training data must contain both correct and failure instances."
            )

        if self.combination_method == "logistic":
            X_scaled = self.scaler.fit_transform(X)
            self.model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
            )
            self.model.fit(X_scaled, y)
            logger.info("Fitted Logistic Regression FailureDetector on %d samples", len(y))

        elif self.combination_method == "mlp":
            X_scaled = self.scaler.fit_transform(X)
            self.model = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True,
                n_iter_no_change=15,
            )
            self.model.fit(X_scaled, y)
            logger.info("Fitted MLP FailureDetector on %d samples", len(y))

        elif self.combination_method == "threshold":
            self._signal_thresholds = {}
            self._signal_polarities = {}

            for idx, signal_name in enumerate(self.signals_to_use):
                col = X[:, idx]
                # Determine polarity: does higher signal correlate with failure?
                # Check mean in failures vs mean in correct samples
                mean_failure = np.mean(col[y == 1])
                mean_correct = np.mean(col[y == 0])

                if mean_failure >= mean_correct:
                    polarity = 1  # higher score indicates failure
                    # Pick median or ROC Youden threshold
                    thresh = float(np.median(col))
                else:
                    polarity = -1  # lower score indicates failure (e.g. MSP)
                    thresh = float(np.median(col))

                self._signal_thresholds[signal_name] = thresh
                self._signal_polarities[signal_name] = polarity

            logger.info(
                "Fitted Threshold FailureDetector with signals: %s",
                self.signals_to_use,
            )

        self._is_fitted = True
        return self

    def predict(self, signal_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute continuous failure risk probability for each sample.

        Args:
            signal_dict: Dictionary containing required signals.

        Returns:
            NumPy array of shape (N,) containing failure probabilities in [0, 1].

        Raises:
            RuntimeError: If called before fit.
        """
        if not self._is_fitted:
            raise RuntimeError("FailureDetector must be fitted before calling predict().")

        X = self._extract_feature_matrix(signal_dict)
        if len(X) == 0:
            return np.array([], dtype=np.float64)

        if self.combination_method in ("logistic", "mlp"):
            X_scaled = self.scaler.transform(X)
            proba = self.model.predict_proba(X_scaled)
            # Second column corresponds to class 1 (failure)
            return proba[:, 1]

        elif self.combination_method == "threshold":
            votes = np.zeros(len(X), dtype=np.float64)
            for idx, signal_name in enumerate(self.signals_to_use):
                col = X[:, idx]
                thresh = self._signal_thresholds[signal_name]
                polarity = self._signal_polarities[signal_name]

                if polarity == 1:
                    is_failure_vote = (col >= thresh).astype(float)
                else:
                    is_failure_vote = (col <= thresh).astype(float)

                votes += is_failure_vote

            # Normalize votes into [0, 1] range
            return votes / max(len(self.signals_to_use), 1)

        else:
            raise RuntimeError(f"Unknown combination method {self.combination_method}")

    def predict_binary(
        self,
        signal_dict: Dict[str, np.ndarray],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary failure flags using a decision threshold.

        Args:
            signal_dict: Dictionary containing required signals.
            threshold: Probability decision threshold.

        Returns:
            Binary NumPy array of shape (N,) where 1 denotes predicted failure.
        """
        prob = self.predict(signal_dict)
        return (prob >= threshold).astype(int)

    def save(self, path: str) -> None:
        """Serialize detector instance to disk using joblib.

        Args:
            path: Destination file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved FailureDetector to %s", path)

    @classmethod
    def load(cls, path: str) -> "FailureDetector":
        """Load serialized detector from disk.

        Args:
            path: Path to serialized detector file.

        Returns:
            Restored FailureDetector instance.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Detector checkpoint not found: {path}")
        detector = joblib.load(path)
        logger.info("Loaded FailureDetector from %s", path)
        return detector
