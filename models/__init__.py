"""TrustLens models package.

Provides architecture definitions and training/evaluation utilities.
"""

from models.classifier import (
    build_resnet18,
    evaluate_model,
    load_model,
    train_model,
)

__all__ = [
    "build_resnet18",
    "train_model",
    "load_model",
    "evaluate_model",
]
