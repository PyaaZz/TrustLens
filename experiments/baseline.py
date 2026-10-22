"""Week 1 Experiment: Baseline Classifier Training and Evaluation.

Trains a ResNet-18 model on CIFAR-10 (or loads an existing checkpoint) and evaluates
it on the clean test set, reporting accuracy, test loss, and inference latency.
Saves predictions, probabilities, training curves, and confusion matrix plots.

Usage:
    python experiments/baseline.py --epochs 100 --batch-size 128 --device auto
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional YAML loader
try:
    import yaml
except ImportError:
    yaml = None

# Imports from project modules
try:
    from models.classifier import (
        build_resnet18,
        evaluate_model,
        load_model,
        train_model,
    )
except ImportError:
    # Graceful fallbacks if classifier defines alternate class/function names
    build_resnet18 = None
    evaluate_model = None
    load_model = None
    train_model = None

try:
    from src.data_loader import (
        CIFAR10_CLASSES,
        get_cifar10_loaders,
    )
except ImportError:
    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    get_cifar10_loaders = None

logger = logging.getLogger("trustlens.baseline")


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across libraries.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file if available.

    Args:
        config_path: Path to config file, or None for default.

    Returns:
        Dictionary of configuration options.
    """
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "experiment.yaml")

    config: Dict[str, Any] = {}
    cfg_file = Path(config_path)
    if cfg_file.exists() and yaml is not None:
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            logger.info("Loaded configuration from %s", cfg_file)
        except Exception as e:
            logger.warning("Failed to load config file %s: %s", cfg_file, e)
    return config


def resolve_device(device_arg: str) -> torch.device:
    """Resolve device string to torch.device.

    Args:
        device_arg: Device choice ('auto', 'cuda', 'cpu', or cuda device index).

    Returns:
        Resolved torch.device.
    """
    if device_arg == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device_arg
    device = torch.device(device_str)
    logger.info("Using compute device: %s", device)
    return device


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Path,
) -> None:
    """Plot and save training/validation loss and accuracy curves.

    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: Destination file path for figure.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax_loss = axes[0]
    if "train_loss" in history and history["train_loss"]:
        ax_loss.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in history and history["val_loss"]:
        ax_loss.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    ax_loss.set_title("Loss vs. Epoch", fontsize=14, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(fontsize=11)

    # Accuracy plot
    ax_acc = axes[1]
    if "train_acc" in history and history["train_acc"]:
        ax_acc.plot(epochs, history["train_acc"], label="Train Acc", color="#2ca02c", linewidth=2)
    if "val_acc" in history and history["val_acc"]:
        ax_acc.plot(epochs, history["val_acc"], label="Val Acc", color="#d62728", linewidth=2, linestyle="--")
    ax_acc.set_title("Accuracy vs. Epoch", fontsize=14, fontweight="bold")
    ax_acc.set_xlabel("Epoch", fontsize=12)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=12)
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("Saved training curves figure to %s", save_path)


def plot_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    save_path: Path,
) -> None:
    """Compute, plot, and save the confusion matrix.

    Args:
        targets: 1D array of ground truth labels.
        predictions: 1D array of predicted class labels.
        class_names: List of class label strings.
        save_path: Destination file path for figure.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, predictions):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title="CIFAR-10 Baseline Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info("Saved confusion matrix figure to %s", save_path)


def evaluate_clean_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Run evaluation on the clean CIFAR-10 test set.

    Args:
        model: Evaluated PyTorch neural network.
        test_loader: DataLoader for the clean test dataset.
        device: Torch device.

    Returns:
        Tuple of (metrics_dict, targets, predictions, probabilities).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item() * targets.size(0)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    elapsed_time = time.perf_counter() - start_time
    avg_loss = total_loss / max(total, 1)
    accuracy = (correct / max(total, 1)) * 100.0
    latency_per_sample_ms = (elapsed_time / max(total, 1)) * 1000.0
    throughput_samples_per_sec = total / max(elapsed_time, 1e-6)

    targets_arr = np.array(all_targets, dtype=np.int64)
    preds_arr = np.array(all_preds, dtype=np.int64)
    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 10))

    metrics = {
        "test_loss": float(avg_loss),
        "test_accuracy": float(accuracy),
        "total_samples": int(total),
        "correct_samples": int(correct),
        "total_eval_time_sec": float(elapsed_time),
        "latency_per_sample_ms": float(latency_per_sample_ms),
        "throughput_samples_per_sec": float(throughput_samples_per_sec),
    }

    logger.info(
        "Clean Test Results: Accuracy=%.2f%%, Loss=%.4f, Latency=%.2f ms/sample (%.1f samples/sec)",
        accuracy,
        avg_loss,
        latency_per_sample_ms,
        throughput_samples_per_sec,
    )
    return metrics, targets_arr, preds_arr, probs_arr


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TrustLens Baseline Classifier Training & Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training/testing")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to CIFAR-10 data directory")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save/load checkpoints")
    parser.add_argument("--results-dir", type=str, default=None, help="Root directory to save results")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--force-train", action="store_true", help="Force training even if checkpoint exists")
    return parser.parse_args()


def main() -> None:
    """Main execution flow for baseline experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    # Load config file and merge CLI overrides
    cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    device_str = args.device or cfg.get("project", {}).get("device", "auto")
    device = resolve_device(device_str)

    # Parameter resolution
    data_dir = args.data_dir or cfg.get("data", {}).get("data_dir", "./data")
    batch_size = args.batch_size or cfg.get("data", {}).get("batch_size", 128)
    epochs = args.epochs or cfg.get("training", {}).get("epochs", 100)
    lr = args.lr or cfg.get("training", {}).get("learning_rate", 0.1)
    checkpoint_dir = Path(args.checkpoint_dir or cfg.get("model", {}).get("checkpoint_dir", "./models/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir or cfg.get("results", {}).get("results_dir", "./results"))
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "baseline_resnet18.pth"

    logger.info("Starting Baseline Classifier Experiment")
    logger.info("Configuration: Epochs=%d, LR=%f, BatchSize=%d, Seed=%d", epochs, lr, batch_size, seed)

    # 1. Obtain Data Loaders
    if get_cifar10_loaders is not None:
        try:
            train_loader, val_loader, test_loader = get_cifar10_loaders(
                data_dir=data_dir,
                batch_size=batch_size,
            )
        except (ValueError, TypeError):
            # In case get_cifar10_loaders returns 2 items (train, test)
            res = get_cifar10_loaders(data_dir=data_dir, batch_size=batch_size)
            if len(res) == 2:
                train_loader, test_loader = res
                val_loader = test_loader
            else:
                train_loader, val_loader, test_loader = res
    else:
        import torchvision
        import torchvision.transforms as transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # 2. Model Initialization or Loading
    model = None
    training_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if checkpoint_path.exists() and not args.force_train:
        logger.info("Found existing checkpoint at %s. Loading model...", checkpoint_path)
        if load_model is not None:
            model = load_model(str(checkpoint_path), device=device)
        else:
            if build_resnet18 is not None:
                model = build_resnet18(num_classes=10)
            else:
                import torchvision.models as tv_models
                model = tv_models.resnet18(num_classes=10)
            state_dict = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
                training_history = state_dict.get("history", training_history)
            else:
                model.load_state_dict(state_dict)
            model.to(device)
    else:
        logger.info("Training baseline ResNet-18 model from scratch...")
        if build_resnet18 is not None:
            model = build_resnet18(num_classes=10)
        else:
            import torchvision.models as tv_models
            model = tv_models.resnet18(num_classes=10)
            # Adjust first conv layer for 32x32 CIFAR-10 images if standard ImageNet resnet
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.to(device)

        if train_model is not None:
            trained_res = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                device=device,
                checkpoint_path=str(checkpoint_path),
            )
            if isinstance(trained_res, tuple) and len(trained_res) >= 2:
                model, training_history = trained_res[0], trained_res[1]
            elif isinstance(trained_res, dict):
                training_history = trained_res
        else:
            # Inline standard training loop
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.CrossEntropyLoss()

            for ep in range(1, epochs + 1):
                model.train()
                running_loss, correct, total = 0.0, 0, 0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * y.size(0)
                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)

                scheduler.step()
                ep_train_loss = running_loss / total
                ep_train_acc = (correct / total) * 100.0
                training_history["train_loss"].append(ep_train_loss)
                training_history["train_acc"].append(ep_train_acc)

                # Validation step
                model.eval()
                val_loss, val_corr, val_tot = 0.0, 0, 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        vout = model(vx)
                        vloss = criterion(vout, vy)
                        val_loss += vloss.item() * vy.size(0)
                        val_corr += (vout.argmax(1) == vy).sum().item()
                        val_tot += vy.size(0)

                ep_val_loss = val_loss / max(val_tot, 1)
                ep_val_acc = (val_corr / max(val_tot, 1)) * 100.0
                training_history["val_loss"].append(ep_val_loss)
                training_history["val_acc"].append(ep_val_acc)

                if ep % 10 == 0 or ep == epochs:
                    logger.info("Epoch [%d/%d]: Train Loss=%.4f, Acc=%.2f%% | Val Loss=%.4f, Acc=%.2f%%",
                                ep, epochs, ep_train_loss, ep_train_acc, ep_val_loss, ep_val_acc)

            # Save checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "history": training_history,
                "epochs": epochs,
                "seed": seed,
            }, checkpoint_path)
            logger.info("Saved trained checkpoint to %s", checkpoint_path)

    # 3. Evaluate on Clean Test Set
    metrics, targets, preds, probs = evaluate_clean_test_set(model, test_loader, device)

    # 4. Save Metrics and Predictions
    metrics_file = metrics_dir / "baseline_metrics.json"
    full_results = {
        "experiment": "baseline",
        "dataset": "CIFAR-10",
        "architecture": "resnet18",
        "seed": seed,
        "metrics": metrics,
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    logger.info("Saved baseline metrics JSON to %s", metrics_file)

    predictions_file = metrics_dir / "baseline_predictions.npz"
    np.savez_compressed(
        predictions_file,
        targets=targets,
        predictions=preds,
        probabilities=probs,
    )
    logger.info("Saved baseline predictions and probabilities to %s", predictions_file)

    # 5. Generate Figures
    if training_history.get("train_loss"):
        curves_path = figures_dir / "baseline_training_curves.png"
        plot_training_curves(training_history, curves_path)

    cm_path = figures_dir / "baseline_confusion_matrix.png"
    plot_confusion_matrix(targets, preds, CIFAR10_CLASSES, cm_path)

    logger.info("Baseline experiment complete. Accuracy: %.2f%%", metrics["test_accuracy"])


if __name__ == "__main__":
    main()
