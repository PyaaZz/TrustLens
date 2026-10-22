"""ResNet-18 classifier architecture adapted for CIFAR-10.

Includes network definition, training loop with Cosine Annealing, checkpointing,
model loading, and batch evaluation utilities.
"""

import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic execution.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug("Random seed set to %d", seed)


def build_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Construct a ResNet-18 model adapted for 32x32 CIFAR-10 images.

    Replaces the standard 7x7 stride-2 first convolution with a 3x3 stride-1
    convolution and replaces the initial max-pooling layer with an Identity
    layer to prevent excessive spatial downsampling on 32x32 inputs.

    Args:
        num_classes: Number of target output classes.
        pretrained: Whether to initialize with ImageNet pretrained weights.

    Returns:
        Adapted ResNet-18 PyTorch nn.Module.
    """
    weights = (
        models.ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )
    model = models.resnet18(weights=weights)

    # CIFAR-10 adaptation: 3x3 conv with stride 1, padding 1
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    # Remove initial max pooling layer
    model.maxpool = nn.Identity()

    # Replace classifier head for CIFAR classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    logger.debug(
        "Built ResNet-18 adapted for CIFAR (num_classes=%d, pretrained=%s)",
        num_classes,
        pretrained,
    )
    return model


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate classifier performance on a given dataset loader.

    Args:
        model: PyTorch model.
        data_loader: DataLoader providing (images, labels).
        device: Device to execute evaluation on ('cpu' or 'cuda').

    Returns:
        Tuple containing:
            - accuracy (float): Overall top-1 classification accuracy [0, 1].
            - mean_loss (float): Average cross-entropy loss per sample.
            - predictions (np.ndarray): Predicted class indices of shape (N,).
            - probabilities (np.ndarray): Softmax probabilities of shape (N, num_classes).
            - labels (np.ndarray): Ground truth labels of shape (N,).
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            total_loss += loss.item()
            total_samples += targets.size(0)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    if total_samples == 0:
        return 0.0, 0.0, np.array([]), np.array([]), np.array([])

    mean_loss = total_loss / total_samples
    predictions = np.concatenate(all_preds, axis=0)
    probabilities = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    accuracy = float(np.mean(predictions == labels))

    return accuracy, mean_loss, predictions, probabilities, labels


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 100,
    lr: float = 0.1,
    device: Union[str, torch.device] = "cpu",
    checkpoint_dir: str = "./models/checkpoints",
    seed: int = 42,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, List[Any]]:
    """Train ResNet-18 classifier with SGD and Cosine Annealing learning rate.

    Args:
        model: PyTorch model instance to train.
        train_loader: DataLoader for training dataset.
        val_loader: DataLoader for validation dataset.
        epochs: Number of training epochs.
        lr: Initial learning rate for SGD.
        device: Device to use for training ('cpu' or 'cuda').
        checkpoint_dir: Directory where checkpoints are saved.
        seed: Random seed for reproducibility.
        checkpoint_path: Optional exact path for saving the checkpoint.

    Returns:
        Dictionary containing epoch-level history for:
        {'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'}
    """
    set_seed(seed)
    device = torch.device(device)
    model.to(device)

    if checkpoint_path is not None:
        best_checkpoint_path = checkpoint_path
        chk_dir = os.path.dirname(checkpoint_path) or checkpoint_dir
    else:
        chk_dir = checkpoint_dir
        best_checkpoint_path = os.path.join(checkpoint_dir, "baseline_resnet18.pth")

    os.makedirs(chk_dir, exist_ok=True)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[Any]] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0

    logger.info("Starting training for %d epochs on device %s", epochs, device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_acc, val_loss, _, _, _ = evaluate_model(model, val_loader, device=device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            "Epoch [%d/%d] | Train Loss: %.4f, Train Acc: %.4f | Val Loss: %.4f, Val Acc: %.4f",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "history": history,
            }
            torch.save(payload, best_checkpoint_path)
            # Also save secondary common name so all scripts can find the file
            for alt_name in ["baseline_resnet18.pth", "best_model.pt"]:
                alt_path = os.path.join(chk_dir, alt_name)
                if os.path.abspath(alt_path) != os.path.abspath(best_checkpoint_path):
                    try:
                        torch.save(payload, alt_path)
                    except Exception:
                        pass
            logger.info(
                "New best validation accuracy: %.4f. Saved checkpoint to %s",
                val_acc,
                best_checkpoint_path,
            )

    return history


def load_model(
    checkpoint_path: str,
    device: Union[str, torch.device] = "cpu",
    num_classes: int = 10,
) -> nn.Module:
    """Load a saved model checkpoint into an adapted ResNet-18 architecture.

    Args:
        checkpoint_path: Path to the .pt or .pth checkpoint file.
        device: Device to place the loaded model on.
        num_classes: Number of output classes.

    Returns:
        Loaded PyTorch model set in eval mode.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = torch.device(device)
    model = build_resnet18(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    elif isinstance(checkpoint, nn.Module):
        model = checkpoint

    model.to(device)
    model.eval()
    logger.info("Loaded model from %s onto %s in eval mode", checkpoint_path, device)
    return model
