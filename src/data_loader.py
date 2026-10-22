"""Data loading utilities for CIFAR-10 and CIFAR-10-C datasets.

Provides dataset loaders, corruptions list, custom dataset wrappers,
and download routines for robustness evaluation.
"""

import logging
import os
import tarfile
import urllib.request
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# CIFAR-10 class labels
CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Standard CIFAR-10-C corruption types (19 corruptions)
CIFAR10C_CORRUPTIONS: List[str] = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
]

# Canonical CIFAR-10 normalization constants
CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

CIFAR10C_DOWNLOAD_URL: str = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"


def get_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Build standard training and testing transforms for CIFAR-10.

    Returns:
        Tuple of (train_transform, test_transform).
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    return train_transform, test_transform


class CIFAR10CDataset(Dataset):
    """Custom Dataset wrapper for CIFAR-10-C corruptions stored in .npy files.

    Handles both pre-loaded numpy arrays or paths to .npy files,
    supporting severity-based slicing (1-5) and custom PyTorch transforms.
    """

    def __init__(
        self,
        data: Union[str, np.ndarray],
        labels: Union[str, np.ndarray],
        severity: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize CIFAR10CDataset.

        Args:
            data: Path to corruption .npy file or loaded numpy array of shape (N, 32, 32, 3).
            labels: Path to labels .npy file or loaded numpy array of shape (N,).
            severity: Corruption severity level (1 to 5). If specified and data
                contains all 50,000 samples (5 levels * 10,000), slices the corresponding 10,000.
            transform: PyTorch transform pipeline. If None, default ToTensor and
                CIFAR-10 normalization are applied.

        Raises:
            ValueError: If severity is not in range [1, 5] or shapes mismatch.
            FileNotFoundError: If file paths are invalid.
        """
        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"CIFAR-10-C data file not found: {data}")
            self.data = np.load(data)
        else:
            self.data = data

        if isinstance(labels, str):
            if not os.path.exists(labels):
                raise FileNotFoundError(f"CIFAR-10-C labels file not found: {labels}")
            self.labels = np.load(labels)
        else:
            self.labels = labels

        # Validate and apply severity slicing if needed
        if severity is not None:
            if not (1 <= severity <= 5):
                raise ValueError(f"Severity must be between 1 and 5, got {severity}")

            # Standard CIFAR-10-C full corruption files have 50,000 samples (10,000 per severity)
            if self.data.shape[0] == 50000:
                start_idx = (severity - 1) * 10000
                end_idx = severity * 10000
                self.data = self.data[start_idx:end_idx]
                if self.labels.shape[0] == 50000:
                    self.labels = self.labels[start_idx:end_idx]
            elif self.data.ndim == 5 and self.data.shape[0] == 5:
                # Shape is (5, 10000, 32, 32, 3)
                self.data = self.data[severity - 1]

        # If labels has 50000 elements but data has 10000, take the first 10000
        if self.labels.shape[0] == 50000 and self.data.shape[0] == 10000:
            self.labels = self.labels[:10000]

        if len(self.data) != len(self.labels):
            raise ValueError(
                f"Data size ({len(self.data)}) does not match labels size ({len(self.labels)})"
            )

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Fetch image tensor and label at the specified index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label).
        """
        img = self.data[idx]
        label = int(self.labels[idx])

        # img shape is (32, 32, 3) uint8 numpy array
        if self.transform is not None:
            img = self.transform(img)

        return img, label


def get_cifar10_loaders(
    batch_size: int = 128,
    val_split: float = 0.1,
    seed: int = 42,
    data_dir: str = "./data",
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders for CIFAR-10.

    Ensures validation data is evaluated without data augmentations while
    training data uses random cropping and horizontal flipping.

    Args:
        batch_size: Mini-batch size for DataLoader.
        val_split: Fraction of training dataset reserved for validation (0.0 to 1.0).
        seed: Random seed for reproducible dataset splitting.
        data_dir: Base directory where CIFAR-10 data is stored or downloaded.
        num_workers: Number of subprocesses for data loading.
        pin_memory: Whether DataLoader pins memory to CUDA page-locked memory.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    os.makedirs(data_dir, exist_ok=True)
    train_transform, test_transform = get_cifar10_transforms()

    # Load CIFAR-10 train set with training transforms and with eval transforms
    raw_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    raw_val_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=test_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    num_total_train = len(raw_train_dataset)
    num_val = int(val_split * num_total_train)
    num_train = num_total_train - num_val

    # Deterministic split indices
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_total_train, generator=generator).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_subset = Subset(raw_train_dataset, train_indices)
    val_subset = Subset(raw_val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        "Loaded CIFAR-10: train=%d, val=%d, test=%d (batch_size=%d)",
        len(train_subset),
        len(val_subset),
        len(test_dataset),
        batch_size,
    )

    return train_loader, val_loader, test_loader


def download_cifar10c(data_dir: str = "./data") -> str:
    """Download and extract CIFAR-10-C archive from Zenodo if not present.

    Args:
        data_dir: Base directory where CIFAR-10-C should be located.

    Returns:
        Absolute path to the extracted CIFAR-10-C directory.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    cifar10c_dir = os.path.join(data_dir, "CIFAR-10-C")
    if os.path.isdir(cifar10c_dir) and os.path.exists(
        os.path.join(cifar10c_dir, "labels.npy")
    ):
        logger.info("CIFAR-10-C dataset already present at %s", cifar10c_dir)
        return cifar10c_dir

    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "CIFAR-10-C.tar")

    logger.info("Downloading CIFAR-10-C from %s to %s", CIFAR10C_DOWNLOAD_URL, tar_path)
    try:
        urllib.request.urlretrieve(CIFAR10C_DOWNLOAD_URL, tar_path)
        logger.info("Extracting %s...", tar_path)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=data_dir)
        logger.info("CIFAR-10-C successfully extracted to %s", cifar10c_dir)
        if os.path.exists(tar_path):
            os.remove(tar_path)
    except Exception as exc:
        logger.error("Failed to download or extract CIFAR-10-C: %s", exc)
        raise RuntimeError(f"CIFAR-10-C download failed: {exc}") from exc

    return cifar10c_dir


def get_cifar10c_loader(
    corruption_type: str,
    severity: int,
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
    pin_memory: bool = True,
    auto_download: bool = False,
) -> DataLoader:
    """Create DataLoader for a specific CIFAR-10-C corruption type and severity.

    Args:
        corruption_type: Name of the corruption (e.g., 'gaussian_noise').
        severity: Corruption severity level from 1 to 5.
        batch_size: Mini-batch size.
        data_dir: Base directory containing CIFAR-10-C/ folder.
        num_workers: Number of workers for DataLoader.
        pin_memory: Whether DataLoader pins memory to CUDA memory.
        auto_download: Whether to attempt automatic download if missing.

    Returns:
        DataLoader for the corrupted dataset.

    Raises:
        ValueError: If corruption_type or severity is invalid.
        FileNotFoundError: If the corruption file or labels file is not found.
    """
    if corruption_type not in CIFAR10C_CORRUPTIONS:
        raise ValueError(
            f"Unknown corruption type '{corruption_type}'. Expected one of: {CIFAR10C_CORRUPTIONS}"
        )
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be an integer between 1 and 5, got {severity}")

    cifar10c_dir = os.path.join(data_dir, "CIFAR-10-C")
    data_file = os.path.join(cifar10c_dir, f"{corruption_type}.npy")
    labels_file = os.path.join(cifar10c_dir, "labels.npy")

    if not os.path.exists(data_file) or not os.path.exists(labels_file):
        if auto_download:
            download_cifar10c(data_dir=data_dir)
        else:
            raise FileNotFoundError(
                f"CIFAR-10-C files not found at {cifar10c_dir}. "
                f"Run download_cifar10c('{data_dir}') or set auto_download=True."
            )

    dataset = CIFAR10CDataset(
        data=data_file,
        labels=labels_file,
        severity=severity,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        "Loaded CIFAR-10-C '%s' (severity=%d): %d samples",
        corruption_type,
        severity,
        len(dataset),
    )
    return loader
