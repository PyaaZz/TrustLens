# Data Directory

This directory stores datasets used by TrustLens.

## CIFAR-10
Automatically downloaded by torchvision on first run.

## CIFAR-10-C
Corrupted versions of CIFAR-10 test images for distribution shift evaluation.

### Automatic Download
```bash
python -c "from src.data_loader import download_cifar10c; download_cifar10c('./data')"
```

### Manual Download
1. Download from: https://zenodo.org/record/2535967
2. Extract to: `data/CIFAR-10-C/`
3. The directory should contain 19 `.npy` files (one per corruption type) and `labels.npy`

### Corruption Types
- **Noise**: gaussian_noise, shot_noise, impulse_noise, speckle_noise
- **Blur**: defocus_blur, glass_blur, motion_blur, zoom_blur, gaussian_blur
- **Weather**: snow, frost, fog, brightness, spatter
- **Digital**: contrast, elastic_transform, pixelate, jpeg_compression, saturate

Each corruption file contains 50,000 images (10,000 images × 5 severity levels).
