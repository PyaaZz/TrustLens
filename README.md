# TrustLens — AI Prediction Reliability Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)

**TrustLens** is a framework for detecting when deep learning classifiers are likely to make mistakes. Rather than blindly trusting raw softmax probabilities under distribution shift or out of distribution (OOD) inputs, TrustLens combines multilayer epistemic uncertainty indicators Max Softmax Probability (MSP), predictive entropy, free energy score, post hoc temperature calibration, and deep Mahalanobis/KNN feature space distances into a unified failure detector that flags untrustworthy predictions before they compromise downstream decision systems.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |             Input Image               |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      ResNet-18 Vision Backbone        |
                      +---------------------------------------+
                                    /           \
               Deep Feature Vectors/             \ Raw Logits
                                  /               \
                                 v                 v
                 +-------------------+     +-------------------+
                 | Feature Extractor |     | Softmax / Scaling |
                 +-------------------+     +-------------------+
                           |                         |
                           v                         v
            +-----------------------------+   +-----------------------------+
            |  Deep Feature Distances     |   | Probability & Energy Signal |
            |  - Class Mahalanobis dist   |   | - Max Softmax Prob (MSP)    |
            |  - KNN Nearest Neighbor     |   | - Predictive Entropy        |
            +-----------------------------+   | - Free Energy Score         |
                           \                  | - Temperature Calibrated ECE|
                            \                 +-----------------------------+
                             \                       /
                              v                     v
                     +---------------------------------------+
                     |        Failure Detector Engine        |
                     |  - Logistic / Ridge / Multi-signal    |
                     |  - Threshold calibrated @ Target FPR  |
                     +---------------------------------------+
                                          |
                                          v
                     +---------------------------------------+
                     |    Prediction Reliability Report      |
                     |  [PASS / RELIABLE]  (Green)           |
                     |  [WARNING / DEGRADED] (Yellow)        |
                     |  [REJECT / UNRELIABLE] (Red)          |
                     +---------------------------------------+
```

---

## Key Features

- **Multi-Signal Reliability Profiling**:
  - **Max Softmax Probability (MSP)**: Baseline confidence indicator ([Hendrycks & Gimpel, 2017](https://arxiv.org/abs/1610.02136)).
  - **Predictive Shannon Entropy**: Dispersal of predicted class probability distributions.
  - **Free Energy Score**: Unnormalized negative free energy tracking log-sum-exp density ([Liu et al., 2020](https://arxiv.org/abs/2010.03759)).
  - **Deep Feature Space Distance**: Mahalanobis distance to class-conditional Gaussian centroids with shared covariance ([Lee et al., 2018](https://arxiv.org/abs/1807.03888)) and k-nearest neighbors.
  - **Temperature-Calibrated Probabilities**: Platt-scaled logits optimizing negative log-likelihood for well-calibrated ECE ([Guo et al., 2017](https://arxiv.org/abs/1706.04599)).
- **Distribution Shift Evaluation**: Systematic benchmarking on CIFAR-10-C across 19 corruption categories at 5 severity levels ([Hendrycks & Dietterich, 2019](https://arxiv.org/abs/1903.12261)).
- **Actionable Thresholding**: Configurable operating points tuned to fixed False Positive Rates (e.g., FPR@95TPR) to minimize safety-critical false acceptances.
- **Interactive Streamlit Dashboard**: Full GUI for real-time single-sample inference, corruption sensitivity heatmaps, ROC/PR curves, and ablation studies.
- **Modular & Extensible**: Modular structure decoupling datasets, models, signal extractors, calibration, and evaluators.

---

## Project Structure

```
TrustLens/
├── configs/
│   └── experiment.yaml          # Hyperparameters and pipeline configuration
├── dashboard/
│   └── app.py                   # Streamlit interactive web dashboard
├── data/
│   └── README.md                # Dataset download instructions (CIFAR-10, CIFAR-10-C)
├── models/
│   ├── __init__.py              # Model interfaces
│   ├── classifier.py            # ResNet-18 architecture, training & evaluation loops
│   └── checkpoints/             # Trained model weights (.pt/.pth)
├── src/
│   ├── __init__.py              # Unified package exports
│   ├── calibration.py           # Temperature scaling, ECE, reliability diagrams
│   ├── data_loader.py           # CIFAR-10 and CIFAR-10-C loaders & down-loaders
│   ├── failure_detector.py      # Failure detection classifiers and thresholding
│   ├── feature_detector.py      # Feature extraction and Mahalanobis/KNN distances
│   ├── inference.py             # Single-sample and batch inference routines
│   ├── metrics.py               # AUROC, AUPR, FPR@95, ECE, MetricsTracker
│   └── signals.py               # Signal calculations (MSP, Entropy, Energy)
├── tests/
│   ├── __init__.py
│   ├── test_calibration.py      # Calibration and ECE unit tests
│   ├── test_metrics.py          # Detection metrics unit tests
│   └── test_signals.py          # Uncertainty signals unit tests
├── results/
│   ├── figures/                 # Generated ROC, PR, and corruption heatmap plots
│   └── metrics/                 # JSON/CSV evaluation metrics
├── .gitignore                   # Git ignore patterns
├── README.md                    # Project documentation
└── requirements.txt             # Project Python dependencies
```

---

## Quick Start

### 1. Installation
Clone repository and install requirements in your environment:
```bash
git clone https://github.com/your-username/TrustLens.git
cd TrustLens
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Dataset Preparation
CIFAR-10 is automatically downloaded by torchvision upon running inference or training.
For CIFAR-10-C corruptions:
```bash
python -c "from src.data_loader import download_cifar10c; download_cifar10c('./data')"
```
Alternatively, follow manual instructions in [`data/README.md`](data/README.md).

### 3. Training the Base Model
Train the ResNet-18 classifier on clean CIFAR-10:
```bash
python -m models.classifier --train --epochs 100 --lr 0.1
```

### 4. Running Failure & OOD Detection Experiments
Evaluate reliability signals across clean and corrupted sets:
```bash
python -m src.failure_detector --config configs/experiment.yaml
```

### 5. Launching the Interactive Dashboard
Launch the TrustLens Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

### 6. Running Unit Tests
Execute the pytest suite:
```bash
pytest tests/ -v
```

---

## Methodology

TrustLens operates on the premise that no single heuristic is universally reliable across all failure modes:

| Signal | Source | Formulation | Strengths & Failure Modes |
| :--- | :--- | :--- | :--- |
| **Max Softmax Probability (MSP)** | Softmax output | $\max_k \frac{e^{z_k}}{\sum_j e^{z_j}}$ | Computationally trivial; suffers from softmax overconfidence on far-OOD inputs. |
| **Predictive Entropy** | Softmax distribution | $-\sum_k p_k \log p_k$ | Measures distribution flatness; high near decision boundaries, but can stay artificially low on far-OOD. |
| **Free Energy Score** | Raw logits | $-T \cdot \log \sum_k e^{z_k / T}$ | Theoretically aligned with input density; robust against softmax normalization distortion. |
| **Mahalanobis Feature Distance** | Penultimate layer | $\min_c (h(x) - \mu_c)^T \Sigma^{-1} (h(x) - \mu_c)$ | Measures true geometric distance in feature space; highly sensitive to distribution shifts and covariate shift. |
| **Temperature Calibration** | Post-hoc scaling | $\sigma(z / T)$ | Recalibrates confidence estimates on clean val set to reflect true posterior probability. |

By fusing these signals through a calibrated detector (e.g. regularized logistic regression on validation misclassifications), TrustLens identifies errors caused by:
1. Borderline instances near decision boundaries.
2. In-distribution model misclassifications.
3. Natural corruptions and covariate shifts (blur, weather, digital noise).
4. Out-of-distribution instances.

---

## Experiments & Benchmarks

1. **Clean Test Baseline**: Baseline classification accuracy and expected calibration error (ECE) before and after temperature scaling.
2. **CIFAR-10-C Robustness**: Performance degradation across 19 corruption types with 5 severity tiers.
3. **Signal Discriminative Power**: AUROC, AUPR, and FPR@95TPR for detecting incorrect classifications.
4. **Generalization to Unseen Corruptions**: Train failure detector on 16 corruptions and test zero-shot on 3 held-out corruption modes (`frost`, `fog`, `brightness`).
5. **Ablation Studies**: Evaluating individual signal performance versus joint multi-signal ensembles.
6. **Inference Latency Tradeoff**: Overhead of feature extraction and distance calculations versus standard inference.

---

## Results & Visualizations

Pre-computed evaluation summaries and plots are stored under `results/`:
- **ROC and Precision-Recall Curves**: Comparison of individual signals vs. TrustLens ensemble.
- **Calibration Diagrams**: Pre- and post-temperature scaling reliability curves.
- **Corruption Breakdown Heatmap**: Performance stability matrix across corruption categories and severity levels.

*(Launch the Streamlit dashboard via `streamlit run dashboard/app.py` for interactive exploration of all benchmark results and live sample diagnostics).*

---

## Limitations

1. **Feature Covariance Inversion**: For ultra-high dimensional representations, calculating full precision empirical covariance matrices can be ill-conditioned without shrinkage or diagonal approximations.
2. **Computational Overhead**: Computing nearest neighbor or full Mahalanobis distances adds inference latency compared to simple forward-pass softmax readouts.
3. **Severe Distribution Shift**: Under extreme semantic shifts, deep feature spaces can undergo unpredictable representations if the backbone has poor domain generalization.

---

## References

1. **Hendrycks, D., & Gimpel, K.** (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*. ICLR 2017. [arXiv:1610.02136](https://arxiv.org/abs/1610.02136)
2. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q.** (2017). *On Calibration of Modern Neural Networks*. ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
3. **Lee, K., Lee, K., Lee, H., & Shin, J.** (2018). *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks*. NeurIPS 2018. [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)
4. **Hendrycks, D., & Dietterich, T.** (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR 2019. [arXiv:1903.12261](https://arxiv.org/abs/1903.12261)
5. **Liu, W., Wang, X., Owens, J., & Li, Y.** (2020). *Energy-based Out-of-distribution Detection*. NeurIPS 2020. [arXiv:2010.03759](https://arxiv.org/abs/2010.03759)

---

## License

This project is licensed under the [MIT License](LICENSE).
