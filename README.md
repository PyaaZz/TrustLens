# TrustLens: AI Prediction Reliability & Failure Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)

Standard neural networks are notoriously overconfident. Under distribution shift—such as blur, bad weather, or sensor noise—a classifier will frequently output a completely wrong prediction with 95%+ confidence. 

**TrustLens** is a monitoring system that flags when a model's predictions shouldn't be trusted. Instead of relying only on raw softmax confidence, it extracts and combines multiple uncertainty signals (softmax entropy, free energy, temperature calibration, and penultimate feature distance) to catch failures before they reach production decisions.

---

## The Problem: Confident Failures

When testing a ResNet-18 trained on CIFAR-10 against corrupted images (CIFAR-10-C):
- Accuracy drops sharply as corruption severity increases (from ~95% down to <60%).
- However, the model still outputs high confidence scores on wrong answers.
- Simple confidence thresholding (MSP) misses many of these failures.

TrustLens extracts complementary signals from different stages of the forward pass to detect when an input has drifted away from familiar training data.

---

## How It Works

TrustLens combines four distinct uncertainty indicators:

| Signal | Source | What it Measures |
| :--- | :--- | :--- |
| **Max Softmax Probability (MSP)** | Output layer | Peak confidence of the model's top prediction. Fast baseline, but prone to overconfidence. |
| **Predictive Entropy** | Probability vector | How spread out the predictions are across all classes. High entropy indicates confusion. |
| **Free Energy Score** | Raw logits | Negative log-sum-exp of logits. Aligns with input density and doesn't suffer from softmax normalization squash. |
| **Mahalanobis Feature Distance** | Penultimate layer | Geometric distance of the image embedding to the nearest training class cluster centroid. Catches covariate shift. |
| **Temperature Scaling** | Post-processing | Recalibrates output probabilities using a learned validation temperature $T$ to reduce Expected Calibration Error (ECE). |

A lightweight failure detector combines these signals to assign each prediction a risk score and a clear status:
- 🟢 **Pass / Reliable:** High confidence, low entropy, sits inside known feature clusters.
- 🟡 **Caution / Degraded:** Elevated entropy or feature drift detected. Human oversight recommended.
- 🔴 **Reject / High Risk:** Severe distribution shift or out-of-distribution input. Automatic decision withheld.

---

## Project Structure

```
TrustLens/
├── configs/             # Experiment parameters and data paths
├── dashboard/           # Streamlit web app (interactive UI)
├── data/                # Dataset loaders and CIFAR-10-C scripts
├── models/              # ResNet-18 architecture adapted for 32x32 images
├── src/                 # Core algorithms: signals, calibration, metrics, detectors
├── experiments/         # Standalone scripts for baseline, shift tests, ablation
├── results/             # Saved evaluation metrics (JSON) and figures (PNG)
└── tests/               # 33 unit tests covering signals, metrics, and calibration
```

---

## Quickstart

### 1. Setup Environment
```bash
git clone https://github.com/your-username/TrustLens.git
cd TrustLens

# Create virtual environment
python -m venv .venv

# Activate environment:
# Windows (cmd):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation
Run the test suite to ensure all signal calculations and calibration routines pass:
```bash
pytest tests/ -v
```

### 3. Launch the Dashboard
Run the interactive Streamlit interface to test custom images and inspect corruption experiments:
```bash
streamlit run dashboard/app.py
```
*(On first run, the app will automatically fetch the pretrained 94.5% ResNet-18 checkpoint).*

---

## Running Experiments

You can reproduce all paper-style benchmarks individually or end-to-end:

### Train the Baseline Classifier (Week 1)
Trains ResNet-18 on clean CIFAR-10 and saves checkpoints and confusion matrices:
```bash
python experiments/baseline.py --epochs 100 --batch-size 128 --device auto
```

### Test Against Distribution Shift (Week 2)
Evaluates accuracy degradation across all 19 CIFAR-10-C corruptions and logs confident failures:
```bash
python -c "from src.data_loader import download_cifar10c; download_cifar10c('./data')"
python experiments/corruption_test.py --device auto
```

### Evaluate Reliability Signals (Week 3)
Computes AUROC, AUPR, and FPR@95%TPR for each individual signal and fits temperature scaling:
```bash
python experiments/evaluate.py --device auto
```

### Run Ablation & Generalization (Weeks 4–5)
Trains the combined failure detector, runs leave-one-out ablations, and tests generalization on held-out corruption modes (`frost`, `fog`, `brightness`):
```bash
python experiments/ablation.py --device auto
```

### Run Everything in Sequence
```bash
python experiments/run_all.py --config configs/experiment.yaml --device auto
```

---

## Key Experimental Takeaways

- **Single signals aren't enough:** MSP performs reasonably on clean data but fails on heavy blur and noise. Feature distance excels at detecting corrupted inputs but is less sensitive to borderline cases. Combining them yields higher AUROC.
- **Generalization to unseen shifts:** The combined detector trained on 16 corruption types maintains >92% AUROC when tested on completely unseen corruption families (frost, fog).
- **Low latency:** Extracting feature distances and computing entropy adds under ~2.5 ms per sample on CPU, making it practical for real-time inference monitoring.

---

## References

- Hendrycks & Gimpel (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*. [arXiv:1610.02136](https://arxiv.org/abs/1610.02136)
- Guo et al. (2017). *On Calibration of Modern Neural Networks*. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- Lee et al. (2018). *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks*. [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)
- Hendrycks & Dietterich (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. [arXiv:1903.12261](https://arxiv.org/abs/1903.12261)
- Liu et al. (2020). *Energy-based Out-of-distribution Detection*. [arXiv:2010.03759](https://arxiv.org/abs/2010.03759)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
