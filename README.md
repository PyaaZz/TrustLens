# TrustLens

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)
![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B.svg)

Image classifiers are bad at knowing when they're wrong. If you take a ResNet-18 trained on CIFAR-10 and feed it noisy, blurred, or shifted images (like CIFAR-10-C), accuracy drops fast—but the model still outputs 90%+ confidence on bad predictions.

**TrustLens** is a project exploring how to catch those silent failures before acting on them. Instead of just thresholding the top softmax score, it pulls a few different signals from the model during inference and feeds them into a lightweight failure detector.

## What it looks at

When an image passes through the ResNet-18 backbone, we extract:

* **Max Softmax Probability (MSP):** The standard baseline—just the top class probability. Fast, but easily fooled by shifted inputs.
* **Predictive Entropy:** Measures how flat the output distribution is across all 10 classes.
* **Energy Score:** Computed directly from the raw logits (`-T * logsumexp(z / T)`) so we don't lose density information to the softmax squash.
* **Temperature Scaling:** A single scalar `T` fit on the validation set to calm down overconfident logits and bring down Expected Calibration Error (ECE).
* **Feature Distance (Mahalanobis / KNN):** Measures how far the penultimate-layer embedding (`512-d` vector before the final linear layer) sits from the training class centroids. This does most of the heavy lifting when images get corrupted.

A logistic regression detector combines these signals and flags each prediction as **Reliable (Pass)**, **Uncertain (Caution)**, or **Unreliable (Reject)**.

## Repo layout

```text
TrustLens/
├── configs/experiment.yaml   # default hyperparams and corruption splits
├── models/classifier.py      # ResNet-18 modified for 32x32 CIFAR inputs
├── src/
│   ├── data_loader.py        # CIFAR-10 and CIFAR-10-C dataset wrappers
│   ├── signals.py            # MSP, entropy, energy, feature distance math
│   ├── calibration.py        # temperature scaling + ECE / Brier score
│   ├── feature_detector.py   # penultimate layer hook + Mahalanobis/KNN
│   ├── failure_detector.py   # combined failure classifier
│   └── metrics.py            # AUROC, AUPR, FPR@95TPR
├── experiments/              # scripts to run the benchmarks and save plots
├── dashboard/app.py          # Streamlit UI for testing images & viewing plots
└── tests/                    # pytest suite for the signal/metric math
```

## Setup & Usage

Clone the repo and install dependencies:

```bash
git clone https://github.com/PyaaZz/TrustLens.git
cd TrustLens
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run the dashboard

The quickest way to try it out is the Streamlit app. If you don't have local weights trained yet, it will automatically grab a pretrained CIFAR-10 ResNet-18 checkpoint (~94.5% clean accuracy) on startup.

```bash
streamlit run dashboard/app.py
```

You can upload custom images or pick preset corruptions to see how the softmax distribution, entropy, and Mahalanobis distance react in real time.

### Run the experiments

If you want to run the full pipeline from scratch:

```bash
# 1. Train or evaluate baseline ResNet-18 on clean CIFAR-10
python experiments/baseline.py --epochs 100 --batch-size 128 --device auto

# 2. Download CIFAR-10-C (~2.6 GB) and run corruption benchmarks
python -c "from src.data_loader import download_cifar10c; download_cifar10c('./data')"
python experiments/corruption_test.py --device auto

# 3. Compare individual uncertainty signals + temperature scaling
python experiments/evaluate.py --device auto

# 4. Run ablation study & test on held-out corruptions (frost, fog, brightness)
python experiments/ablation.py --device auto
```

Or run all steps in order using `python experiments/run_all.py --config configs/experiment.yaml`.

To run the unit tests:

```bash
pytest tests/ -v
```

## Key findings & notes

* **Important split rule:** The failure detector is never trained on the test set or on all corruption types. We hold out entire corruption families (`frost`, `fog`, `brightness`) during detector fitting to verify that it generalizes to unseen shifts rather than just memorizing specific noise patterns.
* **Why combine signals?** MSP and entropy work well for borderline in-distribution mistakes (e.g. a cat that looks like a dog), while Mahalanobis distance is much better at catching severe image corruptions (heavy blur, noise) where softmax still spikes on the wrong class.
* **Tradeoffs:** Computing Mahalanobis/KNN distances adds ~2ms of overhead per batch and requires storing class centroids/covariance from the training set, whereas MSP and entropy are essentially free.

## References

* Hendrycks & Gimpel (2017) – *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*
* Guo et al. (2017) – *On Calibration of Modern Neural Networks*
* Lee et al. (2018) – *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks*
* Hendrycks & Dietterich (2019) – *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*
* Liu et al. (2020) – *Energy-based Out-of-distribution Detection*
